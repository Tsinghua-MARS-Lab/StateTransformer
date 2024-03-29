import pickle
import torch
import torch.nn as nn
import numpy as np
import copy
import os
import re
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import  nested_detach
from transformers.trainer_callback import TrainerState, TrainerControl, IntervalStrategy, DefaultFlowCallback
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import EvalPrediction
from typing import List, Optional, Dict, Any, Tuple, Union
from transformer4planning.utils.nuplan_utils import compute_scores
from transformer4planning.utils.nuplan_utils import normalize_angle
from sklearn.metrics import accuracy_score

FDE_THRESHHOLD = 8 # keep same with nuplan simulation
ADE_THRESHHOLD = 8 # keep same with nuplan simulation
HEADING_ERROR_THRESHHOLD = 0.8 # keep same with nuplan simulation
MISS_RATE_THRESHHOLD = 0.3
MISS_THRESHHOLD = [6, 8, 16]
DISPLACEMENT_WEIGHT = 1
HEADING_WEIGHT = 2
EVAL_LOG_SAVING_PATH = None

def convert_names_to_ids(file_names, t0_frame_ids):
    unique_scenario_ids = list()
    for i, file_name in enumerate(file_names):
        # example file name for nuplan: 2021.06.07.17.49.04_veh-47_02526_02700
        # will convert to 470252602700
        file_names = re.split(r',|-|_', file_name)
        file_numbers = [int(x) for x in file_names if x.isdigit()]
        # convert a list of number into one number
        file_numbers = int("".join(map(str, file_numbers)))
        digits = len(str(int(t0_frame_ids[i])))
        unique_scenario_id = file_numbers * (10 ** digits) + t0_frame_ids[i]
        unique_scenario_ids.append(int(unique_scenario_id))
    return unique_scenario_ids

def batch_angle_normalize(headings):
    for i in range(headings.shape[0]):
        for j in range(headings.shape[1]):
            headings[i, j] = normalize_angle(headings[i, j])
    return headings

# Custom compute_metrics function
def compute_metrics(prediction: EvalPrediction):
    """
    All inputs are finalized and gathered, should run only one time.
    """
    # TODO: Adapt to waymo
    # TODO: Add classification metrics (clf_metrics) for scoring
    eval_result = {}
    item_to_save = {}

    predictions = prediction.predictions
    labels = prediction.label_ids  # nparray: sample_num, 80, 4
    prediction_horizon = labels.shape[1]

    prediction_by_generation = predictions['prediction_generation']  # sample_num, 85, 2/4
    prediction_by_forward = predictions['prediction_forward']  # sample_num, 85, 2/4
    frame_id = predictions['frame_id']
    scenario15s_id = predictions['scenario15s_id']
    item_to_save["scenario15s_id"] = scenario15s_id
    item_to_save["frame_id"] = frame_id

    # select gt key points from gt trajectories
    if 'key_points_logits' in prediction_by_generation:
        selected_indices = predictions['selected_indices'][0]  # 5

    loss_items = predictions['loss_items']
    # check loss items are dictionary
    if isinstance(loss_items, dict):
        for each_key in loss_items:
            # get average loss for each key
            eval_result[each_key] = np.mean(loss_items[each_key])

    # fetch trajectory and key points predictions
    if 'key_points_logits' in prediction_by_generation:
        # first 5 are key points concatentate with trajectory
        prediction_key_points_by_generation = prediction_by_generation["key_points_logits"]  # sample_num, 5, 2/4
        assert prediction_by_forward['traj_logits'].shape[1] == prediction_horizon, f'{prediction_by_forward["traj_logits"].shape[1]} {prediction_horizon}'
        prediction_key_points_by_forward = prediction_by_forward['kp_logits']  # sample_num, 5, 2/4

    prediction_trajectory_by_generation = prediction_by_generation["traj_logits"] # sample_num, 80, 2/4
    prediction_trajectory_by_forward = prediction_by_forward['traj_logits']  # sample_num, 80, 2/4

    # calculate error for generation results
    if 'key_points_logits' in prediction_by_generation:
        assert len(selected_indices) > 1, selected_indices
        label_key_points = labels[:, selected_indices, :]
        # WIP
        if len(selected_indices) > 10:
            ade_x_error_key_points_gen = prediction_key_points_by_generation[:, 10:, 0] - label_key_points[:, 10:, 0]
            ade_y_error_key_points_gen = prediction_key_points_by_generation[:, 10:, 1] - label_key_points[:, 10:, 1]
        else:
            ade_x_error_key_points_gen = prediction_key_points_by_generation[:, :, 0] - label_key_points[:, :, 0]
            ade_y_error_key_points_gen = prediction_key_points_by_generation[:, :, 1] - label_key_points[:, :, 1]
        if selected_indices[0] > selected_indices[-1]:
            # backward
            fde_x_error_key_points_gen = prediction_key_points_by_generation[:, 0, 0] - label_key_points[:, 0, 0]
            fde_y_error_key_points_gen = prediction_key_points_by_generation[:, 0, 1] - label_key_points[:, 0, 1]
        elif selected_indices[0] < selected_indices[-1]:
            # forward
            fde_x_error_key_points_gen = prediction_key_points_by_generation[:, -1, 0] - label_key_points[:, -1, 0]
            fde_y_error_key_points_gen = prediction_key_points_by_generation[:, -1, 1] - label_key_points[:, -1, 1]
        ade_key_points_gen = np.sqrt(ade_x_error_key_points_gen ** 2 + ade_y_error_key_points_gen ** 2).mean()
        eval_result['ade_keypoints_gen'] = ade_key_points_gen
        fde_key_points_gen = np.sqrt(fde_x_error_key_points_gen ** 2 + fde_y_error_key_points_gen ** 2).mean()
        eval_result['fde_keypoints_gen'] = fde_key_points_gen

    ade_x_error_gen = prediction_trajectory_by_generation[:, :, 0] - labels[:, :, 0]
    ade_y_error_gen = prediction_trajectory_by_generation[:, :, 1] - labels[:, :, 1]
    fde_x_error_gen = prediction_trajectory_by_generation[:, -1, 0] - labels[:, -1, 0]
    fde_y_error_gen = prediction_trajectory_by_generation[:, -1, 1] - labels[:, -1, 1]

    # ADE metrics computation
    ade_gen = np.sqrt(copy.deepcopy(ade_x_error_gen) ** 2 + copy.deepcopy(ade_y_error_gen) ** 2)
    ade3_gen = np.mean(copy.deepcopy(ade_gen[:, :30]), axis=1)
    ade5_gen = np.mean(copy.deepcopy(ade_gen[:, :50]), axis=1)
    ade8_gen = np.mean(copy.deepcopy(ade_gen[:, :80]), axis=1)
    avg_ade_gen = (ade3_gen + ade5_gen + ade8_gen)/3
    ade_score = np.ones_like(avg_ade_gen) - avg_ade_gen/ADE_THRESHHOLD
    ade_score = np.where(ade_score < 0, np.zeros_like(ade_score), ade_score)
    item_to_save["ade_score"] = ade_score
    item_to_save['ade_horizon3_gen'] = ade3_gen
    item_to_save['ade_horizon5_gen'] = ade5_gen
    item_to_save['ade_horizon8_gen'] = ade8_gen
    eval_result['ade_horizon3_gen'] = ade3_gen.mean()
    eval_result['ade_horizon5_gen'] = ade5_gen.mean()
    eval_result['ade_horizon8_gen'] = ade8_gen.mean()
    eval_result['metric_ade'] = avg_ade_gen.mean()
    eval_result['ade_score'] = ade_score.mean()

    # FDE metrics computation
    fde3_gen = copy.deepcopy(ade_gen[:, 29])
    fde5_gen = copy.deepcopy(ade_gen[:, 49])
    fde8_gen = np.sqrt(fde_x_error_gen ** 2 + fde_y_error_gen ** 2)
    avg_fde_gen = (fde3_gen + fde5_gen + fde8_gen)/3
    fde_score = np.ones_like(avg_fde_gen) - avg_fde_gen/FDE_THRESHHOLD
    fde_score = np.where(fde_score < 0, np.zeros_like(fde_score), fde_score)
    item_to_save["fde_score"] = fde_score
    item_to_save['fde_horizon3_gen'] = fde3_gen
    item_to_save['fde_horizon5_gen'] = fde5_gen
    item_to_save['fde_horizon8_gen'] = fde8_gen
    eval_result['fde_horizon3_gen'] = fde3_gen.mean()
    eval_result['fde_horizon5_gen'] = fde5_gen.mean()
    eval_result['fde_horizon8_gen'] = fde8_gen.mean()
    eval_result['metric_fde'] = avg_fde_gen.mean()
    eval_result['fde_score'] = fde_score.mean()

    def normalize_angles(angles):
        return np.arctan2(np.sin(angles), np.cos(angles))

    # heading error
    if prediction_trajectory_by_generation.shape[-1] == 4:
        # average heading error computation
        # heading_error_gen = abs(batch_angle_normalize(prediction_trajectory_by_generation[:, :, -1] - labels[:, :, -1]))
        heading_diff_gen = normalize_angles(prediction_trajectory_by_generation[:, :, -1]) - normalize_angles(labels[:, :, -1])
        # normalized_angles = np.fmod(heading_diff_gen + 2 * np.pi, 2 * np.pi)  # normalize to 0, 2pi
        # normalized_angles = np.where(normalized_angles > np.pi, normalized_angles - 2 * np.pi, normalized_angles)  # normalize to -pi, pi
        heading_error_gen = abs(normalize_angles(heading_diff_gen))

        ahe3_gen = np.mean(copy.deepcopy(heading_error_gen[:, :30]), axis=1)
        ahe5_gen = np.mean(copy.deepcopy(heading_error_gen[:, :50]), axis=1)
        ahe8_gen = np.mean(copy.deepcopy(heading_error_gen[:, :80]), axis=1)
        avg_ahe = (ahe3_gen + ahe5_gen + ahe8_gen)/3
        ahe_score = np.ones_like(avg_ahe) - avg_ahe/HEADING_ERROR_THRESHHOLD
        ahe_score = np.where(ahe_score < 0, np.zeros_like(ahe_score), ahe_score)
        item_to_save['ahe_score'] = ahe_score
        item_to_save['ahe_horizon3_gen'] = ahe3_gen
        item_to_save['ahe_horizon5_gen'] = ahe5_gen
        item_to_save['ahe_horizon8_gen'] = ahe8_gen
        eval_result['ahe_horizon3_gen'] = ahe3_gen.mean()
        eval_result['ahe_horizon5_gen'] = ahe5_gen.mean()
        eval_result['ahe_horizon8_gen'] = ahe8_gen.mean()
        eval_result['metric_ahe'] = avg_ahe.mean()
        eval_result['ahe_score'] = ahe_score.mean()

        # final heading error computation
        fhe3_gen = copy.deepcopy(heading_error_gen[:, 29])
        fhe5_gen = copy.deepcopy(heading_error_gen[:, 49])
        fhe8_gen = copy.deepcopy(heading_error_gen[:, 79])
        avg_fhe = (fhe3_gen + fhe5_gen + fhe8_gen)/3
        fhe_score = np.ones_like(avg_fhe) - avg_fhe/HEADING_ERROR_THRESHHOLD
        fhe_score = np.where(fhe_score < 0, np.zeros_like(fhe_score), fhe_score)
        item_to_save['fhe_score'] = fhe_score
        item_to_save['fhe_horizon3_gen'] = fhe3_gen
        item_to_save['fhe_horizon5_gen'] = fhe5_gen
        item_to_save['fhe_horizon8_gen'] = fhe8_gen
        eval_result['fhe_horizon3_gen'] = fhe3_gen.mean()
        eval_result['fhe_horizon5_gen'] = fhe5_gen.mean()
        eval_result['fhe_horizon8_gen'] = fhe8_gen.mean()
        eval_result['metric_fhe'] = avg_fhe.mean()
        eval_result['fhe_score'] = fhe_score.mean()
    
    # missing rate
    max_displacement3 = np.max(ade_gen[:, :30], axis=1)
    max_displacement5 = np.max(ade_gen[:, :50], axis=1)
    max_displacement8 = np.max(ade_gen[:, :80], axis=1)
    # miss = 1, not miss = 0
    miss3 = np.where(max_displacement3 > MISS_THRESHHOLD[0], np.ones_like(max_displacement3), np.zeros_like(max_displacement3))
    miss5 = np.where(max_displacement5 > MISS_THRESHHOLD[1], np.ones_like(max_displacement5), np.zeros_like(max_displacement5))
    miss8 = np.where(max_displacement8 > MISS_THRESHHOLD[2], np.ones_like(max_displacement8), np.zeros_like(max_displacement8))
    miss = np.where(miss3 + miss5 + miss8 >= 1, np.ones_like(miss3), np.zeros_like(miss3))
    item_to_save["miss_score"] = miss
    eval_result["miss_rate3"] = miss3.mean()
    eval_result["miss_rate5"] = miss5.mean()
    eval_result["miss_rate8"] = miss8.mean()
    eval_result["total_miss_rate"] = miss.mean()
    # compute error for forward results
    ade_x_error_for = prediction_trajectory_by_forward[:, :, 0] - labels[:, :, 0]
    ade_y_error_for = prediction_trajectory_by_forward[:, :, 1] - labels[:, :, 1]
    fde_x_error_for = prediction_trajectory_by_forward[:, -1, 0] - labels[:, -1, 0]
    fde_y_error_for = prediction_trajectory_by_forward[:, -1, 1] - labels[:, -1, 1]

    ade_for = np.sqrt(ade_x_error_for ** 2 + ade_y_error_for ** 2).mean()
    eval_result['ade_forward'] = ade_for
    fde_for = np.sqrt(fde_x_error_for ** 2 + fde_y_error_for ** 2).mean()
    eval_result['fde_forward'] = fde_for
    if 'key_points_logits' in prediction_by_generation:
        assert len(selected_indices) > 1, selected_indices
        if len(selected_indices) > 10:
            ade_x_error_key_points_for = prediction_key_points_by_forward[:, 10:, 0] - label_key_points[:, 10:, 0]
            ade_y_error_key_points_for = prediction_key_points_by_forward[:, 10:, 1] - label_key_points[:, 10:, 1]
        else:
            ade_x_error_key_points_for = prediction_key_points_by_forward[:, :, 0] - label_key_points[:, :, 0]
            ade_y_error_key_points_for = prediction_key_points_by_forward[:, :, 1] - label_key_points[:, :, 1]
        if selected_indices[0] > selected_indices[-1]:
            # forward
            fde_x_error_key_points_for = prediction_key_points_by_forward[:, 0, 0] - label_key_points[:, 0, 0]
            fde_y_error_key_points_for = prediction_key_points_by_forward[:, 0, 1] - label_key_points[:, 0, 1]
        elif selected_indices[0] < selected_indices[-1]:
            # backward
            fde_x_error_key_points_for = prediction_key_points_by_forward[:, -1, 0] - label_key_points[:, -1, 0]
            fde_y_error_key_points_for = prediction_key_points_by_forward[:, -1, 1] - label_key_points[:, -1, 1]
        ade_key_points_for = np.sqrt(ade_x_error_key_points_for ** 2 + ade_y_error_key_points_for ** 2).mean()
        eval_result['ade_keypoints_forward'] = ade_key_points_for
        fde_key_points_for = np.sqrt(fde_x_error_key_points_for ** 2 + fde_y_error_key_points_for ** 2).mean()
        eval_result['fde_keypoints_forward'] = fde_key_points_for

        if prediction_key_points_by_forward.shape[-1] == 4:
            heading_error_for = abs(prediction_key_points_by_forward[:, :, -1] - label_key_points[:, :, -1])
            eval_result['heading_error_forward'] = heading_error_for.mean()

    if 'proposal' in prediction_by_generation:
        if 'halfs_intention' in prediction_by_generation:
            # TODO: add waymo proposal accuracy to eval result
            # evaluate classification accuracy and log to eval_result
            # print('debuging trainer compute metrics: ', prediction_by_generation['proposal'], prediction_by_forward['proposal'], prediction_by_generation['halfs_intention'])
            prediction_proposal_class_by_generation = prediction_by_generation["proposal"]  # sample_num, 1
            proposal_labels = prediction_by_generation['halfs_intention']
            accuracy_by_generation = accuracy_score(y_true=proposal_labels.flatten(), y_pred=prediction_proposal_class_by_generation.flatten(), normalize=True)
            # print('inspect trainer compute_metrics: proposal accuracy: ', proposal_labels.flatten(), prediction_proposal_class_by_generation.flatten())
            eval_result['proposal_accuracy'] = accuracy_by_generation
        elif 'intentions' in prediction_by_generation:
            prediction_proposal_class_by_generation = prediction_by_generation["proposal"]  # sample_num, 16
            proposal_labels = prediction_by_generation['intentions']  # sample_num, 16
            accuracy_by_generation = accuracy_score(y_true=proposal_labels.flatten(), y_pred=prediction_proposal_class_by_generation.flatten(), normalize=True)
            # print('inspect trainer compute_metrics: proposal accuracy: ', proposal_labels, prediction_proposal_class_by_generation)
            eval_result['proposal_accuracy'] = accuracy_by_generation
        else:
            print('WARNING: no intention found in generation. ', list(prediction_by_generation.keys()))

    score, miss_score = compute_scores(item_to_save)
    eval_result["average_score"] = score
    eval_result["miss_score"] = miss_score

    # include inputs by passing args.include_inputs_for_metrics = True to save eval result to pickle
    if EVAL_LOG_SAVING_PATH is not None:
        print('Saving eval result to pickle file: ', EVAL_LOG_SAVING_PATH)
        eval_result_to_save = {}
        # get miss scenarios
        miss_indices = np.where(miss == 1)[0]
        scenario15s_id_miss = scenario15s_id[miss_indices]
        # [(file_name, frame_id), ...]
        eval_result_to_save["miss_scenarios"] = scenario15s_id_miss
        # get top fde scenarios (about 10%)
        total_num = len(fde8_gen)
        fde_indices = np.argsort(fde8_gen)[::-1][:int(total_num/10)]
        scenario15s_id_high_fde = scenario15s_id[fde_indices]
        fde8s_value = fde8_gen[fde_indices]
        # [(file_name, frame_id, fde8s_value), ...]
        eval_result_to_save["top_fde_scenarios"] = list(zip(scenario15s_id_high_fde, fde8s_value))
        # save eval result
        with open(EVAL_LOG_SAVING_PATH, "wb") as f:
            pickle.dump(eval_result_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Eval result saved to {EVAL_LOG_SAVING_PATH} with {scenario15s_id_miss.shape[0]} miss scenarios and {scenario15s_id_high_fde.shape[0]} top fde scenarios')
    else:
        assert False, "EVAL_LOG_SAVING_PATH is None"

    return eval_result

def compute_metrics_waymo(prediction: EvalPrediction):
    from transformer4planning.utils.waymo_utils import tensor_to_str, waymo_evaluation
    pred_dicts = prediction.predictions['prediction_generation']
    type_idx_str = {
            1: 'TYPE_VEHICLE',
            2: 'TYPE_PEDESTRIAN',
            3: 'TYPE_CYCLIST',
        }
    
    pred_dict_list = []
    for idx in range(len(pred_dicts["object_type"])):
        single_pred_dict = {}
        for key in pred_dicts.keys():
            if key == "scenario_id":
                single_pred_dict[key] = tensor_to_str(torch.from_numpy(pred_dicts[key][idx]).unsqueeze(0))[0]
            elif key == "object_type":
                single_pred_dict[key] = type_idx_str[pred_dicts[key][idx]]
            elif key == "pred_scores":
                single_pred_dict[key] = pred_dicts[key][idx].reshape(-1)
            else:
                single_pred_dict[key] = pred_dicts[key][idx]
                
        pred_dict_list.append(single_pred_dict)

    result_dict, result_format_str = waymo_evaluation(pred_dicts=pred_dict_list, num_modes_for_eval=6, eval_second=8)
    print(result_format_str)

    metric_names = ['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']
    agent_type = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

    result = {}
    for m in metric_names:  
        record_avg = True 
        for a in agent_type:
            key = f"{m} - {a}"
            if result_dict[key] == -1:
                if record_avg: record_avg = False
            else:
                assert result_dict[key] > 0
                result[key] = result_dict[key]

        if m != "OverlapRate" and record_avg:
            result[m] = result_dict[m]


    loss_items = prediction.predictions['loss_items']
    # check loss items are dictionary
    if isinstance(loss_items, dict):
        for each_key in loss_items:
            # get average loss for each key
            result[each_key] = np.mean(loss_items[each_key])

    return result


class CustomCallback(DefaultFlowCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        # control.should_log = False
        if args.logging_strategy == IntervalStrategy.EPOCH and state.epoch % args.eval_interval == 0:
            control.should_log = True

        # Evaluate
        # control.should_evaluate = False
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch and state.epoch % args.eval_interval == 0:
            control.should_evaluate = True

        # Save
        # control.should_save = False
        if args.save_strategy == IntervalStrategy.EPOCH and state.epoch % args.eval_interval == 0:
            control.should_save = True

        return control


class PlanningTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.

        Overwrite Note:
        To skip the assertion when the batch is empty.
        This might happen due to filters in the preprocess function when batch size is 1.
        Only tested batch size of 1 when evaluation, but training.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            return inputs
            # raise ValueError(
            #     "The batch received was empty, your model won't be able to train on it. Double-check that your "
            #     f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            # )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        return inputs

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Overwriting to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        global EVAL_LOG_SAVING_PATH
        EVAL_LOG_SAVING_PATH = os.path.join(self.args.logging_dir, 'eval_log.pickle')
        if inputs is None:
            return None, None, None
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        # loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        loss_without_labels = True

        inputs = self._prepare_inputs(inputs)
        if len(inputs) == 0:
            return None, None, None
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
                
        ignore_keys = ['past_key_values', 'loss_items']
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise NotImplementedError('Not implemented yet, check source code of Transformers Trainer to adapt it.')
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    loss_items = outputs['loss_items'] if 'loss_items' in outputs else None
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"] + ["loss_items"])  # (logits, pred_dict)
                    else:
                        raise NotImplementedError
                        # logits = outputs[1:]
                else:
                    # # TODO: this needs to be fixed and made cleaner later.
                    raise NotImplementedError

        pred_dict = outputs['pred_dict'] if 'pred_dict' in outputs else logits[-1]

        logits = nested_detach(logits)
        if len(logits) >= 1:
            logits = logits[0]
        logits = torch.as_tensor(logits)
        prediction_generation = self.model.generate(**inputs)

        if logits.shape[0] != self.args.per_device_eval_batch_size:
            # must top to the eval batch size, or will cause error and stuck the whole pipeline
            incorrect_batch_size = logits.shape[0]
            short = self.args.per_device_eval_batch_size - incorrect_batch_size
            if short > 0 :
                for _ in range(short):
                    logits = torch.cat([logits, logits[0].unsqueeze(0)], dim=0)
                    prediction_gen = {}
                    for key in prediction_generation.keys():
                        prediction_gen[key] = torch.cat([prediction_generation[key], prediction_generation[key][0].unsqueeze(0)], dim=0)
                    prediction_generation = prediction_gen
                    labels = torch.cat([labels, labels[0].unsqueeze(0)], dim=0)
            else:
                logits = logits[:self.args.per_device_eval_batch_size]
                prediction_gen = {}
                for key in prediction_generation.keys():
                    prediction_gen[key] = prediction_generation[key][:self.args.per_device_eval_batch_size]
                prediction_generation = prediction_gen
                labels = labels[:self.args.per_device_eval_batch_size]
            # print(f'topping to batch size from {incorrect_batch_size} to {self.args.per_device_eval_batch_size}')
        
        logits_dict = {
            "prediction_forward": pred_dict,
            "prediction_generation": prediction_generation,
            "loss_items": loss_items if loss_items is not None else 0,
        }

        if self.model.config.task == "nuplan":
            if 't0_frame_id' not in inputs or inputs['t0_frame_id'][0] == -1:
                # val14 without 15s in the future
                t0_frame_id = inputs["frame_id"]
            else:
                t0_frame_id = inputs['t0_frame_id']
            scenario15s_ids = convert_names_to_ids(inputs["file_name"], t0_frame_id)
            logits_dict.update({
                "frame_id": inputs["frame_id"],
                "scenario15s_id": torch.tensor(scenario15s_ids, device=logits.device),
            })
            if len(self.model.encoder.selected_indices) != 0:
                logits_dict.update({
                    "selected_indices": torch.tensor(self.model.encoder.selected_indices, device=logits.device).repeat(logits.shape[0], 1),
                })

        return (loss, logits_dict, labels)

    def do_closed_loop_simulation(self,
                                  val1k_datasest,
                                  metric_key_prefix: str = "CLS",):
        """
        Run closed loop simulation and returns the metrics.
        Args:
            val1k_datasest (`Dataset`, *optional*):
                Pass a dataset. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method
                 are automatically removed. It must implement the `__len__` method.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "8sFDE" will be named
                "CLS_8sFDE" if the prefix is "CLS" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        pass

def save_raster(inputs, sample_index, file_index=0,
                prediction_trajectory=None, path_to_save=None,
                high_scale=4, low_scale=0.77,
                prediction_key_point=None,
                prediction_key_point_by_gen=None,
                prediction_trajectory_by_gen=None):
    import cv2
    # save rasters
    image_shape = None
    image_to_save = {
        'high_res_raster': None,
        'low_res_raster': None
    }
    past_frames_num = inputs['context_actions'][sample_index].shape[0]
    agent_type_num = 8
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        """
        each_img = inputs[each_key][sample_index].cpu().numpy()
        goal = each_img[:, :, 0]
        road = each_img[:, :, :21]
        traffic_lights = each_img[:, :, 21:25]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = np.random.randint(0, 255, size=(21, 3)) * 0.5
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float)
        image_shape = target_image.shape

        for i in range(21):
            if i in [0, 11]: continue
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in [0, 11]:
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        if 'high' in each_key:
            scale = high_scale
        elif 'low' in each_key:
            scale = low_scale
        # draw context actions, and trajectory label
        # for each_traj_key in ['context_actions', 'trajectory_label']:
        #     pts = inputs[each_traj_key][sample_index].cpu().numpy()
        #     for i in range(pts.shape[0]):
        #         x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
        #         y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
        #         if x < target_image.shape[0] and y < target_image.shape[1]:
        #             if 'actions' in each_traj_key:
        #                 target_image[x, y, :] = [255, 0, 0]
        #             elif 'label' in each_traj_key:
        #                 target_image[x, y, :] = [0, 255, 0]

        tray_point_size = int(0.75 * scale * 4 / 7)
        key_point_size = int(3 * scale * 4 / 7)
        # draw prediction trajectory
        if prediction_trajectory is not None:
            for i in range(prediction_trajectory.shape[0]):
                x = int(prediction_trajectory[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_trajectory[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x-tray_point_size:x+tray_point_size, y-tray_point_size:y+tray_point_size, 0] += 200

        # draw prediction trajectory by generation
        if prediction_trajectory_by_gen is not None:
            for i in range(prediction_trajectory_by_gen.shape[0]):
                x = int(prediction_trajectory_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_trajectory_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x-tray_point_size:x+tray_point_size, y-tray_point_size:y+tray_point_size, :] += 100

        # draw key points
        if prediction_key_point is not None:
            for i in range(prediction_key_point.shape[0]):
                x = int(prediction_key_point[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x-key_point_size:x+key_point_size, y-key_point_size:y+key_point_size, 1] += 200

        # draw prediction key points during generation
        if prediction_key_point_by_gen is not None:
            for i in range(prediction_key_point_by_gen.shape[0]):
                x = int(prediction_key_point_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_key_point_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x-key_point_size:x+key_point_size, y-key_point_size:y+key_point_size, 2] += 200

        target_image = np.clip(target_image, 0, 255)
        image_to_save[each_key] = target_image
    # import wandb
    if path_to_save is not None:
        for each_key in image_to_save:
            # images = wandb.Image(
            #     image_to_save[each_key],
            #     caption=f"{file_index}-{each_key}"
            # )
            # self.log({"pred examples": images})
            cv2.imwrite(os.path.join(path_to_save, 'test' + '_' + str(file_index) + '_' + str(each_key) + '.png'), image_to_save[each_key])
    else:
        return image_to_save

    print('length of action and labels: ',
          inputs['context_actions'][sample_index].shape, inputs['trajectory_label'][sample_index].shape)
    print('debug images saved to: ', path_to_save, file_index)