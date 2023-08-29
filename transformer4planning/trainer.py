import time
import tqdm
import datetime
import pickle
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import  nested_detach
from transformers.trainer_callback import TrainerState, TrainerControl, IntervalStrategy, DefaultFlowCallback
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import EvalPrediction
from transformer4planning.utils import mtr_utils
from typing import List, Optional, Dict, Any, Tuple, Union
from datasets import Dataset
import torch
import torch.nn as nn
import os
import numpy as np

# Custom compute_metrics function
def compute_metrics_nuplan(prediction: EvalPrediction):
    """
    All inputs are finalized and gathered, should run only one time.
    """
    # TODO: Adapt to waymo
    # TODO: Add classification metrics (clf_metrics) for scoring
    # TODO: Add visualization for prediction results with save_raster method
    eval_result = {}
    predictions = prediction.predictions
    labels = prediction.label_ids  # nparray: sample_num, 80, 4
    prediction_horizon = labels.shape[1]
    valid_mask = predictions["sample_valid_mask"].astype(bool)

    # select gt key points from gt trajectories
    # TODO: merge key points selection and abstract with encoders/base.py select_key_points() to utiles
    # TODO: adoptive with different key point selection strategy
    selected_indices = [79, 39, 19, 9, 4]
    label_key_points = labels[:, selected_indices, :]

    prediction_by_generation = predictions['prediction_generation']  # sample_num, 85, 2/4
    prediction_by_forward = predictions['prediction_forward']  # sample_num, 85, 2/4

    # fetch trajectory and key points predictions
    # TODO: adaptive with no key points predictions
    prediction_key_points_by_generation = prediction_by_generation[:, :-prediction_horizon, :]  # sample_num, 5, 2/4
    prediction_key_points_by_forward = prediction_by_forward[:, :-prediction_horizon, :]  # sample_num, 5, 2/4
    prediction_trajectory_by_generation = prediction_by_generation[:, -prediction_horizon:, :]  # sample_num, 80, 2/4
    prediction_trajectory_by_forward = prediction_by_forward[:, -prediction_horizon:, :]  # sample_num, 80, 2/4

    # calculate error for generation results
    ade_x_error_key_points_gen = prediction_key_points_by_generation[:, :, 0] - label_key_points[:, :, 0]
    ade_y_error_key_points_gen = prediction_key_points_by_generation[:, :, 1] - label_key_points[:, :, 1]
    fde_x_error_key_points_gen = prediction_key_points_by_generation[:, 0, 0] - label_key_points[:, 0, 0]
    fde_y_error_key_points_gen = prediction_key_points_by_generation[:, 0, 1] - label_key_points[:, 0, 1]
    ade_x_error_gen = prediction_trajectory_by_generation[:, :, 0] - labels[:, :, 0]
    ade_y_error_gen = prediction_trajectory_by_generation[:, :, 1] - labels[:, :, 1]
    fde_x_error_gen = prediction_trajectory_by_generation[:, -1, 0] - labels[:, -1, 0]
    fde_y_error_gen = prediction_trajectory_by_generation[:, -1, 1] - labels[:, -1, 1]

    ade_gen = np.sqrt(ade_x_error_gen ** 2 + ade_y_error_gen ** 2)[valid_mask].mean()
    eval_result['ade_gen'] = ade_gen
    fde_gen = np.sqrt(fde_x_error_gen ** 2 + fde_y_error_gen ** 2)[valid_mask].mean()
    eval_result['fde_gen'] = fde_gen
    ade_key_points_gen = np.sqrt(ade_x_error_key_points_gen ** 2 + ade_y_error_key_points_gen ** 2)[valid_mask].mean()
    eval_result['ade_keypoints_gen'] = ade_key_points_gen
    fde_key_points_gen = np.sqrt(fde_x_error_key_points_gen ** 2 + fde_y_error_key_points_gen ** 2)[valid_mask].mean()
    eval_result['fde_keypoints_gen'] = fde_key_points_gen

    if prediction_key_points_by_generation.shape[-1] == 4:
        heading_error_gen = prediction_key_points_by_generation[:, :, -1] - label_key_points[:, :, -1]
        eval_result['heading_error_by_gen'] = heading_error_gen[valid_mask].mean()

    # compute error for forward results
    ade_x_error_key_points_for = prediction_key_points_by_forward[:, :, 0] - label_key_points[:, :, 0]
    ade_y_error_key_points_for = prediction_key_points_by_forward[:, :, 1] - label_key_points[:, :, 1]
    fde_x_error_key_points_for = prediction_key_points_by_forward[:, 0, 0] - label_key_points[:, 0, 0]
    fde_y_error_key_points_for = prediction_key_points_by_forward[:, 0, 1] - label_key_points[:, 0, 1]
    ade_x_error_for = prediction_trajectory_by_forward[:, :, 0] - labels[:, :, 0]
    ade_y_error_for = prediction_trajectory_by_forward[:, :, 1] - labels[:, :, 1]
    fde_x_error_for = prediction_trajectory_by_forward[:, -1, 0] - labels[:, -1, 0]
    fde_y_error_for = prediction_trajectory_by_forward[:, -1, 1] - labels[:, -1, 1]

    ade_for = np.sqrt(ade_x_error_for ** 2 + ade_y_error_for ** 2)[valid_mask].mean()
    eval_result['ade'] = ade_for
    fde_for = np.sqrt(fde_x_error_for ** 2 + fde_y_error_for ** 2)[valid_mask].mean()
    eval_result['fde'] = fde_for
    ade_key_points_for = np.sqrt(ade_x_error_key_points_for ** 2 + ade_y_error_key_points_for ** 2)[valid_mask].mean()
    eval_result['ade_keypoints'] = ade_key_points_for
    fde_key_points_for = np.sqrt(fde_x_error_key_points_for ** 2 + fde_y_error_key_points_for ** 2)[valid_mask].mean()
    eval_result['fde_keypoints'] = fde_key_points_for

    if prediction_key_points_by_forward.shape[-1] == 4:
        heading_error_for = prediction_key_points_by_forward[:, :, -1] - label_key_points[:, :, -1]
        eval_result['heading_error'] = heading_error_for[valid_mask].mean()

    return eval_result

def compute_metrics_waymo(eval_input):
    total_eval_num = eval_input.label_ids.shape[0]
    predictions = eval_input.predictions
    pred_keys = predictions.keys()

    from transformer4planning.utils.mtr_utils import _num_to_str
    predictions["scenario_id"] = _num_to_str(predictions["scenario_id"])
    predictions["object_type"] = _num_to_str(predictions["object_type"])

    pred_list = []
    for i in range(total_eval_num):
        pred = {}
        for key in pred_keys:
            pred[key] = predictions[key][i]

        pred_list.append(pred)

    from dataset_gen.waymo.waymo_eval import waymo_evaluation
    _, result_format_str, final_avg_results = waymo_evaluation(pred_dicts=pred_list, num_modes_for_eval=6)

    print(result_format_str)

    return final_avg_results

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

        Subclass and override to inject custom behavior.

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
        if inputs is None:
            return None, None, None
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
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

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # # TODO: this needs to be fixed and made cleaner later.
                    # if self.args.past_index >= 0:
                    #     self._past = outputs[self.args.past_index - 1]
        if self.model.ar_future_interval > 0:
            prediction_generation = self.model.generate(**inputs)
        else:
            prediction_generation = None
                       
        if self.model.model_args.task == "waymo" and self.model.model_args.encoder_type == "vector":
            from transformer4planning.utils.mtr_utils import rotate_points_along_z, str_to_tensor

            pred_length = inputs['center_gt_trajs_src'].shape[1] - inputs['current_time_index'][0] - 1
            pred_trajs = prediction_generation['logits'][:, :, -pred_length:, :]

            pred_scores = prediction_generation['scores']
            pred_kps = prediction_generation['key_points_logits']
            center_objects_world = inputs['center_objects_world'].type_as(pred_trajs)

            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            # assert num_feat == 7

            pred_trajs_world = rotate_points_along_z(
                points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].view(num_center_objects)
            ).view(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

            num_center_objects, num_modes, num_kps, num_kps_feat = pred_kps.shape
            pred_kps_world = rotate_points_along_z(
                points=pred_kps.view(num_center_objects, num_modes * num_kps, num_kps_feat),
                angle=center_objects_world[:, 6].view(num_center_objects)
            ).view(num_center_objects, num_modes, num_kps, num_kps_feat)
            pred_kps_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

            logits = {
                'scenario_id': str_to_tensor(inputs['scenario_id']).to(pred_trajs.device),
                'pred_trajs': pred_trajs_world[..., 0:2],
                'pred_scores': pred_scores,
                'pred_kps': pred_kps_world,
                'object_id': inputs['center_objects_id'],
                'object_type': str_to_tensor(inputs['center_objects_type']).to(pred_trajs.device),
                'gt_trajs': inputs['center_gt_trajs_src'],
                'track_index_to_predict': inputs['track_index_to_predict'],
            }
        else:
            logits = {
                "prediction_forward": logits[0],
                "prediction_generation": prediction_generation,
            }

        logits = nested_detach(logits)

        sample_valid_mask = torch.ones((self.args.per_device_eval_batch_size,), device=labels.device)
        if len(labels) != self.args.per_device_eval_batch_size:
            incorrect_batch_size = len(labels)
            short = self.args.per_device_eval_batch_size - incorrect_batch_size
            for i in range(short):
                for k in logits.keys():
                    logits[k] = torch.cat([logits[k], logits[k][0].unsqueeze(0)], dim=0)
                labels = torch.cat([labels, labels[0].unsqueeze(0)], dim=0)
            print(f'topping to batch size from {incorrect_batch_size} to {self.args.per_device_eval_batch_size}')
            sample_valid_mask[len(labels):] = 0

        logits["sample_valid_mask"] = sample_valid_mask
        # if prediction_loss_only:
        #     return (loss, None, None)
        return (loss, logits, labels)

    def save_raster(self, path_to_save,
                    inputs, sample_index,
                    prediction_trajectory,
                    file_index,
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

            # print('test: ', past_frames_num, agent_type_num, agent.shape)
            for i in range(past_frames_num):
                for j in range(agent_type_num):
                    # if j == 7:
                    #     print('debug', np.sum(agent[:, :, j * 9 + i]), agent[:, :, j * 9 + i])
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
            for each_traj_key in ['context_actions', 'trajectory_label']:
                pts = inputs[each_traj_key][sample_index].cpu().numpy()
                for i in range(pts.shape[0]):
                    x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                    y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                    if x < target_image.shape[0] and y < target_image.shape[1]:
                        if 'actions' in each_traj_key:
                            target_image[x, y, :] = [255, 0, 0]
                        elif 'label' in each_traj_key:
                            target_image[x, y, :] = [0, 255, 0]

            # draw prediction trajectory
            for i in range(prediction_trajectory.shape[0]):
                x = int(prediction_trajectory[i, 0] * scale) + target_image.shape[0] // 2
                y = int(prediction_trajectory[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, 0] += 100

            # draw key points
            if prediction_key_point is not None:
                for i in range(prediction_key_point.shape[0]):
                    x = int(prediction_key_point[i, 0] * scale) + target_image.shape[0] // 2
                    y = int(prediction_key_point[i, 1] * scale) + target_image.shape[1] // 2
                    if x < target_image.shape[0] and y < target_image.shape[1]:
                        target_image[x, y, 1] += 100

            # draw prediction key points during generation
            if prediction_key_point_by_gen is not None:
                for i in range(prediction_key_point_by_gen.shape[0]):
                    x = int(prediction_key_point_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                    y = int(prediction_key_point_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                    if x < target_image.shape[0] and y < target_image.shape[1]:
                        target_image[x, y, 2] += 100

            # draw prediction trajectory by generation
            if prediction_trajectory_by_gen is not None:
                for i in range(prediction_trajectory_by_gen.shape[0]):
                    x = int(prediction_trajectory_by_gen[i, 0] * scale) + target_image.shape[0] // 2
                    y = int(prediction_trajectory_by_gen[i, 1] * scale) + target_image.shape[1] // 2
                    if x < target_image.shape[0] and y < target_image.shape[1]:
                        target_image[x, y, :] += 100
            target_image = np.clip(target_image, 0, 255)
            image_to_save[each_key] = target_image
        import wandb
        for each_key in image_to_save:
            images = wandb.Image(
                image_to_save[each_key],
                caption=f"{file_index}-{each_key}"
            )
            self.log({"pred examples": images})
            cv2.imwrite(os.path.join(path_to_save, 'test' + '_' + str(file_index) + '_' + str(each_key) + '.png'), image_to_save[each_key])
        print('length of action and labels: ',
              inputs['context_actions'][sample_index].shape, inputs['trajectory_label'][sample_index].shape)
        print('debug images saved to: ', path_to_save, file_index)

    def evaluate_waymo(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self.model.eval()

        dataloader = self.get_eval_dataloader(eval_dataset)
        dataset = dataloader.dataset

        if self.is_world_process_zero:
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

        model_path = self.model.model_args.model_pretrain_name_or_path
        model_name = model_path.split('/')[-3]+'___' + model_path.split('/')[-1]

        eval_output_dir = model_path + '/eval_output/'
        os.makedirs(eval_output_dir, exist_ok=True)
        
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = eval_output_dir + ('%s_log_eval_%s.txt' % (model_name, cur_time))
        logger = mtr_utils.create_logger(log_file, rank=0)

        start_time = time.time()
        logger_iter_interval = 1000

        pred_dicts = []
        for i, batch_dict in enumerate(dataloader):
            with torch.no_grad():
                batch_dict = self._prepare_inputs(batch_dict)
                batch_pred_dicts = self.model.generate(**batch_dict)
                # batch_pred_dicts = self.model(**batch_dict)

                if 'vector' in self.model.model_args.model_name:
                    final_pred_dicts = dataset.generate_prediction_dicts(batch_dict, batch_pred_dicts)
                else:
                    final_pred_dicts = dataset.generate_prediction_dicts({'input_dict': batch_dict}, batch_pred_dicts)

                pred_dicts += final_pred_dicts

            disp_dict = {}

            if self.is_world_process_zero and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
                past_time = progress_bar.format_dict['elapsed']
                second_each_iter = past_time / max(i, 1.0)
                remaining_time = second_each_iter * (len(dataloader) - i)
                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                batch_size = batch_dict.get('batch_size', None)
                logger.info(f'eval: batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                            f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                            f'{disp_str}')
            
            # if i > 100:    
            #     break

        if self.is_world_process_zero:
            progress_bar.close()

        logger.info('*************** Performance of EPOCH *****************' )
        sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
        logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        ret_dict = {}

        with open(eval_output_dir + "result_" + model_name + '_' + cur_time + '.pkl', 'wb') as f:
            pickle.dump(pred_dicts, f)

        result_str, result_dict = dataset.evaluation(
            pred_dicts,
        )

        logger.info(result_str)
        ret_dict.update(result_dict)

        logger.info('Result is save to %s' % eval_output_dir)
        logger.info('****************Evaluation done.*****************')        