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
from transformer4planning.utils import mtr_utils
from typing import List, Optional, Dict, Any, Tuple, Union
from datasets import Dataset
import torch
import torch.nn as nn
import logging
import os
import numpy as np

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
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        # if has_labels or loss_without_labels:
        #     labels = nested_detach(tuple(inputs.get(name) is not None for name in self.label_names))
        #     if len(labels) == 1:
        #         labels = labels[0]
        # else:
        #     labels = None

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
        batch_generate_eval = True
        # if model.mode == 'OA-OA':
        #     batch_generate_eval = False
        #     print('batch generate for OA-OA is not implemented yet')
        # if self.model.model_args.k not in [-1]:
        #     batch_generate_eval = False
        #     print('batch generate for TopK is not implemented yet')
        # if self.model.model_args.k not in [1, -1]:
        #     batch_generate_eval = False
        #     print('batch generate for TopK is not implemented yet')

        if batch_generate_eval:
            # run generate for sequence of actions
            if self.model.model_args.autoregressive:
                # TODO: support different frame interval, change sequence length from 40
                actions_label_in_batch = inputs["trajectory"].clone()
                trajectory_label_in_batch = self.model.compute_normalized_points(actions_label_in_batch)
                if self.model.model_args.k > 0:
                    from transformers import GenerationConfig
                    translation_generation_config = GenerationConfig(
                        num_beams=20,
                        early_stopping=True,
                        decoder_start_token_id=0,
                        eos_token_id=model.config.eos_token_id,
                        pad_token=model.config.pad_token_id,
                        max_new_tokens=50,
                        use_cache=False,
                    )
                    prediction_actions_in_batch = self.model.generate(**inputs, generation_config=translation_generation_config)
                elif self.model.model_args.k == -1:
                    prediction_actions_in_batch = self.model.generate_legacy(**inputs)
                prediction_trajectory_in_batch = self.model.compute_normalized_points(prediction_actions_in_batch)
                ade_x_error = prediction_trajectory_in_batch[:, :, 0] - trajectory_label_in_batch[:, -40:, 0]
                ade_y_error = prediction_trajectory_in_batch[:, :, 1] - trajectory_label_in_batch[:, -40:, 1]
                fde_x_error = prediction_trajectory_in_batch[:, -1, 0] - trajectory_label_in_batch[:, -1, 0]
                fde_y_error = prediction_trajectory_in_batch[:, -1, 1] - trajectory_label_in_batch[:, -1, 1]
            else:
                if self.model.model_args.task == "waymo" and self.model.model_args.encoder_type == "vector":
                    agent_trajs = inputs['agent_trajs']
                    ego_trajs = [traj[inputs["track_index_to_predict"][i], :, :] for i, traj in enumerate(agent_trajs)]
                    ego_trajs = torch.stack(ego_trajs, dim=0).to(self.model.device).squeeze(1)

                    trajectory_label_in_batch = ego_trajs[:, 11:, [0, 1, 2, 6]].clone()
                elif self.model.model_args.task == "nuplan" and self.model.model_args.encoder_type == "raster":
                    trajectory_label_in_batch = inputs["trajectory_label"].clone()
                else:
                    raise NotImplementedError
                
                if isinstance(logits, tuple):
                    prediction_trajectory_in_batch = logits[0]
                else:
                    print('unknown logits type', type(logits), logits)
                if self.model.ar_future_interval > 0:
                    length_of_trajectory = trajectory_label_in_batch.shape[1]
                    prediction_key_points = prediction_trajectory_in_batch[:, :-length_of_trajectory, :]
                    prediction_trajectory_in_batch = prediction_trajectory_in_batch[:, -length_of_trajectory:, :]
                    if self.model.model_args.specified_key_points:
                        # 80, 40, 20, 10, 5
                        if self.model.model_args.forward_specified_key_points:
                            selected_indices = [4, 9, 19, 39, 79]
                        else:
                            selected_indices = [79, 39, 19, 9, 4]
                        future_key_points = trajectory_label_in_batch[:, selected_indices, :]
                    else:
                        future_key_points = trajectory_label_in_batch[:, self.model.ar_future_interval - 1::self.model.ar_future_interval, :]
                    
                    if self.model.model_args.generate_diffusion_dataset_for_key_points_decoder:
                        return None,None,None
                        # no need to return further info.
                    
                    ade_x_error_key_points = prediction_key_points[:, :, 0] - future_key_points[:, :, 0]
                    ade_y_error_key_points = prediction_key_points[:, :, 1] - future_key_points[:, :, 1]
                    fde_x_error_key_points = prediction_key_points[:, -1, 0] - future_key_points[:, -1, 0]
                    fde_y_error_key_points = prediction_key_points[:, -1, 1] - future_key_points[:, -1, 1]
                    ade_x_error = prediction_trajectory_in_batch[:, :, 0] - trajectory_label_in_batch[:, :, 0]
                    ade_y_error = prediction_trajectory_in_batch[:, :, 1] - trajectory_label_in_batch[:, :, 1]
                    fde_x_error = prediction_trajectory_in_batch[:, -1, 0] - trajectory_label_in_batch[:, -1, 0]
                    fde_y_error = prediction_trajectory_in_batch[:, -1, 1] - trajectory_label_in_batch[:, -1, 1]
                    if self.model.model_args.predict_yaw:
                        heading_error = prediction_trajectory_in_batch[:, :, -1] - trajectory_label_in_batch[:, :, -1]
                    if self.model.k >= 1:
                        prediction_trajectory_in_batch_by_gen = self.model.generate(**inputs)
                        length_of_trajectory = trajectory_label_in_batch.shape[1]
                        prediction_key_points_by_gen = prediction_trajectory_in_batch_by_gen[:, :-length_of_trajectory, :]
                        prediction_trajectory_in_batch_by_gen = prediction_trajectory_in_batch_by_gen[:, -length_of_trajectory:, :]
                        ade_x_error_key_points_by_gen = prediction_key_points_by_gen[:, :, 0] - future_key_points[:, :, 0]
                        ade_y_error_key_points_by_gen = prediction_key_points_by_gen[:, :, 1] - future_key_points[:, :, 1]
                        fde_x_error_key_points_by_gen = prediction_key_points_by_gen[:, -1, 0] - future_key_points[:, -1, 0]
                        fde_y_error_key_points_by_gen = prediction_key_points_by_gen[:, -1, 1] - future_key_points[:, -1, 1]
                        ade_x_error_by_gen = prediction_trajectory_in_batch_by_gen[:, :, 0] - trajectory_label_in_batch[:, :, 0]
                        ade_y_error_by_gen = prediction_trajectory_in_batch_by_gen[:, :, 1] - trajectory_label_in_batch[:, :, 1]
                        fde_x_error_by_gen = prediction_trajectory_in_batch_by_gen[:, -1, 0] - trajectory_label_in_batch[:, -1, 0]
                        fde_y_error_by_gen = prediction_trajectory_in_batch_by_gen[:, -1, 1] - trajectory_label_in_batch[:, -1, 1]
                        if self.model.model_args.predict_yaw:
                            heading_error_by_gen = prediction_trajectory_in_batch_by_gen[:, :, -1] - trajectory_label_in_batch[:, :, -1]
                    else:
                        raise ValueError(f'unknown k for auto-regression future keypoints: {self.model.k}')
                else:
                    ade_x_error = prediction_trajectory_in_batch[:, :, 0] - trajectory_label_in_batch[:, :, 0]
                    ade_y_error = prediction_trajectory_in_batch[:, :, 1] - trajectory_label_in_batch[:, :, 1]
                    fde_x_error = prediction_trajectory_in_batch[:, -1, 0] - trajectory_label_in_batch[:, -1, 0]
                    fde_y_error = prediction_trajectory_in_batch[:, -1, 1] - trajectory_label_in_batch[:, -1, 1]
                    if self.model.model_args.predict_yaw:
                        heading_error = prediction_trajectory_in_batch[:, :, -1] - trajectory_label_in_batch[:, :, -1]

            if self.model.model_args.visualize_prediction_to_path is not None and self.is_world_process_zero:
                # WARNING: only use it for one time evaluation, or slowing the evaluation significantly
                # only save and debug at process zero
                path_to_save = self.model.model_args.visualize_prediction_to_path
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                    # empty and re-save at every epoch
                    # os.rmdir(path_to_save)
                batch_size = trajectory_label_in_batch.shape[0]
                for i in range(batch_size):
                    file_number = len(os.listdir(path_to_save))
                    if file_number <= 200:
                        if self.model.ar_future_interval > 0:
                            self.save_raster(path_to_save=path_to_save,
                                             inputs=inputs,
                                             sample_index=i,
                                             file_index=file_number,
                                             prediction_trajectory=prediction_trajectory_in_batch[i],
                                             prediction_key_point=prediction_key_points[i],
                                             prediction_key_point_by_gen=prediction_key_points_by_gen[i],
                                             prediction_trajectory_by_gen=prediction_trajectory_in_batch_by_gen[i])
                        else:
                            # visualize prediction trajectory only
                            self.save_raster(path_to_save=path_to_save,
                                             inputs=inputs,
                                             sample_index=i,
                                             file_index=file_number,
                                             prediction_trajectory=prediction_trajectory_in_batch[i])
                    else:
                        break

            # compute ade
            ade = torch.sqrt(ade_x_error.flatten() ** 2 + ade_y_error.flatten() ** 2)
            ade = ade.mean()
            self.eval_result['ade'].append(float(ade))
            # compute fde
            fde = torch.sqrt(fde_x_error.flatten() ** 2 + fde_y_error.flatten() ** 2)
            fde = fde.mean()
            self.eval_result['fde'].append(float(fde))
            if self.model.model_args.predict_yaw:
                heading_error = torch.abs(heading_error).mean()
                if 'heading_error' not in self.eval_result:
                    self.eval_result['heading_error'] = []
                self.eval_result['heading_error'].append(float(heading_error))

            if self.model.ar_future_interval > 0:
                if 'ade_keypoints' not in self.eval_result:
                    self.eval_result['ade_keypoints'] = []
                    self.eval_result['fde_keypoints'] = []
                # compute key points ade
                ade_key_points = torch.sqrt(ade_x_error_key_points.flatten() ** 2 + ade_y_error_key_points.flatten() ** 2)
                ade_key_points = ade_key_points.mean()
                self.eval_result['ade_keypoints'].append(float(ade_key_points))
                # compute fde
                fde_key_points = torch.sqrt(fde_x_error_key_points.flatten() ** 2 + fde_y_error_key_points.flatten() ** 2)
                fde_key_points = fde_key_points.mean()
                self.eval_result['fde_keypoints'].append(float(fde_key_points))

                if self.model.k >= 1:
                    # evaluate through generate function
                    if 'ade_keypoints_gen' not in self.eval_result:
                        self.eval_result['ade_keypoints_gen'] = []
                        self.eval_result['fde_keypoints_gen'] = []
                        self.eval_result['ade_gen'] = []
                        self.eval_result['fde_gen'] = []
                    # compute ade by gen
                    ade_by_gen = torch.sqrt(ade_x_error_by_gen.flatten() ** 2 + ade_y_error_by_gen.flatten() ** 2)
                    ade_by_gen = ade_by_gen.mean()
                    self.eval_result['ade_gen'].append(float(ade_by_gen))
                    # compute fde
                    fde_by_gen = torch.sqrt(fde_x_error_by_gen.flatten() ** 2 + fde_y_error_by_gen.flatten() ** 2)
                    fde_by_gen = fde_by_gen.mean()
                    self.eval_result['fde_gen'].append(float(fde_by_gen))
                    # compute key points ade
                    ade_key_points_by_gen = torch.sqrt(ade_x_error_key_points_by_gen.flatten() ** 2 + ade_y_error_key_points_by_gen.flatten() ** 2)
                    ade_key_points_by_gen = ade_key_points_by_gen.mean()
                    self.eval_result['ade_keypoints_gen'].append(float(ade_key_points_by_gen))
                    # compute key points fde
                    fde_key_points_by_gen = torch.sqrt(fde_x_error_key_points_by_gen.flatten() ** 2 + fde_y_error_key_points_by_gen.flatten() ** 2)
                    fde_key_points_by_gen = fde_key_points_by_gen.mean()
                    self.eval_result['fde_keypoints_gen'].append(float(fde_key_points_by_gen))
                    if self.model.model_args.predict_yaw:
                        if 'heading_error_by_gen' not in self.eval_result:
                            self.eval_result['heading_error_by_gen'] = []
                        heading_error_by_gen = torch.abs(heading_error_by_gen).mean()
                        self.eval_result['heading_error_by_gen'].append(float(heading_error_by_gen))

        self.eval_itr += 1
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if self.model.model_args.task == "waymo" and self.model.model_args.encoder_type == "vector":
            pred_length = inputs['center_gt_trajs_src'].shape[1] - inputs['current_time_index'][0] - 1
            pred_trajs = logits[:, None, -pred_length:, :]
            pred_scores = torch.ones_like(pred_trajs[:, :, 0, 0])
            center_objects_world = inputs['center_objects_world'].type_as(pred_trajs)
            num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
            # assert num_feat == 7

            pred_trajs_world = mtr_utils.rotate_points_along_z(
                points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].view(num_center_objects)
            ).view(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

            pred_dict = []
            for obj_idx in range(num_center_objects):
                single_pred_dict = {
                    'scenario_id': str(inputs['scenario_id'][obj_idx]),
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                    'object_id': inputs['center_objects_id'][obj_idx],
                    'object_type': str(inputs['center_objects_type'][obj_idx]),
                    'gt_trajs': inputs['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                    'track_index_to_predict': inputs['track_index_to_predict'][obj_idx].cpu().numpy()
                }
                pred_dict.append(single_pred_dict)

            self.pred_dicts_list += pred_dict
        return (loss, logits, None)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        print('Starting evaluation loop with reset eval result')
        if self.is_world_process_zero:
            self.eval_result = {
                'ade': [],
                'fde': [],
            }
        self.eval_itr = 0
        if self.model.model_args.task == "waymo" and self.model.model_args.encoder_type == "vector": 
            prediction_loss_only = False
            self.pred_dicts_list = []

        eval_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        if self.model.model_args.generate_diffusion_dataset_for_key_points_decoder:
            return None
        result = dict()
        if self.model.clf_metrics is not None:
            # run classsification metrics
            result[f"{metric_key_prefix}_accuracy"] = self.model.clf_metrics["accuracy"].compute()
            result[f"{metric_key_prefix}_f1"] = self.model.clf_metrics["f1"].compute(average="macro")
            result[f"{metric_key_prefix}_precision"] = self.model.clf_metrics["precision"].compute(average="macro")
            result[f"{metric_key_prefix}_recall"] = self.model.clf_metrics["recall"].compute(average="macro")
        for each_key in self.eval_result:
            # result[f"{metric_key_prefix}_{each_key}"] = float(self.eval_result[each_key])
            result[f"{metric_key_prefix}_{each_key}"] = sum(self.eval_result[each_key]) / len(self.eval_result[each_key])
        logging.info("***** Eval results *****")
        logging.info(f"{result}")
        self.log(result)

        
        if self.model.model_args.task == "waymo" and self.model.model_args.encoder_type == "vector":
            from dataset_gen.waymo.waymo_eval import waymo_evaluation
            logging.info('*************** Performance of WOMD *****************' )
            try:
                num_modes_for_eval = self.pred_dicts_list[0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=self.pred_dicts_list, num_modes_for_eval=num_modes_for_eval)

            metric_result_str = '\n'
            for key in metric_results:
                metric_results[key] = metric_results[key]
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str

            print(metric_result_str)
            logging.info('****************Evaluation done.*****************')

        return eval_output

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
