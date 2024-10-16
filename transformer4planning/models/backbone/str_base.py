import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from transformer4planning.models.decoder.base import TrajectoryDecoder
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pred_dict: Optional[Dict[str, torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_items: Optional[Dict[str, torch.FloatTensor]] = None


class STRConfig(PretrainedConfig):
    def update_by_model_args(self, model_args):
        for each_key in model_args.__dict__:
            self.__dict__[each_key] = model_args.__dict__[each_key]
        # to be compatible with older models
        attr_list = ["use_key_points", "kp_decoder_type", "separate_kp_encoder", "use_proposal",
                     "autoregressive_proposals", "selected_exponential_past",
                     "rms_norm", "residual_in_fp32", "fused_add_norm", "raster_encoder_type",
                     "vit_intermediate_size", "mean_circular_loss",
                     "camera_image_encoder", "use_speed",
                     "denoise_kp"]
        for each_attr in attr_list:
            if not hasattr(self, each_attr):
                self.__dict__[each_attr] = False

class STR(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.encoder = None
        self.traj_decoder = None
        self.camera_image_encoder = None
        self.k = int(self.config.k)

        self.use_proposal = self.config.use_proposal
        # if self.use_proposal: assert self.config.task == "waymo", "NotImplemented"

        self.use_key_points = self.config.use_key_points
        self.kp_decoder_type = self.config.kp_decoder_type

        self.model_parallel = False
        self.device_map = None
        self.clf_metrics = None
        # Initialize weights and apply final processing
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        if self.config.task == "nuplan":
            if "raster" in self.config.encoder_type:
                from transformer4planning.models.encoder.nuplan_raster_encoder import NuplanRasterizeEncoder
                self.encoder = NuplanRasterizeEncoder(self.config)
            elif "vector" in self.config.encoder_type:
                from transformer4planning.models.encoder.pdm_encoder import PDMEncoder
                pdm_kwargs = dict(
                    hidden_dim=self.config.n_embd,
                    centerline_dim=120,
                    history_dim=20
                )
                self.encoder = PDMEncoder(pdm_kwargs, self.config)
            else:
                raise AttributeError("encoder_type should be either raster or vector")
        elif self.config.task == "waymo":
            from transformer4planning.models.encoder.waymo_vector_encoder import WaymoVectorizeEncoder
            action_kwargs = dict(
                d_embed=self.config.n_embd
            )
            self.encoder = WaymoVectorizeEncoder(action_kwargs, self.config)
        else:
            raise NotImplementedError

    def build_decoder(self):
        # load pretrained diffusion keypoint decoder
        #TODO: add diffusion decoder trained from scratch
        if self.use_proposal:
            if self.config.task == "nuplan":
                from transformer4planning.models.decoder.base import ProposalDecoderCLS
                self.proposal_decoder = ProposalDecoderCLS(self.config, proposal_num=self.use_proposal)
            elif self.config.task == "waymo":
                from transformer4planning.models.decoder.base import ProposalDecoder
                self.proposal_decoder = ProposalDecoder(self.config)

        if self.use_key_points != 'no':
            if self.kp_decoder_type == "diffusion":
                from transformer4planning.models.decoder.diffusion_decoder import KeyPointDiffusionDecoder
                self.key_points_decoder = KeyPointDiffusionDecoder(self.config)
                if self.config.key_points_diffusion_decoder_load_from is not None:
                    print(f"Now loading pretrained key_points_diffusion_decoder from {self.config.key_points_diffusion_decoder_load_from}.")
                    state_dict = torch.load(self.config.key_points_diffusion_decoder_load_from)
                    self.key_points_decoder.model.load_state_dict(state_dict)
                    print("Pretrained keypoint decoder has been loaded!")
                else:
                    print("Now initializing diffusion decoder from scratch. Training will consume lots of time.")
            elif self.kp_decoder_type == "mlp":
                from transformer4planning.models.decoder.base import KeyPointMLPDeocder
                self.key_points_decoder = KeyPointMLPDeocder(self.config)

        self.traj_decoder = TrajectoryDecoder(self.config)

    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def forward(
            self,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            raise NotImplementedError('need to return dict for evaluations in trainer.py')

        input_embeds, info_dict = self.encoder(is_training=self.training, **kwargs)
        transformer_outputs_hidden_state = self.embedding_to_hidden(input_embeds, return_dict=return_dict)
        trajectory_label = info_dict["trajectory_label"]

        loss = torch.tensor(0, dtype=torch.float32, device=transformer_outputs_hidden_state.device)
        traj_loss, traj_logits = self.traj_decoder.compute_traj_loss(transformer_outputs_hidden_state,
                                                                     trajectory_label,
                                                                     info_dict)
        loss += traj_loss
        loss_items = dict(
            traj_loss=traj_loss,
        )

        pred_dict = {"traj_logits": traj_logits}

        if self.use_proposal:
            if self.config.task == "nuplan":
                proposal_loss, pred_proposal_cls = self.proposal_decoder.compute_proposal_loss(transformer_outputs_hidden_state, info_dict)
                loss += proposal_loss
                loss_items["proposal_loss"] = proposal_loss
                pred_dict["proposal"] = pred_proposal_cls
                # debugging
                pred_proposal_score = pred_proposal_cls.softmax(-1)
                topk_score, topk_indx = torch.topk(pred_proposal_score[:, 0, :], dim=-1, k=self.k)
                # print('test inspect model.py forward: ', self.training, info_dict['halfs_intention'], topk_score, topk_indx,
                #       proposal_loss, pred_proposal_score)

            elif self.config.task == "waymo":
                proposal_loss, proposal_loss_logits = self.proposal_decoder.compute_proposal_loss(transformer_outputs_hidden_state, info_dict)
                loss += proposal_loss
                loss += proposal_loss_logits
                loss_items["proposal_loss"] = proposal_loss
                pred_dict["proposal"] = proposal_loss_logits

        if self.config.dense_pred:
            assert self.config.task == "waymo"
            loss += info_dict["dense_pred_loss"]
            loss_items["dense_pred_loss"] = info_dict["dense_pred_loss"]

        if self.use_key_points != 'no':
            if self.config.generate_diffusion_dataset_for_key_points_decoder:
                future_key_points = info_dict["future_key_points"] if self.config.predict_yaw else \
                            info_dict["future_key_points"][..., :2]
                self.key_points_decoder.save_features(input_embeds,info_dict["context_length"],info_dict,future_key_points,transformer_outputs_hidden_state)

            if self.config.kp_decoder_type == "diffusion":
                assert not self.training, "please train diffusion decoder separately."
                # return a dummy loss&kp_logits here. The real data for computing metrics will be computed in the generate function
                kp_loss = torch.tensor(0.0).to(transformer_outputs_hidden_state.device)
                kp_logits = info_dict["future_key_points"].to(transformer_outputs_hidden_state.device) if self.config.predict_yaw else \
                            info_dict["future_key_points"][..., :2].to(transformer_outputs_hidden_state.device)
            else:
                kp_loss, kp_logits = self.key_points_decoder.compute_keypoint_loss(transformer_outputs_hidden_state, info_dict)
                # kp_loss will be 10x larger than traj_loss when converged

            loss += kp_loss
            if 'denoise_kp' in self.encoder.use_key_points:
                # pad zero to the shape from 2 to 4 on the last dimension
                kp_pad_logits = torch.cat([kp_logits, torch.zeros(kp_logits.shape[0], kp_logits.shape[1], 2).to(kp_logits.device)], dim=-1)
                traj_logits = torch.cat([kp_pad_logits, traj_logits], dim=1)
            else:
                traj_logits = torch.cat([kp_logits, traj_logits], dim=1)
            pred_dict["kp_logits"] = kp_logits
            loss_items["kp_loss"] = kp_loss

        # if not return_dict:
        #     output = (traj_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return LTMOutput(
            loss=loss,
            logits=traj_logits,  # deprecated, use pred_dict for evaluation instead
            pred_dict=pred_dict,
            hidden_states=transformer_outputs_hidden_state,
            loss_items=loss_items
        )

    def embedding_to_hidden(self, input_embeds, attention_mask=None, position_ids=None, return_dict=True):
        # default as the gpt2 model output
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            # **kwargs
        )
        return transformer_outputs['last_hidden_state']

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.FloatTensor:
        input_embeds, info_dict = self.encoder(is_training=False, **kwargs)
        batch_size, _, _ = input_embeds.shape
        device = input_embeds.device
        context_length = info_dict["context_length"]

        if self.use_proposal:
            if self.config.autoregressive_proposals:
                # TODO: Training for debugging results
                proposal_result = []
                proposal_scores = []
                assert self.config.task == 'nuplan', 'waymo proposal autoregressive not implemented yet'
                dummy_proposal_embedding = self.encoder.proposal_m_embed(torch.zeros((batch_size, int(self.config.proposal_num), 1), device=device))  # bsz, 16, 256
                input_embeds[:, context_length:context_length+int(self.config.proposal_num), :] = dummy_proposal_embedding
                # loop over each intention for generation
                for i in range(int(self.config.proposal_num)):
                    context_embeds = input_embeds[:, :context_length + 1 + i, :]
                    attention_mask = torch.ones(context_embeds.shape[:2], dtype=torch.long, device=device)
                    position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                    transformer_outputs_hidden_state = self.embedding_to_hidden(context_embeds)
                    proposal_hidden_state = transformer_outputs_hidden_state[:, context_length - 1 + i:context_length - 1 + 1 + i, :]  # (bs, 1, n_embed)
                    proposal_pred_score = self.proposal_decoder.proposal_cls_decoder(proposal_hidden_state).softmax(-1)  # (bs, 1, 5)
                    # WARNING: Only tested with self.k = 1
                    topk_score, topk_indx = torch.topk(proposal_pred_score[:, 0, :], dim=-1, k=self.k)
                    # topk_score: (bs, 5) topk_indx: (bs, 1)
                    proposal_pred_embed = self.encoder.proposal_m_embed(topk_indx.float())  # (bs, n_embed)
                    # print('test generate 1: ', topk_indx.unsqueeze(-1).float().shape, topk_indx, topk_score, proposal_pred_score[:, 0, :])
                    proposal_result.append(topk_indx.unsqueeze(1))  # list of (bs, 1, 1)
                    proposal_scores.append(proposal_pred_score[:, 0, :].unsqueeze(1))  # list of (bs, 1, 13)
                    input_embeds[:, context_length+i:context_length+i+1, :] = proposal_pred_embed.unsqueeze(1)
                proposal_result = torch.cat(proposal_result, dim=1)  # (bs, 13, 1)
                proposal_scores = torch.cat(proposal_scores, dim=1)  # (bs, 13, 5)
            else:
                if self.config.task == "nuplan":
                    dummy_proposal_embedding = self.encoder.proposal_m_embed(torch.zeros((batch_size, 1), device=device)).unsqueeze(1)
                elif self.config.task == 'waymo':
                    dummy_proposal_embedding = self.encoder.proposal_m_embed(torch.zeros((batch_size, 2), device=device)).unsqueeze(1)
                input_embeds[:, context_length:context_length+1, :] = dummy_proposal_embedding

                context_embeds = input_embeds[:, :context_length+1, :]
                attention_mask = torch.ones(context_embeds.shape[:2], dtype=torch.long, device=device)
                transformer_outputs_hidden_state = self.embedding_to_hidden(
                    input_embeds=context_embeds,
                    attention_mask=attention_mask,
                    position_ids=self._prepare_position_ids_for_generation(attention_mask.clone())
                )
                proposal_hidden_state = transformer_outputs_hidden_state[:, context_length-1:context_length-1+1, :] # (bs, 1, n_embed)
                proposal_pred_score = self.proposal_decoder.proposal_cls_decoder(proposal_hidden_state).softmax(-1) # (bs, 1, 64/5)
                if self.config.task == "nuplan":
                    # WARNING: Only tested with self.k = 1
                    topk_score, topk_indx = torch.topk(proposal_pred_score[:, 0, :], dim=-1, k=self.k)
                    # topk_score: (bs, 5), topk_indx: (bs, 1)
                    proposal_pred_embed = self.encoder.proposal_m_embed(topk_indx.float())  # (bs, n_embed)
                    # print('test generate 1: ', topk_indx.unsqueeze(-1).float().shape, topk_indx, topk_score, proposal_pred_score[:, 0, :])
                    proposal_result = topk_indx
                    proposal_scores = proposal_pred_score[:, 0, :]
                    # proposal_pred_embed: (bs, k, n_embed)
                elif self.config.task == 'waymo':
                    proposal_logit = info_dict["center_obj_proposal_pts"] # (bs, 64, 2)
                    topk_score, topk_indx = torch.topk(proposal_pred_score[:, 0, :], dim=-1, k=self.k)
                    proposal_pred_logit = proposal_logit[torch.arange(batch_size)[:, None].repeat(1, self.k).view(-1), topk_indx.view(-1), :].view(batch_size, self.k, 2)
                    proposal_pred_embed = self.encoder.proposal_m_embed(proposal_pred_logit)
                    proposal_result = topk_indx
                    proposal_scores = proposal_pred_score[:, 0, :]

        traj_logits_k = []
        key_points_logits_k = []
        for mode in range(self.k):
            # print('test generate 2: ', proposal_pred_embed.shape)
            if self.use_proposal:
                if self.config.autoregressive_proposals:
                    # already updated in previous step
                    pass
                else:
                    if self.config.task == "nuplan":
                        input_embeds[:, context_length:context_length + 1, :] = proposal_pred_embed.unsqueeze(1)
                    elif self.config.task == 'waymo':
                        input_embeds[:, context_length:context_length + 1, :] = proposal_pred_embed.unsqueeze(2)[:, mode, :, :]
            if self.use_key_points != "no":
                pred_length = info_dict["pred_length"]
                selected_indices = self.encoder.selected_indices
                kp_start_index = context_length
                if self.use_proposal:
                    if self.config.autoregressive_proposals:
                        kp_start_index += int(self.config.proposal_num)
                    else:
                        kp_start_index += 1
                # pass the following infos during generate for one sample (non-batch) generate with KP checking
                map_name = kwargs.get("map", None)
                route_ids = kwargs.get("route_ids", None)
                ego_pose = kwargs.get("ego_pose", None)
                road_dic = kwargs.get("road_dic", None)
                idm_reference_global = kwargs.get("idm_reference_global", None)  # WIP, this was not fulled tested
                trajectory_label_dummy = torch.zeros((batch_size, pred_length, 4), device=device)
                if 'specified' in self.use_key_points:
                    future_key_points = trajectory_label_dummy[:, selected_indices, :]
                elif 'denoise_kp' in self.encoder.use_key_points:
                    future_key_points = trajectory_label_dummy[:, selected_indices, :2]
                else:
                    ar_future_interval = 20
                    future_key_points = trajectory_label_dummy[:, ar_future_interval - 1::ar_future_interval, :]

                assert future_key_points.shape[1] > 0, 'future points not enough to sample'

                if self.config.task == "nuplan" and not self.config.separate_kp_encoder and 'denoise_kp' not in self.use_key_points:
                    if self.config.use_speed:
                        # padding speed, padding the last dimension from 4 to 7
                        future_key_points = torch.cat([future_key_points, torch.zeros_like(future_key_points)[:, :, :3]], dim=-1)
                    future_key_embeds_dummy = self.encoder.action_m_embed(future_key_points)
                else:
                    future_key_embeds_dummy = self.encoder.kps_m_embed(future_key_points)

                key_points_num = future_key_points.shape[1]

                input_embeds[:, kp_start_index:kp_start_index + key_points_num, :] = future_key_embeds_dummy
                pred_key_points_during_generate = []
                for i in range(key_points_num):
                    input_embeds_current = input_embeds[:, :kp_start_index + i, :]
                    attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=input_embeds.device)
                    position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                    transformer_outputs_hidden_state = self.embedding_to_hidden(
                        input_embeds_current,
                        attention_mask,
                        position_ids,
                    )
                    future_key_point_hidden_state = transformer_outputs_hidden_state[:,
                                                    kp_start_index + i - 1,
                                                    :].reshape(batch_size, 1, -1)

                    key_points_logit, _ = self.key_points_decoder.generate_keypoints(future_key_point_hidden_state)
                    if 'denoise_kp' in self.encoder.use_key_points:
                        pred_key_point = key_points_logit.reshape(batch_size, 1, 2)
                    else:
                        pred_key_point = torch.zeros((batch_size, 1, 4), device=device)
                        if self.config.predict_yaw:
                            pred_key_point[:, 0, :] = key_points_logit[:, 0, :]
                        else:
                            pred_key_point[:, 0, :2] = key_points_logit[:, 0, :]

                    off_road_checking = True
                    if off_road_checking and batch_size == 1 and route_ids is not None and road_dic is not None and ego_pose is not None and map_name is not None:
                        from transformer4planning.utils import nuplan_utils
                        if i in [0, 1] and 'backward' in self.use_key_points:
                            # Check key points with map_api
                            # WARNING: WIP, do not use
                            y_inverse = -1 if map_name == 'sg-one-north' else 1
                            pred_key_point_copy = copy.deepcopy(pred_key_point)
                            pred_key_point_copy[0, 0, 1] *= y_inverse
                            pred_key_point_global = nuplan_utils.change_coordination(pred_key_point_copy[0, 0, :2].cpu().numpy(),
                                                                        ego_pose,
                                                                        ego_to_global=True)
                            if isinstance(route_ids, torch.Tensor):
                                route_ids = route_ids.cpu().numpy().tolist()
                            closest_lane_point_on_route, dist, on_road = nuplan_utils.get_closest_lane_point_on_route(pred_key_point_global,
                                                                                                                       route_ids,
                                                                                                                       road_dic)
                            if not on_road:
                                pred_key_point_ego = nuplan_utils.change_coordination(closest_lane_point_on_route,
                                                                                      ego_pose,
                                                                                      ego_to_global=False)
                                pred_key_point_ego[1] *= y_inverse
                                pred_key_point[0, 0, :2] = torch.tensor(pred_key_point_ego, device=pred_key_point.device)
                                print(f'Off Road Detected! Replace {i}th key point')

                    if idm_reference_global is not None and 'backward' in self.use_key_points:
                        # replace last key point with IDM reference
                        ego_state_global = idm_reference_global[selected_indices[i]]
                        idm_reference_lastpt_relative = nuplan_utils.change_coordination(np.array([ego_state_global.rear_axle.x,
                                                                                                ego_state_global.rear_axle.y]),
                                                                                        ego_pose,
                                                                                        ego_to_global=False)
                        print('replace key points with IDM reference, index: ', selected_indices[i], pred_key_point[0, 0, :2], idm_reference_lastpt_relative)  # idm relative has an unusual large negative y value?
                        pred_key_point[0, 0, :2] = torch.tensor(idm_reference_lastpt_relative, device=pred_key_point.device)
                        pred_key_point[0, 0, -1] = nuplan_utils.normalize_angle(ego_state_global.rear_axle.heading - ego_pose[-1])

                    if 'denoise_kp' in self.encoder.use_key_points:
                        key_point_embed = self.encoder.kps_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                    else:
                        if self.config.task == "nuplan" and not self.config.separate_kp_encoder:
                            if self.config.use_speed:
                                # padding speed, padding the last dimension from 4 to 7
                                pred_key_point = torch.cat([pred_key_point, torch.zeros_like(pred_key_point)[:, :, :3]], dim=-1)
                            key_point_embed = self.encoder.action_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                        else:
                            key_point_embed = self.encoder.kps_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                    # replace embed at the next position
                    input_embeds[:, kp_start_index + i, :] = key_point_embed[:, 0, :]
                    if self.config.predict_yaw and 'denoise_kp' not in self.encoder.use_key_points:
                        pred_key_points_during_generate.append(pred_key_point[:, 0, :].unsqueeze(1))
                    else:
                        pred_key_points_during_generate.append(pred_key_point[:, 0, :2].unsqueeze(1))
                key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(batch_size, key_points_num, -1)
                key_points_logits_k.append(key_points_logits)

            # generate remaining trajectory
            transformer_outputs_hidden_state = self.embedding_to_hidden(input_embeds)
            # expected shape for pred trajectory is (b, pred_length, 4)
            if self.traj_decoder is not None:
                traj_logits = self.traj_decoder.generate_trajs(transformer_outputs_hidden_state, info_dict)
                traj_logits_k.append(traj_logits)
            else:
                raise NotImplementedError

        key_points_pred_logits = None
        if self.k == 1:
            traj_pred_logits = traj_logits_k[0]
            if len(key_points_logits_k) > 0:
                # WARNING, k select if not implemented for key points
                assert len(key_points_logits_k) == self.k
                key_points_pred_logits = key_points_logits_k[0]
        else:
            traj_pred_logits = torch.stack(traj_logits_k, dim=1)
            if len(key_points_logits_k) > 0:
                assert len(key_points_logits_k) == self.k
                key_points_pred_logits = torch.stack(key_points_logits_k, dim=1)

        pred_dict = {
            "traj_logits": traj_pred_logits
        }

        if key_points_pred_logits is not None:
            pred_dict.update({"key_points_logits": key_points_pred_logits})

        if self.use_proposal:
            pred_dict.update({"proposal": proposal_result})  # topk results
            pred_dict.update({"proposal_scores": proposal_scores})  # topk scores
            if self.config.task == "nuplan" and 'halfs_intention' in info_dict:
                pred_dict.update({"halfs_intention": info_dict['halfs_intention']})
            elif self.config.task == 'nuplan' and 'intentions' in info_dict:
                pred_dict.update({'intentions': info_dict['intentions']})

        if self.config.task == "waymo":
            center_objects_world = kwargs['center_objects_world'].type_as(traj_pred_logits)
            num_center_objects, num_modes, num_timestamps, num_feat = traj_pred_logits.shape

            from transformer4planning.utils.waymo_utils import rotate_points_along_z, str_to_tensor

            pred_trajs_world = rotate_points_along_z(
                points=traj_pred_logits.view(num_center_objects, num_modes * num_timestamps, num_feat),
                angle=center_objects_world[:, 6].view(num_center_objects)
            ).view(num_center_objects, num_modes, num_timestamps, num_feat)
            pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

            pred_dict = {
                'scenario_id': str_to_tensor(kwargs['scenario_id']).to(device),
                'pred_trajs': pred_trajs_world[:, :, :, 0:2],
                'pred_scores': topk_score,
                'object_id': torch.tensor(kwargs['center_objects_id']).to(device),
                'object_type': torch.tensor(kwargs['center_objects_type']).to(device),
                'gt_trajs': kwargs['center_gt_trajs_src'],
                'track_index_to_predict': kwargs['track_index_to_predict'],
            }

        return pred_dict


def build_models(model_args):
    if 'gpt' in model_args.model_name:
        from transformer4planning.models.backbone.gpt2 import STR_GPT2, STRGPT2Config
        config_p = STRGPT2Config()
        config_p.update_by_model_args(model_args)
        # from transformer4planning.models.backbone.gpt2 import TrajectoryGPT
        ModelCls = STR_GPT2
        tag = 'GPTTrajectory'
        if 'gpt-mini' in model_args.model_name:
            """
            Number of parameters: 300k
            """
            config_p.n_layer = 1
            config_p.n_embd = config_p.d_model = 64
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 1
        elif 'gpt-small' in model_args.model_name:
            """
            Number of parameters: 16M
            """
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
        elif 'gpt-medium' in model_args.model_name:
            """
            Number of parameters: 124M
            """
            config_p.n_layer = 12
            config_p.n_embd = config_p.d_model = 768
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 12
        elif 'gpt-large' in model_args.model_name:
            """
            Number of parameters: 1.5B
            """
            config_p.n_layer = 48
            config_p.n_embd = config_p.d_model = 1600
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 25
        else:
            print('Warning: using default GPT2 config')
            config_p.n_layer = model_args.n_layers
            config_p.n_embd = model_args.d_embed
            config_p.n_inner = model_args.d_inner
            config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function

        if model_args.task == "train_diffusion_decoder":
            from transformer4planning.models.decoder.diffusion_decoder import (KeypointDiffusionModel, T4PTrainDiffWrapper)
            out_features = 4 if model_args.predict_yaw else 2
            diffusion_model = KeypointDiffusionModel(config_p.n_inner,
                                                     config_p.n_embd,
                                                     out_features=out_features,
                                                     key_point_num=1,
                                                     input_feature_seq_lenth=model_args.diffusion_condition_sequence_lenth,
                                                     use_key_points=model_args.use_key_points,
                                                     feat_dim=model_args.key_points_diffusion_decoder_feat_dim,)
            model = T4PTrainDiffWrapper(diffusion_model, num_key_points=model_args.key_points_num, model_args=config_p)
            if model_args.key_points_diffusion_decoder_load_from is not None:
                state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
                model.load_state_dict(state_dict)
                print("Pretrained keypoint decoder has been loaded!")
            print("Only diffusion decoder will be trained singlely!")
            return model
        # whole model training
    elif 'mamba' in model_args.model_name:
        from transformer4planning.models.backbone.mamba import STRMamba, STRMambaConfig
        config_p = STRMambaConfig()
        config_p.update_by_model_args(model_args)
        ModelCls = STRMamba
        tag = 'MambaTrajectory'
        if 'mamba-mini' in model_args.model_name:
            """
            Number of parameters: ?
            """
            config_p.n_layer = 1
            config_p.n_embd = config_p.d_model = 64
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 1
        elif 'mamba-small' in model_args.model_name:
            """
            Number of parameters: 6M (ViT512)
            """
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
        elif 'mamba-medium' in model_args.model_name:
            """
            Number of parameters: 27M (ViT)
            """
            config_p.n_layer = 8
            config_p.n_embd = config_p.d_model = 512
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 16
        elif 'mamba-large' in model_args.model_name:
            """
            WARNING: Gradient WILL CRUSH DURING TRAINING
            Number of parameters: 139M (ViT)
            """
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 1000
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 25
        elif 'mamba-xl' in model_args.model_name:
            """
            Number of parameters: 764M (ViT)
            """
            config_p.n_layer = 24
            config_p.n_embd = config_p.d_model = 2048
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 32
    elif 'mixtral' in model_args.model_name:
        from transformer4planning.models.backbone.mixtral import STR_Mixtral, STRMixtralConfig
        config_p = STRMixtralConfig()
        config_p.update_by_model_args(model_args)
        ModelCls = STR_Mixtral
        tag = 'MixTralTrajectory'
        if 'mixtral-small' in model_args.model_name:
            config_p.n_layer = 4
            config_p.n_embd = config_p.d_model = 256
            config_p.n_inner = config_p.n_embd * 4
            config_p.n_head = 8
            config_p.num_hidden_layers = 4
            config_p.hidden_size = 256
            config_p.intermediate_size = config_p.n_embd * 4
            config_p.num_attention_heads = 8
        elif 'mixtral-medium' in model_args.model_name:
            """
            Number of parameters: 40M (ViT)
            """
            config_p.n_layer = 16
            config_p.n_embd = config_p.d_model = 1024
            config_p.n_inner = 4096
            config_p.n_head = 16
            config_p.num_hidden_layers = 16
            config_p.hidden_size = 1024
            config_p.intermediate_size = 4096
            config_p.num_attention_heads = 16
        elif 'mixtral-large' in model_args.model_name:
            """
            WARNING: Gradient WILL CRUSH DURING TRAINING
            Number of parameters: 1.3B
            """
            config_p.n_layer = 32
            config_p.n_embd = config_p.d_model = 2048
            config_p.n_inner = 8192
            config_p.n_head = 32
            config_p.num_hidden_layers = 32
            config_p.hidden_size = 2048
            config_p.intermediate_size = 8192
            config_p.num_attention_heads = 32
    else:
        raise ValueError("Model name must choose from ['scratch', 'pretrain'] + ['nonauto-gpt', 'transxl', 'gpt', 'xlnet']!")

    if 'scratch' in model_args.model_name:
        model = ModelCls(config_p)
        print('Scratch ' + tag + ' Initialized!')
    elif 'pretrain' in model_args.model_name:
        model = ModelCls.from_pretrained(model_args.model_pretrain_name_or_path, config=config_p)
        print('Pretrained ' + tag + 'from {}'.format(model_args.model_pretrain_name_or_path))
        if model_args.key_points_diffusion_decoder_load_from is not None:
                print(f"Now loading pretrained key_points_diffusion_decoder from {model_args.key_points_diffusion_decoder_load_from}.")
                state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
                model.key_points_decoder.model.load_state_dict(state_dict)
    elif 'transfer' in model_args.model_name:
        model = ModelCls(config_p)
        print('Transfer' + tag + ' from {}'.format(model_args.model_pretrain_name_or_path))
    return model


def interpolate_yaw(pred_traj, mode, yaw_change_upper_threshold=0.1):
    # deprecated
    if mode == "normal":
        return pred_traj
    elif mode == "interplate" or mode == "hybrid":
        # Warning: this function is tested not better than normal mode
        assert False, "Warning: this function is tested not better than normal mode, to be updated in the future"
        # generating yaw angle from relative_traj
        dx = pred_traj[:, 4::5, 0] - pred_traj[:, :-4:5, 0]
        dy = pred_traj[:, 4::5, 1] - pred_traj[:, :-4:5, 1]
        distances = torch.sqrt(dx ** 2 + dy ** 2)
        relative_yaw_angles = torch.where(distances > 0.1, torch.arctan2(dy, dx), 0)
        # accumulate yaw angle
        # relative_yaw_angles = yaw_angles.cumsum()
        relative_yaw_angles_full = relative_yaw_angles.repeat_interleave(5, dim=1)
        if mode == "interplate":
            pred_traj[:, :, -1] = relative_yaw_angles_full
        else:
            pred_traj[:, :, -1] = torch.where(torch.abs(pred_traj[:, :, -1]) > yaw_change_upper_threshold, relative_yaw_angles_full, pred_traj[:, :, -1])
    return pred_traj
