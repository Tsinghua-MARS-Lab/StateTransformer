import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from transformers import (GPT2Model, GPT2PreTrainedModel, GPT2Config)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformer4planning.libs.models.mlp import DecoderResCat
from transformer4planning.utils import nuplan_utils 

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_items: Optional[Dict[str, torch.FloatTensor]] = None

class TrajectoryGPT(GPT2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.transformer = GPT2Model(config)
        self.model_args = kwargs["model_args"]
        self.traj_decoder = None
        self.k = int(self.model_args.k)
        self.ar_future_interval = self.model_args.ar_future_interval
        self.model_parallel = False
        self.device_map = None

        self.next_token_scorer_decoder = None
        self.key_points_decoder = None
        out_features = 4 if self.model_args.predict_yaw else 2
        if not self.model_args.pred_key_points_only:
            self.traj_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=out_features)
        if self.ar_future_interval > 0:
            self.key_points_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=out_features * self.k)
        if self.k > 1:
            self.next_token_scorer_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.k)

        self.diff_series = self.model_args.diff_series

        self.clf_metrics = None
        # Initialize weights and apply final processing
        self.post_init()
        self.build_encoder()
        
    def build_encoder(self):
        if self.model_args.task == "nuplan":
            tokenizer_kwargs = dict(
                dirpath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer', 'gpt2-tokenizer'),
                d_embed=self.config.n_embd,
            )
            if "raster" in self.model_args.encoder_type:
                from transformer4planning.models.encoder.nuplan_raster_encoder import NuplanRasterizeEncoder
                cnn_kwargs = dict(
                    d_embed=self.config.n_embd // 2,
                    in_channels=self.model_args.raster_channels,
                    resnet_type=self.model_args.resnet_type, 
                    pretrain=self.model_args.pretrain_encoder
                )
                action_kwargs = dict(
                    d_embed=self.config.n_embd
                )
                self.encoder = NuplanRasterizeEncoder(cnn_kwargs, action_kwargs, tokenizer_kwargs, self.model_args)
            elif "vector" in self.model_args.encoder_type:
                from transformer4planning.models.encoder.pdm_encoder import PDMEncoder
                pdm_kwargs = dict(
                    hidden_dim=self.config.n_embd,
                    centerline_dim=120,
                    history_dim=20
                )
                self.encoder = PDMEncoder(pdm_kwargs, tokenizer_kwargs, self.model_args)
            else:
                raise AttributeError("encoder_type should be either raster or vector")
        elif self.model_args.task == "waymo":
            from transformer4planning.models.encoder.mtr_encoder import WaymoVectorizeEncoder
            from dataset_gen.waymo.config import cfg_from_yaml_file, cfg
            cfg_from_yaml_file(self.model_args.mtr_config_path, cfg)
            action_kwargs = dict(
                d_embed=self.config.n_embd
            )
            tokenizer_kwargs = dict(
                dirpath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpt2-tokenizer'),
                d_embed=self.config.n_embd,
                max_token_len=self.model_args.max_token_len,
            ) if self.model_args.token_scenario_tag else None
            self.encoder = WaymoVectorizeEncoder(cfg, action_kwargs, tokenizer_kwargs, self.model_args)
        else:
            raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError

    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def from_joint_to_marginal(self, hidden_state, info_dict):
        agents_num_per_scenario = info_dict["agents_num_per_scenario"]
        scenario_num, _, _ = hidden_state.shape
        assert len(agents_num_per_scenario) == scenario_num
        hidden_state_marginal = []
        for i in range(scenario_num):
            agents_num = agents_num_per_scenario[i]
            for j in range(agents_num):
                hidden_state_marginal.append(hidden_state[i, j::agents_num, :])
        hidden_state_marginal = torch.stack(hidden_state_marginal)
        return hidden_state_marginal

    def forward(
            self,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        # gpt non-autoregression version
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_embeds, info_dict = self.encoder(**kwargs)

        attention_mask = info_dict["instance_valid_mask"] if self.model_args.task == "waymo" else None
        # attention_mask = None
        # print(attention_mask.shape, input_embeds.shape)
        # exit()
        device = input_embeds.device
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=return_dict,
            # **kwargs
        )

        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        
        pred_length = info_dict["pred_length"]
        trajectory_label = info_dict["trajectory_label"]
        context_length = info_dict["context_length"]
        
        if not self.diff_series: context_length = context_length * 2

        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length - 1:-1, :]
        # expected shape for pred trajectory is (b, pred_length, 4)
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        if 'mse' in self.model_args.loss_fn:
            loss_fct = nn.MSELoss(reduction="mean")
        elif 'l1' in self.model_args.loss_fn:
            loss_fct = nn.SmoothL1Loss()
        if not self.model_args.pred_key_points_only:
            traj_logits = self.traj_decoder(traj_hidden_state)
            if self.model_args.task == "waymo":
                trajectory_label_mask = info_dict["trajectory_label_mask"]
                loss_fct = nn.MSELoss(reduction="none")
                _loss = (loss_fct(traj_logits[..., :2], trajectory_label[..., :2].to(device)) * trajectory_label_mask).sum() / (
                            trajectory_label_mask.sum() + 1e-7)
                loss += _loss
            else:
                if self.model_args.predict_yaw:
                    loss += loss_fct(traj_logits, trajectory_label.to(device)) * self.model_args.trajectory_loss_rescale
                else:
                    loss += loss_fct(traj_logits[..., :2], trajectory_label[...,
                                                           :2].to(device)) * self.model_args.trajectory_loss_rescale
        else:
            traj_logits = torch.zeros_like(trajectory_label[..., :2])

        if self.ar_future_interval > 0:
            """
            for example:
            context_length = 2
            FutureKeyPoints = 2
            input_embed: [O, A, O, A, FutureKey1, FutureKey2, Traj1(Given0), Traj2(Given0)..]
            output_embed: [A, O, A, FutureKey1, FutureKey2, Traj1, Traj2.., x(Attentionally Blank)]
            """
            future_key_points = info_dict["future_key_points"]
            additional_token_num = 0
            additional_token_num += self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
            additional_token_num += 1 if self.model_args.route_in_separate_token else 0
            future_key_points_hidden_state = transformer_outputs_hidden_state[:, additional_token_num + context_length - 1:additional_token_num + context_length + future_key_points.shape[1] - 1, :]
            key_points_logits = self.key_points_decoder(future_key_points_hidden_state)  # b, s, 4/2*k

            if self.k == 1:
                if self.model_args.predict_yaw:
                    loss_to_add = loss_fct(key_points_logits, future_key_points.to(device))
                else:
                    loss_to_add = loss_fct(key_points_logits, future_key_points[..., :2].to(device))
                if self.model_args.task == "waymo":
                    future_key_points_gt_mask = info_dict["future_key_points_gt_mask"]
                    loss_to_add = (loss_to_add* future_key_points_gt_mask).sum() / (future_key_points_gt_mask.sum() + 1e-7)
                loss += loss_to_add
                traj_logits = torch.cat([key_points_logits, traj_logits], dim=1)
            else:
                b, s, c = future_key_points.shape
                k_results = key_points_logits.reshape(b, s, self.k, -1)

                # get loss of minimal loss from k results
                k_future_key_points = future_key_points.unsqueeze(2).repeat(1, 1, self.k, 1).reshape(b, s, self.k, -1)
                loss_fct_key_points = nn.MSELoss(reduction="none")
                if self.model_args.predict_yaw:
                    loss_to_add = loss_fct_key_points(k_results, k_future_key_points.to(device))
                else:
                    loss_to_add = loss_fct_key_points(k_results, k_future_key_points[..., :2].to(device))
                # add loss on x, y (the last dimension)
                loss_to_add = loss_to_add.sum(dim=-1)  # b, s, k
                min_loss, min_loss_indices = torch.min(loss_to_add, dim=2)  # b, s
                if self.model_args.task == "waymo":
                    future_key_points_gt_mask = info_dict["future_key_points_gt_mask"]
                    loss += (min_loss.unsqueeze(-1) * future_key_points_gt_mask).sum() / (future_key_points_gt_mask.sum() + 1e-7)
                else:
                    loss += min_loss.mean()
                if self.next_token_scorer_decoder is not None:
                    pred_logits = self.next_token_scorer_decoder(future_key_points_hidden_state.to(device))  # b, s, k
                    loss_fct = nn.CrossEntropyLoss(reduction="mean")
                    loss_to_add = loss_fct(pred_logits.reshape(b * s, self.k).to(torch.float64), min_loss_indices.reshape(-1).long())
                    loss += loss_to_add
                    if self.training:
                        # concatenate the key points with predicted trajectory for evaluation
                        selected_key_points = key_points_logits.reshape(b * s, self.k, -1)[torch.arange(b * s),
                                              min_loss_indices.reshape(-1), :].reshape(b, s, -1)
                    else:
                        # concatenate the key points with predicted trajectory selected from the classifier for evaluation
                        selected_key_points = key_points_logits.reshape(b * s, self.k, -1)[torch.arange(b * s),
                                              pred_logits.argmax(dim=-1).reshape(-1), :].reshape(b, s, -1)
                    traj_logits = torch.cat([selected_key_points, traj_logits], dim=1)
                else:
                    print('WARNING: Randomly select key points for evaluation, try to use next_token_scorer_decoder')
                    traj_logits = torch.cat([key_points_logits[0].reshape(b, s, -1), traj_logits], dim=1)

        # evaluate accuracy if on eval
        if not self.training and self.clf_metrics is not None:
            if self.next_token_scorer_decoder is not None:
                # classification on k predictions
                predictions = torch.argmax(pred_logits, dim=-1)  # b, s, k
                for _, metric in self.clf_metrics.items():
                    metric.add_batch(references=min_loss_indices.reshape(-1), predictions=predictions.reshape(-1))

        if not return_dict:
            output = (traj_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate(self, **kwargs) -> torch.FloatTensor:
        """
        For nuplan generation, the input include those nuplan encoder requires; 
        additionally, it also requires: `map_api`, `route_ids`, `ego_pose`, `road_dic`, `idm_reference_global`
        to post process the generated trajectory which are out of route or out of road

        For waymo generation, the input include a `input_dict` and waymo encoder processes it in its 
        forward function.
        """
        # pass the following infos during generate for one sample (non-batch) generate with KP checking
        map_name = kwargs.get("map", None)
        route_ids = kwargs.get("route_ids", None)
        ego_pose = kwargs.get("ego_pose", None)
        road_dic = kwargs.get("road_dic", None)
        # idm_reference_global = kwargs.get("idm_reference_global", None)  # WIP, this was not fulled tested
        """
        Used for generate with key points
        """
        input_embeds, info_dict = self.encoder(**kwargs)

        selected_indices = info_dict["selected_indices"]
        pred_length = info_dict["pred_length"]
        trajectory_label = info_dict["trajectory_label"]
        context_length = info_dict["context_length"]
        
        if not self.diff_series: context_length = context_length * 2

        device = input_embeds.device
        batch_size = trajectory_label.shape[0]

        additional_token_num = 0
        additional_token_num += self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
        additional_token_num += 1 if self.model_args.route_in_separate_token else 0

        assert self.ar_future_interval > 0, 'ar_future_interval should be larger than 0, else do not use generate'
        trajectory_label_dummy = torch.zeros((batch_size, pred_length, 4), device=device)
        if self.model_args.specified_key_points:
            future_key_points = trajectory_label_dummy[:, selected_indices, :]
        else:
            future_key_points = trajectory_label_dummy[:, self.ar_future_interval - 1::self.ar_future_interval, :]
        assert future_key_points.shape[1] > 0, 'future points not enough to sample'
        future_key_embeds_dummy = self.encoder.action_m_embed(future_key_points)
        key_points_num = future_key_points.shape[1]

        if self.model_args.interaction:
            input_embeds = self.from_joint_to_marginal(input_embeds, info_dict)
        input_embeds[:, additional_token_num + context_length:additional_token_num + context_length + key_points_num, :] = future_key_embeds_dummy
        pred_key_points_during_generate = []

        if self.model_args.task == "waymo":
            length_before_keypoints = additional_token_num + context_length
            if self.model_args.generation_method == 'beam':
                pred_key_points_during_generate, input_embeds_kpts, kpts_scores, _ = self.beam_search(input_embeds, length_before_keypoints, key_points_num, attention_mask=info_dict.get("instance_valid_mask", None))
            else:
                raise NotImplementedError
            
            all_traj_logits = []
            all_kps_logits = []
            n_mode = input_embeds_kpts.shape[1]
            
            for m_i in range(n_mode):
                input_embeds[:, length_before_keypoints:length_before_keypoints+key_points_num, :] = input_embeds_kpts[:, m_i, :, :] # (bs, num_kpts, n_embdes)
                transformer_output = self.transformer(
                    inputs_embeds=input_embeds,
                    attention_mask=info_dict.get("instance_valid_mask", None),
                    position_ids=None,
                )
                transformer_outputs_hidden_state = transformer_output['last_hidden_state']
                
                # get traj_logits
                traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length-1:-1, :]
                traj_logits = self.traj_decoder(traj_hidden_state) # (bs, pred_len, 2)
                all_traj_logits.append(traj_logits[:, None, :, :])
                
            all_traj_logits = torch.cat(all_traj_logits, dim=1) # (bs, n_mode, pred_len, 2)
            all_kps_logits = pred_key_points_during_generate  # (bs, n_mode, kps_num, 4/2)
            
            kpts_scores = kpts_scores.softmax(dim=1) # (bs, k, num_kps)
            
            # use accumulated score
            all_traj_scores = torch.ones((batch_size, n_mode), device=device)
            
            for k_i in range(key_points_num):
                all_traj_scores *= kpts_scores[:, :, k_i]
            all_traj_scores = all_traj_scores / all_traj_scores.sum()
            
            # use last score
            # all_traj_scores = kpts_scores[:, :, -1] # (bs, n_mode)
            return {'key_points_logits': all_kps_logits, 'logits': all_traj_logits, 'scores': all_traj_scores}
        else:
            # Loop for generation
            for i in range(key_points_num):
                input_embeds_current = input_embeds[:, :additional_token_num + context_length + i, :]
                attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=input_embeds.device)
                position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                transformer_output = self.transformer(
                    inputs_embeds=input_embeds_current,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                transformer_outputs_hidden_state = transformer_output['last_hidden_state']
                future_key_point_hidden_state = transformer_outputs_hidden_state[:,
                                                additional_token_num + context_length + i - 1,
                                                :].reshape(batch_size, 1, -1)

                if self.k > 1:
                    key_points_logit = self.key_points_decoder(future_key_point_hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2*k
                    pred_logits = self.next_token_scorer_decoder(future_key_point_hidden_state.to(device)).reshape(batch_size, 1, -1)  # b, 1, k
                    selected_key_point = key_points_logit.reshape(batch_size, self.k, -1)[torch.arange(batch_size),
                                        pred_logits.argmax(dim=-1).reshape(-1), :].reshape(batch_size, 1, -1)
                    key_points_logit = selected_key_point
                else:
                    key_points_logit = self.key_points_decoder(future_key_point_hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2
                pred_key_point = torch.zeros((batch_size, 1, 4), device=device)
                if self.model_args.predict_yaw:
                    pred_key_point[:, 0, :] = key_points_logit[:, 0, :]
                else:
                    pred_key_point[:, 0, :2] = key_points_logit[:, 0, :]

                # self.model_args.generate_with_offroad_correction = True
                if self.model_args.generate_with_offroad_correction and batch_size == 1 and map_name is not None and route_ids is not None and road_dic is not None and map_name != 'sg-one-north':
                    # currently only support batch size 1, i.e. single sample generation
                    pred_key_point_global = change_coordination(pred_key_point[0, 0, :2].cpu().numpy(),
                                                                ego_pose,
                                                                ego_to_global=True)
                    # find closest point on the route
                    closest_point = project_point_to_nearest_lane_on_route(road_dic, route_ids, pred_key_point_global)
                    closest_point_relative = change_coordination(closest_point, ego_pose, ego_to_global=False)
                    pred_key_point[0, 0, :2] = torch.tensor(closest_point_relative[:2], device=pred_key_point.device)
                # if idm_reference_global is not None and i == key_points_num - 1 and not self.model_args.forward_specified_key_points:
                #     # replace last key point with IDM reference
                #     ego_state_global = idm_reference_global[selected_indices[-1]]
                #     idm_reference_lastpt_relative = nuplan_utils.change_coordination(np.array([ego_state_global.rear_axle.x,
                #                                                                                ego_state_global.rear_axle.y]),
                #                                                                      ego_pose,
                #                                                                      ego_to_global=False)
                #     print('replace last key point with IDM reference, index: ', selected_indices[-1], pred_key_point[0, 0, :2], idm_reference_lastpt_relative)  # idm relative has an unusual large negative y value?
                #     pred_key_point[0, 0, :2] = torch.tensor(idm_reference_lastpt_relative, device=pred_key_point.device)
                key_point_embed = self.encoder.action_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                # replace embed at the next position
                input_embeds[:, additional_token_num + context_length + i, :] = key_point_embed[:, 0, :]
                if self.model_args.predict_yaw:
                    pred_key_points_during_generate.append(pred_key_point[:, 0, :].unsqueeze(1))
                else:
                    pred_key_points_during_generate.append(pred_key_point[:, 0, :2].unsqueeze(1))

            # generate remaining trajectory
            transformer_output = self.transformer(
                inputs_embeds=input_embeds,
                attention_mask=None,
                position_ids=None,
            )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']

            traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length - 1:-1, :]
            # expected shape for pred trajectory is (b, pred_length, 4)
            if self.traj_decoder is not None:
                traj_logits = self.traj_decoder(traj_hidden_state)
            else:
                traj_logits = trajectory_label_dummy[..., :2]
            future_key_points_hidden_state = transformer_outputs_hidden_state[:,
                                            additional_token_num + context_length - 1:additional_token_num + context_length +
                                                                                        future_key_points.shape[1] - 1, :]

            if self.k > 1:
                b, _, c = pred_key_points_during_generate[0].shape
                key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(b, -1, c)
            elif self.k == 1:
                key_points_logits = self.key_points_decoder(future_key_points_hidden_state)  # b, s, 4/2
                # use previous prediction during generation
                # print('inspect kp: ', key_points_logits, pred_key_points_during_generate)
                key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(key_points_logits.shape)
            else:
                raise ValueError("illegal k while generating trajectory", self.k)
            # print('Inspect shape in model generate: ', key_points_logits.shape, traj_logits.shape)
            return torch.cat([key_points_logits, traj_logits], dim=1)
    
    def beam_search(self, input_embeds, tot_scenario_contenxt_len, key_points_num, attention_mask=None, num_beam=None, out_num_mode=None, center_obj_anchor_pts=None):
        '''
        input_embeds: (bs, tot_scenario_context_length + num_kps + num_future_frame, n_embed)
        
        return:
            k_input_embeds_kpts: (bs, num_beam, num_kps, n_embed)
            k_kpts_scores: (bs, num_beam, num_kps)
            pred_key_points_during_generate: (bs, num_beam, num_kps, 4)
        '''

        assert self.k > 1
        if num_beam is None:
            num_beam = self.k
            
        if out_num_mode is None:
            out_num_mode = num_beam 
        
        assert num_beam <= self.k
        assert out_num_mode <= num_beam

        self.use_anchor = False
        self.pred_dim = 2
        self.beam_search_temp = 1.0
        
        if self.use_anchor:
            key_points_num += self.anchor_len
        
        device = input_embeds.device
        batch_size, tot_len, n_embed = input_embeds.shape
        pred_key_points_during_generate = torch.zeros((batch_size, num_beam, key_points_num, self.pred_dim), device=device)
        
        k_kpts_scores = torch.zeros((batch_size, num_beam, key_points_num), device=device)
        k_input_embeds = input_embeds[:, None, :, :].repeat(1, num_beam, 1, 1)
        if attention_mask is not None: k_attention_mask = attention_mask[:, None, :].repeat(1, num_beam, 1)
        
        k_kpts_index = torch.zeros((batch_size, num_beam, key_points_num), device=device)
        
        for i in range(key_points_num):
            # prepare attention mask
            k_input_embeds_current = k_input_embeds[:, :, :tot_scenario_contenxt_len + i, :].view(batch_size*num_beam, -1, n_embed)
            if attention_mask is None: 
                k_attention_mask = torch.ones(k_input_embeds_current.shape[:2], dtype=torch.long, device=device)
            else:
                k_attention_mask = k_attention_mask[:, :, :tot_scenario_contenxt_len + i].view(batch_size*num_beam, -1)
            position_ids = self._prepare_position_ids_for_generation(k_attention_mask.clone())
            transformer_output = self.transformer(
                inputs_embeds=k_input_embeds_current,
                attention_mask=k_attention_mask,
                position_ids=position_ids
            )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']
            future_key_point_hidden_state = transformer_outputs_hidden_state[:, [tot_scenario_contenxt_len + i - 1], :] # (bs*num_beam, 1, n_embed)

            if self.use_anchor and i == 0:
                pred_kps_score = self.anchor_cls_decoder(future_key_point_hidden_state).view(batch_size, num_beam, 64).softmax(-1) # (bs, num_beam, 64)
                pred_kps_logit = center_obj_anchor_pts.repeat(1, num_beam, 1) # (bs, num_beam*64, 2)
            else:
                # get topk kps
                pred_kps_logit = self.key_points_decoder(future_key_point_hidden_state) # (bs*num_beam, 1, 4/2 *k)
                pred_kps_logit = pred_kps_logit.view(batch_size, num_beam*self.k, self.pred_dim) # (bs, num_beam*k, 2)
                
                pred_kps_score = self.next_token_scorer_decoder(future_key_point_hidden_state)  # (bs*num_beam, 1, k)
                pred_kps_score = (pred_kps_score.view(batch_size, num_beam, self.k)/self.beam_search_temp).softmax(-1) # (bs, num_beam, k)
            
            if i == 0:
                topk_score, topk_indx = torch.topk(pred_kps_score[:, 0, :], dim=-1, k =num_beam) # (bs, num_beam)
            else:
                # pred_kps_score_accum = k_kpts_scores[:, :, [i-1]].repeat(1, 1, num_beam) * pred_kps_score
                # pred_kps_score = pred_kps_score_accum.view(batch_size, num_beam*self.k) # (bs, num_beam*k)
                pred_kps_score = pred_kps_score.view(batch_size, num_beam*self.k) # (bs, num_beam*k)
                topk_score, topk_indx = torch.topk(pred_kps_score, dim=-1, k =num_beam) # (bs, num_beam)
            
            # topk_score = topk_score.softmax(-1)
            topk_group = torch.div(topk_indx, self.k, rounding_mode='floor')
            
            # pred_kps_logit_topk = []
            # for k_ in range(num_beam):
            #     pred_kps_logit_topk.append(pred_kps_logit[torch.arange(batch_size), topk_indx[:, k_], :][:, None, :]) 
            # pred_kps_logit_topk = torch.cat(pred_kps_logit_topk, dim=1) # (bs, num_beam, 2)
            
            pred_kps_logit_topk = pred_kps_logit[torch.arange(batch_size)[:, None].repeat(1, num_beam).view(-1), topk_indx.view(-1), :].view(batch_size, num_beam, 2) # # (bs, num_beam, 2)

            pred_kps_logit_topk = torch.cat((pred_kps_logit_topk, torch.zeros((batch_size, num_beam, 2), device=device)), dim=-1) # (bs, num_beam, 4)

            # get kps topk embeds
            pred_kps_logit_topk_embed = self.encoder.action_m_embed(pred_kps_logit_topk)  # b, num_beam, n_embed
            
            k_input_embeds[:, :, tot_scenario_contenxt_len + i, :] = pred_kps_logit_topk_embed
            k_kpts_scores[:, :, i] = topk_score
            
            if i > 0:
                k_input_embeds_kpts_prev = torch.zeros((batch_size, num_beam, i, n_embed), device=device)
                k_kpts_scores_prev = torch.zeros((batch_size, num_beam, i), device=device)
                
                for p_i in range(num_beam):
                    k_input_embeds_kpts_prev[:, p_i, :, :] = k_input_embeds[torch.arange(batch_size), topk_group[:, p_i], tot_scenario_contenxt_len: tot_scenario_contenxt_len + i, :]
                    k_kpts_scores_prev[:, p_i, :] = k_kpts_scores[torch.arange(batch_size), topk_group[:, p_i], :i]
                
                k_input_embeds[:, :, tot_scenario_contenxt_len: tot_scenario_contenxt_len + i, :] = k_input_embeds_kpts_prev
                k_kpts_scores[:, :, :i] = k_kpts_scores_prev
                
                k_kpts_scores[:, :, i] *= k_kpts_scores[:, :, i-1]
            
            pred_key_points_during_generate[:, :, i, :] = pred_kps_logit_topk[:, :, :self.pred_dim]
            k_input_embeds_kpts = k_input_embeds[:, :, tot_scenario_contenxt_len: tot_scenario_contenxt_len + key_points_num, :]
            
            k_kpts_index[:, :, i] = topk_indx
        
        return pred_key_points_during_generate[:, 0:out_num_mode, ...], k_input_embeds_kpts[:, 0:out_num_mode, ...], k_kpts_scores[:, 0:out_num_mode, ...], k_kpts_index
    
def query_current_lane(map_api, target_point):
    """
    Query the current road_block id and lane id given a point on the map with map_api from NuPlan.
    Args:
        map_api: NuPlan's Map Api
        target_point: [x, y, ..] in global coordination
    Returns:
        {
            'road_id': int,
            'lane_id': int,
            'distance_to_road_block': float,
            'distance_to_lane': float
        }
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
    point2d = Point2D(target_point[0], target_point[1])
    nearest_road_block_id, distance_to_road_block = map_api.get_distance_to_nearest_map_object(
        point=point2d,
        layer=SemanticMapLayer.ROADBLOCK
    )
    nearest_road_blockc_id, distance_to_road_block_c = map_api.get_distance_to_nearest_map_object(
        point=point2d,
        layer=SemanticMapLayer.ROADBLOCK_CONNECTOR
    )
    nearest_lane_id, distance_to_lane = map_api.get_distance_to_nearest_map_object(
        point=point2d,
        layer=SemanticMapLayer.LANE
    )
    nearest_lanec_id, distance_to_lanec = map_api.get_distance_to_nearest_map_object(
        point=point2d,
        layer=SemanticMapLayer.LANE_CONNECTOR
    )
    # check if on route
    if distance_to_road_block < distance_to_road_block_c:
        nearest_road_blockc_id = int(nearest_road_block_id)
        dist_to_road_block = distance_to_road_block
    else:
        nearest_road_blockc_id = int(nearest_road_blockc_id)
        dist_to_road_block = distance_to_road_block_c
    if distance_to_lane < distance_to_lanec:
        nearest_lane = int(nearest_lane_id)
        dist_to_nearest_lane = distance_to_lane
    else:
        nearest_lane = int(nearest_lanec_id)
        dist_to_nearest_lane = distance_to_lanec
    return {
        'road_id': nearest_road_blockc_id,
        'lane_id': nearest_lane,
        'distance_to_road_block': dist_to_road_block,
        'distance_to_lane': dist_to_nearest_lane
    }


def project_point_to_nearest_lane_on_route(road_dic, route_ids, org_point):
    import numpy as np
    points_of_lane = []
    for each_road_id in route_ids:
        each_road_id = int(each_road_id)
        if each_road_id not in road_dic:
            continue
        road_block = road_dic[each_road_id]
        lanes_in_block = road_block['lower_level']
        for each_lane in lanes_in_block:
            each_lane = int(each_lane)
            if each_lane not in road_dic:
                continue
            points_of_lane.append(road_dic[each_lane]['xyz'])
    if len(points_of_lane) <= 1:
        print('Warning: No lane found in route, return original point.')
        return org_point
    points_np = np.vstack(points_of_lane)
    total_points = points_np.shape[0]
    dist_xy = abs(points_np[:, :2] - np.repeat(org_point[np.newaxis, :], total_points, axis=0))
    dist = dist_xy[:, 0] + dist_xy[:, 1]
    minimal_index = np.argmin(dist)
    minimal_point = points_np[minimal_index, :2]
    min_dist = min(dist)
    return minimal_point
    # return minimal_point if min_dist < 10 else org_point


def build_models(model_args):
    if 'vector' in model_args.model_name and 'gpt' in model_args.model_name:
        config_p = None
        if not model_args.autoregressive:
            from transformer4planning.models.vector_model import GPTNonAutoRegressiveModelVector, GPTAutoRegressiveModelVector
            ModelCls = GPTNonAutoRegressiveModelVector
            tag = 'Vector GPT nonauto'
        else:
            ModelCls = GPTAutoRegressiveModelVector
            tag = 'Vector GPT auto'
    elif 'gpt' in model_args.model_name:
        config_p = GPT2Config()
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
            config_p.n_layer = model_args.n_layers
            config_p.n_embd = model_args.d_embed
            config_p.n_inner = model_args.d_inner
            config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function

        if not model_args.generate_diffusion_dataset_for_key_points_decoder:
            if 'diffusion_KP_decoder' not in model_args.model_name:
                ModelCls = TrajectoryGPT
                tag = 'GPTTrajectory'
            else:
                from transformer4planning.models.diffusion_KP_model import TrajectoryGPTDiffusionKPDecoder
                ModelCls = TrajectoryGPTDiffusionKPDecoder
                tag = 'GPTTrajectory with Diffusion Key Point Decoder'
        else:
            from transformer4planning.models.diffusion_KP_model import TrajectoryGPTFeatureGen
            print("Now generating features.")
            ModelCls = TrajectoryGPTFeatureGen
            tag = 'GPTTrajectory: Model For Generating and Saving Features used for training DiffusionKeyPointDecoder'
    elif 'transxl' in model_args.model_name:
        config_p = TransfoXLConfig()
        config_p.pad_token_id = 0
        config_p.eos_token_id = 0
        config_p.n_layer = model_args.n_layers
        config_p.d_embed = model_args.d_embed
        config_p.d_model = model_args.d_model
        config_p.d_inner = model_args.d_inner
        ModelCls = TransfoXLModelNuPlan
        tag = 'TransformerXL'
    elif 'xlnet' in model_args.model_name:
        config_p = XLNetConfig()
        config_p.d_model = model_args.d_model
        config_p.d_inner = model_args.d_inner
        config_p.n_layer = model_args.n_layers
        config_p.ff_activation = model_args.activation_function
        ModelCls = XLNetModelNuplan
        tag = 'XLNet'
    elif 't5' in model_args.model_name:
        config_p = T5Config()
        config_p.num_heads = model_args.n_heads
        config_p.d_model = model_args.d_model
        config_p.d_kv = model_args.d_model // config_p.num_heads
        config_p.d_ff = model_args.d_inner
        config_p.num_layers = model_args.n_layers
        ModelCls = T5ModelNuplan
        tag = 'T5'
    elif 'bert' in model_args.model_name:
        config_p = DebertaV2Config()
        config_p.hidden_size = model_args.d_model
        config_p.intermediate_size = model_args.d_inner
        config_p.num_hidden_layers = model_args.n_layers
        config_p.hidden_act = model_args.activation_function
        config_p.num_attention_heads = model_args.n_heads
        ModelCls = DeBertaNuplan
        tag = 'DeBerta'
    elif 'mmtransformer' in model_args.model_name:
        config_p = GPT2Config()
        config_p.n_layer = model_args.n_layers
        config_p.n_embd = model_args.d_embed
        config_p.n_inner = model_args.d_inner
        config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function
        from transformer4planning.models.mmtransformer.model import MMTransformer
        ModelCls = MMTransformer
        tag = 'mmtransformer'
    else:
        raise ValueError("Model name must choose from ['scratch', 'pretrain'] + ['nonauto-gpt', 'transxl', 'gpt', 'xlnet']!")
    if 'scratch' in model_args.model_name:
        model = ModelCls(config_p, model_args=model_args)
        print('Scratch ' + tag + ' Initialized!')
    elif 'pretrain' in model_args.model_name:
        model = ModelCls.from_pretrained(model_args.model_pretrain_name_or_path, model_args=model_args, config=config_p)
        print('Pretrained ' + tag + 'from {}'.format(model_args.model_pretrain_name_or_path))
    elif 'transfer' in model_args.model_name:
        model = ModelCls(config_p, model_args=model_args)
        print('Transfer' + tag + ' from {}'.format(model_args.model_pretrain_name_or_path))
        
    if model_args.key_points_diffusion_decoder_load_from is not None:
        assert type(model) == TrajectoryGPTDiffusionKPDecoder, ''
        print(f"Now loading pretrained key_points_diffusion_decoder from {model_args.key_points_diffusion_decoder_load_from}.")
        from transformer4planning.models.decoder.diffusion_decoder import DiffusionDecoderTFBasedForKeyPoints
        state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
        pretrained_key_points_diffusion_decoder = DiffusionDecoderTFBasedForKeyPoints(1024, 256, out_features=4 if model_args.predict_yaw else 2, feat_dim=model_args.key_points_diffusion_decoder_feat_dim, num_key_points=model_args.key_points_num, input_feature_seq_lenth=model_args.diffusion_condition_sequence_lenth)
        pretrained_key_points_diffusion_decoder.load_state_dict(state_dict)
        model.key_points_decoder = pretrained_key_points_diffusion_decoder

    return model
