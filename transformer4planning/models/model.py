import os
import torch
import numpy as np
from typing import Tuple, Optional, Dict
from transformers import (GPT2Model, GPT2PreTrainedModel, GPT2Config)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformer4planning.libs.mlp import DecoderResCat
from transformer4planning.utils import nuplan_utils 
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from dataclasses import dataclass

@dataclass
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

        self.clf_metrics = None
        # Initialize weights and apply final processing
        self.post_init()
        self.build_encoder()
        self.build_decoder()
        
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
        # load pretrained diffusion keypoint decoder
        #TODO: add diffusion decoder trained from scratch
        self.decoder_type = self.model_args.decoder_type
        if self.decoder_type == "diffusion":
            from transformer4planning.models.decoder.diffusion_decoder import DiffusionKPTrajDecoder
            self.decoder = DiffusionKPTrajDecoder(self.model_args, self.config)
            if self.model_args.key_points_diffusion_decoder_load_from is not None:
                print(f"Now loading pretrained key_points_diffusion_decoder from {self.model_args.key_points_diffusion_decoder_load_from}.")
                state_dict = torch.load(self.model_args.key_points_diffusion_decoder_load_from)
                self.decoder.load_state_dict(state_dict) 
                print("Pretrained keypoint decoder has been loaded!")
            else:
                print("Now initializing diffusion decoder from scratch. Training will consume lots of time.")
        elif self.decoder_type == "mlp":
            from transformer4planning.models.decoder.mlp_decoder import MultiTrajDecoder
            self.decoder = MultiTrajDecoder(self.model_args, self.config)

        
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

        attention_mask = info_dict["input_embeds_mask"] if self.model_args.interaction else None
        
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=return_dict,
            # **kwargs
        )

        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        
        trajectory_label = info_dict["trajectory_label"]

        traj_logits, loss, loss_items = self.decoder(transformer_outputs_hidden_state, trajectory_label, info_dict)

        if not return_dict:
            output = (traj_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return LTMOutput(
            loss=loss,
            logits=traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            loss_items=loss_items
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
        map_api = kwargs.get("map_api", None)
        route_ids = kwargs.get("route_ids", None)
        ego_pose = kwargs.get("ego_pose", None)
        road_dic = kwargs.get("road_dic", None)
        idm_reference_global = kwargs.get("idm_reference_global", None)
        """
        Used for generate with key points
        """
       
        input_embeds, info_dict  = self.encoder(**kwargs)

        selected_indices = info_dict["selected_indices"]
        pred_length = info_dict["pred_length"]
        trajectory_label = info_dict["trajectory_label"]
        context_length = info_dict.get("context_length", None)
        if context_length is None: # pdm encoder
            input_length = info_dict.get("input_length", None)

        device = input_embeds.device
        batch_size = trajectory_label.shape[0]

        scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0

        # Loop for generation with mlp decoder. Generate key points in autoregressive way.
        if self.decoder_type == "mlp" and self.ar_future_interval > 0:
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
            kp_start_index = scenario_type_len + context_length * 2 if context_length is not None else scenario_type_len + input_length * 2
            input_embeds[:, kp_start_index:kp_start_index + key_points_num, :] = future_key_embeds_dummy
            pred_key_points_during_generate = []
            for i in range(key_points_num):
                input_embeds_current = input_embeds[:, :kp_start_index + i, :]
                attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=input_embeds.device)
                position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
                transformer_output = self.transformer(
                    inputs_embeds=input_embeds_current,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                transformer_outputs_hidden_state = transformer_output['last_hidden_state']
                future_key_point_hidden_state = transformer_outputs_hidden_state[:,
                                                kp_start_index + i - 1,
                                                :].reshape(batch_size, 1, -1)

                if self.k > 1:
                    key_points_logit, pred_logits = self.decoder.generate_keypoints(future_key_point_hidden_state)
                    selected_key_point = key_points_logit.reshape(batch_size, self.k, -1)[torch.arange(batch_size),
                                        pred_logits.argmax(dim=-1).reshape(-1), :].reshape(batch_size, 1, -1)
                    key_points_logit = selected_key_point
                else:
                    key_points_logit, _ = self.decoder.generate_keypoints(future_key_point_hidden_state)
                pred_key_point = torch.zeros((batch_size, 1, 4), device=device)
                if self.model_args.predict_yaw:
                    pred_key_point[:, 0, :] = key_points_logit[:, 0, :]
                else:
                    pred_key_point[:, 0, :2] = key_points_logit[:, 0, :]

                off_road_checking = False
                if off_road_checking and batch_size == 1 and map_api is not None and route_ids is not None and road_dic is not None:
                    # Check key points with map_api
                    # WARNING: WIP, do not use
                    pred_key_point_global = nuplan_utils.change_coordination(pred_key_point[0, 0, :2].cpu().numpy(),
                                                                ego_pose,
                                                                ego_to_global=True)
                    closest_lane_road_dic = query_current_lane(map_api=map_api, target_point=pred_key_point_global)
                    nearest = closest_lane_road_dic['road_id']
                    nearest_lane = closest_lane_road_dic['lane_id']
                    dist = closest_lane_road_dic['distance_to_road_block']
                    if nearest not in route_ids or dist > 0.5:
                        # off-road, move to nearest lane according to PDMPath
                        dist = nuplan_utils.euclidean_distance(pred_key_point[0, 0, :2].cpu().numpy(), [0, 0])
                        interpolate_point = center_path.interpolate(np.array([dist]))[0]
                        print('test offroad correction: ', pred_key_point[0, 0, :2].cpu().numpy(), interpolate_point)
                        pred_key_point[0, 0, :2] = torch.tensor(interpolate_point, device=pred_key_point.device)

                if idm_reference_global is not None and i == key_points_num - 1 and not self.model_args.forward_specified_key_points:
                    # replace last key point with IDM reference
                    ego_state_global = idm_reference_global[selected_indices[-1]]
                    idm_reference_lastpt_relative = nuplan_utils.change_coordination(np.array([ego_state_global.rear_axle.x,
                                                                                ego_state_global.rear_axle.y]),
                                                                        ego_pose,
                                                                        ego_to_global=False)
                    print('replace last key point with IDM reference, index: ', selected_indices[-1], pred_key_point[0, 0, :2], idm_reference_lastpt_relative)  # idm relative has an unusual large negative y value?
                    pred_key_point[0, 0, :2] = torch.tensor(idm_reference_lastpt_relative, device=pred_key_point.device)
                key_point_embed = self.encoder.action_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
                # replace embed at the next position
                input_embeds[:, kp_start_index + i, :] = key_point_embed[:, 0, :]
                if self.model_args.predict_yaw:
                    pred_key_points_during_generate.append(pred_key_point[:, 0, :].unsqueeze(1))
                else:
                    pred_key_points_during_generate.append(pred_key_point[:, 0, :2].unsqueeze(1))
            key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(batch_size, key_points_num, -1)
        
        elif self.decoder_type == "diffusion":
            # TODO:confirm the attention mask here
            transformer_output = self.transformer(
                    inputs_embeds=input_embeds,
                    attention_mask=None,
                    position_ids=None,
                )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']
            key_points_logits, pred_logits = self.decoder.generate_keypoints(transformer_outputs_hidden_state, info_dict)
        else:
            key_points_logits = None
        # predict the whole trajectory
        if self.model_args.interaction:
            input_embeds = self.encoder.from_marginal_to_joint(input_embeds, info_dict, update_info_dict=False)
            attention_mask = info_dict["input_embeds_mask"]
        else:
            attention_mask = None
        # generate remaining trajectory
        transformer_output = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=None,
        )
        transformer_outputs_hidden_state = transformer_output['last_hidden_state']

        if self.model_args.interaction: 
            transformer_outputs_hidden_state = self.from_joint_to_marginal(transformer_outputs_hidden_state, info_dict)

        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length - 1:-1, :]
        # expected shape for pred trajectory is (b, pred_length, 4)
        if self.traj_decoder is not None:
            traj_logits = self.traj_decoder(traj_hidden_state)
        else:
            traj_logits = trajectory_label_dummy[..., :2]

        if key_points_logits is not None:
            return torch.cat([key_points_logits, traj_logits], dim=1)
        else: # predict trajectory directly
            return traj_logits

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


def build_models(model_args):
    if 'vector' in model_args.model_name and 'gpt' in model_args.model_name:
        config_p = GPT2Config()
        config_p.n_layer = model_args.n_layers
        config_p.n_embd = model_args.d_embed
        config_p.n_inner = model_args.d_inner
        config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function
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
        
        if "diffusion" in model_args.task:
            from transformer4planning.models.decoder.diffusion_decoder import (KeypointDiffusionModel, DiffusionWrapper)
            out_features = 4 if model_args.predict_yaw else 2
            diffusion_model = KeypointDiffusionModel(config_p.n_inner, 
                                                    config_p.n_embd, 
                                                    out_features=model_args.k * out_features,
                                                    key_point_num=model_args.key_points_num,
                                                    input_feature_seq_lenth=model_args.diffusion_condition_sequence_lenth,
                                                    specified_key_points=model_args.specified_key_points,
                                                    forward_specified_key_points=model_args.forward_specified_key_points
                                                    )
            model = DiffusionWrapper(diffusion_model, num_key_points=model_args.key_points_num)
            if model_args.key_points_diffusion_decoder_load_from is not None:
                state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
                model.load_state_dict(state_dict)
                print("Pretrained keypoint decoder has been loaded!")
            print("Only diffusion decoder will be trained singlely!")
            return model
        # whole model training
        else:
            ModelCls = TrajectoryGPT
            tag = 'GPTTrajectory'
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
        
    return model
