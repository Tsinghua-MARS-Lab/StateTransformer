from transformers import GPT2Model, GPT2PreTrainedModel
from transformer4planning.models.GPT2.models import *
from transformer4planning.models.decoders import *
from transformer4planning.models.utils import *
from transformer4planning.utils import *
import torch.nn as nn


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

    def build_encoder(self):
        if self.model_args.task == "nuplan":
            from transformer4planning.models.encoders import (NuplanRasterizeEncoder, )
            # TODO: add raster/vector encoder configuration item
            if "raster" in self.model_args.encoder_type:
                cnn_kwargs = dict(
                    d_embed=self.config.n_embd // 2,
                    in_channels=self.model_args.raster_channels,
                    resnet_type=self.model_args.resnet_type,
                    pretrain=self.model_args.pretrain_encoder
                )
                action_kwargs = dict(
                    d_embed=self.config.n_embd
                )
                tokenizer_kwargs = dict(
                    dirpath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpt2-tokenizer'),
                    d_embed=self.config.n_embd,
                )
                self.encoder = NuplanRasterizeEncoder(cnn_kwargs, action_kwargs, tokenizer_kwargs, self.model_args)
            elif "vector" in self.model_args.encoder_type:
                raise NotImplementedError
            else:
                raise AttributeError("encoder_type should be either raster or vector")
        else:
            raise NotImplementedError

    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids
    
    def forward(
            self,
            trajectory_label: Optional[torch.FloatTensor] = None,
            context_actions: Optional[torch.FloatTensor] = None,
            high_res_raster: Optional[torch.LongTensor] = None,
            low_res_raster: Optional[torch.LongTensor] = None,
            scenario_type: Optional[str] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        # gpt non-autoregression version
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = high_res_raster.device
        batch_size, pred_length = trajectory_label.shape[:2]
        context_length = context_actions.shape[1]  # past_interval=10, past_frames=2 * 20, context_length = 40/10=4
        feature_inputs = dict(
            high_res_raster=high_res_raster,
            low_res_raster=low_res_raster,
            context_actions=context_actions,
            trajectory_label=trajectory_label,
            scenario_type=scenario_type,
            pred_length=pred_length,
            context_length=context_length,
        )
        input_embeds, future_key_points, _ = self.encoder(**feature_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            return_dict=return_dict,
            # **kwargs
        )

        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']

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
                loss_fct = MSELoss(reduction="none")
                y_mask = ((trajectory_label != -1).sum(-1) > 0).view(batch_size, pred_length, 1)
                _loss = (loss_fct(traj_logits[..., :2], trajectory_label[..., :2].to(device)) * y_mask).sum() / (y_mask.sum() + 1e-7)
                loss += _loss
            else:
                if self.model_args.predict_yaw:
                    loss += loss_fct(traj_logits, trajectory_label.to(device)) * self.model_args.trajectory_loss_rescale
                else:
                    loss += loss_fct(traj_logits[..., :2], trajectory_label[..., :2].to(device)) * self.model_args.trajectory_loss_rescale
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
            scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
            future_key_points_hidden_state = transformer_outputs_hidden_state[:, scenario_type_len + context_length * 2 - 1:scenario_type_len + context_length * 2 + future_key_points.shape[1] - 1, :]
            key_points_logits = self.key_points_decoder(future_key_points_hidden_state)  # b, s, 4/2*k

            if self.k == 1:
                if self.model_args.predict_yaw:
                    loss_to_add = loss_fct(key_points_logits, future_key_points.to(device))
                else:
                    loss_to_add = loss_fct(key_points_logits, future_key_points[..., :2].to(device))
                loss += loss_to_add
                traj_logits = torch.cat([key_points_logits, traj_logits], dim=1)
            else:
                b, s, c = future_key_points.shape
                k_results = key_points_logits.reshape(b, s, self.k, -1)

                # get loss of minimal loss from k results
                k_future_key_points = future_key_points.unsqueeze(2).repeat(1, 1, self.k, 1).reshape(b, s, self.k, -1)
                loss_fct_key_points = MSELoss(reduction="none")
                if self.model_args.predict_yaw:
                    loss_to_add = loss_fct_key_points(k_results, k_future_key_points.to(device))
                else:
                    loss_to_add = loss_fct_key_points(k_results, k_future_key_points[..., :2].to(device))
                # add loss on x, y (the last dimension)
                loss_to_add = loss_to_add.sum(dim=-1)  # b, s, k
                min_loss, min_loss_indices = torch.min(loss_to_add, dim=2)  # b, s
                loss += min_loss.mean()
                if self.next_token_scorer_decoder is not None:
                    pred_logits = self.next_token_scorer_decoder(future_key_points_hidden_state.to(device))  # b, s, k
                    loss_fct = CrossEntropyLoss(reduction="mean")
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
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        pred_length = kwargs.get("pred_length", None)
        trajectory_label = kwargs.get("trajectory_label", None)
        context_actions = kwargs.get("context_actions", None)
        # pass the following infos during generate for one sample (non-batch) generate with KP checking
        map_api = kwargs.get("map_api", None)
        route_ids = kwargs.get("route_ids", None)
        ego_pose = kwargs.get("ego_pose", None)
        road_dic = kwargs.get("road_dic", None)
        scenario_type = kwargs.get("scenario_type", None)
        idm_reference_global = kwargs.get("idm_reference_global", None)
        """
        Used for generate with key points
        """
        device = high_res_raster.device
        batch_size, pred_length = trajectory_label.shape[:2]
        context_length = context_actions.shape[1]
        feature_inputs = dict(
            high_res_raster=high_res_raster,
            low_res_raster=low_res_raster,
            context_actions=context_actions,
            trajectory_label=trajectory_label,
            scenario_type=scenario_type,
            pred_length=pred_length,
            context_length=context_length,
        )
        input_embeds, _, selected_indices = self.encoder(**feature_inputs)
        scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0

        assert self.ar_future_interval > 0, 'ar_future_interval should be larger than 0, else do not use generate'
        trajectory_label_dummy = torch.zeros((batch_size, pred_length, 4), device=device)
        if self.model_args.specified_key_points:
            future_key_points = trajectory_label_dummy[:, selected_indices, :]
        else:
            future_key_points = trajectory_label_dummy[:, self.ar_future_interval - 1::self.ar_future_interval, :]
        assert future_key_points.shape[1] > 0, 'future points not enough to sample'
        future_key_embeds_dummy = self.encoder.action_m_embed(future_key_points)
        key_points_num = future_key_points.shape[1]
        input_embeds[:, scenario_type_len + context_length * 2:scenario_type_len + context_length * 2 + key_points_num, :] = future_key_embeds_dummy
        pred_key_points_during_generate = []
        # Loop for generation
        for i in range(key_points_num):
            input_embeds_current = input_embeds[:, :scenario_type_len + context_length * 2 + i, :]
            attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=input_embeds.device)
            position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
            transformer_output = self.transformer(
                inputs_embeds=input_embeds_current,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']
            future_key_point_hidden_state = transformer_outputs_hidden_state[:,
                                            scenario_type_len + context_length * 2 + i - 1,
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

            off_road_checking = False
            if off_road_checking and batch_size == 1 and map_api is not None and route_ids is not None and road_dic is not None:
                # Check key points with map_api
                # WARNING: WIP, do not use
                pred_key_point_global = change_coordination(pred_key_point[0, 0, :2].cpu().numpy(),
                                                            ego_pose,
                                                            ego_to_global=True)
                closest_lane_road_dic = query_current_lane(map_api=map_api, target_point=pred_key_point_global)
                nearest = closest_lane_road_dic['road_id']
                nearest_lane = closest_lane_road_dic['lane_id']
                dist = closest_lane_road_dic['distance_to_road_block']
                if nearest not in route_ids or dist > 0.5:
                    # off-road, move to nearest lane according to PDMPath
                    dist = euclidean_distance(pred_key_point[0, 0, :2].cpu().numpy(), [0, 0])
                    interpolate_point = center_path.interpolate(np.array([dist]))[0]
                    print('test offroad correction: ', pred_key_point[0, 0, :2].cpu().numpy(), interpolate_point)
                    pred_key_point[0, 0, :2] = torch.tensor(interpolate_point, device=pred_key_point.device)

            if idm_reference_global is not None and i == key_points_num - 1 and not self.model_args.forward_specified_key_points:
                # replace last key point with IDM reference
                ego_state_global = idm_reference_global[selected_indices[-1]]
                idm_reference_lastpt_relative = change_coordination(np.array([ego_state_global.rear_axle.x,
                                                                              ego_state_global.rear_axle.y]),
                                                                    ego_pose,
                                                                    ego_to_global=False)
                print('replace last key point with IDM reference, index: ', selected_indices[-1], pred_key_point[0, 0, :2], idm_reference_lastpt_relative)  # idm relative has an unusual large negative y value?
                pred_key_point[0, 0, :2] = torch.tensor(idm_reference_lastpt_relative, device=pred_key_point.device)
            key_point_embed = self.encoder.action_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
            # replace embed at the next position
            input_embeds[:, scenario_type_len + context_length * 2 + i, :] = key_point_embed[:, 0, :]
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
        future_key_points_hidden_state = transformer_outputs_hidden_state[:, scenario_type_len + context_length * 2 - 1:scenario_type_len + context_length * 2 + future_key_points.shape[1] - 1, :]

        if self.k > 1:
            key_points_logits = self.key_points_decoder(future_key_points_hidden_state)  # b, s, 4/2*k
            pred_logits = self.next_token_scorer_decoder(future_key_points_hidden_state.to(device))  # b, s, k
            selected_key_points = key_points_logits.reshape(batch_size * key_points_num, self.k, -1)[
                                  torch.arange(batch_size * key_points_num),
                                  pred_logits.argmax(dim=-1).reshape(-1),
                                  :].reshape(batch_size, key_points_num, -1)
            key_points_logits = selected_key_points
        elif self.k == 1:
            key_points_logits = self.key_points_decoder(future_key_points_hidden_state)  # b, s, 4/2
            # use previous prediction during generation
            # print('inspect kp: ', key_points_logits, pred_key_points_during_generate)
            key_points_logits = torch.cat(pred_key_points_during_generate, dim=1).reshape(key_points_logits.shape)
        else:
            raise ValueError("illegal k while generating trajectory", self.k)
        # print('Inspect shape in model generate: ', key_points_logits.shape, traj_logits.shape)
        return torch.cat([key_points_logits, traj_logits], dim=1)


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
    elif 'waymomodel' in model_args.model_name:
        config_p = GPT2Config()
        config_p.n_layer = model_args.n_layers
        config_p.n_embd = model_args.d_embed
        config_p.n_inner = model_args.d_inner
        config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function
        from .waymo_model import GPTModelWaymo
        ModelCls = GPTModelWaymo
        tag = 'waymomodel'
    elif 'demo' in model_args.model_name:
        config_p = GPT2Config()
        config_p.n_layer = model_args.n_layers
        config_p.n_embd = model_args.d_embed
        config_p.n_inner = model_args.d_inner
        config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function
        from .waymo_model import GPTModelDemo
        ModelCls = GPTModelDemo
        tag = 'demo'
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
        from transformer4planning.models.diffusion_decoders import DiffusionDecoderTFBasedForKeyPoints
        state_dict = torch.load(model_args.key_points_diffusion_decoder_load_from)
        pretrained_key_points_diffusion_decoder = DiffusionDecoderTFBasedForKeyPoints(1024, 256, out_features = 4 if model_args.predict_yaw else 2, feat_dim=model_args.key_points_diffusion_decoder_feat_dim, num_key_points = model_args.key_points_num, input_feature_seq_lenth = model_args.diffusion_condition_sequence_lenth)
        pretrained_key_points_diffusion_decoder.load_state_dict(state_dict)
        model.key_points_decoder = pretrained_key_points_diffusion_decoder
        
    return model
