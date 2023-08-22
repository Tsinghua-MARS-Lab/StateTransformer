import os
import pickle
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SmoothL1Loss
from transformer4planning.models.GPT2.models import *
from transformer4planning.models.decoders import DecoderResCat
from transformer4planning.models.encoder.mtr_encoder import MTREncoder

class GPTAutoRegressiveModelVector(GPT2PreTrainedModel):
    def forward(self, **kwargs):
        pass
    
    
class GPTNonAutoRegressiveModelVector(GPT2PreTrainedModel):
    def __init__(self, config, **kwargs):
        
        model_args = kwargs["model_args"]
        self.model_args = model_args
        vector_model_cfg = model_args.vector_model_cfg
        
        # LLM transformer
        llm_config = GPT2Config()
        llm_config.n_layer = vector_model_cfg.LLM_TRANSFORMER.n_layer
        llm_config.n_embd = vector_model_cfg.LLM_TRANSFORMER.n_embd
        llm_config.n_inner = vector_model_cfg.LLM_TRANSFORMER.n_inner
        llm_config.n_head = vector_model_cfg.LLM_TRANSFORMER.n_head
        llm_config.activation_function = vector_model_cfg.LLM_TRANSFORMER.activation_function
        
        super().__init__(llm_config)
        
        self.transformer = GPT2Model(llm_config)
        
        # encoder
        self.context_encoder = MTREncoder(vector_model_cfg.CONTEXT_ENCODER)
        
        # load intention points
        intention_points_file = vector_model_cfg.MOTION_DECODER.INTENTION_POINTS_FILE
        with open(intention_points_file, 'rb') as f:
            intention_points_dict = pickle.load(f)

        self.intention_points = {}
        for cur_type in vector_model_cfg.MOTION_DECODER.OBJECT_TYPE:
            cur_intention_points = intention_points_dict[cur_type]
            cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
            self.intention_points[cur_type] = cur_intention_points
        
        # decoder
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        self.ar_future_interval = model_args.ar_future_interval
        self.task = model_args.task
        self.action_m_embed = nn.Sequential(nn.Linear(4, llm_config.n_embd), nn.Tanh())
        self.llm_n_embd = llm_config.n_embd

        self.traj_decoder = None
        self.k = int(self.model_args.k)

        self.next_token_scorer_decoder = None
        self.key_points_decoder = None
        out_features = 4 if model_args.predict_yaw else 2
        
        if self.predict_trajectory:
            self.traj_decoder = DecoderResCat(llm_config.n_inner, llm_config.n_embd, out_features=out_features)
            
        if self.ar_future_interval > 0:
            self.key_points_decoder = DecoderResCat(llm_config.n_inner, llm_config.n_embd, out_features=out_features * self.k)
            
        if self.k > 1:
            self.next_token_scorer_decoder = DecoderResCat(llm_config.n_inner, llm_config.n_embd, out_features=self.k)
        
        self.use_anchor = True
        
        if self.use_anchor:
            self.anchor_num = 64
            self.anchor_cls_decoder = DecoderResCat(llm_config.n_inner, llm_config.n_embd, out_features= self.anchor_num)
            self.anchor_logits_decoder = DecoderResCat(llm_config.n_inner, llm_config.n_embd, out_features= out_features * self.anchor_num)
            self.anchor_len = 1
            self.cls_anchor_loss = CrossEntropyLoss(reduction="none")
            self.logits_anchor_loss = MSELoss(reduction="none")
        else:
            self.anchor_len = 0

        self.clf_metrics = None

        # Initialize weights and apply final processing
        self.model_parallel = False
        self.device_map = None
        self.with_traffic_light = model_args.with_traffic_light
        
        if self.model_args.token_scenario_tag:
            self.tag_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpt2-tokenizer'))
            self.tag_tokenizer.pad_token = self.tag_tokenizer.eos_token
            self.tag_embedding = nn.Embedding(self.tag_tokenizer.vocab_size, config.n_embd)
            
            self.scenario_type_len = self.model_args.max_token_len
        else:
            self.scenario_type_len = 0
            
        self.post_init()
        
        # loss
        if 'mse' in self.loss_fn:
            self.reg_trj_loss = MSELoss(reduction="none")
        elif 'l1' in self.loss_fn:
            self.reg_trj_loss = nn.SmoothL1Loss()
            
        if self.ar_future_interval > 0:
            self.reg_kps_loss = self.reg_trj_loss
        
        self.cls_kps_loss = CrossEntropyLoss(reduction="mean")
        self.cls_kps_loss_weight = 1
        self.beam_search_temp = 1.0
        
        if self.model_args.predict_yaw:
            self.pred_dim = 4
        else:
            self.pred_dim = 2 
        
        # self.generate_method = "greedy_search"
        self.generate_method = 'beam_search'
        
        self.tot_iter_num = 0
        self.debug = True

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.traj_decoder = self.traj_decoder.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.traj_decoder = self.traj_decoder.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def _prepare_OA_inputs(self, input_dict, batch_sample_count, scenario_type):
        batch_size = input_dict['obj_trajs'].shape[0]
        device = input_dict['obj_trajs'].device
        
        batch_dict = self.context_encoder({'input_dict': input_dict, 
                                           'batch_sample_count': batch_sample_count})
        
        obj_feature = batch_dict['obj_feature']
        map_feature = batch_dict['map_feature']
        state_embeds = torch.cat((map_feature, obj_feature), dim=1) # (bs, num_poly+num_obj, num_timestamp, 256)
        state_embeds = state_embeds.max(dim=1)[0]
        
        # traj
        trajectory_label = input_dict['trajectory_label']
        trajectory_label_mask = input_dict['center_gt_trajs_mask'].unsqueeze(-1)
        
        # action context
        context_actions = input_dict['center_objects_past']
        if self.model_args.x_random_walk > 0 and self.training:
            x_noise = torch.rand(context_actions.shape, device=device) * self.model_args.x_random_walk * 2 - self.model_args.x_random_walk
            context_actions[:, :, 0] += x_noise[:, :, 0]
        if self.model_args.y_random_walk > 0 and self.training:
            y_noise = torch.rand(context_actions.shape, device=device) * self.model_args.y_random_walk * 2 - self.model_args.y_random_walk
            context_actions[:, :, 1] += y_noise[:, :, 1]

        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1]  # past_interval=10, past_frames=2 * 20, context_length = 40/10=4

        # create OAOAOA..
        input_embeds = torch.zeros(
            (batch_size, context_length * 2, self.llm_n_embd),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds  # index: 0, 2, 4, .., 18
        input_embeds[:, 1::2, :] = action_embeds  # index: 1, 3, 5, .., 19
        
        # add scenario_embedding
        if self.model_args.token_scenario_tag:
            scenario_tag_ids = list()
            for i in range(batch_size):
                scenario_tag_ids.append(torch.tensor(self.tag_tokenizer(scenario_type[i], max_length=self.model_args.max_token_len, padding='max_length')["input_ids"]).unsqueeze(0))
            scenario_tag_ids = torch.stack(scenario_tag_ids, dim=0).to(device)
            scenario_tag_embeds = self.tag_embedding(scenario_tag_ids).squeeze(1)
            input_embeds = torch.cat([scenario_tag_embeds, input_embeds], dim=1)
        
        return input_embeds, context_length, trajectory_label, trajectory_label_mask
        
    def forward(
            self,
            batch_size, input_dict, batch_sample_count,
            context_actions: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
            ):
        
        batch_size = input_dict['obj_trajs'].shape[0]
        device = input_dict['obj_trajs'].device
        scenario_type = kwargs.get("scenario_type", None)
        
        input_embeds, context_length, trajectory_label, trajectory_label_mask = self._prepare_OA_inputs(input_dict, batch_sample_count, scenario_type)
        pred_length = trajectory_label.shape[1]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # add anchor embedding
        if self.use_anchor:
            center_obj_types = input_dict['center_objects_type']
            center_obj_anchor_pts = [self.intention_points[center_obj_types[i]].unsqueeze(0) for i in range(batch_size)]
            center_obj_anchor_pts = torch.cat(center_obj_anchor_pts, dim=0) # (bs, 64, 2)
            dist2GT = torch.norm(trajectory_label[:, [-1], :2] - center_obj_anchor_pts, dim=2)
            anchor_GT_cls = dist2GT[:, :].argmin(dim = 1) # (bs, )
            anchor_GT_logits = center_obj_anchor_pts[torch.arange(batch_size), anchor_GT_cls, :] # (bs, 2)
            anchor_embedding = self.action_m_embed(torch.cat([anchor_GT_logits, torch.zeros((batch_size, 2), device=device)], dim = 1)).unsqueeze(1)
            input_embeds = torch.cat([input_embeds, anchor_embedding], dim=1)
            
            gt_anchor_mask = trajectory_label_mask[:, -1, :] # (bs, 1)

        # add future traj embedding
        if self.ar_future_interval == 0:
            # to keep input and output at the same dimension
            input_embeds = torch.cat([input_embeds,
                                      torch.zeros((batch_size, pred_length, self.llm_n_embd), device=device)], dim=1)
            future_key_points = None
            future_key_points_gt_mask = None
        elif self.ar_future_interval > 0:
            # use autoregressive future interval
            if self.model_args.specified_key_points:
                # 80, 40, 20, 10, 5
                if self.model_args.forward_specified_key_points:
                    selected_indices = [4, 9, 19, 39, 79]
                else:
                    selected_indices = [79, 39, 19, 9, 4]
                future_key_points = trajectory_label[:, selected_indices, :]
                future_key_points_gt_mask = trajectory_label_mask[:, selected_indices, :]
            else:
                future_key_points = trajectory_label[:, self.ar_future_interval - 1::self.ar_future_interval, :]
                future_key_points_gt_mask = trajectory_label_mask[:, self.ar_future_interval - 1::self.ar_future_interval, :]
                
            assert future_key_points.shape[1] != 0, 'future points not enough to sample'

            future_key_points_aug = future_key_points.clone()
            if self.model_args.arf_x_random_walk > 0 and self.training:
                x_noise = torch.rand(future_key_points.shape, device=device) * self.model_args.arf_x_random_walk * 2 - self.model_args.arf_x_random_walk
                # add progressive scale, the future points the larger noise
                if self.model_args.specified_key_points:
                    indices = torch.tensor(selected_indices, device=device, dtype=float) / 80.0
                else:
                    indices = torch.arange(future_key_points.shape[1], device=device) / future_key_points.shape[1]
                expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
                x_noise = x_noise * expanded_indices
                future_key_points_aug[:, :, 0] += x_noise[:, :, 0]
            if self.model_args.arf_y_random_walk > 0 and self.training:
                y_noise = torch.rand(future_key_points.shape, device=device) * self.model_args.arf_y_random_walk * 2 - self.model_args.arf_y_random_walk
                expanded_indices = indices.unsqueeze(0).unsqueeze(-1).expand(future_key_points.shape)
                y_noise = y_noise * expanded_indices
                future_key_points_aug[:, :, 1] += y_noise[:, :, 1]

            if not self.model_args.predict_yaw:
                # keep the same information when generating future points
                future_key_points_aug[:, :, 2:] = 0

            future_key_embeds = self.action_m_embed(future_key_points_aug)
            input_embeds = torch.cat([input_embeds, future_key_embeds,
                                      torch.zeros((batch_size, pred_length, self.llm_n_embd), device=device)], dim=1)
        else:
            raise ValueError("ar_future_interval should be non-negative", self.ar_future_interval)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']

        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length - 1:-1, :]
        pred_traj_logits = None
        if self.predict_trajectory:
            pred_traj_logits = self.traj_decoder(traj_hidden_state)   # (bs, pred_length, 2 * k)
        pred_kps_logits = None
        pred_kps_cls = None
        
        tot_scenario_contenxt_len = self.scenario_type_len + context_length * 2
        tot_scenario_contenxt_anchor_len = self.scenario_type_len + context_length * 2 + self.anchor_len
        
        if self.use_anchor:
            pred_anchor_embed = transformer_outputs_hidden_state[:, tot_scenario_contenxt_len-1 : tot_scenario_contenxt_len-1+self.anchor_len, :] # (bs, anchor_len, n_embed)
            pred_anchor_cls = self.anchor_cls_decoder(pred_anchor_embed) # (bs, anchor_len, 64)
            pred_anchor_logits = self.anchor_logits_decoder(pred_anchor_embed) # (bs, anchor_len, 64 * 2)

        if self.ar_future_interval > 0:
            """
            for example:
            context_length = 2
            FutureKeyPoints = 2
            input_embed: [O, A, O, A, FutureKey1, FutureKey2, Traj1(Given0), Traj2(Given0)..]
            output_embed: [A, O, A, FutureKey1, FutureKey2, Traj1, Traj2.., x(Attentionally Blank)]
            """

            future_key_points_hidden_state = transformer_outputs_hidden_state[:, tot_scenario_contenxt_anchor_len - 1:tot_scenario_contenxt_anchor_len + future_key_points.shape[1] - 1, :]
            pred_kps_logits = self.key_points_decoder(future_key_points_hidden_state)  # (b, num_kps, 4/2*k)
            
            if self.k > 1:
                pred_kps_cls = self.next_token_scorer_decoder(future_key_points_hidden_state.to(device))  # (b, num_kps, k)
        
        # get loss
        pred_traj_logits, loss = self.calc_loss(device=device, pred_traj_logits=pred_traj_logits, 
                              gt_traj=trajectory_label, 
                              gt_traj_mask=trajectory_label_mask, 
                              pred_kps_logits=pred_kps_logits,
                              gt_kps=future_key_points, 
                              gt_kps_mask=future_key_points_gt_mask,
                              pred_kps_cls=pred_kps_cls,
                              pred_anchor_cls=pred_anchor_cls,
                              gt_anchor_cls=anchor_GT_cls,
                              gt_anchor_mask=gt_anchor_mask,
                              pred_anchor_logits=pred_anchor_logits, # (bs, anchor_len, 64 * 2)
                              gt_anchor_logits = anchor_GT_logits, #(bs, 2)
                              )

        if not return_dict:
            output = (pred_traj_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=pred_traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate(self, batch_size, input_dict, batch_sample_count, **kwargs) -> torch.FloatTensor:
        
        batch_size = input_dict['obj_trajs'].shape[0]
        device = input_dict['obj_trajs'].device
        scenario_type = kwargs.get("scenario_type", None)
        
        # input_embeds: (bs, context_length*2 + scenario_len, n_embd)
        input_embeds, context_length, trajectory_label, trajectory_label_mask = self._prepare_OA_inputs(input_dict, batch_sample_count, scenario_type) 
        pred_length = trajectory_label.shape[1]
        
        # anchor embedding
        tot_scenario_contenxt_len = self.scenario_type_len + context_length * 2
        tot_scenario_contenxt_anchor_len = self.scenario_type_len + context_length * 2 + self.anchor_len
        
        if self.use_anchor:
            dummy_anchor_embedding = self.action_m_embed(torch.zeros((batch_size, 4), device=device)).unsqueeze(1) # (bs, 1, n_embed)
            input_embeds = torch.cat([input_embeds, dummy_anchor_embedding], dim=1)
            
            center_obj_types = input_dict['center_objects_type']
            center_obj_anchor_pts = [self.intention_points[center_obj_types[i]].unsqueeze(0) for i in range(batch_size)]
            center_obj_anchor_pts = torch.cat(center_obj_anchor_pts, dim=0) # (bs, 64, 2)
            
            dist2GT = torch.norm(trajectory_label[:, [-1], :2] - center_obj_anchor_pts, dim=2)
            anchor_GT_cls = dist2GT[:, :].argmin(dim = 1) # (bs, )
            
            gt_anchor_mask = trajectory_label_mask[:, -1, :] # (bs, 1)
        else:
            center_obj_anchor_pts = None

        if not self.use_anchor:
            assert self.ar_future_interval > 0, 'ar_future_interval should be larger than 0, else do not use generate'
        if self.ar_future_interval > 0:
            # use autoregressive future interval
            trajectory_label_dummy = torch.zeros((batch_size, pred_length, 4), device=device)
            if self.model_args.specified_key_points:
                # 80, 40, 20, 10, 5
                if self.model_args.forward_specified_key_points:
                    selected_indices = [4, 9, 19, 39, 79]
                else:
                    selected_indices = [79, 39, 19, 9, 4]
                future_key_points = trajectory_label_dummy[:, selected_indices, :]
                future_key_points_gt = trajectory_label[:, selected_indices, :]
            else:
                future_key_points = trajectory_label_dummy[:, self.ar_future_interval - 1::self.ar_future_interval, :]
                future_key_points_gt = trajectory_label[:, self.ar_future_interval - 1::self.ar_future_interval, :]
            assert future_key_points.shape[1] > 0, 'future points not enough to sample'
            future_key_embeds_dummy = self.action_m_embed(future_key_points)
            input_embeds = torch.cat([input_embeds, future_key_embeds_dummy,
                                    torch.zeros((batch_size, pred_length, self.llm_n_embd), device=device)], dim=1)
            key_points_num = future_key_points.shape[1]
        else:
            input_embeds = torch.cat([input_embeds,
                                    torch.zeros((batch_size, pred_length, self.llm_n_embd), device=device)], dim=1)
        
        if self.use_anchor:
            pred_key_points_during_generate, input_embeds_kpts, kpts_scores, kpts_idx = self.beam_search_anchor_only(input_embeds, tot_scenario_contenxt_len, out_num_mode=6,
                                                                                               center_obj_anchor_pts=center_obj_anchor_pts)
            key_points_num = 0
        elif self.generate_method == 'greedy_search':
            pred_key_points_during_generate, input_embeds_kpts, kpts_scores = self.greedy_search(input_embeds, tot_scenario_contenxt_len, key_points_num,
                                                                                                 center_obj_anchor_pts=center_obj_anchor_pts)
        elif self.generate_method == 'beam_search':
            pred_key_points_during_generate, input_embeds_kpts, kpts_scores = self.beam_search(input_embeds, tot_scenario_contenxt_len, key_points_num,
                                                                                               num_beam=self.k, out_num_mode=self.k,
                                                                                               center_obj_anchor_pts=center_obj_anchor_pts)
        else:
            raise ValueError("generate_method has not yet been implemented ", self.generate_method)
        
        all_traj_logits = []
        all_kps_logits = []
        n_mode = input_embeds_kpts.shape[1]
        
        all_kps_logits = pred_key_points_during_generate  # (bs, n_mode, kps_num, 4/2)
        
        # use accumulated score
        all_traj_scores = torch.ones((batch_size, n_mode), device=device) # (bs, n_mode)
        
        # for k_i in range(key_points_num):
        #     all_traj_scores *= kpts_scores[:, :, k_i]
        # all_traj_scores = all_traj_scores / all_traj_scores.sum()
        
        # kpts_score: accumulated score
        all_traj_scores = kpts_scores[:, :, -1] # (bs, n_mode)
        # all_traj_scores = all_traj_scores / all_traj_scores.sum()
        all_traj_scores = (all_traj_scores*200).softmax(-1)
        
        if self.use_anchor:
            hard_match_num = ((kpts_idx[:, 0] == anchor_GT_cls) * gt_anchor_mask[:, 0]).sum()
            soft_match_vec = (kpts_idx[:, 0] == anchor_GT_cls)
            for m_i in range(1, 6):
                soft_match_vec |= (kpts_idx[:, m_i] == anchor_GT_cls)
            soft_match_num = (soft_match_vec * gt_anchor_mask[:, 0]).sum()
            
        out_res = {'key_points_logits': all_kps_logits, 'scores': all_traj_scores, 'anchor_hard_match_num': hard_match_num, 'anchor_soft_match_num': soft_match_num, 'tot_num': batch_size}
        
        if not self.predict_trajectory:
            return out_res
        
        for m_i in range(n_mode):
            input_embeds[:, tot_scenario_contenxt_len:tot_scenario_contenxt_anchor_len+key_points_num, :] = input_embeds_kpts[:, m_i, :, :] # (bs, num_kpts, n_embdes)
            transformer_output = self.transformer(
                inputs_embeds=input_embeds,
                attention_mask=None,
                position_ids=None,
            )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']
            
            # get traj_logits
            traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length-1:-1, :]
            traj_logits = self.traj_decoder(traj_hidden_state) # (bs, pred_len, 2)
            all_traj_logits.append(traj_logits[:, None, :, :])
            
        all_traj_logits = torch.cat(all_traj_logits, dim=1) # (bs, n_mode, pred_len, 2)
        
        out_res.update(logits=all_traj_logits)

        return out_res
    
    def calc_loss(self, device, pred_traj_logits, gt_traj, gt_traj_mask=None, 
                  pred_kps_logits=None, gt_kps=None, gt_kps_mask=None,
                  pred_kps_cls=None,
                  pred_anchor_cls=None,
                  gt_anchor_cls=None,
                  gt_anchor_mask=None,
                  pred_anchor_logits=None,
                  gt_anchor_logits=None
                  ):
        """_summary_

        Args:
            pred_traj_logits (bs, pred_len, 2): _description_
            gt_traj (bs, pred_len, 4): _description_
            gt_traj_mask (bs, pred_len, 1): _description_. Defaults to None.
            pred_kps_logits (bs, num_kps, k*2): _description_. Defaults to None.
            gt_kps (bs, num_kps, 4): _description_. Defaults to None.
            gt_kps_mask (bs, num_kps, 1): _description_. Defaults to None.
            pred_kps_cls (bs, num_kps, 6): _description_. Defaults to None.
            
            pred_anchor_cls=None, # (bs, anchor_len, 64)
            gt_anchor_cls=None # (bs, )

        Returns:
            _type_: _description_
        """
        
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        loss_traj = torch.tensor(0, dtype=torch.float32, device=device)

        # loss traj
        if self.predict_trajectory:
            loss_traj = self.reg_trj_loss(pred_traj_logits[..., :self.pred_dim], gt_traj[..., :self.pred_dim].to(device)) * self.model_args.trajectory_loss_rescale
            loss_traj = (loss_traj * gt_traj_mask).sum() / (gt_traj_mask.sum() + 1e-7)
            loss += loss_traj

        # loss kps
        if self.ar_future_interval > 0:
            if self.k == 1:
                loss_keypoints = self.reg_kps_loss(pred_kps_logits, gt_kps[..., :self.pred_dim].to(device))
                loss_keypoints = (loss_keypoints* gt_kps_mask).sum() / (gt_kps_mask.sum() + 1e-7)
                
                loss += loss_keypoints
                pred_traj_logits = torch.cat([pred_kps_logits, pred_traj_logits], dim=1)
            else:
                b, num_kps, c = gt_kps.shape
                k_results = pred_kps_logits.reshape(b, num_kps, self.k, -1)

                # get loss of minimal loss from k results
                k_future_key_points = gt_kps.unsqueeze(2).repeat(1, 1, self.k, 1).reshape(b, num_kps, self.k, -1)

                loss_keypoints = self.reg_kps_loss(k_results, k_future_key_points[..., :self.pred_dim].to(device))
                # add loss on x, y (the last dimension)
                loss_keypoints = loss_keypoints.sum(dim=-1)  # b, num_kps, k
                min_loss_kp, min_loss_kp_indices = torch.min(loss_keypoints, dim=2)  # b, num_kps
                # min_loss_kp = loss_keypoints.mean(dim=2) # option 1
                min_loss_kp = (min_loss_kp.unsqueeze(-1) * gt_kps_mask).sum() / (gt_kps_mask.sum() + 1e-7)
                
                loss += min_loss_kp
                
                if self.next_token_scorer_decoder is not None:
                    pred_kps_cls_masked = pred_kps_cls[gt_kps_mask[...,0].to(torch.bool)] 
                    min_loss_kp_indices_masked = min_loss_kp_indices[gt_kps_mask[...,0].to(torch.bool)]
                    loss_kp_cls = self.cls_kps_loss(pred_kps_cls_masked.reshape(-1, self.k).to(torch.float64), min_loss_kp_indices_masked.reshape(-1).long()) * self.cls_kps_loss_weight
                    loss += loss_kp_cls
                    
                    if self.training:
                        # concatenate the key points with predicted trajectory for evaluation
                        selected_key_points = pred_kps_logits.reshape(b * num_kps, self.k, -1)[torch.arange(b * num_kps),
                                              min_loss_kp_indices.reshape(-1), :].reshape(b, num_kps, -1)
                    else:
                        # concatenate the key points with predicted trajectory selected from the classifier for evaluation
                        selected_key_points = pred_kps_logits.reshape(b * num_kps, self.k, -1)[torch.arange(b * num_kps),
                                              pred_kps_cls.argmax(dim=-1).reshape(-1), :].reshape(b, num_kps, -1)
                    pred_traj_logits = torch.cat([selected_key_points, pred_traj_logits], dim=1)
                else:
                    print('WARNING: Randomly select key points for evaluation, try to use next_token_scorer_decoder')
                    pred_traj_logits = torch.cat([pred_kps_logits[0].reshape(b, num_kps, -1), pred_traj_logits], dim=1)
        
        # loss anchor
        if self.use_anchor:
            loss_anchor = self.cls_anchor_loss(pred_anchor_cls.reshape(-1, self.anchor_num).to(torch.float64), gt_anchor_cls.reshape(-1).long())
            loss_anchor = (loss_anchor * gt_anchor_mask.view(-1)).sum()/ (gt_anchor_mask.sum()+1e-7)
            loss += loss_anchor
            
            bs = gt_anchor_cls.shape[0]
            pred_anchor_logits = pred_anchor_logits.view(bs, self.anchor_num, 2)
            pred_pos_anchor_logits = pred_anchor_logits[torch.arange(bs), gt_anchor_cls, :] # (bs, 2)
            loss_anchor_logits = self.logits_anchor_loss(pred_pos_anchor_logits, gt_anchor_logits)
            loss_anchor_logits = (loss_anchor_logits * gt_anchor_mask).sum() / (gt_anchor_mask.sum() + 1e-7)
            loss += loss_anchor_logits
        
        # evaluate accuracy if on eval
        if not self.training and self.clf_metrics is not None:
            if self.k > 1:
                # classification on k predictions
                predictions = torch.argmax(pred_kps_cls, dim=-1)  # b, num_kps, k
                for _, metric in self.clf_metrics.items():
                    metric.add_batch(references=min_loss_kp_indices.reshape(-1), predictions=predictions.reshape(-1))
        
        if self.debug and loss.device.index == 4:
            self.tot_iter_num += 1
            if self.tot_iter_num % 100 ==0 and self.use_anchor:
                print("loss traj ", loss_traj, " loss anchor ", loss_anchor, " loss_anchor_logits ", loss_anchor_logits, ' tot loss ', loss)
            elif self.tot_iter_num % 100 ==0 and self.k > 1:
                print("loss traj ", loss_traj, " loss kpts logits ", min_loss_kp, " loss kpts cls ", loss_kp_cls, ' tot loss ', loss)
            elif self.tot_iter_num % 100 ==0 and self.k == 1:
                print("loss traj ", loss_traj, " loss kpts logits ", loss_keypoints,  ' tot loss ', loss)
                
            if self.tot_iter_num >= 1e6:
                self.tot_iter_num = 0
                    
        return pred_traj_logits, loss
    
    def greedy_search(self, input_embeds, tot_scenario_contenxt_len, key_points_num,
                      center_obj_anchor_pts=None):
        '''
        input_embeds: (bs, tot_scenario_context_length + num_kps + num_future_frame, n_embed)
        
        return:
            input_embeds_kpts: (bs, 1, num_kps, n_embed)
            kpts_scores: (bs, 1, num_kps)
            pred_key_points_during_generate: (bs, self.k, num_kps, 4)
        '''
        
        device = input_embeds.device
        batch_size, cur_len, n_embed = input_embeds.shape
        pred_key_points_during_generate = torch.zeros((batch_size, 1, key_points_num, self.pred_dim), device=device)
        
        input_embeds_kpts = torch.zeros((batch_size, 1, key_points_num, n_embed), device=device)
        kpts_scores = torch.zeros((batch_size, 1, key_points_num), device=device)
        
        if self.use_anchor:
            input_embeds_current = input_embeds[:, :tot_scenario_contenxt_len, :]
            attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=device)
            position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
            transformer_output = self.transformer(
                inputs_embeds=input_embeds_current,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']
            pred_anchor_embed = transformer_outputs_hidden_state[:, tot_scenario_contenxt_len - 1, :] # (bs, n_embd)
            pred_anchor_cls = self.anchor_cls_decoder(pred_anchor_embed).argmax(1) # (bs, )
            pred_anchor_logits = center_obj_anchor_pts[torch.arange(batch_size), pred_anchor_cls, : ] # (bs, 2)
            
            correct_anchor_embedding = self.action_m_embed(torch.cat([pred_anchor_logits, torch.zeros((batch_size, 2), device=device)], dim = 1)).unsqueeze(1) # (bs, 1, n_embed)
            input_embeds = torch.cat([input_embeds, correct_anchor_embedding], dim=1)
        
        tot_scenario_contenxt_anchor_len = tot_scenario_contenxt_len + self.anchor_len
        for i in range(key_points_num):
            # prepare attention mask
            input_embeds_current = input_embeds[:, :tot_scenario_contenxt_anchor_len + i, :]
            attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=device)
            position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
            transformer_output = self.transformer(
                inputs_embeds=input_embeds_current,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            transformer_outputs_hidden_state = transformer_output['last_hidden_state']
            future_key_point_hidden_state = transformer_outputs_hidden_state[:, tot_scenario_contenxt_anchor_len + i - 1, :].reshape(batch_size, 1, -1)

            if self.k > 1:
                key_points_logit = self.key_points_decoder(future_key_point_hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2*k
                pred_kps_score = self.next_token_scorer_decoder(future_key_point_hidden_state.to(device)).reshape(batch_size, 1, -1)  # b, 1, k
                
                # delta = (key_points_logit.reshape(batch_size, self.k, -1) - future_key_points_gt[:, [i], :2])
                # dist = -delta[..., 0]*delta[..., 0] - delta[..., 1]*delta[..., 1]
                # pred_kps_score = dist[:, None, :]
                
                # pred_kps_score_index = pred_kps_score.argsort(dim=-1)
                # selected_key_point = torch.zeros((batch_size, 1, 2), device=pred_kps_score.device, dtype=pred_kps_score.dtype)
                
                # for s_ind in range(3):
                #     selected_key_point += key_points_logit.reshape(batch_size, self.k, -1)[torch.arange(batch_size), pred_kps_score_index[:, 0, s_ind].reshape(-1), :].reshape(batch_size, 1, -1)
                
                # selected_key_point /= 3.0
                
                selected_key_point = key_points_logit.reshape(batch_size, self.k, -1)[torch.arange(batch_size), pred_kps_score.argmax(dim=-1).reshape(-1), :].reshape(batch_size, 1, -1)    
                key_points_logit = selected_key_point
            else:
                key_points_logit = self.key_points_decoder(future_key_point_hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2
            pred_key_point = torch.zeros((batch_size, 1, 4), device=device)
            pred_key_point[:, 0, :self.pred_dim] = key_points_logit[:, 0, :]

            key_point_embed = self.action_m_embed(pred_key_point).reshape(batch_size, 1, -1)  # b, 1, n_embed
            # replace embed at the next position
            input_embeds[:, tot_scenario_contenxt_anchor_len + i, :] = key_point_embed[:, 0, :]
            input_embeds_kpts[:, 0, i, :] = key_point_embed[:, 0, :]
            kpts_scores[:, :, i] = pred_kps_score.max(-1)[0]
            pred_key_points_during_generate[:, 0, i, :] = pred_key_point[:, 0, :self.pred_dim]
            
        return pred_key_points_during_generate, input_embeds_kpts, kpts_scores
    
    def beam_search(self, input_embeds, tot_scenario_contenxt_len, key_points_num, num_beam=None, out_num_mode=None, center_obj_anchor_pts=None):
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
        
        if self.use_anchor:
            key_points_num += self.anchor_len
        
        device = input_embeds.device
        batch_size, tot_len, n_embed = input_embeds.shape
        pred_key_points_during_generate = torch.zeros((batch_size, num_beam, key_points_num, self.pred_dim), device=device)
        
        k_kpts_scores = torch.zeros((batch_size, num_beam, key_points_num), device=device)
        k_input_embeds = input_embeds[:, None, :, :].repeat(1, num_beam, 1, 1)
        
        for i in range(key_points_num):
            # prepare attention mask
            k_input_embeds_current = k_input_embeds[:, :, :tot_scenario_contenxt_len + i, :].view(batch_size*num_beam, -1, n_embed)
            attention_mask = torch.ones(k_input_embeds_current.shape[:2], dtype=torch.long, device=device)
            position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
            transformer_output = self.transformer(
                inputs_embeds=k_input_embeds_current,
                attention_mask=attention_mask,
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
            pred_kps_logit_topk_embed = self.action_m_embed(pred_kps_logit_topk)  # b, num_beam, n_embed
            
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
        
        return pred_key_points_during_generate[:, 0:out_num_mode, ...], k_input_embeds_kpts[:, 0:out_num_mode, ...], k_kpts_scores[:, 0:out_num_mode, ...]
    
    def beam_search_anchor_only(self, input_embeds, tot_scenario_contenxt_len, out_num_mode=6, center_obj_anchor_pts=None):
        '''
        input_embeds: (bs, tot_scenario_context_length + num_kps + num_future_frame, n_embed)
        
        return:
            k_input_embeds_kpts: (bs, num_beam, num_kps, n_embed)
            k_kpts_scores: (bs, num_beam, num_kps)
            pred_key_points_during_generate: (bs, num_beam, num_kps, 4)
        '''

        
        device = input_embeds.device
        batch_size, tot_len, n_embed = input_embeds.shape

        # prepare attention mask
        input_embeds_current = input_embeds[:, :tot_scenario_contenxt_len, :].view(batch_size, -1, n_embed)
        attention_mask = torch.ones(input_embeds_current.shape[:2], dtype=torch.long, device=device)
        position_ids = self._prepare_position_ids_for_generation(attention_mask.clone())
        transformer_output = self.transformer(
            inputs_embeds=input_embeds_current,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        transformer_outputs_hidden_state = transformer_output['last_hidden_state']
        future_key_point_hidden_state = transformer_outputs_hidden_state[:, [tot_scenario_contenxt_len - 1], :] # (bs, 1, n_embed)

        pred_kps_score = self.anchor_cls_decoder(future_key_point_hidden_state).softmax(-1) # (bs, 1, 64)
        pred_kps_logit = center_obj_anchor_pts # (bs, 64, 2)

        topk_score, topk_indx = torch.topk(pred_kps_score[:, 0, :], dim=-1, k =out_num_mode) # (bs, num_beam)
    
        pred_kps_logit_topk = pred_kps_logit[torch.arange(batch_size)[:, None].repeat(1, out_num_mode).view(-1), topk_indx.view(-1), :].view(batch_size, out_num_mode, 2) # (bs, out_num_mode, 2)
        pred_kps_logit_topk = torch.cat((pred_kps_logit_topk, torch.zeros((batch_size, out_num_mode, 2), device=device)), dim=-1) # (bs, out_num_mode, 4)

        # get kps topk embeds
        pred_kps_logit_topk_embed = self.action_m_embed(pred_kps_logit_topk).unsqueeze(2)  # (b, out_num_mode, 1, n_embed)
        
        # wrarp res
        k_kpts_scores = torch.zeros((batch_size, out_num_mode, 1), device=device)
        k_kpts_scores[:, :, 0] = topk_score
        
        k_kpts_index = torch.zeros((batch_size, out_num_mode, 1), device=device)
        k_kpts_index[:, :, 0] = topk_indx
        
        return pred_kps_logit_topk[:, :, None, :self.pred_dim], pred_kps_logit_topk_embed, k_kpts_scores, topk_indx
    