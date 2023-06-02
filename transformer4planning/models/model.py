from transformers.models.xlnet import XLNetConfig, XLNetModel, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetLMHeadModelOutput
from transformers.models.t5 import T5Config,T5Model, T5PreTrainedModel
from transformers.models.deberta_v2 import DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput

from transformer4planning.models.TransformerXL.model import *
from transformer4planning.models.GPT2.models import *
from transformer4planning.models.encoders import *
from transformer4planning.models.decoders import *


import torch.nn as nn
from transformers import GPT2Model,GPT2PreTrainedModel
_CHECKPOINT_FOR_DOC = "transfo-xl-wt103"
_CONFIG_FOR_DOC = "TransfoXLConfig"

def cat_raster_seq(raster:Optional[torch.LongTensor], framenum=9):
    """
    input raster can be either high resolution raster or low resolution raster
    expected input size: [bacthsize, channel, h, w], and channel is consisted of goal(1d)+roadtype(20d)+agenttype*time(8*9d)
    """
    b, c, h, w = raster.shape
    agent_type = 8
    road_type = 20

    goal_raster = raster[:, 0, :, :].reshape(b, 1, h, w)
    road_ratser = raster[:, 1:21, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type + 1, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, 1 + road_type + i::framenum, :, :]
        raster_i = torch.cat([goal_raster, road_ratser, agent_raster], dim = 1) # expected format (b, 1+20+8, h, w)
        result[:, i, :, :, :] = raster_i
    # return format (batchsize, history_frame_number, channels_per_frame, h, w)
    return result

class TransfoXLModelNuPlan(TransfoXLPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = TransfoXLModel(config)
        model_args = kwargs['model_args']
        self.predict_trajectory = model_args.predict_trajectory

        self.loss_fn = model_args.loss_fn
        in_channels = 29 # raster: goal + road_type + agent_type
        n_embed = config.d_embed // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.d_embed), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            embed_sz = config.d_embed
            self.traj_decoder = DecoderResCat(config.d_inner, embed_sz, out_features=4)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLNuPlanNSMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # input_ids: Optional[torch.LongTensor] = None,
        intended_maneuver_vector: Optional[torch.LongTensor] = None,
        current_maneuver_vector: Optional[torch.LongTensor] = None,
        trajectory_label: Optional[torch.LongTensor] = None,
        context_actions:Optional[torch.LongTensor] = None,
        intended_maneuver_label: Optional[torch.LongTensor] = None,
        current_maneuver_label: Optional[torch.LongTensor] = None,
        high_res_raster: Optional[torch.LongTensor] = None,
        low_res_raster: Optional[torch.LongTensor] = None,
        intended_maneuver_gt: Optional[torch.LongTensor] = None,
        current_maneuver_gt: Optional[torch.LongTensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, TransfoXLNuPlanNSMOutput]:


        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        device = high_res_raster.device
        # with history menuever label input
        intended_maneuver_vector = None
        current_maneuver_vector = None
        intended_maneuver_label = None
        current_maneuver_label = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, h, w, total_channels = high_res_raster.shape
        ## action embedding
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        # embed with the format of (batchsize*history, n_embed) => (batchsize, history, n_embed): both high and low res => (batchsize, history, 2*n_embed)
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)

        state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)

        if trajectory_label is not None:
            trajectory_label = trajectory_label[:, 1::2, :] # downsample the 20hz trajectory to 10hz
            pred_length = trajectory_label.shape[1]
        else:
            pred_length = 80

        # n_embed is 2/4 multiple because different embeddings are concated togaher at the same timestep.
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2 - 1, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # to keep input and output at the same dimension
        input_embeds = torch.cat([input_embeds, torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)

        transformer_outputs = self.transformer(
            None,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        if self.traj_decoder is not None:
            traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length:, :]
            # expected shape for pred trajectory is (b, pred_length, 4)
            traj_pred = self.traj_decoder(traj_hidden_state)
        else:
            traj_pred = None

        loss = torch.tensor(0, dtype=torch.float32, device=device)
        self.config_problem_type = 'NuPlan_Planning'

        if trajectory_label is not None and self.traj_decoder is not None:
            if 'mse' in self.loss_fn:
                loss_fct = MSELoss(reduction="mean")
            elif 'l1' in self.loss_fn:
                loss_fct = SmoothL1Loss()
            loss += loss_fct(traj_pred, trajectory_label.to(device))

        pooled_logits = [0, 0, traj_pred]
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TransfoXLNuPlanNSMOutput(
            loss=loss,
            logits=traj_pred,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            all_logits=pooled_logits
        )

class GPTNonAutoRegressiveModelNuplan(GPT2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        model_args = kwargs["model_args"]
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        in_channels = 29  # raster: goal + road_type + agent_type
        n_embed = config.n_embd // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.n_embd), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            self.traj_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=4)
            # self.traj_decoder = DecoderResCat(config.n_inner, embed_sz, out_features=4)

        # end of added
        # Initialize weights and apply final processing
        self.model_parallel = False
        self.device_map = None
        self.post_init()
    
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
        self.cnn_downsample = self.cnn_downsample.to(self.transformer.first_device)
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
        self.cnn_downsample = self.cnn_downsample.to("cpu")
        self.traj_decoder = self.traj_decoder.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(
            self,
            intended_maneuver_vector: Optional[torch.LongTensor] = None,
            trajectory_label: Optional[torch.LongTensor] = None,
            context_actions:Optional[torch.LongTensor] = None,
            intended_maneuver_label: Optional[torch.LongTensor] = None,
            high_res_raster: Optional[torch.LongTensor] = None,
            low_res_raster: Optional[torch.LongTensor] = None,
            intended_maneuver_gt: Optional[torch.LongTensor] = None,
            current_maneuver_gt: Optional[torch.LongTensor] = None,
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
            # gpt non-autoregression version
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = high_res_raster.device
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)

        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)

        trajectory_label = trajectory_label[:, 1::2, :]
        pred_length = trajectory_label.shape[1]
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2 - 1, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # to keep input and output at the same dimension
        input_embeds = torch.cat([input_embeds, torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)

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
        
        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length:, :]
        # expected shape for pred trajectory is (b, pred_length, 4)
        traj_logits = self.traj_decoder(traj_hidden_state)
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        
        if 'mse' in self.loss_fn:
            loss_fct = MSELoss(reduction="mean")
        elif 'l1' in self.loss_fn:
            loss_fct = SmoothL1Loss()

        loss += loss_fct(traj_logits, trajectory_label.to(device))
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        
class XLNetModelNuplan(XLNetPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = XLNetModel(config)
        model_args = kwargs["model_args"]
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        in_channels = 29  # raster: goal + road_type + agent_type
        n_embed = config.d_model // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.d_model), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            self.traj_decoder = DecoderResCat(config.d_inner, config.d_model, out_features=4)
        
        self.post_init()

    def forward(
        self,
        intended_maneuver_vector: Optional[torch.Tensor] = None,
        trajectory_label: Optional[torch.Tensor] = None,
        context_actions:Optional[torch.Tensor] = None,
        intended_maneuver_label: Optional[torch.Tensor] = None,
        high_res_raster: Optional[torch.Tensor] = None,
        low_res_raster: Optional[torch.Tensor] = None,
        intended_maneuver_gt: Optional[torch.Tensor] = None,
        current_maneuver_gt: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_mems: Optional[bool] = True,
        mems: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = high_res_raster.device
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device),context_length)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device),context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)
        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)
        trajectory_label = trajectory_label[:, 1::2, :]
        pred_length = trajectory_label.shape[1]
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2 - 1, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # to keep input and output at the same dimension
        input_embeds = torch.cat([input_embeds, torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            mems=mems,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            return_dict=return_dict,
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        
        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length:, :]
        # expected shape for pred trajectory is (b, pred_length, 4)
        traj_logits = self.traj_decoder(traj_hidden_state)
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        
        if 'mse' in self.loss_fn:
            loss_fct = MSELoss(reduction="mean")
        elif 'l1' in self.loss_fn:
            loss_fct = SmoothL1Loss()
        loss += loss_fct(traj_logits, trajectory_label.to(device))
        
        return XLNetLMHeadModelOutput(
            loss=loss,
            logits=traj_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class T5ModelNuplan(T5PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = T5Model(config)
        model_args = kwargs["model_args"]
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        in_channels = 29  # raster: goal + road_type + agent_type
        n_embed = config.d_model // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.d_model), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            self.traj_decoder = DecoderResCat(config.d_ff, config.d_model, out_features=4)
        self.model_parallel = False
        self.device_map = None
        self.post_init()
    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.transformer.parallelize(device_map)
        self.model_parallel = True
        self.device_map = self.transformer.device_map

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()

    def forward(
        self,
        intended_maneuver_vector: Optional[torch.Tensor] = None,
        trajectory_label: Optional[torch.Tensor] = None,
        context_actions:Optional[torch.Tensor] = None,
        intended_maneuver_label: Optional[torch.Tensor] = None,
        high_res_raster: Optional[torch.Tensor] = None,
        low_res_raster: Optional[torch.Tensor] = None,
        intended_maneuver_gt: Optional[torch.Tensor] = None,
        current_maneuver_gt: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = high_res_raster.device
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device),context_length)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device),context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)
        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)
        trajectory_label = trajectory_label[:, 1::2, :]
        pred_length = trajectory_label.shape[1]
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2 - 1, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # to keep input and output at the same dimension
        input_embeds = torch.cat([input_embeds, torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length:, :]
        # expected shape for pred trajectory is (b, pred_length, 4)
        traj_logits = self.traj_decoder(traj_hidden_state)
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        
        if 'mse' in self.loss_fn:
            loss_fct = MSELoss(reduction="mean")
        elif 'l1' in self.loss_fn:
            loss_fct = SmoothL1Loss()
        loss += loss_fct(traj_logits, trajectory_label.to(device))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            decoder_attentions=transformer_outputs.decoder_attentions,
            decoder_hidden_states=transformer_outputs.decoder_hidden_states,
            cross_attentions=transformer_outputs.cross_attentions,
            encoder_attentions=transformer_outputs.encoder_attentions,
            encoder_last_hidden_state=transformer_outputs.encoder_last_hidden_state,
            encoder_hidden_states=transformer_outputs.encoder_hidden_states
        )

class DeBertaNuplan(DebertaV2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = DebertaV2Model(config)
        model_args = kwargs["model_args"]
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        in_channels = 29  # raster: goal + road_type + agent_type
        n_embed = config.hidden_size // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.hidden_size), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            self.traj_decoder = DecoderResCat(config.intermediate_size, config.hidden_size, out_features=4)
        self.post_init()

    def forward(
        self,
        intended_maneuver_vector: Optional[torch.Tensor] = None,
        trajectory_label: Optional[torch.Tensor] = None,
        context_actions:Optional[torch.Tensor] = None,
        intended_maneuver_label: Optional[torch.Tensor] = None,
        high_res_raster: Optional[torch.Tensor] = None,
        low_res_raster: Optional[torch.Tensor] = None,
        intended_maneuver_gt: Optional[torch.Tensor] = None,
        current_maneuver_gt: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = high_res_raster.device
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device),context_length)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device),context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)

        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)
        trajectory_label = trajectory_label[:, 1::2, :]
        pred_length = trajectory_label.shape[1]
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2 - 1, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # to keep input and output at the same dimension
        input_embeds = torch.cat([input_embeds, torch.zeros((batch_size, pred_length, n_embed), device=device)], dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']
        
        traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length:, :]
        # expected shape for pred trajectory is (b, pred_length, 4)
        traj_logits = self.traj_decoder(traj_hidden_state)
        
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        if self.training:
            if 'mse' in self.loss_fn:
                loss_fct = MSELoss(reduction="mean")
            elif 'l1' in self.loss_fn:
                loss_fct = SmoothL1Loss()
            loss += loss_fct(traj_logits, trajectory_label.to(device))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=traj_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
class GPTModelNuPlan(GPT2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        model_args = kwargs['model_args']
        self.predict_trajectory = model_args.predict_trajectory
        self.recover_obs = model_args.recover_obs

        in_channels = 29 # raster: goal + road_type + agent_type
        n_embed = config.n_embd // 2

        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.n_embd), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            self.traj_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=4)
        if self.recover_obs:
            self.obs_embed_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=config.n_embd)
        # end of added
        # Initialize weights and apply final processing
        self.model_parallel = False
        self.device_map = None
        self.post_init()

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
        self.cnn_downsample = self.cnn_downsample.to(self.transformer.first_device)
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
        self.cnn_downsample = self.cnn_downsample.to("cpu")
        self.traj_decoder = self.traj_decoder.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    @property
    def mode(self):
        # pred mode: Obs-Maneuver-Action Pair: [m,a | o,m,a | ... | o,m,a]
        # pred mode: Only Action
        if self.predict_trajectory and not self.recover_obs:
            return "PRED-A"

        elif self.predict_trajectory and self.recover_obs:
            return "PRED-OA"

    def forward(
        self,
        intended_maneuver_vector: Optional[torch.Tensor] = None,
        current_maneuver_vector: Optional[torch.Tensor] = None,
        high_res_raster: Optional[torch.Tensor] = None,
        low_res_raster: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
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
        past_seq: Optional[int] = 8,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        intended_maneuver_vector:  batch_size, seq
        current_maneuver_vector: batch_size, seq, 12
        high_res_raster: batch_size, seq, h, w, c (c=29)
        low_res_raster: batch_size, seq, h, w, c (c=29)
        trajectory: batch_size, seq, 4
        """
        if len(high_res_raster.shape) == 4: # convert (b, h, w, seq*c) ->(b, seq, h, w, c)
            _b, _h, _w, _= high_res_raster.shape
            high_res_raster = high_res_raster.reshape(_b, _h, _w, -1, 29).permute(0, 3, 1, 2, 4)
            low_res_raster = low_res_raster.reshape(_b, _h, _w, -1, 29).permute(0, 3, 1, 2, 4)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = high_res_raster.device

        ## ratser embedding and concat to state embedding
        high_res_raster = high_res_raster.permute(0, 1, 4, 2, 3)
        low_res_raster = low_res_raster.permute(0, 1, 4, 2, 3)
        batch_size, seq, c, h, w = high_res_raster.shape
        future_seq = seq - past_seq
        # embed with the format of (batchsize*history, n_embed) => (batchsize, history, n_embed): both high and low res => (batchsize, history, 2*n_embed)
        high_res_embed = self.cnn_downsample(high_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        low_res_embed = self.cnn_downsample(low_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)

        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)
        ## action embedding
        action_embeds = self.action_m_embed(trajectory)
        n_embed = action_embeds.shape[-1]

        # concat state embeding, maneuver embeding, action embeding
        input_embeds_past = torch.cat((
            torch.zeros_like(state_embeds[:, :past_seq+1]), torch.zeros_like(action_embeds[:, :past_seq, :])
        ), dim=1)
        input_embeds_past[:, ::2, :] = state_embeds[:, :past_seq+1, :]
        input_embeds_past[:, 1::2, :] = action_embeds[:, :past_seq, :]

        total_past_length = input_embeds_past.shape[1]
        if self.mode == "PRED-OA":
            input_embeds_future = torch.cat((
                torch.zeros_like(state_embeds[:, past_seq+1:, :]), torch.zeros_like(action_embeds[:, past_seq:, :])
            ), dim=1)
            input_embeds_future[:, ::2, :] = action_embeds[:, past_seq:, :]
            input_embeds_future[:, 1::2, :] = state_embeds[:, past_seq+1:, :]
        elif self.mode == "PRED-A":
            input_embeds_future = action_embeds[:, past_seq:, :]
        input_embeds = torch.cat((input_embeds_past, input_embeds_future), dim=1)

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
        hidden_states = transformer_outputs[0]
        # compute correspond hidden states to predict
        action_hidden_states_past = hidden_states[:, :total_past_length-1, :][:, ::2, :]
        if self.mode == "PRED-OA":
            action_hidden_states = hidden_states[:, ::2, :]
            obs_recover_hidden_states = hidden_states[:, 1::2, :]
        elif self.mode == "PRED-A":
            action_hidden_states_future = hidden_states[:, total_past_length-1:-1, :]
            action_hidden_states = torch.cat((action_hidden_states_past, action_hidden_states_future), dim=1)

        traj_logits = None

        if self.recover_obs:
            obs_labels = state_embeds[:, 1:, :]
            recovered_obs_embd = self.obs_embed_decoder(obs_recover_hidden_states[:, :-1, :])

        loss = torch.tensor(0, dtype=torch.float32, device=device)

        ## input recover supervision
        if self.predict_trajectory and self.traj_decoder is not None:
            loss_fct = MSELoss(reduction="mean")
            loss_to_add = loss_fct(traj_logits[:, :, :2], trajectory[:, :, :2].to(device))
            loss += loss_to_add
            # yaw_loss = loss_fct(traj_logits[:, :, -1]*1000, trajectory[:, :, -1].to(device)*1000)
            # loss += yaw_loss
            # gt_normalized_pts = self.compute_normalized_points(trajectory)
            # pred_normalized_pts = self.compute_normalized_points(traj_logits)
            # final_pt_loss = loss_fct(pred_normalized_pts[:, -1, :2], gt_normalized_pts[:, -1, :2].to(device))
            # loss += final_pt_loss
            # world_coor_loss = loss_fct(pred_normalized_pts[:, :, :2], gt_normalized_pts[:, :, :2]).to(device)
            # loss += world_coor_loss

        if self.recover_obs:
            loss_fct = MSELoss(reduction="mean")
            loss_to_add = loss_fct(recovered_obs_embd, obs_labels)
            loss += loss_to_add

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

    def generate(self,
                intended_maneuver_vector: Optional[torch.Tensor] = None,
                current_maneuver_vector: Optional[torch.Tensor] = None,
                high_res_raster: Optional[torch.Tensor] = None,
                low_res_raster: Optional[torch.Tensor] = None,
                trajectory: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = True,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                seq_length: Optional[int] = 33,
                **kwargs):
        """
        all the input items only include the historic contents
        """
        device = high_res_raster.device
        if len(high_res_raster.shape) == 4: # convert (b, h, w, seq*c) ->(b, seq, h, w, c)
            _b, _h, _w, _= high_res_raster.shape
            high_res_raster = high_res_raster.reshape(_b, _h, _w, -1, 29).permute(0, 3, 1, 2, 4)
            low_res_raster = low_res_raster.reshape(_b, _h, _w, -1, 29).permute(0, 3, 1, 2, 4)

        ## ratser embedding and state embedding concat
        high_res_raster = high_res_raster.permute(0, 3, 4, 1, 2)
        low_res_raster = low_res_raster.permute(0, 3, 4, 1, 2)
        batch_size, seq, c, h, w = high_res_raster.shape
        # embed with the format of (batchsize*history, n_embed) => (batchsize, history, n_embed): both high and low res => (batchsize, history, 2*n_embed)
        high_res_embed = self.cnn_downsample(high_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        low_res_embed = self.cnn_downsample(low_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)
        ## maneuver embedding
        maneuver_embeds = None
        ## action embedding
        action_embeds = self.action_m_embed(trajectory)
        input_embeds = torch.cat((torch.zeros_like(state_embeds, dtype=torch.float32, device=device),
                                  torch.zeros_like(action_embeds, dtype=torch.float32, device=device)), dim=1)

        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # result dict
        result_to_return = dict()
        result_to_return["trajectory"] = list()
        result_to_return["intend_maneuver"] = list()
        result_to_return["current_maneuver"] = list()
        step = 0
        while True:
            # TODO: attention mask prepare and position_ids prepare
            attention_mask = self._prepare_attention_mask_for_generation(input_embeds)
            position_ids = self._prepare_position_ids_for_generation(attention_mask)
            transformer_outputs = self.transformer(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_state = transformer_outputs[0]
            # pred mode: Obs-Maneuver-Action Pair: [m,a | o,m,a | ... | o,m,a]
            if self.mode == "PRED-OA":
                if step > 2 * seq_length - 1:
                    break
                if step % 2 == 0:
                    if self.predict_trajectory:
                        traj_logits = self.traj_decoder(hidden_state[:, -1, :].unsqueeze(1))
                    result_to_return["trajectory"].append(traj_logits)
                    next_embed = self.action_m_embed(traj_logits)
                if step % 2 == 1:
                    next_embed = self.obs_embed_decoder(hidden_state[:, -1, :].unsqueeze(1))
            # pred mode : Only Action
            elif self.mode == "PRED-A":
                if step > seq_length - 1:
                    break
                if self.predict_trajectory:
                    traj_logits = self.traj_decoder(hidden_state[:, -1, :].unsqueeze(1))
                result_to_return["trajectory"].append(traj_logits)
                next_embed = self.action_m_embed(traj_logits)

            input_embeds = torch.cat((input_embeds, next_embed), dim=1)
            step += 1

        result_to_return["trajectory"] = torch.cat(result_to_return["trajectory"], dim=1)
        return result_to_return

    # def _prepare_model_inputs(self, input)
    def _prepare_attention_mask_for_generation(self, input_embeds):
        return torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

    def _prepare_position_ids_for_generation(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids

    def compute_normalized_points(self, trajectory, yaw=0):
        bsz = trajectory.shape[0]
        device = trajectory.device
        ego_trajectory = torch.zeros((bsz, 1, 4), device=device)
        ego_trajectory[-1] = yaw
        next_normalized_trajectories = list()
        for idx in range(0, trajectory.shape[1]):
            cos_, sin_ = torch.cos(-ego_trajectory[:, -1, -1]), torch.sin(-ego_trajectory[:, -1, -1])
            cos_.to(device)
            sin_.to(device)
            delta_yaw = torch.arctan(torch.divide(trajectory[:, idx, 1], trajectory[:, idx, 0]))
            offset_x = trajectory[:, idx, 0] * cos_ + trajectory[:, idx, 1] * sin_
            offset_y = trajectory[:, idx, 1] * cos_ - trajectory[:, idx, 0] * sin_
            next_ego_traj = torch.stack([ego_trajectory[:, -1, 0] + offset_x,
                                        ego_trajectory[:, -1, 1] + offset_y,
                                        torch.zeros_like(ego_trajectory[:, -1, 1]),
                                        # ego_trajectory[:, -1, -1] + delta_yaw],dim=-1)
                                        ego_trajectory[:, -1, -1] + trajectory[:, idx, -1]],dim=-1)
            ego_trajectory = torch.cat((ego_trajectory, torch.tensor(next_ego_traj).reshape(bsz, 1, -1)), dim=1)
            next_normalized_trajectories.append(next_ego_traj)

        next_normalized_trajectories = torch.stack(next_normalized_trajectories).permute(1, 0, 2)
        # bsz, seq, 4
        return next_normalized_trajectories

    def compute_yaw(self, point):
        yaw = torch.arctan(point[:, 1]/point[:, 0])
        return yaw

def build_models(model_args):
    if 'gpt' in model_args.model_name:
        config_p = GPT2Config()
        config_p.n_layer = model_args.n_layers
        config_p.n_embd = model_args.d_embed
        config_p.n_inner = model_args.d_inner
        config_p.n_head = model_args.n_heads
        config_p.activation_function = model_args.activation_function
        if 'nonauto' in model_args.model_name:
            ModelCls = GPTNonAutoRegressiveModelNuplan
            tag = 'GPT nonauto'
        else:
            ModelCls = GPTModelNuPlan
            tag = 'GPT auto'
    elif 'transxl' in model_args.model_name:
        config_p = TransfoXLConfig()
        config_p.pad_token_id = 0
        config_p.eos_token_id = 0
        config_p.n_layer = model_args.n_layers
        config_p.d_embed = model_args.d_embed
        config_p.d_model = model_args.d_model
        config_p.d_inner = model_args.d_inner
        ModelCls= TransfoXLModelNuPlan
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
        config_p.num_heads=model_args.n_heads
        config_p.d_model = model_args.d_model
        config_p.d_kv = model_args.d_model//config_p.num_heads
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
        model = ModelCls.from_pretrained(model_args.model_pretrain_name_or_path, model_args=model_args)
        print('Pretrained ' + tag + 'from {}'.format(model_args.model_pretrain_name_or_path))
    return model    

if  __name__ == '__main__':
    import datasets
    import argparse, time, pickle
    import matplotlib.pyplot as plt
    from transformers import HfArgumentParser
    from transformer4planning.utils import ModelArguments
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args()
    model_args.d_embed = 256
    model_args.d_model = 256
    model_args.d_inner = 1024
    model_args.n_layers = 4
    model_args.n_heads = 8
    model_args.model_name = "scratch-mmtransformer"
    model_args.task = "waymo"

    model = build_models(model_args)

    def compute_world_points(pred_traj, yaw=0):
        ego_trajectory = np.zeros((1, 3))
        ego_trajectory[-1] = yaw
        next_world_coor_trajectories = list()
        for idx in range(0, pred_traj.shape[0]):
            cos_, sin_ = math.cos(-ego_trajectory[-1][2]), math.sin(-ego_trajectory[-1][2])
            offset_x = pred_traj[idx, 0] * cos_ + pred_traj[idx, 1] * sin_
            offset_y = pred_traj[idx, 1] * cos_ - pred_traj[idx, 0] * sin_
            next_ego_traj = [ego_trajectory[-1][0] + offset_x,
                            ego_trajectory[-1][1] + offset_y,
                            ego_trajectory[-1][2] + pred_traj[idx, -1]]
            ego_trajectory = np.concatenate((ego_trajectory, np.array(next_ego_traj.copy()).reshape(1, -1)), axis=0)
            next_world_coor_trajectories.append(next_ego_traj)

        next_world_coor_trajectories = np.array(next_world_coor_trajectories)
        next_world_coor_points = next_world_coor_trajectories[::2]
        next_world_coor_x = next_world_coor_trajectories[:,0]
        next_world_coor_y = next_world_coor_trajectories[:,1]
        return next_world_coor_x - yaw, next_world_coor_y - yaw
    
    # dataset = datasets.load_from_disk("/media/shiduozhang/My Passport/nuplan/5hz_boston/")
    dataset = datasets.load_from_disk("/home/shiduozhang/waymo/t4p_waymo")
    # print(dataset.features)
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    example = dataset['train'][0]
    result = model(
        trajectory_label=example['trajectory_label'].unsqueeze(0),
        context_actions=example['context_actions'].unsqueeze(0),
        high_res_raster=example['high_res_raster'].unsqueeze(0),
        low_res_raster=example['low_res_raster'].unsqueeze(0),
        return_dict=True,
    )
    pred_traj = result.logits
    gt_traj = example['trajectory_label'][1::2].cpu().numpy()
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred_traj, example['trajectory_label'][1::2])
    print("loss", loss)
    pred_x, pred_y = pred_traj[0, :, 0].detach().cpu().numpy(), pred_traj[0, :, 1].detach().cpu().numpy()
    gt_x, gt_y = gt_traj[:, 0], gt_traj[:, 1]
    fig = plt.figure(figsize=(200,100))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlim([-100, 100])
    ax1.set_ylim([-100, 100])
    ax1.scatter(gt_x[::4], gt_y[::4], color='green')
    ax1.scatter(pred_x[::4], pred_y[::4], color='red')
    plt.show()


    ## ground truth inverse computation
    with open("visulization/rasters/test/frame1k.pkl", "rb") as f:
        example = pickle.load(f)
        trajectory = example["trajectory"]
        tan = np.divide(trajectory[:, 1], trajectory[:, 0])
        yaw_np = np.arctan(tan)
        delta_yaw = yaw_np - trajectory[:, 2]
        gt = example["gt"]
        yaw = example["world_yaw"]
    gt_x, gt_y = gt[:, 0], gt[:, 1]
    world_trajectory = model.compute_normalized_points(torch.tensor(trajectory).unsqueeze(0), yaw)
    x = world_trajectory[0, :, 0].numpy() - yaw
    y = world_trajectory[0, :, 1].numpy() - yaw
    # x, y = compute_world_points(trajectory, yaw)
    diff_x = x - gt_x
    diff_y = y - gt_y
    fig = plt.figure(figsize=(200,100))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlim([-100, 100])
    ax1.set_ylim([-100, 100])
    ax1.plot(gt_x, gt_y, color='green')
    ax1.plot(x, y, color='red')
    plt.show()

    print("done")
