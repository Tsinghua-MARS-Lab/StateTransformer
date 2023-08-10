from transformer4planning.models.TransformerXL.model import *
from transformers import GPT2Model,GPT2PreTrainedModel
from transformer4planning.models.GPT2.models import *
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput
from transformers.models.xlnet import XLNetConfig, XLNetModel, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetLMHeadModelOutput
from transformers.models.t5 import T5Config,T5Model, T5PreTrainedModel
from transformers.models.deberta_v2 import DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel
from transformer4planning.models.encoder.encoders import *
from transformer4planning.models.decoders import *
from transformer4planning.models.utils import *
from transformers.generation.configuration_utils import GenerationConfig


import torch.nn as nn
import torch.nn.functional as F
from datasets import Value
import evaluate
import copy

class TransfoXLModelNuPlan(TransfoXLPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = TransfoXLModel(config)
        model_args = kwargs['model_args']
        self.predict_trajectory = model_args.predict_trajectory

        self.loss_fn = model_args.loss_fn
        if model_args.with_traffic_light:
            in_channels = 33 # raster: goal + road_type + traffic light +agent_type
        else:
            in_channels = 29
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

class XLNetModelNuplan(XLNetPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = XLNetModel(config)
        model_args = kwargs["model_args"]
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        if model_args.with_traffic_light:
            in_channels = 33 # raster: goal + road_type + traffic light +agent_type
        else:
            in_channels = 29
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
        if model_args.with_traffic_light:
            in_channels = 33 # raster: goal + road_type + traffic light +agent_type
        else:
            in_channels = 29
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
        if model_args.with_traffic_light:
            in_channels = 33 # raster: goal + road_type + traffic light +agent_type
        else:
            in_channels = 29
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
        self.model_args = model_args
        self.predict_trajectory = model_args.predict_trajectory
        self.recover_obs = model_args.recover_obs
        if model_args.with_traffic_light:
            self.in_channels = 33  # raster: goal + road_type + traffic light +agent_type
        else:
            self.in_channels = 29
        # TODO: add parameter to conifg past_seq
        self.past_seq = model_args.past_seq  # 20 frames / 4 = 5 frames per second, 5 * 2 seconds = 10 frames
        n_embed = config.n_embd // 2

        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=self.in_channels)
        if self.model_args.tokenize_label:
            self.action_m_embed = nn.Sequential(nn.Linear(40 * 80, config.n_embd), nn.Tanh())
        else:
            self.action_m_embed = nn.Sequential(nn.Linear(2, config.n_embd), nn.Tanh())

        self.traj_decoder = None
        self.k = int(self.model_args.k)
        if self.predict_trajectory:
            if self.model_args.k == -1:
                # do classification
                self.traj_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=80 * 40)
            elif self.model_args.k == 1:
                self.traj_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=2)
            else:
                self.traj_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=2*self.k)
        self.next_token_scorer_decoder = None
        if self.model_args.next_token_scorer and self.k > 1:
            self.next_token_scorer_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=self.k)

        if self.recover_obs:
            self.obs_embed_decoder = DecoderResCat(model_args.d_inner, config.n_embd, out_features=config.n_embd)
        # end of added
        # Initialize weights and apply final processing
        self.model_parallel = False
        self.device_map = None
        self.token_map = None
        self.post_init()

        self.clf_metrics = None

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
        if self.next_token_scorer_decoder is not None:
            self.next_token_scorer_decoder = self.next_token_scorer_decoder.to(self.transformer.first_device)
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
        if self.next_token_scorer_decoder is not None:
            self.next_token_scorer_decoder = self.next_token_scorer_decoder.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    @property
    def mode(self):
        # pred mode: Obs-Maneuver-Action Pair: [m,a | o,m,a | ... | o,m,a]
        # pred mode: Only Action
        if self.predict_trajectory and self.recover_obs:
            return "PRED-OA"
        elif self.predict_trajectory and self.model_args.teacher_forcing_obs:
            return "OA-OA"
        elif self.predict_trajectory:
            return "PRED-A"

    def tokenize(self, trajectory):
        """
        Default token space is: x [0, 3], interval 0.1, 30 token; y [-1, 1], interval is 0.05
        """
        x_range = DEFAULT_TOKEN_CONFIG["x_range"]
        y_range = DEFAULT_TOKEN_CONFIG["y_range"]
        x_class = DEFAULT_TOKEN_CONFIG["x_class"]
        y_class = DEFAULT_TOKEN_CONFIG["y_class"]
        x, y = trajectory[..., 0], trajectory[..., 1]
        x = torch.where(x > x_range[1], torch.ones_like(x) * x_range[1], x)
        x = torch.where(x < x_range[0], torch.ones_like(x) * x_range[0], x)
        y = torch.where(y > y_range[1], torch.ones_like(y) * y_range[1], y)
        y = torch.where(y < y_range[0], torch.ones_like(y) * y_range[0], y)
        x_index = (x - x_range[0])/((x_range[1] - x_range[0])/x_class)
        y_index = (y - y_range[0])/((y_range[1] - y_range[0])/y_class)
        x_index = torch.where(x_index >= x_class, torch.ones_like(x_index) * (x_class - 1), x_index).to(torch.int32)
        y_index = torch.where(y_index >= y_class, torch.ones_like(y_index) * (y_class - 1), y_index).to(torch.int32)
        labels = x_index + y_index * x_class
        if self.token_map is None:
            self.token_map = dict(
                x_interval = (x_range[1] - x_range[0])/x_class,
                y_interval = (y_range[1] - y_range[0])/y_class,
                y_class = y_class,
                x_class = x_class,
                x_range = x_range,
                y_range = y_range
            )
        return labels

    def _prepare_model_inputs(self,
                              high_res_raster,
                              low_res_raster,
                              trajectory):
        """
        Prepare the inputs for the model.
        """
        past_seq = self.past_seq
        if len(high_res_raster.shape) == 4:  # convert (b, h, w, seq*c) ->(b, seq, c, h, w)
            _b, _h, _w, _ = high_res_raster.shape
            high_res_raster = high_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)
            low_res_raster = low_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)

        device = high_res_raster.device

        batch_size, seq, c, h, w = high_res_raster.shape
        future_seq = seq - past_seq
        # embed with the format of (batchsize*history, n_embed) => (batchsize, history, n_embed): both high and low res => (batchsize, history, 2*n_embed)
        high_res_embed = self.cnn_downsample(high_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        low_res_embed = self.cnn_downsample(low_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)

        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)
        # action embedding, shape is (b, seq), seq is default to 51 with 5hz
        copy_trajectory = trajectory.clone()
        if self.model_args.x_random_walk > 0:
            x_noise = torch.rand(trajectory.shape, device=device) * self.model_args.x_random_walk * 2 - self.model_args.x_random_walk
            copy_trajectory[:, past_seq:, 0] += x_noise[:, past_seq:, 0]
        if self.model_args.y_random_walk > 0:
            y_noise = torch.rand(trajectory.shape, device=device) * self.model_args.y_random_walk * 2 - self.model_args.y_random_walk
            copy_trajectory[:, past_seq:, 1] += y_noise[:, past_seq:, 1]

        if self.model_args.tokenize_label:
            action_label = self.tokenize(copy_trajectory)
            action_token = F.one_hot(action_label.to(torch.int64), self.token_map["x_class"] * self.token_map["y_class"])
            action_embeds = self.action_m_embed(action_token.to(torch.float32))
        else:
            action_embeds = self.action_m_embed(copy_trajectory[..., :2].to(torch.float32))  # (b, seq, emd)

        # concat state embedding, action embedding as input embedding
        ## past state embedding shape is (b, seq+1，emd), while past action embedding shape is (b, seq+1, emd), seq is defalutly set to 10
        input_embeds_past = torch.cat((
            torch.zeros_like(state_embeds[:, :past_seq + 1]), torch.zeros_like(action_embeds[:, :past_seq, :])
        ), dim=1)
        input_embeds_past[:, ::2, :] = state_embeds[:, :past_seq + 1, :]
        input_embeds_past[:, 1::2, :] = action_embeds[:, :past_seq, :]

        total_past_length = input_embeds_past.shape[1]
        if self.mode == "PRED-OA" or self.mode == 'OA-OA':
            input_embeds_future = torch.cat((
                torch.zeros_like(state_embeds[:, past_seq + 1:, :]), torch.zeros_like(action_embeds[:, past_seq:, :])
            ), dim=1)
            input_embeds_future[:, ::2, :] = action_embeds[:, past_seq:, :]
            input_embeds_future[:, 1::2, :] = state_embeds[:, past_seq + 1:, :]
        elif self.mode == "PRED-A":
            input_embeds_future = action_embeds[:, past_seq:, :]

        input_embeds = torch.cat((input_embeds_past, input_embeds_future), dim=1)
        return input_embeds, total_past_length

    def forward(
        self,
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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        intended_maneuver_vector:  batch_size, seq
        current_maneuver_vector: batch_size, seq, 12
        high_res_raster: batch_size, seq, h, w, c (c=29)
        low_res_raster: batch_size, seq, h, w, c (c=29)
        trajectory: batch_size, seq, 4
        """
        device = high_res_raster.device
        input_embeds, total_past_length = self._prepare_model_inputs(high_res_raster, low_res_raster, trajectory)

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
        # in PRED-A mode, the hidden states' shape is (b, (2*past_seq+1)+(future seq), emb), such as (1, 62, 256), in PRED-OA mode, the shape is (b, 2*total_seq, emb), such as (1, 2*51, 256)
        hidden_states = transformer_outputs[0]
        # compute correspond hidden states to predict
        action_hidden_states_past = hidden_states[:, :total_past_length-1, :][:, ::2, :]
        if self.mode == "PRED-OA":
            action_hidden_states = hidden_states[:, ::2, :]
            obs_recover_hidden_states = hidden_states[:, 1::2, :]
        elif self.mode == "PRED-A":
            action_hidden_states_future = hidden_states[:, total_past_length-1:-1, :]
            action_hidden_states = torch.cat((action_hidden_states_past, action_hidden_states_future), dim=1)
        elif self.mode == "OA-OA":
            action_hidden_states = hidden_states[:, ::2, :]

        if self.traj_decoder is not None:
            action_logits = self.traj_decoder(action_hidden_states.to(device))

        if self.recover_obs:
            obs_labels = state_embeds[:, 1:, :]
            recovered_obs_embd = self.obs_embed_decoder(obs_recover_hidden_states[:, :-1, :])

        loss = torch.tensor(0, dtype=torch.float32, device=device)
        ## input recover supervision
        if self.predict_trajectory and self.traj_decoder is not None:
            b, s, c = action_logits.shape
            if self.k == -1:
                # compute classification loss
                loss_fct = CrossEntropyLoss(reduction="mean")
                loss_to_add = loss_fct(action_logits.reshape(b*s, c).to(torch.float64), action_label.reshape(-1).to(device).long())
                loss += loss_to_add
            elif self.k == 1:
                # testing smooth_l1 loss
                loss_fct = nn.SmoothL1Loss()
                loss_to_add = loss_fct(action_logits, trajectory[:, :, :2].to(device)) * 100
                loss += loss_to_add
            else:
                k_results = action_logits.reshape(b, s, self.k, 2)
                loss_fct = nn.SmoothL1Loss()
                losses = []  # length of b * s
                min_loss_indices = []  # length of b
                for i in range(b):
                    per_batch_losses = []  # length of s, [x, x, ..]
                    per_batch_indices = []  # length of s, [3, 2, 1, 0, ..]
                    for j in range(s):
                        per_sequence_losses = []  # length of k
                        for k in range(self.k):
                            loss_to_add = loss_fct(k_results[i, j, k, :], trajectory[i, j, :2].to(device)) * 100
                            per_sequence_losses.append(loss_to_add)
                        min_loss = min(per_sequence_losses)
                        min_loss_index = per_sequence_losses.index(min_loss)
                        per_batch_losses.append(min_loss)
                        per_batch_indices.append(min_loss_index)
                    losses += per_batch_losses
                    min_loss_indices.append(per_batch_indices)
                loss += sum(losses) / b / s
                min_loss_indices = torch.tensor(min_loss_indices).to(device)  # b, s

                if self.next_token_scorer_decoder is not None:
                    pred_logits = self.next_token_scorer_decoder(action_hidden_states.to(device))  # b, s, k
                    loss_fct = CrossEntropyLoss(reduction="mean")
                    loss_to_add = loss_fct(pred_logits.reshape(b*s, self.k).to(torch.float64), min_loss_indices.reshape(-1).long()) * 0.1
                    loss += loss_to_add

        if self.recover_obs:
            loss_fct = MSELoss(reduction="mean")
            loss_to_add = loss_fct(recovered_obs_embd, obs_labels)
            loss += loss_to_add

        # evaluate accuracy if on eval
        if not self.training and self.clf_metrics is not None:
            if self.next_token_scorer_decoder is not None:
                # classification on k predictions
                predictions = torch.argmax(pred_logits, dim=-1)  # b, s, k
                for _, metric in self.clf_metrics.items():
                    metric.add_batch(references=min_loss_indices.reshape(-1), predictions=predictions.reshape(-1))
            else:
                # classification on action logits
                predictions = torch.argmax(action_logits, dim=-1)
                for _, metric in self.clf_metrics.items():
                    metric.add_batch(references=action_label.reshape(-1), predictions=predictions.reshape(-1))

        if not return_dict:
            output = (action_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=action_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            generation_config: Optional[GenerationConfig] = None,
            **kwargs
    ) -> torch.FloatTensor:

        r"""

        Generates sequences of poses for models with a pose decoder head.

        This is derived from the original generate function from language modeling tasks.
        See the Higging Face's Official Repo for potential updates at:
        https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/generation/utils.py#L1111

        Provide at least 2 seconds of Observation and Action pair to generate a trajectory.
        Frequency should be greater than the data used for training.

        Parameters:
            TBD (TODO)
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                As a decoeder-only model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()

        # 2. Set generation parameters if not already defined
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # 3. Define model inputs
        # input_embeds has to be defined
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        trajectory = kwargs.get("trajectory", None)
        input_embeds, _ = self._prepare_model_inputs(high_res_raster, low_res_raster, trajectory.clone())
        batch_size = trajectory.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        assert not generation_config.use_cache, "Generation with caches is currently not supported. Please set `use_cache=False`."

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

        if self.device.type != input_embeds.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_embeds.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        if self.k == 1:
            return self.generate_without_score(kwargs)
        elif self.k > 1:
            return self.generate_with_score(kwargs)
        else:
            raise NotImplementedError("TopK generation is not implemented yet.")

    def generate_without_score(self, kwargs):
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        trajectory = kwargs.get("trajectory", None)
        device = high_res_raster.device
        past_length = 11 if kwargs.get("past_length", None) is None else kwargs.get("past_length", None)
        provided_length = past_length
        seq_length = 40 if kwargs.get("seq_length", None) is None else kwargs.get("seq_length", None)

        if len(high_res_raster.shape) == 4:  # convert (b, h, w, seq*c) ->(b, seq, c, w, h)
            _b, _h, _w, _ = high_res_raster.shape
            high_res_raster = high_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)[:,
                              :past_length, ...]
            low_res_raster = low_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)[:,
                             :past_length, ...]
        batch_size, seq, c, h, w = high_res_raster.shape
        high_res_embed = self.cnn_downsample(high_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        low_res_embed = self.cnn_downsample(low_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)
        trajectory_to_loop = trajectory.clone()[:, :past_length, :2]  # b, past_length, 2

        looping_embeds = None
        for _ in range(seq_length):
            if looping_embeds is None:
                ## action embedding
                trajectory_tokens = self.tokenize(trajectory_to_loop[:, :past_length, ...])
                tokenized_trajectory = F.one_hot(trajectory_tokens.to(torch.int64), 80 * 40)
                action_embeds = self.action_m_embed(tokenized_trajectory.to(torch.float32))
                looping_embeds = torch.cat((torch.zeros_like(state_embeds, dtype=torch.float32, device=device),
                                          torch.zeros_like(action_embeds, dtype=torch.float32, device=device)), dim=1)

                looping_embeds[:, ::2, :] = state_embeds
                looping_embeds[:, 1::2, :] = action_embeds
            else:
                ## action embedding
                prev_token = self.tokenize(trajectory_to_loop[:, -1, ...].unsqueeze(1))
                prev_tokenized_action = F.one_hot(prev_token.to(torch.int64), 80 * 40)
                action_embeds = self.action_m_embed(prev_tokenized_action.to(torch.float32))
                looping_embeds = torch.cat((looping_embeds, action_embeds), dim=1)

            attention_mask = self._prepare_attention_mask_for_generation(looping_embeds)
            position_ids = self._prepare_position_ids_for_generation(attention_mask)
            transformer_output = self.transformer(
                inputs_embeds=looping_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                # **input_kwargs
            )
            transformer_hidden_state = transformer_output[0]
            output = self.traj_decoder(transformer_hidden_state[:, -1, :]).unsqueeze(1)  # b, 1, 2

            trajectory_to_loop = torch.cat((trajectory_to_loop, output), dim=1)
            past_length += 1

        return trajectory_to_loop[:, provided_length:, :2]

    def generate_with_score(self, kwargs):
        assert self.next_token_scorer_decoder is not None, 'generate k depends on scores prediction'
        high_res_raster = kwargs.get("high_res_raster", None)
        low_res_raster = kwargs.get("low_res_raster", None)
        trajectory = kwargs.get("trajectory", None)
        device = high_res_raster.device
        past_length = 11 if kwargs.get("past_length", None) is None else kwargs.get("past_length", None)
        provided_length = past_length
        seq_length = 40 if kwargs.get("seq_length", None) is None else kwargs.get("seq_length", None)

        if len(high_res_raster.shape) == 4:  # convert (b, h, w, seq*c) ->(b, seq, c, w, h)
            _b, _h, _w, _ = high_res_raster.shape
            high_res_raster = high_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)[:,
                              :past_length, ...]
            low_res_raster = low_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)[:,
                             :past_length, ...]
        batch_size, seq, c, h, w = high_res_raster.shape
        high_res_embed = self.cnn_downsample(high_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        low_res_embed = self.cnn_downsample(low_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)
        trajectory_to_loop = trajectory.clone()[:, :past_length, :2]  # b, past_length, 2

        looping_embeds = None
        for _ in range(seq_length):
            if looping_embeds is None:
                ## action embedding
                if self.model_args.tokenize_label:
                    trajectory_tokens = self.tokenize(trajectory_to_loop[:, :past_length, ...])
                    tokenized_trajectory = F.one_hot(trajectory_tokens.to(torch.int64), 80 * 40)
                    action_embeds = self.action_m_embed(tokenized_trajectory.to(torch.float32))
                else:
                    action_embeds = self.action_m_embed(trajectory_to_loop[:, :past_length, :2].to(torch.float32))
                looping_embeds = torch.cat((torch.zeros_like(state_embeds, dtype=torch.float32, device=device),
                                          torch.zeros_like(action_embeds, dtype=torch.float32, device=device)), dim=1)

                looping_embeds[:, ::2, :] = state_embeds
                looping_embeds[:, 1::2, :] = action_embeds
            else:
                if self.model_args.tokenize_label:
                    ## action embedding
                    prev_token = self.tokenize(trajectory_to_loop[:, -1, ...].unsqueeze(1))
                    prev_tokenized_action = F.one_hot(prev_token.to(torch.int64), 80 * 40)
                    action_embeds = self.action_m_embed(prev_tokenized_action.to(torch.float32))
                else:
                    action_embeds = self.action_m_embed(trajectory_to_loop[:, -1, :2].unsqueeze(1).to(torch.float32))
                looping_embeds = torch.cat((looping_embeds, action_embeds), dim=1)

            attention_mask = self._prepare_attention_mask_for_generation(looping_embeds)
            position_ids = self._prepare_position_ids_for_generation(attention_mask)
            transformer_output = self.transformer(
                inputs_embeds=looping_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                # **input_kwargs
            )
            transformer_hidden_state = transformer_output[0]
            output = self.traj_decoder(transformer_hidden_state[:, -1, :])  # b, k*2
            predict_actions = output.reshape(batch_size, -1, 2)  # b, k, 2
            pred_logits = self.next_token_scorer_decoder(transformer_hidden_state[:, -1, :])  # b, k
            selected_pred = torch.argmax(pred_logits, dim=-1)  # b
            selected_actions = []
            for i in range(batch_size):
                selected_actions.append(predict_actions[i, selected_pred[i], :].unsqueeze(0))
            selected_actions = torch.cat(selected_actions, dim=0).reshape(batch_size, 1, 2)
            trajectory_to_loop = torch.cat((trajectory_to_loop, selected_actions), dim=1)
            past_length += 1

        return trajectory_to_loop[:, provided_length:, :2]

    def generate_legacy(self,
                high_res_raster: Optional[torch.Tensor] = None,
                low_res_raster: Optional[torch.Tensor] = None,
                trajectory: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = True,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                past_length: Optional[int] = 11,
                seq_length: Optional[int] = 40,
                **kwargs):
        """
        all the input items only include the historic contents
        """
        device = high_res_raster.device
        if len(high_res_raster.shape) == 4: # convert (b, h, w, seq*c) ->(b, seq, c, w, h)
            _b, _h, _w, _= high_res_raster.shape
            high_res_raster = high_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)[:, :past_length, ...]
            low_res_raster = low_res_raster.reshape(_b, _h, _w, -1, self.in_channels).permute(0, 3, 4, 1, 2)[:, :past_length, ...]
        batch_size, seq, c, h, w = high_res_raster.shape
        high_res_embed = self.cnn_downsample(high_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        low_res_embed = self.cnn_downsample(low_res_raster.to(torch.float32).reshape(batch_size * seq, c, h, w)).reshape(batch_size, seq, -1)
        state_embeds = torch.cat((high_res_embed, low_res_embed), dim=-1).to(torch.float32)
        input_kwargs = dict(
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.model_args.tokenize_label:
            ## action embedding
            trajectory = self.tokenize(trajectory[:, :past_length, ...])
            tokenized_trajectory = F.one_hot(trajectory.to(torch.int64), 80 * 40)
            action_embeds = self.action_m_embed(tokenized_trajectory.to(torch.float32))
        else:
            assert False, "not implemented"
        input_embeds = torch.cat((torch.zeros_like(state_embeds, dtype=torch.float32, device=device),
                                  torch.zeros_like(action_embeds, dtype=torch.float32, device=device)), dim=1)

        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        # result dict
        step = 0
        beam = self.beam_search(input_embeds, input_kwargs, max_length=seq_length, beam_width=6)
        best_seq, _ = beam[0]
        # TODO OA pair
        best_seq = best_seq[:, -seq_length:, :]
        action_labels = self.traj_decoder(best_seq)
        action_logits = F.softmax(action_labels, dim=-1)
        action_labels = torch.argmax(action_logits, dim = -1)
        return self.token2action(action_labels)

    def beam_search(self, input_embeds, input_kwargs, max_length, beam_width=6):
            # input_embeds shape is (bsz, seq_length, hidden_size)
            self.eval()
            batch_size = input_embeds.shape[0]
            with torch.no_grad():
                # init beams and scores
                beam = [(input_embeds, 0)]

                for _ in range(max_length):
                    candidates = []

                    for seq, score in beam:
                        attention_mask = self._prepare_attention_mask_for_generation(seq)
                        position_ids = self._prepare_position_ids_for_generation(attention_mask)
                        transformer_output = self.transformer(
                            inputs_embeds=seq,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **input_kwargs
                        )
                        transformer_hidden_state = transformer_output[0]
                        output = self.traj_decoder(transformer_hidden_state[:, -1, :])
                        prob_dist = torch.log_softmax(output, dim=1)

                        topk_probs, topk_class = prob_dist.topk(beam_width, dim=-1)
                        topk_class = topk_class.reshape(topk_class.shape[0], 1, -1)
                        for j in range(beam_width):
                            action = F.one_hot(topk_class[:, :, j].to(torch.int64), 80 * 40).to(torch.float32)
                            next_token_embed = self.action_m_embed(action)
                            # TODO OA pair
                            candidate_seq = torch.cat((seq, next_token_embed), dim=1)
                            candidate_score = score + topk_probs[:, j]
                            candidates.append((candidate_seq, candidate_score))

                    candidate_perbatch_list = []
                    for i in range(batch_size):
                        candidate_perbatch = []
                        for candidate in candidates:
                            cls, score = candidate[0][i], candidate[1][i]
                            candidate_perbatch.append((cls, score))
                        candidate_perbatch = sorted(candidate_perbatch, key=lambda x: x[1], reverse=True)[:beam_width]
                        candidate_perbatch_list.append(candidate_perbatch)

                    reshaped_candidates = list()
                    for i in range(beam_width): # reshape the tensor by order
                        candidate_cls_order = []
                        candidate_score_order = []
                        for candidate_perbatch in candidate_perbatch_list:
                            candidate_cls_order.append(candidate_perbatch[i][0].unsqueeze(0))
                            candidate_score_order.append(candidate_perbatch[i][1].unsqueeze(0))
                        candidate_cls_order = torch.cat(candidate_cls_order, dim=0)
                        candidate_score_order = torch.cat(candidate_score_order, dim=0)
                        reshaped_candidates.append((candidate_cls_order, candidate_score_order))

                    beam = reshaped_candidates

                return beam

    def token2action(self, tokens):
        y_label = torch.floor(torch.div(tokens, self.token_map["x_class"]))
        x_label = tokens - y_label * self.token_map["x_class"]
        y_values = self.token_map["y_range"][0] + y_label * self.token_map["y_interval"]
        x_values = self.token_map["x_range"][0] + x_label * self.token_map["x_interval"]
        return torch.cat([x_values.unsqueeze(-1), y_values.unsqueeze(-1)], dim=-1)

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

        # low velocity filter
        # TODO: set dynamic threshold
        trajectory[:, :, 0] = torch.where(trajectory[:, :, 0] < 0.1, 0, trajectory[:, :, 0])
        trajectory[:, :, 1] = torch.where(trajectory[:, :, 0] < 0.1, 0, trajectory[:, :, 1])

        for idx in range(0, trajectory.shape[1]):
            # loop each point in trajectory
            cos_, sin_ = torch.cos(-ego_trajectory[:, -1, -1]).to(device), torch.sin(-ego_trajectory[:, -1, -1]).to(device)
            delta_yaw = torch.arctan(torch.divide(trajectory[:, idx, 1], trajectory[:, idx, 0]))
            offset_x = trajectory[:, idx, 0] * cos_ + trajectory[:, idx, 1] * sin_
            offset_y = trajectory[:, idx, 1] * cos_ - trajectory[:, idx, 0] * sin_
            next_ego_traj = torch.stack([ego_trajectory[:, -1, 0] + offset_x,
                                        ego_trajectory[:, -1, 1] + offset_y,
                                        torch.zeros_like(ego_trajectory[:, -1, 1]),
                                        # ego_trajectory[:, -1, -1] + delta_yaw],dim=-1)
                                        ego_trajectory[:, -1, -1] + trajectory[:, idx, -1]],dim=-1)
            ego_trajectory = torch.cat((ego_trajectory, torch.tensor(next_ego_traj).reshape(bsz, 1, -1)), dim=1)
        # bsz, seq, 4
        return ego_trajectory[:, 1:, :]

    def compute_yaw(self, point):
        yaw = torch.arctan(point[:, 1]/point[:, 0])
        return yaw