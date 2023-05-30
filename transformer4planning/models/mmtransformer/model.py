from transformers.models.xlnet import XLNetConfig, XLNetModel, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetLMHeadModelOutput
from transformers.models.t5 import T5Config,T5Model, T5PreTrainedModel
from transformers.models.deberta_v2 import DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput

from transformer4planning.models.TransformerXL.model import *
from transformer4planning.models.GPT2.models import *
from transformer4planning.models.nsm import NSMDecoder
from transformer4planning.models.encoders import *
from transformer4planning.models.decoders import *


import torch.nn as nn
from transformers import GPT2Model,GPT2PreTrainedModel
_CHECKPOINT_FOR_DOC = "transfo-xl-wt103"
_CONFIG_FOR_DOC = "TransfoXLConfig"


from .stacked_transformer import STF
from .criterion import Loss


class MMTransformer(GPT2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        model_args = kwargs["model_args"]
        self.use_nsm = model_args.use_nsm
        self.with_future_intend_maneuver_with_encoder = model_args.with_future_intend_maneuver_with_encoder
        self.with_future_current_maneuver = model_args.with_future_current_maneuver
        self.predict_trajectory = model_args.predict_trajectory
        self.predict_trajectory_with_stopflag = model_args.predict_trajectory_with_stopflag
        self.loss_fn = model_args.loss_fn
        in_channels = 29 # raster: goal + road_type + agent_type
        if self.use_nsm and self.predict_trajectory_with_stopflag:
            n_embed = config.n_embd // 3 # high res + low res + intented m
        else:
            n_embed = config.n_embd // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)
        self.intended_m_embed = nn.Sequential(nn.Embedding(num_embeddings=30, embedding_dim=n_embed), nn.Tanh())
        assert not (self.with_future_intend_maneuver_with_encoder and self.with_future_intend_maneuver_with_decoder) # choose up to one of intend and weights m
        if self.with_future_intend_maneuver_with_encoder or self.with_future_intend_maneuver_with_decoder:
            self.future_intended_m_embed = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.n_embd), nn.Tanh())

        if self.predict_trajectory_with_stopflag:
            self.stop_flag_embed = nn.Sequential(nn.Embedding(num_embeddings=30, embedding_dim=config.n_embd), nn.Tanh())

        self.traj_decoder = None
        if self.predict_trajectory:
            self.traj_decoder = STF(config.n_embd, 6, future_num_frames=80)
        
        self.loss = Loss(K=6, future_num_frames=80)

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
        self.intended_m_embed = self.intended_m_embed.to(self.transformer.first_device)
        self.intended_m_decoder = self.intended_m_decoder.to(self.transformer.first_device)
        self.current_m_decoder = self.current_m_decoder.to(self.transformer.first_device)
        self.nsm_decoder = self.nsm_decoder.to(self.transformer.first_device)
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
        self.intended_m_embed = self.intended_m_embed.to("cpu")
        self.intended_m_decoder = self.intended_m_decoder.to("cpu")
        self.current_m_decoder = self.current_m_decoder.to("cpu")
        self.nsm_decoder = self.nsm_decoder.to("cpu")
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
        # with future manuever label input
        if self.with_future_intend_maneuver_with_encoder or self.with_future_intend_maneuver_with_decoder:
            future_maneuver_embed = self.future_intended_m_embed(intended_maneuver_gt.unsqueeze(-1).to(device).to(torch.float32))
        if self.predict_trajectory_with_stopflag and self.use_nsm:
            stopflag = torch.eq(intended_maneuver_label, 1) # bsz,  -> bsz,
            stopflag_embed = self.stop_flag_embed(stopflag.to(device).long())
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)
        
        if self.use_nsm and self.predict_trajectory_with_stopflag:
            intended_maneuver_embed = self.intended_m_embed(intended_maneuver_vector.to(device))  # [bsz, hidden_size]
            state_embeds = torch.cat((intended_maneuver_embed,
                                    high_res_embed,
                                    low_res_embed), dim=-1)
        else:
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
        if self.with_future_intend_maneuver_with_encoder:
            input_embeds = torch.cat([input_embeds, future_maneuver_embed], dim=1)
        else:
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
        traj_coords, traj_logits = self.traj_decoder(traj_hidden_state)
        
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        
        loss += self.loss((traj_coords, traj_logits), trajectory_label.to(device))
        traj_coords, traj_logits = traj_coords[:,0], traj_logits[:,0] # a hack
        
        return custom_output(
            loss=loss,
            logits=traj_coords,
            scores=traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class custom_output(CausalLMOutputWithCrossAttentions):
    scores = None