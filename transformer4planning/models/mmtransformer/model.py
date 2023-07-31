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

from torch.distributions.multivariate_normal import MultivariateNormal
from transformer4planning.models.mmtransformer.simple_decoder import SimpleTrajectoryDecoder
from transformer4planning.models.mmtransformer.stacked_transformer import STF
from transformer4planning.models.mmtransformer.criterion import Loss
from dataclasses import dataclass, field

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

def cat_raster_seq_for_waymo(raster, framenum=11):
    b, c, h, w = raster.shape
    agent_type = 6
    road_type = 20
    road_raster = raster[:, :road_type, :, :]
    result = torch.zeros((b, framenum, agent_type + road_type, h, w), device=raster.device)
    for i in range(framenum):
        agent_raster = raster[:, road_type + i::framenum, :, :]
        raster_i = torch.cat([road_raster, agent_raster], dim=1)
        assert raster_i.shape[1] == agent_type + road_type
        result[:, i, :, :, :] = raster_i
    return result

class MMTransformer(GPT2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        model_args = kwargs["model_args"]
        if "transfer" in model_args.model_name:
            self.transformer = GPT2Model.from_pretrained(model_args.model_pretrain_name_or_path)
            print("Backbone init by transfered one")
        else:
            self.transformer = GPT2Model(config)
        self.predict_trajectory = model_args.predict_trajectory
        self.loss_fn = model_args.loss_fn
        self.task = model_args.task
        if model_args.task == 'nuplan':
            in_channels = 29  # raster: goal + road_type + agent_type
        else:
            in_channels = 26 # raster: road_type(20) + agent_type(6)
        n_embed = config.n_embd // 2
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=in_channels)

        self.action_m_embed = nn.Sequential(nn.Linear(4, config.n_embd), nn.Tanh())

        self.traj_decoder = None
        
        self.loss = Loss(K=6, future_num_frames=80)

        # end of added
        # Initialize weights and apply final processing
        self.model_parallel = False
        self.device_map = None
        self.post_init()
        
        self.K = 6
        self.modality_embeds = nn.Embedding(self.K, config.n_embd)
        # Use mmTransformer decoder as default.
        self.mmT = True
        nn.init.orthogonal_(self.modality_embeds.weight)
        if self.predict_trajectory:
            if self.mmT:
                self.traj_decoder = STF(config.n_embd, 6, future_num_frames=80)
            else:
                self.traj_decoder = SimpleTrajectoryDecoder(config.n_embd)
    
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
            trajectory_label: Optional[torch.LongTensor] = None,
            context_actions:Optional[torch.LongTensor] = None,
            high_res_raster: Optional[torch.LongTensor] = None,
            low_res_raster: Optional[torch.LongTensor] = None,
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
        if self.task == "waymo":
            context_actions[:, :, -1] = 0
        # inspect actions
        max_action = torch.max(context_actions[:, :, :2])
        min_action = torch.min(context_actions[:, :, :2])
        if abs(max_action) > 1000 or abs(min_action) > 1000:
            print(context_actions.shape)
            print(context_actions[:10, :])
            assert False, f'Invalid actions to filter: {max_action}, {min_action}'

        # inspect labels
        max_label = torch.max(trajectory_label[:, :, :2])
        min_label = torch.min(trajectory_label[:, :, :2])
        if abs(max_label) > 1000 or abs(min_label) > 1000:
            print(trajectory_label.shape)
            print(trajectory_label[:80, :])
            assert False, f'Invalid labels to filter: {max_label}, {min_label}'


        device = high_res_raster.device
        action_embeds = self.action_m_embed(context_actions)
        context_length = context_actions.shape[1] + 1
        if self.task == "nuplan":
            high_res_seq = cat_raster_seq(high_res_raster.permute(0, 3, 2, 1).to(device), context_length)
            low_res_seq = cat_raster_seq(low_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        elif self.task == "waymo":
            high_res_seq = cat_raster_seq_for_waymo(high_res_raster.permute(0, 3, 2, 1).to(device), context_length)
            low_res_seq = cat_raster_seq_for_waymo(low_res_raster.permute(0, 3, 2, 1).to(device), context_length)
        batch_size, context_length, c, h, w = high_res_seq.shape
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size * context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)
        # assert not high_res_embed.isnan().any(), "high embedding is NAN"
        # assert not low_res_embed.isnan().any(), "low embedding is NAN"
        state_embeds = torch.cat((high_res_embed,
                                  low_res_embed), dim=-1).to(torch.float32)
        if self.task == "waymo":
            trajectory_label = trajectory_label[:, :, :2]
        else:
            trajectory_label = trajectory_label[:, 1::2, :2]
        pred_length = trajectory_label.shape[1]
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2 - 1, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds
        # assert not state_embeds.isnan().any(), "state embedding is NAN"
        # assert not action_embeds.isnan().any(), "action embedding is NAN"
        
        # to keep input and output at the same dimension
        if self.mmT:
            decode_embeds = torch.zeros((batch_size, pred_length, n_embed), device=device)
            input_embeds = torch.cat([input_embeds, decode_embeds], dim=1)
        else:
            modality_embeds = self.modality_embeds.weight
            m = MultivariateNormal(modality_embeds,torch.eye(modality_embeds.shape[-1],device=device)[None].repeat(self.K,1,1)*0.01)
            modality_embeds = m.sample((batch_size,)).view(batch_size*self.K,1,n_embed)
            
            decode_embeds = torch.zeros((batch_size*self.K, pred_length+1, n_embed), device=device)
            input_embeds = input_embeds.repeat_interleave(self.K, dim=0)
            # input_embeds = torch.cat([modality_embeds, input_embeds, decode_embeds], dim=1)
            input_embeds = torch.cat([input_embeds, modality_embeds, decode_embeds], dim=1)
         
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
        # assert not transformer_outputs_hidden_state.isnan().any(), "Hidden state is NAN"
        if not self.mmT:
            traj_hidden_state = transformer_outputs_hidden_state[:, -(pred_length+1):, :]
        else:
            traj_hidden_state = transformer_outputs_hidden_state[:, -pred_length:, :]
        
        traj_coords, traj_logits = self.traj_decoder(traj_hidden_state)
        
        # simple version
        if not self.mmT:
            traj_coords, traj_logits = \
                traj_coords.view(batch_size,self.K,pred_length,2)[:,None], traj_logits.view(batch_size,self.K)[:,None]
        
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        
        _loss, _, _ = self.loss((traj_coords, traj_logits[:,None]), trajectory_label.to(device))
        
        loss += _loss
        traj_coords, traj_logits = traj_coords[:,0], traj_logits[:,0] # a hack
        # get the coords with highest score
        # idx_highest = torch.argmax(traj_logits, dim=-1).detach().cpu().numpy()
        # traj_coords = traj_coords[list(np.arange(traj_coords.shape[0])), list(idx_highest), :]
        return CustomOutput(
            loss=loss,
            logits=traj_coords,
            scores=traj_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

@dataclass
class CustomOutput(ModelOutput):
    scores: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


    