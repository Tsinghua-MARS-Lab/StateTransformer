from .model import GPTModelNuPlan
from .decoders import DecoderResCat
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        merge_bt = False
        if len(polylines.shape) == 5:
            merge_bt = True
            B, T, num_polylines, num_points_each_polylines, C = polylines.shape
            polylines = polylines.view(B * T, num_polylines, num_points_each_polylines, C)
            polylines_mask = polylines_mask.view(B * T, num_polylines, num_points_each_polylines)

        assert len(polylines.shape) == 4, polylines.shape

        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid

        if merge_bt: 
            feature_buffers = feature_buffers.view(B, T, num_polylines, -1)

        return feature_buffers

class GPTModelWaymo(GPTModelNuPlan):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        if self.model_args.tokenize_label:
            self.ego_encoder = nn.Sequential(nn.Linear(40 * 80, 128), nn.Tanh())
        else:
            self.ego_encoder = nn.Sequential(nn.Linear(2, 128), nn.Tanh())

        self.agent_encoder = PointNetPolylineEncoder(
            in_channels=2,
            hidden_dim=64,
            num_layers=3,
            out_channels=64
        )

        self.map_encoder = PointNetPolylineEncoder(
            in_channels=9,
            hidden_dim=64,
            num_layers=5,
            num_pre_layers=3,
            out_channels=64
        )

        self.map_downsampling = build_mlps(768 * 64, [128 * 64, 64])

        self.traj_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.k * 80 * 40)

    def forward(
        self,
        agent_trajs,
        track_index_to_predict,
        map_polyline, 
        map_polylines_mask,

        past_key_values = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        agent_trajs: num_egos, num_frames, num_agents (128), c (c=10)
        map_polyline: num_egos, num_frames, num_polylines (768), num_points (20), 9
        """
        device = agent_trajs.device
        input_embeds, ego_mask, gt = self._prepare_model_inputs(agent_trajs, track_index_to_predict, map_polyline, map_polylines_mask)
        attention_mask = ego_mask[:, :-1]
        gt_mask = ego_mask[:, 1:].to(torch.bool)

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
        b, s, _ = hidden_states.shape

        if self.traj_decoder is not None:
            action_logits = self.traj_decoder(hidden_states.to(device))

        loss = torch.tensor(0, dtype=torch.float32, device=device)

        if self.model_args.tokenize_label:
            pred = self.traj_decoder(hidden_states).view(b * s * self.k, -1)
            pred_loss_fct = nn.CrossEntropyLoss(reduce=False)
            pred_loss = pred_loss_fct(pred, gt.repeat(1, 1, self.k).view(b * s * self.k).long()).view(b, s, self.k)
            pred_loss = pred_loss[gt_mask]
            min_indices = torch.argmin(pred_loss, dim=-1)
            loss += torch.mean(pred_loss)
        else:
            k_results = action_logits.reshape(b, s, self.k, 2)
            loss_fct = torch.nn.SmoothL1Loss()
            losses = []  # length of b * s
            min_indices = []  # length of b
            for i in range(b):
                per_batch_losses = []  # length of s, [x, x, ..]
                per_batch_indices = []  # length of s, [3, 2, 1, 0, ..]
                for j in range(s):
                    per_sequence_losses = []  # length of k
                    for k in range(self.k):
                        loss_to_add = loss_fct(k_results[i, j, k, :], gt[i, j, :2].to(device)) * 100
                        per_sequence_losses.append(loss_to_add)
                    min_loss = min(per_sequence_losses)
                    min_loss_index = per_sequence_losses.index(min_loss)
                    per_batch_losses.append(min_loss)
                    per_batch_indices.append(min_loss_index)
                losses += per_batch_losses
                min_indices.append(per_batch_indices)
            loss += sum(losses) / b / s
            min_indices = torch.tensor(min_indices).to(device)

        if self.next_token_scorer_decoder is not None:
            pred_logits = self.next_token_scorer_decoder(hidden_states)  # b, s, k
            pred_logits = pred_logits[gt_mask]
            logit_loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss += logit_loss_fct(pred_logits, min_indices.reshape(-1).long())

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=action_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def _prepare_model_inputs(self, agent_trajs, track_index_to_predict, map_polyline, map_polylines_mask):
        """
        agent_trajs: num_egos, num_frames, num_agents (128), c (c=10)
        map_polyline: num_egos, num_frames, num_polylines (768), num_points (20), 9
        """
        device = agent_trajs.device
        num_egos, num_frames, _, _ = agent_trajs.shape
        ego_trajs = []
        other_trajs = []
        for i, traj in enumerate(agent_trajs):
            ego_trajs.append(traj[:, track_index_to_predict[i], :])
            other_trajs.append(torch.cat((traj[:, :track_index_to_predict[i], :], traj[:, track_index_to_predict[i] + 1:, :]), dim=1))
        
        ego_trajs = torch.stack(ego_trajs, dim=0).to(device).squeeze(2)
        ego_mask = ego_trajs[..., -1]
        if self.model_args.tokenize_label:
            ego_token = self.tokenize(ego_trajs[..., :2])
            gt = ego_token[:, 1:]
            ego_token_input = ego_token[:, :-1]
            ego_token_one_hot = F.one_hot(ego_token_input.to(torch.int64), self.token_map["x_class"] * self.token_map["y_class"])
            ego_embedding = self.ego_encoder(ego_token_one_hot.to(torch.float32))
        else:
            print(ego_trajs.shape)
            ego_input = ego_trajs[:, :-1, :2]
            gt = ego_trajs[:, 1:, :2]
            print(ego_input.shape, gt.shape)
            ego_embedding = self.ego_encoder(ego_input)

        other_trajs = torch.stack(other_trajs, dim=0).to(device)
        other_trajs = other_trajs[:, :-1, :, :]
        agent_mask = other_trajs[..., -1].to(torch.bool)
        other_trajs_pos = other_trajs[..., :2]
        agent_embedding = self.agent_encoder(other_trajs_pos, agent_mask)

        map_embedding = self.map_encoder(map_polyline, map_polylines_mask.to(torch.bool))
        map_embedding = self.map_downsampling(map_embedding.view(num_egos, -1))
        map_embedding = map_embedding.unsqueeze(1).repeat(1, num_frames - 1, 1)
        
        input_embedding = torch.cat((ego_embedding, agent_embedding, map_embedding), dim=-1)

        return input_embedding, ego_mask, gt


