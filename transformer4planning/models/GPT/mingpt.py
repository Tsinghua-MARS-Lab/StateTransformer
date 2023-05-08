# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging
import math
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from rl_frame.model.utils import soft_update_params

logger = logging.getLogger(__name__)

import numpy as np


def mask_inverse(config):
    start, unit_size, remove = 2, 2, 1

    mask = torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
    for i in range(start, config.block_size, unit_size):
        for j in range(1, remove + 1):
            mask[i, i - j] = 0
    return mask


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """base GPT config, params common to all GPT versions."""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def _convert2dic(self):
        dic = dict((name, getattr(self, name)) for name in dir(self) if not name.startswith('_')) 
        dic["embd_pdrop"] = 0.1
        dic["resid_pdrop"] = 0.1
        dic["attn_pdrop"] = 0.1
        return dic
    
class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params."""

    n_layer = 12
    n_head = 12
    n_embd = 768


def build_mlp(
    input_size,
    output_size,
    n_layers,
    size=512,
    activation=nn.ReLU(),
    output_activation=nn.Identity(),
    init_method=None,
    bias=True,
):
    layers = []
    in_size = input_size
    for _ in range(n_layers - 1):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size, bias=bias)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)

    return nn.Sequential(*layers)


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end.

    It is possible to use torch.nn.MultiheadAttention here but I am including an explicit
    implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1))
        # self.register_buffer("inverse_mask", mask_inverse(config)
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, inputs):
        x, mask = inputs
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        outputs = x, mask
        return outputs


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.block_size = config.block_size
        self.model_type = config.model_type  # act based on rtgs ('reward_conditioned') or not ('naive')
        self.ct = 0

        # input embedding stem
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # pos embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # normalization
        self.ln_f = nn.LayerNorm(config.n_embd)

        # action prediction head
        # if config.linear_rtg:
        #     self.reward_conditioned_head = nn.Linear(config.n_embd * 2, config.vocab_size, bias=False) # predict action conditioned on rtg
        # else:
        #     self.reward_conditioned_head = nn.Sequential(
        #         nn.Linear(config.n_embd * 2, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, config.vocab_size),
        #     )
        # self.naive_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # predict action with state embedding
        # # forward prediction head
        # self.forward_pred_head = nn.Linear(config.n_embd * 2, config.n_embd, bias=True)
        # # inverse prediction head
        # self.inverse_pred_head = nn.Linear(config.n_embd * 2, config.vocab_size, bias=False)
        # # reward prediction head
        # self.reward_pred_head = nn.Linear(config.n_embd * 2, 1, bias=False)

        # rtg-based action prediction head
        self.reward_conditioned_head = build_mlp(config.n_embd * 2, config.vocab_size, config.rtg_layers, bias=False)
        # naive action prediction head (for behavior cloning)
        self.naive_head = build_mlp(config.n_embd, config.vocab_size, config.bc_layers, bias=False)
        # forward prediction head
        self.forward_pred_head = build_mlp(config.n_embd * 2, config.n_embd, config.pred_layers, bias=True)
        # inverse prediction head
        self.inverse_pred_head = build_mlp(config.n_embd * 2, config.vocab_size, config.pred_layers, bias=False)
        # reward prediction head
        self.reward_pred_head = build_mlp(config.n_embd * 2, 1, config.pred_layers, bias=False)
        # if config.use_rand_inverse:
        self.rand_inverse_pred_head = build_mlp(config.n_embd, config.vocab_size, config.pred_layers, bias=True)

        self.apply(self._init_weights)

        # observation embedding encoder
        self.state_encoder = nn.Sequential(nn.Linear(config.obs_dim, config.n_embd), nn.Tanh())
        self.target_state_encoder = nn.Sequential(nn.Linear(config.obs_dim, config.n_embd), nn.Tanh())
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())

        # rtg embedding encoder
        self.ret_encoder = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        # action embedding encoder
        if self.config.cont_action:
            self.action_encoder = nn.Sequential(nn.Linear(config.vocab_size, config.n_embd), nn.Tanh())
        else:
            self.action_encoder = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())

        nn.init.normal_(self.action_encoder[0].weight, mean=0.0, std=0.02)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size + 1, config.block_size + 1)).view(
                1, 1, config.block_size + 1, config.block_size + 1
            ),
        )
        self.register_buffer(
            "inverse_mask", mask_inverse(config).view(1, 1, config.block_size + 1, config.block_size + 1)
        )
        # print("mask", self.mask)
        # print("inverse mask", self.inverse_mask)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        states,
        actions,
        targets=None,
        rtgs=None,
        timesteps=None,
        rewards=None,
        pred_forward=False,
        pred_inverse=False,
        pred_reward=False,
        pred_rand_inverse=False,
        rand_mask_size=1,
        mask_obs_size=0,
        forward_weight=1,
    ):
        # states: (batch, context_length, 3*84*84) / (bacth, context_length, n_emdd)
        # actions: (batch, context_length, n_actions) / (bacth, context_length, n_embd)
        # targets: (batch, context_length, 1) / (bacth, context_length, n_embd)
        # rtgs: (batch, context_length, 1) / (batch, context_length, n_embd)
        # timesteps: (batch, 1, 1)

        self.ct += 1  # for debug

        is_testing = (actions is None) or (actions.shape[1] != states.shape[1])

        # (batch * context_length, n_embd)
        state_embeddings = self.state_encoder(states)

        if actions is not None:
            if self.config.cont_action:
                action_embeddings = self.action_encoder(actions)
            else:
                action_embeddings = self.action_encoder(
                    actions.type(torch.long).squeeze(-1)
                )  # (batch, context_length, n_embd)
            # token shape (bacth, 2*history, embed dim)
            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(is_testing), self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(is_testing) :, :]
        else:  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(
            self.global_pos_emb, batch_size, dim=0
        )  # batch_size, traj_length, n_embd
        position_embeddings = (
            torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
            + self.pos_emb[:, : token_embeddings.shape[1], :]
        )

        final_embeddings = self.drop(token_embeddings + position_embeddings)

        x, _ = self.blocks((final_embeddings, self.mask))
        x = self.ln_f(x)

        if actions is not None:
            state_output = x[:, ::2, :]  # predictions from state embeddings
            action_output = x[:, 1::2, :]  # predictions from action embeddings
        else:
            state_output = x  # for completeness
            action_output = None
        
        # return the hidden represention from the last hidden state
        return state_output
        # print("token_embeddings", token_embeddings.size())
        # print("final_embeddings", final_embeddings.size())
        # print("state_output", state_output.size())
        # print("action_output", action_output.size())

        ## act
        rtg_action_logits, naive_action_logits = None, None
        ## compute losses
        losses = defaultdict(float)

        if self.model_type == "reward_conditioned":
            # act according to rtgs
            rtg_embeddings = self.ret_encoder(rtgs.type(torch.float32))
            rtg_action_logits = self.reward_conditioned_head(torch.cat((state_output, rtg_embeddings), dim=2))
            logits = rtg_action_logits
            if targets is not None:
                if self.config.cont_action:
                    losses["acton_rtg"] = F.mse_loss(rtg_action_logits, targets)
                else:
                    losses["acton_rtg"] = F.cross_entropy(
                        rtg_action_logits.reshape(-1, rtg_action_logits.size(-1)), targets.reshape(-1)
                    )

        elif self.model_type == "naive":
            # act without rtgs
            naive_action_logits = self.naive_head(state_output)
            logits = naive_action_logits
            if targets is not None:
                if self.config.cont_action:
                    losses["acton_naive"] = F.mse_loss(naive_action_logits, targets)
                else:
                    losses["acton_naive"] = F.cross_entropy(
                        naive_action_logits.reshape(-1, naive_action_logits.size(-1)), targets.reshape(-1)
                    )
        else:
            raise NotImplementedError()

        if pred_forward:    
            next_state_embeddings = self.state_encoder(states).detach()
            next_state_embeddings = next_state_embeddings[:, 1:, :]  # (batch, context_length-1, n_embd)
            forward_pred = self.forward_pred_head(
                torch.cat((state_output[:, :-1, :], action_output[:, : -1 + int(is_testing), :]), dim=2)
            )
            losses["forward_pred"] = (
                F.mse_loss(
                    forward_pred.reshape(-1, forward_pred.size(-1)),
                    next_state_embeddings.reshape(-1, next_state_embeddings.size(-1)),
                )
                * forward_weight
            )
            soft_update_params(self.state_encoder, self.target_state_encoder, 0.005)

        if pred_reward:
            reward_pred = self.reward_pred_head(torch.cat((state_output, action_output), dim=2))
            losses["reward_pred"] = nn.BCEWithLogitsLoss()(reward_pred, rewards)

        if pred_inverse:
            inv_x, _ = self.blocks((final_embeddings, self.inverse_mask))
            inv_x = self.ln_f(inv_x)
            cur_state_output = state_output[:, :-1, :]  # predictions from cur-state embeddings
            next_state_output = inv_x[:, 2::2, :]  # predictions from next-state embeddings
            inverse_action_logits = self.inverse_pred_head(torch.cat((cur_state_output, next_state_output), dim=2))
            inverse_target = actions[:, : -1 + int(is_testing), :]
            if self.config.cont_action:
                losses["inverse_pred"] = F.mse_loss(inverse_action_logits, inverse_target)
            else:
                losses["inverse_pred"] = F.cross_entropy(
                    inverse_action_logits.reshape(-1, inverse_action_logits.size(-1)), inverse_target.reshape(-1)
                )

        if pred_rand_inverse:
            # randomly mask past actions and predict them
            rand_mask_idx = np.random.choice(actions.shape[1], rand_mask_size, replace=False)
            masked_token = token_embeddings.clone()
            for j in range(rand_mask_size):
                masked_token[:, 1 + 2 * rand_mask_idx[j], :] = -1

            if mask_obs_size > 0:
                assert actions.shape[1] > 2
                rand_mask_obs_idx = np.random.choice(list(range(1, actions.shape[1] - 1)), mask_obs_size, replace=False)
                for j in range(mask_obs_size):
                    masked_token[:, 2 * rand_mask_obs_idx[j], :] = -1
            # batch_size = states.shape[0]
            # all_global_pos_emb = torch.repeat_interleave(
            #     self.global_pos_emb, batch_size, dim=0
            # )  # batch_size, traj_length, n_embd
            # position_embeddings = (
            #     torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
            #     + self.pos_emb[:, : token_embeddings.shape[1], :]
            # )

            final_masked_embeddings = self.drop(masked_token + position_embeddings)

            temp_mask = (
                torch.ones((self.config.block_size + 1, self.config.block_size + 1))
                .view(1, 1, self.config.block_size + 1, self.config.block_size + 1)
                .to(masked_token.device)
            )
            masked_x, _ = self.blocks((final_masked_embeddings, temp_mask))
            x = self.ln_f(masked_x)
            rand_inverse_logits = self.rand_inverse_pred_head(x[:, rand_mask_idx, :])
            rand_inverse_action_targets = actions[:, rand_mask_idx, :]
            if self.config.cont_action:
                losses["rand_inverse_pred"] = F.mse_loss(rand_inverse_logits, rand_inverse_action_targets)
            else:
                losses["rand_inverse_pred"] = F.cross_entropy(
                    rand_inverse_logits.reshape(-1, rand_inverse_logits.size(-1)),
                    rand_inverse_action_targets.reshape(-1),
                )

        return logits, losses

    def get_embeddings(self, states, actions, timesteps):
        if actions is not None and actions.shape[1] == 0:
            actions = None
        is_testing = (actions is None) or (actions.shape[1] != states.shape[1])

        # (batch * context_length, n_embd)
        if hasattr(self.config, "vector_obs") and self.config.vector_obs:
            state_embeddings = self.state_encoder(states)
        else:
            state_embeddings = self.state_encoder(
                states.reshape(-1, self.config.channels, 84, 84).type(torch.float32).contiguous()
            )
            # (batch, context_length, n_embd)
            state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)

        if actions is not None:
            if self.config.cont_action:
                action_embeddings = self.action_encoder(actions)
            else:
                action_embeddings = self.action_encoder(
                    actions.type(torch.long).squeeze(-1)
                )  # (batch, context_length, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(is_testing), self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings  # [:, -states.shape[1]:, :]
        else:  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(
            self.global_pos_emb, batch_size, dim=0
        )  # batch_size, traj_length, n_embd
        position_embeddings = (
            torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
            + self.pos_emb[:, : token_embeddings.shape[1], :]
        )

        final_embeddings = self.drop(token_embeddings + position_embeddings)

        x, _ = self.blocks((final_embeddings, self.mask))
        x = self.ln_f(x)

        return x

    def configure_optimizers(self, hparams):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:     https://pytorch-
        lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters {} were not separated into either decay/no_decay set!".format(
            str(param_dict.keys() - union_params)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": hparams["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=hparams["lr"], betas=hparams["betas"])
        return optimizer

    def configure_naive_optimizer(self, hparams):
        optim_groups = [
            {"params": list(self.naive_head.parameters()), "weight_decay": hparams["weight_decay"]},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=hparams["lr"], betas=hparams["betas"])
        return optimizer
