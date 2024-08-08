#######################################################################################
# receive the checkpoint from the original STR and combine it with the diffusion model#
#######################################################################################
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from einops import rearrange
from transformer4planning.models.backbone.str_base import STR
from transformer4planning.models.algorithms.diffusion_forcing.df_base import DiffusionForcingBase
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import logging
from typing import Tuple, Optional, Dict
from random import random

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    trajectory_label: torch.FloatTensor = None
    


class StrDiff(DiffusionForcingBase):
    def __init__(self, cfg, str_model:STR):
        super().__init__(cfg=cfg)
        logging.info(f"STRDiff: {cfg}")
        self.str_model = str_model
        self._build_model()
        self.configure_optimizers()
        
    def forward(self,
            return_dict: Optional[bool] = None,
            **kwargs):
        """
        input should be the same as the STR model
        `high_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `low_res_raster`: torch.Tensor, shape (batch_size, 224, 224, seq)
        `context_actions`: torch.Tensor, shape (batch_size, seq, 4 / 6)
        `trajectory_label`: torch.Tensor, shape (batch_size, seq, 2/4), depend on whether pred yaw value
        `pred_length`: int, the length of prediction trajectory
        `context_length`: int, the length of context actions
        """
        
        # Phase1: get the result from the STR model
        str_result = self.str_model.generate(**kwargs)
        trajectory_prior = str_result["traj_logits"]
        trajectory_label = kwargs.get("trajectory_label", None)
        batch = [trajectory_label, trajectory_prior]
        
        # Phase2:convey the result to the diffusion model
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch, is_train=True)
        n_frames, batch_size, _, *_ = xs.shape
        
        xs_pred = []
        loss = []
        z = init_z
        cum_snr = None
        for t in range(0, n_frames):
            deterministic_t = None
            if random() <= self.gt_cond_prob or (t == 0 and random() <= self.gt_first_frame):
                deterministic_t = 0
            # print(f"log the input data, z shape:{z.shape},xs shape:{xs[t].shape}, condition shape: {conditions[t].shape}")
            z_next, x_next_pred, l, cum_snr = self.transition_model(
                z, xs[t], conditions[t], deterministic_t=deterministic_t, cum_snr=cum_snr
            )

            z = z_next
            xs_pred.append(x_next_pred)
            loss.append(l)
        xs_pred = torch.stack(xs_pred)
        loss = torch.stack(loss)
        x_loss = self.reweigh_loss(loss)
        loss = x_loss
        # output_dict = {
        #     "loss": loss,
        #     "xs_pred": self._unnormalize_x(xs_pred),
        #     "xs": self._unnormalize_x(xs),
        # }
        output_dict = LTMOutput(
            loss=loss,
            logits=self._unnormalize_x(xs_pred),
            trajectory_label=self._unnormalize_x(xs)
        )
        return output_dict

    def _preprocess_batch(self, batch, is_train=True):
        if is_train:
            batch[1]= batch[1]
            return super()._preprocess_batch(batch)
        else:
            batch[0]= batch[0]
            batch_size, n_frames = batch[0].shape[:2]
            xs = torch.randn(batch_size, n_frames, *self.x_shape)

            if n_frames % self.frame_stack != 0:
                raise ValueError("Number of frames must be divisible by frame stack size")
            if self.context_frames % self.frame_stack != 0:
                raise ValueError("Number of context frames must be divisible by frame stack size")

            nonterminals = batch[-1]
            nonterminals = nonterminals.bool().permute(1, 0)
            masks = torch.cumprod(nonterminals, dim=0).contiguous()
            n_frames = n_frames // self.frame_stack

            if self.external_cond_dim:
                conditions = batch[0]
                conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
                conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
            else:
                conditions = [None for _ in range(n_frames)]

            xs = self._normalize_x(xs)
            xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

            if self.learnable_init_z:
                init_z = self.init_z[None].expand(batch_size, *self.z_shape)
            else:
                init_z = torch.zeros(batch_size, *self.z_shape)
                init_z = init_z.to(xs.device)

            return xs, conditions, masks, init_z
    
    @torch.no_grad()
    def generate(self, **kwargs):

        
        # Phase1: get the result from the STR model
        str_result = self.str_model.generate(**kwargs)
        trajectory_prior = str_result["traj_logits"]
        
        
        # Phase2:convey the result to the diffusion model
        batch = [trajectory_prior]
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch, is_train=False)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        xs_pred_all = []
        z = init_z
        
        # prediction
        while len(xs_pred) < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)

            chunk = [
                torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
            ]

            pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
            pyramid = np.zeros((pyramid_height, horizon), dtype=int)
            for m in range(pyramid_height):
                for t in range(horizon):
                    pyramid[m, t] = m - int(t * self.uncertainty_scale)
            pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

            for m in range(pyramid_height):
                if self.transition_model.return_all_timesteps:
                    xs_pred_all.append(chunk)

                z_chunk = z.detach()
                for t in range(horizon):
                    i = min(pyramid[m, t], self.sampling_timesteps - 1)

                    chunk[t], z_chunk = self.transition_model.ddim_sample_step(
                        chunk[t], z_chunk, conditions[len(xs_pred) + t], i
                    )

                    # theoretically, one shall feed new chunk[t] with last z_chunk into transition model again 
                    # to get the posterior z_chunk, and optionaly, with small noise level k>0 for stablization. 
                    # However, since z_chunk in the above line already contains info about updated chunk[t] in 
                    # our simplied math model, we deem it suffice to directly take this z_chunk estimated from 
                    # last z_chunk and noiser chunk[t]. This saves half of the compute from posterior steps. 
                    # The effect of the above simplification already contains stablization: we always stablize 
                    # (ddim_sample_step is never called with noise level k=0 above)

            z = z_chunk
            xs_pred += chunk
        xs_pred = torch.stack(xs_pred)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = self._unnormalize_x(xs_pred)
        pred_dict = {
            "traj_logits": traj_pred_logits
        }
        return pred_dict

        
        
        
        

        
        
        
