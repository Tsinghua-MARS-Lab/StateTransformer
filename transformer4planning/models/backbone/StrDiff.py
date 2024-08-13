#######################################################################################
# receive the checkpoint from the original STR and combine it with the diffusion model#
#######################################################################################
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Union, Sequence
from einops import rearrange
from transformer4planning.models.backbone.str_base import STR
from transformer4planning.models.algorithms.diffusion_forcing.df_base import DiffusionForcingBase
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformer4planning.models.algorithms.diffusion_forcing.models.diffusion_transition import DiffusionTransitionModel
import logging
from typing import Tuple, Optional, Dict
from random import random

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    trajectory_label: torch.FloatTensor = None
    loss_items: Optional[torch.FloatTensor] = None
    pred_dict: Optional[Dict] = None
    


class StrDiff(PreTrainedModel):
    def __init__(self, config):
        from transformer4planning.utils.common_utils import load_config
        print("=================StrDiff===================")
        print("loading the diffusion config...")
        cfg=load_config(config.diffusion_config)
        self.cfg = cfg
        ########################
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_shape) == 3  # pixel
        self.gt_cond_prob = cfg.gt_cond_prob  # probability to condition one-step diffusion o_t+1 on ground truth o_t
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.chunk_size = cfg.chunk_size
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.learnable_init_z = cfg.learnable_init_z
        ########################
        super().__init__(config)
        from transformer4planning.models.backbone.mixtral import STR_Mixtral
        print("loading the STR model as a part of the whole model...")
        self.str_model = STR_Mixtral.from_pretrained(config.model_pretrain_name_or_path,config=config)

        self.freeze_parameters(self.str_model) # freeze the STR model
        self._build_model()
        self.configure_optimizers()
        
    @property
    def encoder(self):
        return self.str_model.encoder
    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
        # build the core diffusion transition model and store the data mean and std
    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.learnable_init_z:
            self.init_z = nn.Parameter(torch.randn(list(self.z_shape)), requires_grad=True)
    
    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            transition_params.append(self.init_z)
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )

        return optimizer_dynamics
    
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
        maps_info = str_result["maps_info"]
        trajectory_label = kwargs.get("trajectory_label", None)
        batch = [trajectory_label, trajectory_prior, maps_info]
        
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

        # output_dict = {
        #     "loss": loss,
        #     "xs_pred": self._unnormalize_x(xs_pred),
        #     "xs": self._unnormalize_x(xs),
        # }
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> b (fs t) c ...", fs=self.frame_stack)
        pred_dict = {
            "traj_logits": xs_pred.to(dtype=torch.float)
        }
        
        output_dict = LTMOutput(
            loss=x_loss,
            loss_items=loss,
            logits=self._unnormalize_x(xs_pred).view(-1, *self.x_shape),
            trajectory_label=self._unnormalize_x(xs),
            pred_dict=pred_dict,
        )
        return output_dict

    def _preprocess_batch(self, batch, is_train=True):
        batch_size, n_frames = batch[0].shape[:2]
        if is_train:
            prior_indice = 1
            maps_indice = 2
            label_indice = 0
            xs = batch[1]
        else:
            prior_indice = 0
            maps_indice = 1
            label_indice = None
            xs = torch.randn(batch_size, n_frames, *self.x_shape)
        

        # check if number of frames is divisible by frame stack size
        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[prior_indice]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()

        else:
            conditions = [None for _ in range(n_frames)]
        
        # print(f"log the input data, xs shape:{xs.shape}, conditions shape: {conditions.shape}")
            


        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        if self.learnable_init_z:
            init_z = self.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)

        return xs, conditions, None, init_z
    
    @torch.no_grad()
    def generate(self, **kwargs)-> torch.FloatTensor:

        
        # Phase1: get the result from the STR model
        str_result = self.str_model.generate(**kwargs)
        trajectory_prior = str_result["traj_logits"]
        maps_info = str_result["maps_info"]
        
        
        # Phase2:convey the result to the diffusion model
        batch = [trajectory_prior, maps_info]
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
        xs_pred = xs_pred.reshape(batch_size, n_frames*self.frame_stack, *self.x_shape)
        pred_dict = {
            "traj_logits": xs_pred.to(dtype=torch.float)
        }
        return pred_dict
    
    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return xs * std + mean
        
        
        # reweigh the loss
    def reweigh_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(weight, "(t fs) b ... -> t b fs ..." + " 1" * expand_dim, fs=self.frame_stack)
            loss = loss * weight

        return loss.mean()

        
    def register_data_mean_std(
        self, mean: Union[str, float, Sequence], std: Union[str, float, Sequence], namespace: str = "data"
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))


        
        
        
        

        
        
        
