#######################################################################################
# receive the checkpoint from the original STR and combine it with the diffusion model#
#######################################################################################
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Union, Sequence
from einops import rearrange
from transformer4planning.models.algorithms.common.base_pytorch_algo import BasePytorchAlgo
from transformer4planning.models.algorithms.diffusion_forcing.df_base import DiffusionForcingBase
from transformer4planning.models.backbone.str_base import STR
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import logging
from einops import rearrange, reduce
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from transformer4planning.models.diffusion_loss.diffusion_transition import DiffusionTransitionModel
from transformer4planning.models.backbone.str_base import STR
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    trajectory_label: torch.FloatTensor = None
    loss_items: Optional[torch.FloatTensor] = None
    pred_dict: Optional[Dict] = None
    


class StrDiffDiT(PreTrainedModel):
    def __init__(self, config):
        from transformer4planning.utils.common_utils import load_config
        print("=================StrDiffDiT===================")
        print("loading the diffusion config...")
        cfg=load_config(config.diffusion_config)
        self.cfg = cfg
        ########################
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.external_cond_dim = cfg.external_cond_dim
        self.learnable_init_z = cfg.learnable_init_z
        ########################
        super().__init__(config)
        from transformer4planning.models.backbone.mixtral import STR_Mixtral
        print("loading the STR model as a part of the whole model...")
        self.str_model = STR_Mixtral.from_pretrained(config.model_pretrain_name_or_path,config=config)
        self.freeze_parameters(self.str_model) # freeze the STR model
        
        self.frame_stack = self.cfg.frame_stack
        self.learnable_init_z = self.cfg.learnable_init_z
        self.map_cond = self.cfg.map_cond
        self._build_model()
        self.configure_optimizers()
        
    @property
    def encoder(self):
        return self.str_model.encoder
    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(self.cfg)
        
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.learnable_init_z:
            self.init_z = nn.Parameter(torch.randn(list(self.cfg.z_shape)), requires_grad=True)

    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            transition_params.append(self.init_z)
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )

        return optimizer_dynamics
    
        # process the lr for the optimizer
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr
    
    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
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
        xs, prior, masks, *_, init_z, cond = self._preprocess_batch(batch, is_train=True)
        n_frames, batch_size, _, *_ = xs.shape
        
        xs_pred = []
        loss = []
        z = init_z
        for t in range(0, n_frames):
            z_next, x_next_pred, target= self.transition_model(
                z, xs[t], cond[t], prior[t]
            )
            z = z_next
            xs_pred.append(x_next_pred)
            # print("x_next_pred:",x_next_pred[0])
            # print("target:",target[0])
            # calculate the loss
            l = F.mse_loss(x_next_pred, target, reduction='none')
            loss.append(l)
        xs_pred = torch.stack(xs_pred)
        xs_pred = self._unnormalize_x(xs_pred)
        xs_pred = rearrange(xs_pred, "t b fs c ... -> b (t fs) c ...", fs=self.frame_stack)
        loss = torch.stack(loss)
        x_loss = self.reweigh_loss(loss)

        # output_dict = {
        #     "loss": loss,
        #     "xs_pred": self._unnormalize_x(xs_pred),
        #     "xs": self._unnormalize_x(xs),
        # }
        pred_dict = {
            "traj_logits": xs_pred
        }
        output_dict = LTMOutput(
            loss=x_loss,
            loss_items=loss,
            logits=xs_pred,
            pred_dict=pred_dict,
            trajectory_label=self._unnormalize_x(xs)
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
        if self.cfg.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        n_frames = n_frames // self.frame_stack

        if self.cfg.cond_dim:
            conditions = batch[prior_indice]

            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b fs d", fs=self.frame_stack).contiguous()
            if self.map_cond:
                maps_info = batch[maps_indice]
                maps_info = rearrange(maps_info, "b t d -> 1 b t d").repeat(n_frames, 1, 1, 1)
                
                
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b fs c ...", fs=self.frame_stack).contiguous()
        if self.learnable_init_z:
            init_z = self.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)

        return xs, conditions, None, init_z, maps_info
    
    @torch.no_grad()
    def generate(self, **kwargs)-> torch.FloatTensor:

        
        # Phase1: get the result from the STR model
        str_result = self.str_model.generate(**kwargs)
        trajectory_prior = str_result["traj_logits"]
        maps_info = str_result["maps_info"]
        
        
        # Phase2:convey the result to the diffusion model
        batch = [trajectory_prior, maps_info]
        xs, prior, masks, *_, init_z, cond = self._preprocess_batch(batch, is_train=False)
        n_frames, batch_size, _, *_ = xs.shape
        
        xs_pred = []
        z = init_z
        
        # prediction
        for t in range(0, n_frames):
            z_next, x_next_pred= self.transition_model.generate(
                xs[t], z, cond[t], prior[t]
            )

            z = z_next
            xs_pred.append(x_next_pred)
        xs_pred = torch.stack(xs_pred)
        xs_pred = rearrange(xs_pred, "t b fs c ... -> b (t fs) c ...", fs=self.frame_stack)
        xs_pred = self._unnormalize_x(xs_pred)
        pred_dict = {
            "traj_logits": xs_pred.to(dtype=torch.float)
        }
        print("xs_pred:",xs_pred[0,:10,0])
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
