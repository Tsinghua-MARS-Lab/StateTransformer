#######################################################################################
# receive the checkpoint from the original STR and combine it with the diffusion model#
#######################################################################################
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Union, Sequence
from einops import rearrange
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import logging
from einops import rearrange, reduce
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from transformer4planning.models.diffusion_loss.traj_diffusion import TrajDiffusion
from transformers.modeling_utils import PreTrainedModel

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    trajectory_label: torch.FloatTensor = None
    loss_items: Optional[torch.FloatTensor] = None
    pred_dict: Optional[Dict] = None
    


class ExplicitDiT(PreTrainedModel):
    def __init__(self, config):
        from transformer4planning.utils.common_utils import load_config
        print("=================ExplicitDiT===================")
        print("loading the diffusion config...")
        cfg=load_config(config.diffusion_config)
        
        ###########config the attribute#############
        self.cfg = cfg
        self.trajectory_dim = cfg.trajectory_dim
        self.frame_stack = cfg.frame_stack
        self.learnable_init_traj = cfg.learnable_init_traj
        self.map_cond = self.cfg.map_cond
        self.config = config
        
        # init the pretrained model
        super().__init__(config)
        
        # load the STR model
        from transformer4planning.models.backbone.mixtral import STR_Mixtral
        print("loading the STR model as a part of the whole model...")
        self.str_model = STR_Mixtral.from_pretrained(config.model_pretrain_name_or_path,config=config)
        
        # freeze the STR model
        self.freeze_parameters(self.str_model) # freeze the STR model
        
        self._build_model()
        
    @property
    def encoder(self):
        return self.str_model.encoder
    
    def _build_model(self):
        self.diffusion = TrajDiffusion(self.cfg)
        
        if self.config.learnable_std_mean:
            self.mean = nn.Parameter(torch.zeros(1))
            self.std = nn.Parameter(torch.ones(1))
        else:
            self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        
        if self.learnable_init_traj:
            self.init_x = nn.Parameter(torch.randn(list([self.trajectory_dim*self.frame_stack])), requires_grad=True)
    
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
        
        # Phase2:convey the result to the diffusion model
        label, trajectory_prior, maps_info, init_traj = self._preprocess_batch(trajectory_prior=trajectory_prior, maps_info=maps_info, label=trajectory_label, is_train=True)
        n_frames, batch_size, _, *_ = label.shape
        
        final_predict = []
        loss = []
        transition_info = init_traj
        # into the diffusion model
        # label: (8, b, 40)
        # trajectory_prior: (8, b, 40)
        # maps_info: (8, b, 788, 512)
        # init_traj: (b, 40)
        for t in range(0, n_frames):
            l, prediction= self.diffusion(
                label[t], transition_info, trajectory_prior[t], maps_info[t]   
            ) # (b, 40)
            transition_info = prediction
            loss.append(l)
            final_predict.append(prediction)
            
        # Notice the final_predict is not the final result, it is the intermediate result
        final_predict = torch.stack(final_predict)
        final_predict = rearrange(final_predict, "t b (fs c) ... -> b (t fs) c ...", fs=self.frame_stack)
        final_predict = self._unnormalize_x(final_predict)
        
        loss = torch.stack(loss)
        x_loss = loss.mean()

        pred_dict = {
            "traj_logits": final_predict
        }
        
        output_dict = LTMOutput(
            loss=x_loss,
            loss_items=loss,
            logits=final_predict,
            pred_dict=pred_dict,
            trajectory_label=trajectory_label
        )
        return output_dict

    def _preprocess_batch(self, trajectory_prior, maps_info, label= None, is_train=True):
        # trajectory_prior: (b, 80, 4)
        batch_size, n_frames = trajectory_prior.shape[:2]
        
        if not is_train:
            label = torch.randn(batch_size, n_frames, self.trajectory_dim, device=trajectory_prior[0].device)
        
        # check if number of frames is divisible by frame stack size
        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        n_frames = n_frames // self.frame_stack

        trajectory_prior = rearrange(trajectory_prior, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        
        if self.map_cond:
            maps_info = rearrange(maps_info, "b t d -> 1 b t d").repeat(n_frames, 1, 1, 1)

        label = self._normalize_x(label)
        label = rearrange(label, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()
        
        if self.learnable_init_traj:
            init_traj = self.init_x[None].expand(batch_size, *self.init_x.shape)

        return label, trajectory_prior, maps_info, init_traj
    
    @torch.no_grad()
    def generate(self, **kwargs)-> torch.FloatTensor:

        
        # Phase1: get the result from the STR model
        str_result = self.str_model.generate(**kwargs)
        trajectory_prior = str_result["traj_logits"]
        maps_info = str_result["maps_info"]
        
        
        # Phase2:convey the result to the diffusion model
        label, trajectory_prior, maps_info, init_traj = self._preprocess_batch(trajectory_prior=trajectory_prior, maps_info=maps_info, is_train=False)
        n_frames, batch_size, _, *_ = label.shape   
        
        final_predict = []
        transition_info = init_traj
        
        # prediction
        for t in range(0, n_frames):
            prediction= self.diffusion.generate(
                label[t], transition_info, trajectory_prior[t], maps_info[t]
            ) # torch.Size([8, 1, 40])
            transition_info = prediction
            final_predict.append(prediction)
        final_predict = torch.stack(final_predict)
        
        # unnormalize after rearrange
        final_predict = rearrange(final_predict, "t b (fs c) ... -> b (t fs) c ...", fs=self.frame_stack)
        final_predict = self._unnormalize_x(final_predict)
        pred_dict = {
            "traj_logits": final_predict.to(dtype=torch.float)
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
