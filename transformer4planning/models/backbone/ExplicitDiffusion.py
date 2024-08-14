###################################
# using the diffusion from liderun#
###################################
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
from transformer4planning.models.diffusion_loss.TrajRefiner import TrajectoryRefiner
from transformer4planning.models.diffusion_loss.mlp import SimpleMLP
from transformers.modeling_utils import PreTrainedModel

@dataclass
class LTMOutput(CausalLMOutputWithCrossAttentions):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    trajectory_label: torch.FloatTensor = None
    loss_items: Optional[torch.FloatTensor] = None
    pred_dict: Optional[Dict] = None
    


class ExplicitDiffusion(PreTrainedModel):
    def __init__(self, config):
        from transformer4planning.utils.common_utils import load_config
        print("=================DiffusionRefiner===================")
        print("loading the diffusion config...")
        cfg=load_config(config.diffusion_config)
        
        ###########config the attribute#############
        self.cfg = cfg
        self.trajectory_dim = cfg.trajectory_dim
        self.frame_stack = cfg.frame_stack
        self.learnable_init_traj = cfg.learnable_init_traj
        self.map_cond = config.map_cond
        self.config = config
        self.cfg.map_cond = self.map_cond
        
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
        self.diffusion = TrajectoryRefiner(self.cfg)
        
        
        if self.learnable_init_traj:
            self.init_x = nn.Parameter(torch.randn(list([self.frame_stack, self.trajectory_dim])), requires_grad=True)
        else:
            self.init_x = torch.zeros(list([self.trajectory_dim*self.frame_stack]))
    
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
        # label: (8, b, 10, 4)
        # trajectory_prior: (8, b, 10, 4)
        # maps_info: (8, b, 788, 512)
        # init_traj: (b, 40)
        self.diffusion.train()
        for t in range(0, n_frames):
            l, prediction= self.diffusion(
                label[t], transition_info, trajectory_prior[t], maps_info
            ) # (b, 10, 4)
            transition_info = prediction
            loss.append(l)
            final_predict.append(prediction)
        # Notice the final_predict is not the final result, it is the intermediate result
        final_predict = torch.stack(final_predict)
        final_predict = rearrange(final_predict, "t b fs c ... -> b (t fs) c ...", fs=self.frame_stack)
        
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

        if self.learnable_init_traj:
            init_traj = self.init_x[None].expand(batch_size, *self.init_x.shape)
        else:
            init_traj = self.init_x[None].expand(batch_size, *self.init_x.shape)
        
        trajectory_prior = rearrange(trajectory_prior, "b (t fs) c ... -> t b fs c ...", fs=self.frame_stack)
        label = rearrange(label, "b (t fs) c ... -> t b fs c ...", fs=self.frame_stack)
        

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
        self.diffusion.eval()
        # prediction
        for t in range(0, n_frames):
            prediction= self.diffusion.generate(
                label[t], transition_info, trajectory_prior[t], maps_info
            ) # torch.Size([8, 1, 40])
            transition_info = prediction
            final_predict.append(prediction)
        final_predict = torch.stack(final_predict)
        
        
        # unnormalize after rearrange
        final_predict = rearrange(final_predict, "t b fs c ... -> b (t fs) c ...", fs=self.frame_stack)
        print("final predict:", final_predict[0,:2])
        pred_dict = {
            "traj_logits": final_predict.to(dtype=torch.float)
        }
        return pred_dict


        
