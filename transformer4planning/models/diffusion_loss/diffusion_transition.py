import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple
from einops import rearrange, reduce

from transformer4planning.models.diffusion_loss.diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionTransitionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._build_model()
    def _build_model(self):
        self.diffusion = TransformerForDiffusion(
        input_dim=self.cfg.z_shape[-1],
        output_dim=self.cfg.z_shape[-1],
        horizon=self.cfg.frame_stack,
        n_cond_steps=self.cfg.n_maps_token,
        cond_dim=self.cfg.cond_dim,
        n_prior_steps=self.cfg.frame_stack,
        prior_dim=self.cfg.prior_dim,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps  = self.cfg.scheduler.num_train_timesteps,
            beta_start=self.cfg.scheduler.beta_start,
            beta_end=self.cfg.scheduler.beta_end,
            beta_schedule=self.cfg.scheduler.beta_schedule,
            variance_type=self.cfg.scheduler.variance_type,
            clip_sample=self.cfg.scheduler.clip_sample,
            prediction_type=self.cfg.scheduler.prediction_type,
        )
        self.x_from_z = nn.Linear(self.cfg.z_shape[0]*self.cfg.z_shape[1], self.cfg.x_shape[0]*self.cfg.frame_stack)
        
    
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, prior=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        diffusion_model = self.diffusion
        x_from_z = self.x_from_z
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        z_output = None
        for t in scheduler.timesteps:
            # 1. apply conditioning

            # 2. predict model output
            z_output = diffusion_model(trajectory, t, cond, prior)
            z_output = z_output.view(z_output.shape[0], -1)
            model_output = x_from_z(z_output)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            

        return z_output, trajectory
    
    @torch.no_grad()
    def generate(self, z, cond, prior) -> Dict[str, torch.Tensor]:
        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None

        # run sampling
        z_next, x_next_pred = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            prior=prior,
            **self.kwargs)
        

        return z_next, x_next_pred

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        prior: torch.Tensor,
    ):
        noises = torch.randn(z.shape, device=z.device)
        bsz = z.shape[0]
        
        timesteps = torch.randint(
            1,self.noise_scheduler.config.num_train_timesteps,
            (bsz,),device=x.device
        )
        
        noisy_z = self.noise_scheduler.add_noise(z,noises, timesteps) # z: (b, 10, 4)
        pred_z = self.diffusion(noisy_z, timesteps, cond, prior)
        pred_z = pred_z.view(pred_z.shape[0], -1)
        pred_x = self.x_from_z(pred_z)
        
        pred_type = self.noise_scheduler.config.prediction_type
        
        if pred_type == 'epsilon':
            target = noises
        elif pred_type == 'sample':
            target = x
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        pred_x = pred_x.view(z.shape)
        return pred_z, pred_x
        
        
        