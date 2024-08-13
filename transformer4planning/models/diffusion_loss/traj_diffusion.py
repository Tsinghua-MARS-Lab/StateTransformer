import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple
from einops import rearrange, reduce

from transformer4planning.models.diffusion_loss.diffusion import DiffusionForTraj
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class TrajDiffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._build_model()
        
    def _build_model(self):
        self.diffusion = DiffusionForTraj(
        input_dim=self.cfg.trajectory_dim*self.cfg.frame_stack,
        output_dim=self.cfg.trajectory_dim*self.cfg.frame_stack,
        horizon=1,
        
        n_cond_steps=self.cfg.n_maps_token,
        cond_dim=self.cfg.maps_dim,
        
        n_prior_steps=1,
        prior_dim=self.cfg.trajectory_dim*self.cfg.frame_stack,
        
        causal_attn=True,
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
        
    # ========= inference  ============
    @torch.no_grad()
    def generate(self, label, transition_info, trajectory_prior, maps_info) -> Dict[str, torch.Tensor]:
        """
        label: (bsz, 40)
        transition_info: (bsz, 40)
        trajectory_prior: (bsz, 40)
        maps_info: (bsz, 788, 512)
        """
        diffusion_model = self.diffusion
        scheduler = self.noise_scheduler

        # unsqueeze to add time dimension
        label = label.unsqueeze(1)
        if len(transition_info.shape) == 2:
           transition_info = transition_info.unsqueeze(1)
        trajectory_prior = trajectory_prior.unsqueeze(1)

        trajectory = torch.randn(
            size=label.shape, 
            dtype=label.dtype,
            device=label.device) # (bsz, 1, 40)

        # set step values
        scheduler.set_timesteps(self.cfg.scheduler.num_inference_steps)
        
        for t in scheduler.timesteps:
            # not convey information during one step: z
            model_output = diffusion_model(trajectory, t, transition_info, trajectory_prior, maps_info) # z_output: (b, 40)
            # model_output: (bsz, 40)
            model_output = model_output.unsqueeze(1)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                ).prev_sample
            
        return trajectory.squeeze(1)

    def forward(
        self,
         label, transition_info, trajectory_prior, maps_info
    ):
        """
        label: (bsz, 40)
        transition_info: (bsz, 40)
        trajectory_prior: (bsz, 40)
        maps_info: (bsz, 788, 512)
        """
        
        label = label.unsqueeze(1)
        if len(transition_info.shape) == 2:
           transition_info = transition_info.unsqueeze(1)
        trajectory_prior = trajectory_prior.unsqueeze(1)
        
        
        noises = torch.randn(label.shape, device=label.device)
        bsz = label.shape[0]
        
        timesteps = torch.randint(
            1,self.noise_scheduler.config.num_train_timesteps,
            (bsz,),device=label.device
        )
        
        noisy_x = self.noise_scheduler.add_noise(label,noises, timesteps)  
        pred_x = self.diffusion(noisy_x, timesteps, transition_info, trajectory_prior, maps_info)
        
        pred_type = self.noise_scheduler.config.prediction_type
        
        if pred_type == 'epsilon':
            target = noises
        elif pred_type == 'sample':
            target = label
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # compute loss
        loss = F.mse_loss(pred_x, target.squeeze(1), reduction='none')
        
        
        return loss, pred_x
        
        
        