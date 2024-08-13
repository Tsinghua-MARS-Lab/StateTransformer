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
        input_dim=self.cfg.x_shape[-1],
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
        self.x_from_z = nn.Linear(self.cfg.z_shape[1], self.cfg.x_shape[0])
        
    
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,z,
            cond=None, prior=None, generator=None,
            # keyword arguments to scheduler.step

            ):
        diffusion_model = self.diffusion
        x_from_z = self.x_from_z
        scheduler = self.noise_scheduler

        # trajectory = torch.randn(
        #     size=condition_data.shape, 
        #     dtype=condition_data.dtype,
        #     device=z.device,
        #     generator=generator)
        trajectory = prior.clone()
    
        # set step values
        scheduler.set_timesteps(self.cfg.scheduler.num_inference_steps)
        z_output = None
        for t in scheduler.timesteps:
            # 1. apply conditioning

            # 2. predict model output
            # condition_data: (b, 10, 4)
            
            z_output = diffusion_model(trajectory, t, z, cond, prior) # z_output: (b, 10, 4)
            model_output = x_from_z(z_output) # model_output: (b, 10, 4)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                ).prev_sample
            

        return z_output, trajectory
    
    @torch.no_grad()
    def generate(self, xs, z, cond, prior) -> Dict[str, torch.Tensor]:
        # handle different ways of passing observation

        # run sampling
        z_next, x_next_pred = self.conditional_sample(
            xs, 
            z=z,
            cond=cond,
            prior=prior,
            )
        

        return z_next, x_next_pred

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor,
        prior: torch.Tensor,
    ):
        """
        z: (bsz, 10, 768)
        x: (bsz, 10, 4)
        cond: (bsz, 10, 256)->(bsz, 10, 768)
        prior: (bsz, 10, 4)->(bsz, 10, 768)
        
        """
        noises = torch.randn(x.shape, device=z.device)
        bsz = z.shape[0]
        
        timesteps = torch.randint(
            1,self.noise_scheduler.config.num_train_timesteps,
            (bsz,),device=x.device
        )
        
        noisy_x = self.noise_scheduler.add_noise(x,noises, timesteps)
        pred_z = self.diffusion(noisy_x, timesteps, z, cond, prior)
        pred_x = self.x_from_z(pred_z)
        
        pred_type = self.noise_scheduler.config.prediction_type
        
        if pred_type == 'epsilon':
            target = noises
        elif pred_type == 'sample':
            target = x
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
    
        return pred_z, pred_x, target
        
        
        