import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple
from einops import rearrange, reduce

from transformer4planning.models.diffusion_loss.transformer import Transformer
from transformer4planning.models.utils import *

class TrajectoryRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._build_model()
        
    def _build_model(self):
        
        # initailize the diffusion model
        model = Transformer(
            input_dim=self.cfg.trajectory_dim,
            output_dim=self.cfg.trajectory_dim,
            horizon=self.cfg.frame_stack,
            n_cond_steps=self.cfg.n_maps_token,
            cond_dim=self.cfg.maps_dim,
            n_prior_steps=self.cfg.frame_stack,
            prior_dim=self.cfg.trajectory_dim,
            causal_attn=True,
            map_cond=self.cfg.map_cond,
        )
        self.diffusion = DiffusionWrapper(
            model,
            beta_schedule=self.cfg.beta_schedule,
            n_timesteps=self.cfg.diffusion_timesteps,
            predict_range=self.cfg.frame_stack,
        )

    # ========= inference  ============
    @torch.no_grad()
    def generate(self, label, transition_info, trajectory_prior, maps_info):
        """
        label: (bsz, 10, 4)
        transition_info: (bsz, 10, 4)
        trajectory_prior: (bsz, 10, 4)
        maps_info: (bsz, 788, 512)
        """
        diffusion_model = self.diffusion
        batch_size = label.shape[0]
        # concatenate the trajectory prior and the transition info to form the condition
        condition = {
                     "maps_info": maps_info,
                        "transition_info": transition_info,
                        "trajectory_prior": trajectory_prior
                     }
        trajectory = diffusion_model(condition,
                                              batch_size=batch_size,
                                              label=None,
                                              get_inter=self.cfg.visualize_diffusion_intermedians)
        return trajectory
    
    def forward(self, label, transition_info, trajectory_prior, maps_info) -> Dict[str, torch.Tensor]:
        """
        label: (bsz, 10, 4)
        transition_info: (bsz, 10, 4)
        trajectory_prior: (bsz, 10, 4)
        maps_info: (bsz, 788, 512)
        """
        diffusion_model = self.diffusion
        batch_size = label.shape[0]
        # concatenate the trajectory prior and the transition info to form the condition
        condition = {
                     "maps_info": maps_info,
                        "transition_info": transition_info,
                        "trajectory_prior": trajectory_prior
                     }
        traj_logits = torch.zeros_like(label)
        traj_loss = None
        
        traj_loss = diffusion_model(condition, batch_size = batch_size, label = label, info_dict=None)
        
        traj_loss = traj_loss.mean()
        traj_loss *= self.cfg.trajectory_loss_rescale
        return traj_loss, traj_logits
    
class DiffusionWrapper(nn.Module):
    def __init__(self, 
                 model,
                 beta_schedule='linear', 
                 linear_start=0.01,
                 linear_end=0.9,
                 n_timesteps=10,
                 clip_denoised=True,
                 predict_epsilon=True,
                 max_action=100,
                 predict_range=None, 
                 model_args=None
                 ):
        super(DiffusionWrapper, self).__init__()
        self.model = model
        self.model_args = model_args
        self.n_timesteps = int(n_timesteps)
        self.action_dim = model.out_features
        self.predict_range = predict_range
        assert clip_denoised, ''
        assert predict_epsilon, ''
        self.clip_denoised = clip_denoised
        self.max_action = max_action
        self.predict_epsilon = predict_epsilon

        if beta_schedule == "linear": betas = linear_beta_schedule(n_timesteps, beta_start=linear_start, beta_end=linear_end)
        elif beta_schedule == "cosine": betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        print("We are now using diffusion model for decoder.")
        print("When training, the forward method of the diffusion decoder returns loss, and while testing the forward method returns predicted trajectory.")

        self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, hidden_state, batch_size=0, label=None, info_dict=None, get_inter=False, **kwargs):
        if self.training:
            return self.train_forward(hidden_state, label, info_dict=info_dict)
        else:
            assert batch_size > 0
            return self.sample_forward(hidden_state, batch_size, info_dict=info_dict, get_inter=get_inter, **kwargs)

    # ------------------------- Train -------------------------
    def train_forward(self, hidden_state, trajectory_label, info_dict=None):
        trajectory_label = normalize(trajectory_label)
        return self.train_loss(trajectory_label, hidden_state, info_dict=info_dict)

    def train_loss(self, x, state, info_dict=None):
        batch_size=len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device = x.device).long()
        return self.p_losses(x, state, t, info_dict=info_dict)

    def p_losses(self, x_start, state, t, info_dict=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start = x_start, t=t, noise = noise)
        x_recon = self.model(x_noisy, t, state, info_dict=info_dict)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)
        return loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    # ------------------------- Sample -------------------------
    def sample_forward(self, state, batch_size, info_dict=None, get_inter=False, **kwargs):
        assert not self.training, 'sample_forward should not be called directly during training.'
        seq_len = self.predict_range if self.predict_range is not None else state.shape[-2]
        shape = (batch_size, seq_len, self.action_dim)
        action, cls = self.p_sample_loop(state, shape, info_dict=info_dict, get_inter=get_inter, **kwargs)
        action = denormalize(action)
        return action

    def p_sample_loop(self, state, shape, info_dict=None, verbose=False, return_diffusion=False, cal_elbo=False, mc_num=1, determin=True, get_inter=False):
        if get_inter: x_inters = []
        if mc_num == 1:
            device = self.betas.device
            batch_size = shape[0]
            if determin == False:
                x = torch.randn(shape, device=device)
            else:
                x = torch.zeros(shape, device=device)
            total_cls = -(torch.mean(x.detach()**2, dim=1)) # Consider the prior score
            assert not verbose, 'not supported'
            assert not cal_elbo, 'not supported'
            assert not return_diffusion, 'not supported'

            # for i in tqdm(reversed(range(0,self.n_timesteps))):
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, cls = self.p_sample(x, timesteps, state, info_dict=info_dict, determin=determin)
                if get_inter: x_inters.append(x)
                total_cls = total_cls + cls
        else:
            device = self.betas.device
            batch_size = shape[0]
            shape = tuple([batch_size * mc_num] + list(shape[1:]))
            # assert not determin, 'It does not make sense to use deterministic sampling with mc_num > 1'
            x = torch.randn(shape, device=device)

            total_cls = -(torch.mean(x.detach()**2, dim=(1, 2))) # Consider the prior score

            # repeat state in dim 0 for mc_num times.
            # we don't know the shape of state, so we use torch.repeat_interleave
            state = torch.repeat_interleave(state, mc_num, dim=0)
            # then we reshape state into shape()
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size * mc_num,), i, device=device, dtype=torch.long)
                x, cls = self.p_sample(x, timesteps, state, info_dict=info_dict, determin=determin)
                if get_inter: x_inters.append(x.reshape(batch_size, mc_num, *x.shape[1:]))
                total_cls = total_cls + cls
                
            # reshape x into shape (batch_size, mc_num, ...)
            x = x.reshape(batch_size, mc_num, *x.shape[1:])
            total_cls = total_cls.reshape(batch_size, mc_num)
        
        if get_inter: 
            x_inters = torch.stack(x_inters, dim=0)
            return x_inters, total_cls
        
        return x, total_cls
        
    def p_sample(self, x, t, s, info_dict=None, determin=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, info_dict=info_dict)
        if determin == False:
            noise = 1 * torch.randn_like(x)
            # print("2")
            # this is what is printed during sampling
            # sjjjjjj 0313: double the noise
        else:
            noise = torch.zeros(x.shape).to(device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, -(torch.mean(noise.detach()**2, dim=1))
    
    def p_mean_variance(self, x, t, s, info_dict=None):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s, info_dict=info_dict))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
   
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
