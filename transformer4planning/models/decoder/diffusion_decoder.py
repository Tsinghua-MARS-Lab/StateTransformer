import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict
from transformer4planning.libs.mlp import DecoderResCat
from transformer4planning.models.utils import *

from transformer4planning.utils.modify_traj_utils import modify_func

class SinusoidalPosEmb(nn.Module):
    """
    Sin positional embedding, where the noisy time step are encoded as an pos embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class BaseDiffusionModel(nn.Module):
    def __init__(self,
                 config,
                 out_features=4,):
        # TODO: use argments to manage model layer dimension.
        super().__init__()
        self.out_features = out_features
        feat_dim = config.n_embd
        self.t_dim = feat_dim
        
        if feat_dim == 512:
            connect_dim = 800
        elif feat_dim == 1024:
            connect_dim = 1600
        else:
            connect_dim = int(feat_dim * 1.5)
            
        self.state_encoder = nn.Sequential(
            DecoderResCat(config.n_inner, config.n_embd, connect_dim),
            DecoderResCat(2 * connect_dim, connect_dim, feat_dim)
        ) 

        self.time_encoder = nn.Sequential( 
            SinusoidalPosEmb(self.t_dim),
            nn.Linear(self.t_dim, self.t_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.t_dim * 2, self.t_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.t_dim * 2, self.t_dim),
        )

        self.x_encoder = nn.Sequential(
            DecoderResCat(64, out_features, 128),
            DecoderResCat(256, 128, 512),
            DecoderResCat(1024, 512, 768),
            DecoderResCat(2048, 768, feat_dim),
        )
        # self.backbone = nn.ModuleList()
        # for i in range(layer_num):
        #     self.backbone.append(nn.TransformerEncoderLayer(d_model=feat_dim, 
        #                                                     nhead=8,
        #                                                     dim_feedforward=4 * feat_dim,
        #                                                     batch_first=True))
        # self.backbone = nn.Sequential(*self.backbone)
        # Use transformer decoder backbone for cross attention between scenarios and trajectories.
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=config.n_head, dim_feedforward=4 * feat_dim, batch_first=True)
        self.self_attn = nn.TransformerEncoder(transformer_encoder_layer, num_layers=config.n_layer)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=config.n_head, dim_feedforward=4 * feat_dim, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(transformer_decoder_layer, num_layers=config.n_layer)
        
        self.x_decoder = nn.Sequential(
            DecoderResCat(2 * feat_dim, feat_dim, 512),
            DecoderResCat(1024, 512, 256),
            DecoderResCat(512, 256, 128),
            DecoderResCat(256, 128, 64),
            DecoderResCat(128, 64, 32),
            DecoderResCat(64, 32, out_features),
        )
    
    def forward(self, x, t, state):
        """
        x: input noise
        t: time step
        state: input hidden state from backbone
        """
        raise NotImplementedError

class TrajDiffusionModel(BaseDiffusionModel):
    def __init__(self,
                config,
                out_features=4,
                predict_range=80):
        super().__init__(config, out_features)
        if 'specified' in config.use_key_points:
            if 'forward' in config.use_key_points:
                position_embedding = torch.flip(exp_positional_embedding(predict_range, config.n_embd).unsqueeze(0), [-2])
            else:
                position_embedding = exp_positional_embedding(predict_range, config.n_embd).unsqueeze(0)
        else:
            position_embedding = uniform_positional_embedding(predict_range, config.n_embd).unsqueeze(0)
            
        # trainable positional embedding, initialized by cos/sin positional embedding.
        self.position_embedding = torch.nn.Parameter(position_embedding, requires_grad=True)
    
    def forward(self, x, t, state):
        # input encoding
        state_embedding = self.state_encoder(state) # B * seq_len * feat_dim
        x_embedding = self.x_encoder(x) # B * seq_len * feat_dim
        t_embedding = self.time_encoder(t) # B * feat_dim
        t_embedding = t_embedding.unsqueeze(1) # -> B * 1 * feat_dim
        
        # concat input embedding
        # seq = torch.cat([state_embedding, x_embedding], dim=-2) # B * (2*seq_len) * feat_dim
        # seq = seq + t_embedding + self.position_embedding # B * (2*seq_len) * feat_dim

        # feature = self.backbone(seq)
        # feature = feature[..., -x.shape[-2]:, :]
        x_embedding = x_embedding + t_embedding + self.position_embedding
        feature = self.self_attn(x_embedding)
        feature = self.cross_attn(feature, state_embedding + self.position_embedding)
        
        output = self.x_decoder(feature)
        return output

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
        assert beta_schedule == 'linear', ''
        self.clip_denoised = clip_denoised
        self.max_action = max_action
        self.predict_epsilon = predict_epsilon

        betas = linear_beta_schedule(n_timesteps, beta_start=linear_start, beta_end=linear_end)
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
    
    def forward(self, hidden_state, batch_size=0, label=None, **kwargs):
        if self.training:
            return self.train_forward(hidden_state, label)
        else:
            assert batch_size > 0
            return self.sample_forward(hidden_state, batch_size, **kwargs)

    # ------------------------- Train -------------------------
    def train_forward(self, hidden_state, trajectory_label):
        trajectory_label = normalize(trajectory_label)
        return self.train_loss(trajectory_label, hidden_state)

    def train_loss(self, x, state):
        batch_size=len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device = x.device).long()
        return self.p_losses(x, state, t)

    def p_losses(self, x_start, state, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start = x_start, t=t, noise = noise)
        x_recon = self.model(x_noisy, t, state)
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
    def sample_forward(self, state, batch_size, **kwargs):
        assert not self.training, 'sample_forward should not be called directly during training.'
        seq_len = self.predict_range if self.predict_range is not None else state.shape[-2]
        shape = (batch_size, seq_len, self.action_dim)
        action, cls = self.p_sample_loop(state, shape, **kwargs)
        action = denormalize(action)
        return action, cls

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False, cal_elbo=False, mc_num=1, determin=True):
        if mc_num == 1:
            device = self.betas.device
            batch_size = shape[0]
            if determin == False:
                x = torch.randn(shape, device=device)
            else:
                x = torch.zeros(shape, device=device)
            total_cls = -(torch.mean(x.detach()**2, dim=1))         # Consider the prior score
            assert not verbose, 'not supported'
            assert not cal_elbo, 'not supported'
            assert not return_diffusion, 'not supported'

            # for i in tqdm(reversed(range(0,self.n_timesteps))):
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, cls = self.p_sample(x, timesteps, state, determin=determin)
                total_cls = total_cls + cls

            return x, total_cls
        else:
            device = self.betas.device
            batch_size = shape[0]
            shape = tuple([batch_size * mc_num] + list(shape[1:]))
            # assert not determin, 'It does not make sense to use deterministic sampling with mc_num > 1'
            x = torch.randn(shape, device=device)

            total_cls = -(torch.mean(x.detach()**2, dim=(1, 2)))         # Consider the prior score

            # repeat state in dim 0 for mc_num times.
            # we don't know the shape of state, so we use torch.repeat_interleave
            state = torch.repeat_interleave(state, mc_num, dim=0)
            # then we reshape state into shape()
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size * mc_num,), i, device=device, dtype=torch.long)
                x, cls = self.p_sample(x, timesteps, state, determin=determin)
                total_cls = total_cls + cls
                
            # reshape x into shape (batch_size, mc_num, ...)
            x = x.reshape(batch_size, mc_num, *x.shape[1:])
            total_cls = total_cls.reshape(batch_size, mc_num)
            return x, total_cls
        
    def p_sample(self, x, t, s, determin=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        if determin == False:
            noise = 1 * torch.randn_like(x)
            # print("2")
            # this is what is printed during sampling
            # sjjjjjj 0313: double the noise
        else:
            noise = torch.zeros(x.shape).to(device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, -(torch.mean(noise.detach()**2, dim=(1, 2)))
    
    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

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

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_features = 4 if config.predict_yaw else 2
        self.k = config.k
        
        if config.future_select == "no":
            self.predict_range = 80
        elif config.future_select == "next_1":
            self.predict_range = 1
        elif config.future_select == "next_10":
            self.predict_range = 10
        else: raise NotImplementedError
            
        diffusion_model = TrajDiffusionModel(config,
                                            out_features=self.out_features,
                                            predict_range=self.predict_range)

        self.model = DiffusionWrapper(diffusion_model, predict_range=self.predict_range)

    def compute_traj_loss(self, 
                          hidden_output,
                          label, 
                          info_dict,
                          device=None):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        pred_length = info_dict.get("pred_length", label.shape[1])
        traj_hidden_state = hidden_output[:, -pred_length-1:-1, :]
        if device is None:
            device = traj_hidden_state.device
        
        if not self.config.predict_yaw: label = label[..., :2].clone()
        
        traj_logits = torch.zeros_like(label)
        traj_loss = None
        # compute trajectory loss conditioned on gt keypoints
        if not self.config.pred_key_points_only:
            if self.config.task == "waymo" or self.config.task == "interaction" or self.config.task == "simagents":
                trajectory_label_mask = info_dict.get("trajectory_label_mask", None)
                assert trajectory_label_mask is not None, "trajectory_label_mask is None"
                
                traj_loss = self.model.train_forward(traj_hidden_state, label)
                traj_loss = (traj_loss * trajectory_label_mask).sum() / (trajectory_label_mask.sum() + 1e-7)
                
            elif self.config.task == "nuplan":
                raise NotImplementedError
                    
            traj_loss *= self.config.trajectory_loss_rescale
        
        return traj_loss, traj_logits
    
    def generate_trajs(self, hidden_output, info_dict):
        pred_length = info_dict.get("pred_length", 0)
        pred_traj_num = info_dict.get("trajectory_label", None).shape[0]
        assert pred_length > 0 and pred_traj_num > 0
        traj_hidden_state = hidden_output[:, -pred_length-1:-1, :]
        
        if self.k == 1:
            traj_logits, scores = self.model(traj_hidden_state, batch_size=pred_traj_num, determin=True)
        else:
            traj_logits, scores = self.model(traj_hidden_state, batch_size=pred_traj_num, determin=False, mc_num=self.config.mc_num)

            if self.config.mc_num > self.k:
                reg_sigma_cls_dict = modify_func(
                    output = dict(
                        reg = [traj_p for traj_p in traj_logits.detach().unsqueeze(1)],
                        cls = [cls for cls in scores.detach().unsqueeze(1)],
                    ),
                    num_mods_out = self.k,
                    EM_Iter = 25,
                )
                traj_logits = torch.cat(reg_sigma_cls_dict["reg"],dim=0)
                scores = torch.cat(reg_sigma_cls_dict["cls"],dim=0)

        return traj_logits, scores

