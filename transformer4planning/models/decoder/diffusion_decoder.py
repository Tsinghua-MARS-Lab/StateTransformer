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
                 hidden_size,
                 in_features,
                 out_features=4,
                 layer_num=7,
                 feat_dim=1024):
        # TODO: use argments to manage model layer dimension.
        super().__init__()
        self.out_features = out_features
        self.t_dim = feat_dim
        if feat_dim == 512:
            connect_dim = 800
        elif feat_dim == 1024:
            connect_dim = 1600
        else:
            connect_dim = int(feat_dim * 1.5)
            
        self.state_encoder = nn.Sequential(
            DecoderResCat(hidden_size, in_features, connect_dim),
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
        self.backbone = nn.ModuleList()
        for i in range(layer_num):
            self.backbone.append(nn.TransformerEncoderLayer(d_model=feat_dim, 
                                                            nhead=8,
                                                            dim_feedforward=4 * feat_dim,
                                                            batch_first=True))
        self.backbone = nn.Sequential(*self.backbone)
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

class KeypointDiffusionModel(BaseDiffusionModel):
    """
    Keypoints are low-dimension representation.
    """
    def __init__(self,
                hidden_size,
                in_features,
                out_features=4,
                layer_num=7,
                feat_dim=1024,
                input_feature_seq_lenth=40,
                key_point_num=5,
                use_key_points='no',):
                # specified_key_points=True, 
                # forward_specified_key_points=False):
        super().__init__(hidden_size, in_features, out_features, layer_num, feat_dim)
        if 'specified' in use_key_points:
            if 'forward' in use_key_points:
                key_points_position_embedding = torch.flip(exp_positional_embedding(key_point_num, feat_dim).unsqueeze(0), [-2])
            else:
                key_points_position_embedding = exp_positional_embedding(key_point_num, feat_dim).unsqueeze(0)
        else:
            key_points_position_embedding = uniform_positional_embedding(key_point_num, feat_dim).unsqueeze(0)
            
        # trainable positional embedding, initialized by cos/sin positional embedding.
        position_embedding = torch.cat([
            torch.zeros([1,input_feature_seq_lenth, feat_dim]), 
            key_points_position_embedding
        ], dim=-2)
        self.position_embedding = torch.nn.Parameter(position_embedding, requires_grad=True)
    
    def forward(self, x, t, state):
        # input encoding
        state_embedding = self.state_encoder(state) # B * seq_len * feat_dim
        x_embedding = self.x_encoder(x) # B * seq_len * feat_dim
        t_embedding = self.time_encoder(t) # B * feat_dim
        t_embedding = t_embedding.unsqueeze(1) # -> B * 1 * feat_dim
        # concat input embedding
        seq = torch.cat([state_embedding, x_embedding], dim=-2) # B * (2*seq_len) * feat_dim
        seq = seq + t_embedding + self.position_embedding # B * (2*seq_len) * feat_dim

        feature = self.backbone(seq)
        feature = feature[..., -x.shape[-2]:, :]
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
                 num_key_points=None, 
                 model_args=None
                 ):
        super(DiffusionWrapper, self).__init__()
        self.model = model
        self.model_args = model_args
        self.n_timesteps = int(n_timesteps)
        self.action_dim = model.out_features
        self.num_key_points = num_key_points
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

        self.loss_fn = nn.MSELoss(reduction='mean')
    
    def forward(self, hidden_state, label=None, **kwargs):
        if self.training:
            return self.train_forward(hidden_state, label)
        else:
            return self.sample_forward(hidden_state, **kwargs)

    # ------------------------- Train -------------------------
    def train_forward(self, hidden_state, trajectory_label):
        trajectory_label = trajectory_label.float()
        trajectory_label = normalize(trajectory_label)
        return self.train_loss(trajectory_label, hidden_state)

    def train_loss(self, x, state):
        batch_size=len(x)
        t = torch.randint(0,self.n_timesteps,(batch_size,), device = x.device).long()
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
    def sample_forward(self, state, **kwargs):
        assert not self.training, 'sample_forward should not be called directly during training.'
        batch_size = state.shape[0]
        seq_len = self.num_key_points if self.num_key_points is not None else state.shape[-2]
        shape = (batch_size, seq_len,self.action_dim)
        action, cls = self.p_sample_loop(state, shape, **kwargs)
        action = denormalize(action)
        return action, cls

    def p_sample_loop(self,state, shape, verbose=False, return_diffusion=False, cal_elbo=False, mc_num=1, determin=True):
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
            assert not determin, 'It does not make sense to use deterministic sampling with mc_num > 1'
            x = torch.randn(shape, device=device)
            total_cls = -(torch.mean(x.detach()**2, dim=1))         # Consider the prior score
            # repeat state in dim 0 for mc_num times.
            # we don't know the shape of state, so we use torch.repeat_interleave
            state = torch.repeat_interleave(state, mc_num, dim=0)
            # then we reshape state into shape()
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
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
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, -(torch.mean(noise.detach()**2, dim=1))
    
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

class T4PTrainDiffWrapper(DiffusionWrapper):
    def __init__(self, 
                 model,
                 beta_schedule='linear', 
                 linear_start=0.01,
                 linear_end=0.9,
                 n_timesteps=10,
                 clip_denoised=True,
                 predict_epsilon=True,
                 max_action=100,
                 num_key_points=None,
                 model_args=None 
                 ):
        super(T4PTrainDiffWrapper, self).__init__(model=model,
                                                beta_schedule=beta_schedule, 
                                                linear_start=linear_start,
                                                linear_end=linear_end,
                                                n_timesteps=n_timesteps,
                                                clip_denoised=clip_denoised,
                                                predict_epsilon=predict_epsilon,
                                                max_action=max_action,
                                                num_key_points=num_key_points,
                                                model_args=model_args)
    
    def forward(self, hidden_state, label=None, **kwargs):
        value = super().forward(hidden_state, label=label, **kwargs)
        if self.training:
            return dict(loss=value)
        else:
            return dict(logits=value[0], scores=value[1])

class KeyPointDiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_features = 4 if config.predict_yaw else 2
        self.k = config.k
        self.use_key_points = config.use_key_points
        diffusion_model = KeypointDiffusionModel(config.n_inner,
                                                 config.n_embd,
                                                 out_features=self.out_features * self.config.k,
                                                 key_point_num=1,
                                                 feat_dim=self.config.key_points_diffusion_decoder_feat_dim,
                                                 input_feature_seq_lenth=self.config.diffusion_condition_sequence_lenth,
                                                 use_key_points=config.use_key_points,)

        self.model = DiffusionWrapper(diffusion_model, num_key_points=1,)# self.model_args.key_points_num)
        if 'mse' in self.config.loss_fn:
            self.loss_fct = nn.MSELoss(reduction="mean")
        elif 'l1' in self.config.loss_fn:
            self.loss_fct = nn.SmoothL1Loss()
        else:
            raise NotImplementedError


    
    def generate_keypoints(self, 
                            hidden_output,
                            info_dict:Dict=None):
        '''
            Input:
                hidden_output: batch_size *1 * n_embd
                info_dict: Dict
            Output:
                tuple of key_points_logits and scores
                    key_points_logits: batch_size * self.k * num_key_points * 2/4
                    scores: batch_size * self.k. assert torch.sum(scores, dim=1) == 1, ''
                    
        '''
        assert self.use_key_points != 'no'
        assert not self.training
        # hidden state to predict future kp is different from mlp decoder
        # context_length = info_dict.get("context_length", None)
        # if context_length is None: # pdm encoder
        #    input_length = info_dict.get("input_length", None)
        # kp_end_index = scenario_type_len + context_length * 2 if context_length is not None \
        #             else scenario_type_len + input_length
        # future_key_points_hidden_state = hidden_output[:, :kp_end_index, :]
        future_key_points_hidden_state = hidden_output
        if self.k == 1:
            key_points_logits, scores = self.model(future_key_points_hidden_state, determin = True)
        else:
            key_points_logits, scores = self.model(future_key_points_hidden_state, determin = False, mc_num = self.config.mc_num)
            reg_sigma_cls_dict = modify_func(
                output = dict(
                    reg = [traj_p for traj_p in key_points_logits.detach().unsqueeze(1)],
                    cls = [cls for cls in scores.detach().unsqueeze(1)],
                ),
                num_mods_out = self.k,
                EM_Iter = 25,
            )
            key_points_logits = torch.cat(reg_sigma_cls_dict["reg"],dim=0)
            scores = torch.cat(reg_sigma_cls_dict["cls"],dim=0)
        return key_points_logits, scores 

    

