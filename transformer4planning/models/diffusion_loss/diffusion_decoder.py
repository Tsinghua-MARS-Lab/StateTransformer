import torch
import torch.nn as nn
import numpy as np

from transformer4planning.libs.mlp import DecoderResCat
from transformer4planning.libs.diffusionTransformer import DiTBlock, MM_DiTBlock, DiT_FinalLayer
from transformer4planning.libs.embeds import SinusoidalPosEmb
from transformer4planning.models.utils import *

    
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
            DecoderResCat(connect_dim, config.trajectory_dim, config.n_embd),
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
    
    def forward(self, x, t, state, info_dict=None):
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
        self.config = config
        
        position_embedding = uniform_positional_embedding(predict_range, config.n_embd).unsqueeze(0)
      
        # trainable positional embedding, initialized by cos/sin positional embedding.
        self.position_embedding = torch.nn.Parameter(position_embedding, requires_grad=True)
    
    def forward(self, x, t, state, info_dict=None):
        # input encoding
        state_embedding = self.state_encoder(state) # B * seq_len * feat_dim
        x_embedding = self.x_encoder(x) # B * seq_len * feat_dim
        t_embedding = self.time_encoder(t) # B * feat_dim
        t_embedding = t_embedding.unsqueeze(1) # -> B * 1 * feat_dim
        

        x_embedding = x_embedding + t_embedding + self.position_embedding
        feature = self.self_attn(x_embedding) # the self-attention layer for noisy trajectory
        feature = self.cross_attn(feature, state_embedding + self.position_embedding)
    
        output = self.x_decoder(feature)
        return output


class TransformerDiffusionModel(nn.Module):
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
        if config.diffusion_backbone == 'DiT':
            self.blocks = nn.ModuleList([
                DiTBlock(feat_dim, config.n_head, mlp_ratio=4) for _ in range(config.n_layer)
            ])
            if config.debug_scenario_decoding:
                self.agent_blocks = nn.ModuleList([
                    DiTBlock(feat_dim, config.n_head, mlp_ratio=4) for _ in range(config.n_layer)
                ])
        elif config.diffusion_backbone == 'MM-DiT':
            self.blocks = nn.ModuleList([
                MM_DiTBlock(feat_dim, config.n_head, mlp_ratio=4) for _ in range(config.n_layer)
            ])
            if config.debug_scenario_decoding:
                self.agent_blocks = nn.ModuleList([
                    MM_DiTBlock(feat_dim, config.n_head, mlp_ratio=4) for _ in range(config.n_layer)
                ])
        else:
            raise ValueError('Augment of diffusion backbone is invalid!')
        self.final_layer = DiT_FinalLayer(feat_dim, out_features)
        
    
    def forward(self, x, t, state):
        """
        x: input noise
        t: time step
        state: input hidden state from backbone
        """
        raise NotImplementedError


class Transformer_TrajDiffusionModel(TransformerDiffusionModel):
    def __init__(self,
                config,
                out_features=4,
                predict_range=80):
        super().__init__(config, out_features)
        self.config = config
        if config.debug_scene_level_prediction:
            assert MAX_SCENE_LENGTH is not None
            position_embedding = uniform_positional_embedding(predict_range // MAX_SCENE_LENGTH, config.n_embd).unsqueeze(0)
            if config.debug_scene_permute:
                position_embedding = position_embedding.repeat_interleave(MAX_SCENE_LENGTH, dim=1)
            else:
                position_embedding = position_embedding.repeat(1, MAX_SCENE_LENGTH, 1)
        else:
            position_embedding = uniform_positional_embedding(predict_range, config.n_embd).unsqueeze(0)
      
        if config.debug_use_adapter:
            self.adapter = MotionAdapter(config.n_embd, conditioned=True)
            
        # trainable positional embedding, initialized by cos/sin positional embedding.
        self.position_embedding = torch.nn.Parameter(position_embedding, requires_grad=True)
        self.initialize_weights()
    
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize state embedding table:
        nn.init.normal_(self.state_encoder[0].mlp.linear.weight, std=0.02)
        nn.init.normal_(self.state_encoder[0].fc.weight, std=0.02)
        nn.init.normal_(self.state_encoder[1].mlp.linear.weight, std=0.02)
        nn.init.normal_(self.state_encoder[1].fc.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_encoder[1].weight, std=0.02)
        nn.init.normal_(self.time_encoder[3].weight, std=0.02)
        nn.init.normal_(self.time_encoder[5].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        if self.config.diffusion_backbone == 'DiT':
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if self.config.debug_scenario_decoding:
                for block in self.agent_blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        elif self.config.diffusion_backbone == 'MM-DiT':
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation_x[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_x[-1].bias, 0)
                nn.init.constant_(block.adaLN_modulation_c[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_c[-1].bias, 0)
            if self.config.debug_scenario_decoding:
                for block in self.agent_blocks:
                    nn.init.constant_(block.adaLN_modulation_x[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation_x[-1].bias, 0)
                    nn.init.constant_(block.adaLN_modulation_c[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation_c[-1].bias, 0)
        else:
            raise ValueError('Augment of diffusion backbone is invalid!')

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, state, info_dict=None):
        # input encoding
        state_embedding = self.state_encoder(state) # B * seq_len * feat_dim
        x_embedding = self.x_encoder(x) # B * seq_len * feat_dim
        t_embedding = self.time_encoder(t) # B * feat_dim
        t_embedding = t_embedding.unsqueeze(1) # -> B * 1 * feat_dim
        
        if self.config.diffusion_backbone == 'DiT':
            c = t_embedding + state_embedding
            x = x_embedding + self.position_embedding
            for block in self.blocks:
                x = block(x, c)
                
            output = self.final_layer(x, c)
        elif self.config.diffusion_backbone == 'MM-DiT':
            y = t_embedding + state_embedding
            c = state_embedding
            x = x_embedding + self.position_embedding
            for block in self.blocks:

                x, c = block(x, c, y)

            output = self.final_layer(x, y)
        else:
            raise ValueError('Augment of diffusion backbone is invalid!')
        
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

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_features = 4 if config.predict_yaw else 2
        self.k = config.k
        
        self.predict_range = config.predict_range
        
        if config.diffusion_backbone == 'DDPM':
            diffusion_model = TrajDiffusionModel(config,
                                                out_features=self.out_features,
                                                predict_range=self.predict_range)
        elif 'DiT' in config.diffusion_backbone:
            diffusion_model = Transformer_TrajDiffusionModel(config,
                                                out_features=self.out_features,
                                                predict_range=self.predict_range)
        else:
            raise ValueError('Augment of diffusion backbone is invalid!')
        self.model = DiffusionWrapper(diffusion_model, beta_schedule=config.beta_schedule, n_timesteps=config.diffusion_timesteps, predict_range=self.predict_range)

    def forward(self, label, 
                          hidden_output,
                          info_dict=None,
                          device=None):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        pred_length =  label.shape[1]
        traj_hidden_state = hidden_output
        if device is None:
            device = traj_hidden_state.device
        
        if not self.config.predict_yaw: label = label[..., :2].clone()
        
        traj_logits = torch.zeros_like(label)
        traj_loss = None
        # compute trajectory loss conditioned on gt keypoints

        
        traj_loss = self.model.train_forward(traj_hidden_state, label, info_dict=info_dict)
        traj_loss = traj_loss.sum() 
            

                
        traj_loss *= self.config.trajectory_loss_rescale
        
        return  traj_loss,traj_logits
    
    def generate(self, hidden_output, info_dict=None):
        pred_length = self.config.predict_range
        pred_traj_num = hidden_output.shape[0]
        assert pred_length > 0 and pred_traj_num > 0
        traj_hidden_state = hidden_output
        
        if self.k == 1:
            traj_logits = self.model(traj_hidden_state, info_dict=info_dict, batch_size=pred_traj_num, determin=True, get_inter=self.config.visualize_diffusion_intermedians)
        else:
            traj_logits, scores = self.model(traj_hidden_state, info_dict=info_dict, batch_size=pred_traj_num, determin=False, mc_num=self.config.mc_num, get_inter=self.config.visualize_diffusion_intermedians)

            if self.config.mc_num > self.k:
                if self.config.visualize_diffusion_intermedians:
                    traj_logits_list, scores_list = [], []
                    for traj in traj_logits:
                        traj_k, scores_k = select_k_from_mc(traj, scores, self.k)
                        traj_logits_list.append(traj_k)
                        scores_list.append(scores_k)
                    
                    traj_logits = torch.stack(traj_logits_list, dim=0)
                    scores = torch.stack(scores_list, dim=0)
                else:
                    traj_logits, scores = select_k_from_mc(traj_logits, scores, self.k)

        return traj_logits