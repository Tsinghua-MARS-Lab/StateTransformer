import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import einops

from transformer4planning.models.decoders import DecoderResCat
def normalize(x):
    y = torch.zeros_like(x)
    # mean(x[...,0]) = 9.517, mean(sqrt(x[...,0]**2))=9.517
    y[...,0] += (x[...,0] / 10)
    y[...,0] -= 0
    # mean(x[...,1]) = -0.737, mean(sqrt(x[...,1]**2))=0.783
    y[...,1] += (x[...,1] / 10)
    y[...,1] += 0
    if x.shape[-1]==2:
        return y
    # mean(x[...,2]) = 0, mean(sqrt(x[...,2]**2)) = 0
    y[...,2] = x[...,2] * 10
    # mean(x[...,3]) = 0.086, mean(sqrt(x[...,3]**2))=0.090
    y[...,3] += x[...,3] / 2
    y[...,3] += 0
    return y

def denormalize(y):
    x = torch.zeros_like(y)
    x[...,0] = (y[...,0]) * 10
    x[...,1] = (y[...,1]) * 10
    if y.shape[-1]==2:
        return x
    x[...,2] = y[...,2] / 10
    x[...,3] = y[...,3] * 2
    return x
# def normalize(x):
#     y = torch.zeros_like(x)
#     # mean(x[...,0]) = 9.517, mean(sqrt(x[...,0]**2))=9.517
#     y[...,0] += x[...,0] / 9.517
#     y[...,0] -= 1
#     # mean(x[...,1]) = -0.737, mean(sqrt(x[...,1]**2))=0.783
#     y[...,1] += x[...,1] / 0.783
#     y[...,1] += 0.941
#     if x.shape[-1]==2:
#         return y
#     # mean(x[...,2]) = 0, mean(sqrt(x[...,2]**2)) = 0
#     y[...,2] = x[...,2] * 10
#     # mean(x[...,3]) = 0.086, mean(sqrt(x[...,3]**2))=0.090
#     y[...,3] += x[...,3] / 0.09
#     y[...,3] += 0.958
#     return y

# def denormalize(y):
#     x = torch.zeros_like(y)
#     x[...,0] = (y[...,0] + 1) * 9.517
#     x[...,1] = (y[...,1]-0.941) * 0.783
#     if y.shape[-1]==2:
#         return x
#     x[...,2] = y[...,2] / 10
#     x[...,3] = (y[...,3]-0.958) * 0.09
#     return x

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class SinusoidalPosEmb(nn.Module):
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


class NewDecoderTFBased(nn.Module):
    def __init__(self,hidden_size,in_features,out_features=4,layer=7,feat_dim=512):
        super(NewDecoderTFBased,self).__init__()

        # we first encode cond, time, x, then add them together.
        # Then we use transformer encoder blocks.
        # finally we use decoders to decode the trajectory.

        # cond: B * 40 * 1600 -> B * 40 * 800 -> B * 40 * feat_dim
        self.out_features = out_features
        if feat_dim == 512:
            self.state_encoder1 = DecoderResCat(hidden_size,in_features,800)
            self.state_encoder2 = DecoderResCat(1600,800,feat_dim)
        elif feat_dim == 1024:
            # print("This is suggested and used: feat_dim=1024.")
            self.state_encoder1 = DecoderResCat(hidden_size,in_features,1600)
            self.state_encoder2 = DecoderResCat(3200,1600,feat_dim)
        else:
            self.state_encoder1 = DecoderResCat(hidden_size,in_features,int(feat_dim * 1.5))
            self.state_encoder2 = DecoderResCat(feat_dim*3, int(feat_dim*1.5),feat_dim)

        # time: -> B * feat_dim
        t_dim = feat_dim
        self.time_encoder = nn.Sequential( # tdim = 512
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(t_dim * 2, t_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # x: B * 40 * out_features -> B * 40 * 200 -> B * 40 * feat_dim

        self.x_encoder1 = DecoderResCat(400,out_features,400)
        self.x_encoder2 = DecoderResCat(800,400,feat_dim)

        self.transformerblock_list = nn.ModuleList()
        for i in range(layer):
            self.transformerblock_list.append(nn.TransformerEncoderLayer(d_model=feat_dim,nhead=8,dim_feedforward=4*feat_dim,batch_first=True))

        self.x_decoder = nn.Sequential(
            DecoderResCat(2*feat_dim,feat_dim,128),
            DecoderResCat(256,128,64),
            DecoderResCat(128,64,64),
            DecoderResCat(128,64,32),
            DecoderResCat(64,32,out_features),
        )

    def forward(self,x,t,state):

        # First encode the cond, time, x.
        state_embedding = self.state_encoder2(self.state_encoder1(state)) # B * 40 * feat_dim
        x_embedding = self.x_encoder2(self.x_encoder1(x)) # B * 40 * feat_dim
        time_embedding = self.time_encoder(t) # B * feat_dim.
        # change the shape of time_embedding to B * 1 * feat_dim.
        time_embedding = einops.rearrange(time_embedding,'... d -> ... 1 d')
        seq = state_embedding + x_embedding + time_embedding # B * 40 * feat_dim


        # Then use transformer blocks.
        for block in self.transformerblock_list:
            seq = block(seq)


        # Then decode the trajectory.
        x = self.x_decoder(seq)
        return x

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, targ):
        loss = self._loss(pred, targ)
        return loss.mean()
    def _loss(self, pred, targ):
        return F.mse_loss(pred,targ,reduction='none')

class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None


class DiffusionDecoderTFBased(nn.Module):
    def __init__(self, hidden_size, in_features, out_features = 60,
                 beta_schedule = 'linear', linear_start=0.01,linear_end=0.9,n_timesteps=10,
                 loss_type='l2', clip_denoised=True,predict_epsilon=True,max_action = 100, feat_dim=1024,
                 ):
        # if during sampling mc_num > 1, then we return traj and cls of shape batch_size * mc_num * traj_lenth * traj_point_dim.
        # else the returned shape of traj is batch_size * traj_lenth * traj_point_dim.


        super(DiffusionDecoderTFBased, self).__init__()
        self.model = NewDecoderTFBased(hidden_size, in_features, out_features,feat_dim=feat_dim)
        self.n_timesteps = int(n_timesteps)
        self.action_dim = out_features
        assert clip_denoised, ''
        assert predict_epsilon, ''
        assert beta_schedule == 'linear', ''
        self.clip_denoised = clip_denoised
        self.max_action = max_action
        self.predict_epsilon = predict_epsilon

        betas = linear_beta_schedule(n_timesteps,beta_start=linear_start, beta_end=linear_end)
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

        self.loss_fn = L2Loss()
    def fowrard(self, *args):
        if self.training:
            return self.train_forward(self,*args)
        else:
            return self.sample_forward(self,*args)


    # ------------------------- Train -------------------------
    def train_forward(self,hidden_states,trajectory_label):
        trajectory_label = trajectory_label.float()
        trajectory_label = normalize(trajectory_label)
        # print(trajectory_label.shape)
        # print(trajectory_label)
        # print("xsuifhqwouedcnakjdafeawliu")
        # assert False, 'debugging 2'
        # returns loss
        return self.train_loss(trajectory_label, hidden_states)

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
    def sample_forward(self,state,*args, **kwargs):
        assert not self.training, 'sample_forward should not be called directly during training.'
        # returns predicted trajectory
        batch_size = state.shape[0]
        # print(state.shape)
        # print("s;oadnawo4ijasdaufwuo4fhawejf")
        # assert False, 'debugging'

        shape = (batch_size, state.shape[-2] ,self.action_dim)
        action, cls = self.p_sample_loop(state, shape, *args, **kwargs)
        action = denormalize(action)
        return action, cls

    def p_sample_loop(self,state,shape, verbose=False, return_diffusion=False, cal_elbo=False, mc_num=1, determin=True):
        if mc_num == 1:
            device = self.betas.device
            batch_size = shape[0]
            if determin == False:
                x = torch.randn(shape, device=device)
                # print("1")
                # sjjjjjj 0313 try: double the noise.
            else:
                x = torch.zeros(shape, device=device)
            total_cls = -(torch.mean(x.detach()**2, dim=1))         # Consider the prior score
            assert not verbose, 'not supported'
            assert not cal_elbo, 'not supported'
            assert not return_diffusion, 'not supported'
            progress = Silent()
            # for i in tqdm(reversed(range(0,self.n_timesteps))):
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, cls = self.p_sample(x, timesteps, state, determin=determin)
                total_cls = total_cls + cls
                progress.update({'t': i})

            return x, total_cls
        else:
            device = self.betas.device
            batch_size = shape[0]
            shape = tuple([batch_size * mc_num] + list(shape[1:]))
            assert not determin, 'It does not make sense to use deterministic sampling with mc_num > 1'
            x = torch.randn(shape, device=device)
            total_cls = -(torch.mean(x.detach()**2, dim=1))         # Consider the prior score
            progress = Silent()
            # repeat state in dim 0 for mc_num times.
            # we don't know the shape of state, so we use torch.repeat_interleave
            state = torch.repeat_interleave(state, mc_num, dim=0)
            # then we reshape state into shape()
            for i in reversed(range(0,self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, cls = self.p_sample(x, timesteps, state, determin=determin)
                total_cls = total_cls + cls
                progress.update({'t': i})

            # reshape x into shape (batch_size, mc_num, ...)
            x = x.reshape(batch_size, mc_num, *x.shape[1:])
            total_cls = total_cls.reshape(batch_size, mc_num)
            return x, total_cls

    def p_sample(self,x,t,s,determin=True):
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
        # print("x.shape==",x.shape)
        # print("t.shape==",t.shape)
        # print("s.shape==",s.shape)
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






class DiffusionDecoderTFBasedForKeyPoints(DiffusionDecoderTFBased):
    def __init__(self, hidden_size, in_features, out_features = 2,
                 beta_schedule = 'linear', linear_start=0.01,linear_end=0.9,n_timesteps=10,
                 loss_type='l2', clip_denoised=True,predict_epsilon=True,max_action = 100,feat_dim=1024,num_key_points = 5,input_feature_seq_lenth = 16,
                 ):
        super(DiffusionDecoderTFBasedForKeyPoints, self).__init__(hidden_size, in_features, out_features = out_features,
                 beta_schedule = beta_schedule, linear_start=linear_start,linear_end=linear_end,n_timesteps=n_timesteps,
                 loss_type=loss_type, clip_denoised=clip_denoised,predict_epsilon=predict_epsilon,max_action = max_action,feat_dim=feat_dim,)
        self.input_feature_seq_lenth = input_feature_seq_lenth
        self.num_key_points = num_key_points
        self.model = NewDecoderTFBasedForKeyPoints(hidden_size, in_features, out_features,feat_dim=feat_dim,input_feature_seq_lenth = input_feature_seq_lenth,key_point_num=num_key_points)
    # ------------------------- Sample -------------------------
    def sample_forward(self,state,*args, **kwargs):
        assert not self.training, 'sample_forward should not be called directly during training.'
        # returns predicted trajectory
        batch_size = state.shape[0]
        # print(state.shape)
        # print("s;oadnawo4ijasdaufwuo4fhawejf")
        # assert False, 'debugging'

        shape = (batch_size, self.num_key_points ,self.action_dim)
        action, cls = self.p_sample_loop(state, shape, *args, **kwargs)
        action = denormalize(action)
        return action, cls


def exp_positional_embedding(key_point_num, feat_dim):
    point_num = key_point_num
    # Creating the position tensor where the first position is 2^point_num, the second is 2^{point_num-1}, and so on.
    position = torch.tensor([2 ** (point_num - i - 1) for i in range(point_num)]).unsqueeze(1).float()

    # Create a table of divisors to divide each position. This will create a sequence of values for the divisor.
    div_term = torch.exp(torch.arange(0, feat_dim, 2).float() * (-math.log(100.0) / feat_dim))

    # Generate the positional encodings
    # For even positions use sin, for odd use cos
    pos_embedding = torch.zeros((point_num, feat_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    return pos_embedding



class NewDecoderTFBasedForKeyPoints(nn.Module):
    def __init__(self,hidden_size,in_features,out_features=4,layer=7,feat_dim=1024,input_feature_seq_lenth=16,key_point_num=5):
        super().__init__()
        self.out_features = out_features
        if feat_dim == 512:
            self.state_encoder1 = DecoderResCat(hidden_size,in_features,800)
            self.state_encoder2 = DecoderResCat(1600,800,feat_dim)
        elif feat_dim == 1024:
            # print("This is suggested and used: feat_dim=1024.")
            self.state_encoder1 = DecoderResCat(hidden_size,in_features,1600)
            self.state_encoder2 = DecoderResCat(3200,1600,feat_dim)
        else:
            self.state_encoder1 = DecoderResCat(hidden_size,in_features,int(feat_dim * 1.5))
            self.state_encoder2 = DecoderResCat(feat_dim*3, int(feat_dim*1.5),feat_dim)

        t_dim = feat_dim
        input_feature_seq_lenth = input_feature_seq_lenth

        self.time_encoder = nn.Sequential( # tdim = 512
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(t_dim * 2, t_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )


        self.x_encoder = nn.Sequential(
            DecoderResCat(64, out_features, 128),
            DecoderResCat(256, 128, 512),
            DecoderResCat(1024, 512, 768),
            DecoderResCat(2048, 768, feat_dim),
        )


        self.transformerblock_list = nn.ModuleList()
        for i in range(layer):
            self.transformerblock_list.append(nn.TransformerEncoderLayer(d_model=feat_dim,nhead=8,dim_feedforward=4*feat_dim,batch_first=True))

        self.x_decoder = nn.Sequential(
            DecoderResCat(2*feat_dim,feat_dim,512),
            DecoderResCat(1024,512,256),
            DecoderResCat(512,256,128),
            DecoderResCat(256,128,64),
            DecoderResCat(128,64,32),
            DecoderResCat(64,32,out_features),
        )
        key_points_position_embedding = exp_positional_embedding(key_point_num, feat_dim).unsqueeze(0)
        position_embedding = torch.cat([torch.zeros([1,input_feature_seq_lenth, feat_dim]),key_points_position_embedding],dim=-2)
        self.position_embedding = torch.nn.Parameter(position_embedding, requires_grad=True)

    def forward(self,x,t,state):
        seq_lenth = x.shape[-2]
        # First encode the cond, time, x.
        state_embedding = self.state_encoder2(self.state_encoder1(state)) # B * 40 * feat_dim
        x_embedding = self.x_encoder(x) # B * 40 * feat_dim
        time_embedding = self.time_encoder(t) # B * feat_dim.
        # change the shape of time_embedding to B * 1 * feat_dim.
        time_embedding = einops.rearrange(time_embedding,'... d -> ... 1 d')
        seq = torch.cat([state_embedding, x_embedding],dim = -2)
        seq = seq + time_embedding + self.position_embedding


        for block in self.transformerblock_list:
            seq = block(seq)

        x_features = seq[..., -seq_lenth:, :]
        x = self.x_decoder(x_features)
        return x
