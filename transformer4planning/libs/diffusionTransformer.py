import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        input_type = x.dtype
        variace = x.to(torch.float32).pow(2).mean(-1, keepdim = True)
        x = x * torch.rsqrt(variace + self.eps)
        return (x * self.weight).to(input_type)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.TransformerEncoderLayer(d_model=hidden_size, 
                                          nhead=num_heads, 
                                          dim_feedforward=4 * hidden_size, 
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential( 
                                    nn.Linear(hidden_size, mlp_hidden_dim),
                                    nn.GELU(approximate="tanh"),
                                    nn.Linear(mlp_hidden_dim, hidden_size),
                                    nn.GELU(approximate="tanh"),
                                )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), src_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT_Attention(nn.Module):
    def __init__(self, d_model, nhead, k_dim=None, v_dim=None):
        super(DiT_Attention, self).__init__()
        self.nhead = nhead
        self.k_dim = k_dim if k_dim else d_model
        self.v_dim = v_dim if v_dim else d_model
        
        self.Wq_x = nn.Linear(d_model, self.k_dim * nhead, bias=False)
        self.Wk_x = nn.Linear(d_model, self.k_dim * nhead, bias=False)
        self.Wv_x = nn.Linear(d_model, self.v_dim * nhead, bias=False)
        
        self.Wq_c = nn.Linear(d_model, self.k_dim * nhead, bias=False)
        self.Wk_c = nn.Linear(d_model, self.k_dim * nhead, bias=False)
        self.Wv_c = nn.Linear(d_model, self.v_dim * nhead, bias=False)

        self.ResNorm_x_q = RMSNorm(self.k_dim)
        self.ResNorm_x_k = RMSNorm(self.k_dim)
        self.ResNorm_c_q = RMSNorm(self.k_dim)
        self.ResNorm_c_k = RMSNorm(self.k_dim)

        self.proj = nn.Linear(self.v_dim * nhead * 2, d_model)

    def cal_QKV(self, x, c):
        batch_size, seq_len, d_model_x = x.size()

        # q_x = self.ResNorm_x_q(self.Wq_x(x).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3))
        # k_x = self.ResNorm_x_k(self.Wk_x(x).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3))
        q_x = self.Wq_x(x).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3)
        k_x = self.Wk_x(x).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3)
        v_x = self.Wv_x(x).view(batch_size, seq_len, self.nhead, self.v_dim).permute(0, 2, 1, 3)
        
        # q_c = self.ResNorm_c_q(self.Wq_c(c).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3))
        # k_c = self.ResNorm_c_k(self.Wk_c(c).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3))
        q_c = self.Wq_c(c).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3)
        k_c = self.Wk_c(c).view(batch_size, seq_len, self.nhead, self.k_dim).permute(0, 2, 1, 3)
        v_c = self.Wv_c(c).view(batch_size, seq_len, self.nhead, self.v_dim).permute(0, 2, 1, 3)

        q = torch.cat([q_x, q_c], dim=-1)
        k = torch.cat([k_x, k_c], dim=-1)
        v = torch.cat([v_x, v_c], dim=-1)

        return q, k, v

    def forward(self, x, c, attn_mask=None):
        batch_size, seq_len, d_model_x = x.size()

        Q, K, V = self.cal_QKV(x, c)

        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.k_dim**0.5
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool and attn_mask.shape == attn.shape[-2:]
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
            attn = attn + attn_mask
            
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.proj(output)

        return output


class MM_DiTBlock(nn.Module):
    """
    A MM DiT block with conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.norm1_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_x = nn.Sequential( 
            nn.Linear(hidden_size, hidden_size, bias=True), 
            nn.GELU(approximate="tanh")
        )
        self.norm2_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_x = nn.Sequential( 
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.GELU(approximate="tanh"),
        )

        self.attn = DiT_Attention(
            d_model=hidden_size, 
            nhead=num_heads
            )
        
        self.adaLN_modulation_c = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.norm1_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_c = nn.Sequential( 
            nn.Linear(hidden_size, hidden_size, bias=True), 
            nn.GELU(approximate="tanh")
        )
        self.norm2_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_c = nn.Sequential( 
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x, c, y, attn_mask=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_x(y).chunk(6, dim=2)
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation_c(y).chunk(6, dim=2)

        out_attn = self.attn(modulate(self.norm1_x(x), shift_msa_x, scale_msa_x), modulate(self.norm1_c(c), shift_msa_c, scale_msa_c), attn_mask=attn_mask)

        x_attn = x + gate_msa_x * self.linear_x(out_attn)
        x_out = x + gate_mlp_x * self.mlp_x(modulate(self.norm2_x(x_attn), shift_mlp_x, scale_mlp_x))

        c_attn = c + gate_msa_c * self.linear_c(out_attn)
        c_out = c + gate_mlp_c * self.mlp_c(modulate(self.norm2_c(c_attn), shift_mlp_c, scale_mlp_c))

        return x_out, c_out


class DiT_FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x