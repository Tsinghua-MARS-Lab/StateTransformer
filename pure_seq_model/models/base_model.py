import math
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, input_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = input_size
        self.linear = nn.Linear(input_size, out_features)
        self.layer_norm = LayerNorm(out_features)
        # self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, input_states):
        input_states = self.linear(input_states)
        input_states = self.layer_norm(input_states)
        input_states = torch.nn.functional.relu(input_states)
        # input_states = self.lrelu(input_states)
        return input_states

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

    def forward(self, embed_states, attn_mask=None):
        embed_states, atten_weights = self.multi_head_attention(embed_states, embed_states, embed_states, attn_mask=attn_mask)
        return embed_states

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0):
        super(CrossAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        embed_states, atten_weights = self.multi_head_attention(query, key, value, key_padding_mask=key_padding_mask)
        return embed_states
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, hidden_size)
        self.w_2 = nn.Linear(hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, input_dims=3, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(max_len*2.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if input_dims==3:
            pe = pe.unsqueeze(0)
        elif input_dims==4:
            pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[..., :x.size(-2), :].requires_grad_(False)
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, self_attn, feed_forward, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_self_attn = LayerNorm(embed_dim)
        self.norm_pff = LayerNorm(embed_dim)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_pff = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = self.norm_self_attn(x + self.dropout_self_attn(self.self_attn(x, attn_mask=attn_mask)))
        x = self.norm_pff(x + self.dropout_pff(self.feed_forward(x)))
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, self_attn, cross_attn, feed_forward, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward

        self.norm_self_attn = LayerNorm(embed_dim)
        self.norm_cross_attn = LayerNorm(embed_dim)
        self.norm_pff = LayerNorm(embed_dim)
        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.dropout_pff = nn.Dropout(dropout)

    def forward(self, x, latent, key_padding_mask=None):
        x = self.norm_self_attn(x + self.dropout_self_attn(self.self_attn(x)))
        x = self.norm_cross_attn(x + self.dropout_cross_attn(self.cross_attn(x, latent, latent, key_padding_mask=key_padding_mask)))
        x = self.norm_pff(x + self.dropout_pff(self.feed_forward(x)))
        return x
