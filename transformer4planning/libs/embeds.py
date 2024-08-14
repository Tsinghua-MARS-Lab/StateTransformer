import math
import torch
import torch.nn as nn

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
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CustomPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, downsampling):
        super(CustomPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.downsampling = downsampling
        self.max_len = 11 + 80 // downsampling

        # 生成位置编码
        pe = torch.zeros(self.max_len, embed_dim)
        position = torch.zeros(self.max_len)

        # 前11个位置的编码
        for i in range(11):
            position[i] = i

        # 后16个位置的编码
        sampled_indices = torch.linspace(11, 91, 80 // downsampling).long()
        position[11:] = sampled_indices

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状是 (batch_size, sequence_length, embed_dim)
        x = x + self.pe[:x.size(1), :].unsqueeze(0)  # 添加位置编码
        return x