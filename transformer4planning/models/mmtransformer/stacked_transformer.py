import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TF_utils import (Decoder, DecoderLayer, GeneratorWithParallelHeads626, MultiHeadAttention,
                      PointerwiseFeedforward, PositionalEncoding)


class STF(nn.Module):
    def __init__(self, d_model: int = 128, num_queries: int = 6, future_num_frames: int = 80):
        super(STF, self).__init__()
        "Helper: Construct a model from hyperparameters."

        dec_out_size = future_num_frames*2

        # Hyperparameters predefined
        d_ff = d_model*2
        num_layers = 2
        h = 2
        dropout = 0

        c = copy.deepcopy
        dropout_atten = dropout
        #dropout_atten = 0.1
        attn = MultiHeadAttention(h, d_model, dropout=dropout_atten)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)

        self.pos_emb = PositionalEncoding(d_model, dropout)

        self.cross_attn = Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout), num_layers)

        # self.g = Generator(d_model*2, dec_out_size)
        self.prediction_header = GeneratorWithParallelHeads626(
            d_model, dec_out_size, dropout)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        self.query_embed = nn.Embedding(self.num_queries, d_model)
        self.query_embed.weight.requires_grad == False
        nn.init.orthogonal_(self.query_embed.weight)

    # input: [inp, dec_inp, src_att, trg_att]

    def forward(self, hs):
        '''
            Args:
                hs: torch.Tensor [batch size, num query, hidden size]

            Returns:
                outputs_coord: [batch size, 1, num_query, 30, 2]
                outputs_class: [batch size, 1, num_query]
        '''
        bs, len, h = hs.shape

        query_batches = self.query_embed.weight.view(
            1, *self.query_embed.weight.shape).repeat(bs, 1, 1)

        # Prepare inputs
        pos = self.pos_emb(torch.zeros_like(hs))
        mem = pos+hs

        # decoder
        query_batches = self.cross_attn(query_batches, mem, None, None)

        # Prediction head
        outputs_coord, outputs_class = self.prediction_header(
            query_batches[:, None])

        return outputs_coord, outputs_class
