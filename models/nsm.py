from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ExpertWeights(nn.Module):
    def __init__(self, shape):
        super(ExpertWeights, self).__init__()
        """shape"""
        self.weight_shape = shape  # expert_components * out * in
        self.bias_shape = (shape[0], shape[1], 1)  # expert_components * out * 1

        """alpha and beta"""
        self.alpha =  self.initial_alpha()
        self.beta = self.initial_beta()

    """initialize parameters for experts i.e. alpha and beta"""

    def initial_alpha(self):
        shape = self.weight_shape
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = (- alpha_bound - alpha_bound) * torch.rand(shape) + alpha_bound
        return alpha

    def initial_beta(self):
        return torch.zeros(self.bias_shape)

    def get_NNweight(self, controlweights, batch_size, device):
        a = self.alpha.unsqueeze(1).to(device)  # expert_components*out*in   -> expert_components*1*out*in
        a = a.repeat(1, batch_size, 1, 1)  # expert_components*1*out*in -> expert_components*?*out*in
        w = controlweights.unsqueeze(-1).unsqueeze(-1)  # expert_components*?        -> expert_components*?*1*1
        r = w * a  # expert_components*?*1*1 m expert_components*?*out*in
        return torch.sum(r, 0)  # ?*out*in

    def get_NNbias(self, controlweights, batch_size, device):
        b = self.beta.unsqueeze(1).to(device)  # expert_components*out*1   -> expert_components*1*out*1
        b = b.repeat(1, batch_size, 1, 1)  # expert_components*1*out*1 -> expert_components*?*out*1
        w = controlweights.unsqueeze(-1).unsqueeze(-1)  # expert_components*?        -> expert_components*?*1*1
        r = w * b  # expert_components*?*1*1 m expert_components*?*out*1
        return torch.sum(r, 0)  # ?*out*1

    def forward(self, input):
        return input


class ComponentNN(nn.Module):
    def __init__(self, num_experts, dim_layers, activation, dropout_ratio):
        super(ComponentNN, self).__init__()
        """
        Define ComponentNN based on https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_Asia_2019/TensorFlow/NSM/Lib_Expert/ComponentNN.py
        :param num_experts: number of experts
        :param dim_layers: dimension of each layer including the dimension of input and output
        :param activation: activation function of each layer
        :param dropout_ratio: for drop out
        """
        """dropout"""
        self.dropout_ratio = dropout_ratio

        """"NN structure"""
        self.num_experts = num_experts
        self.dim_layers = dim_layers
        self.num_layers = len(dim_layers) - 1
        self.activation = activation

        """Build NN"""
        self.experts = self.initExperts()

    def initExperts(self):
        experts = []
        for i in range(self.num_layers):
            expert = ExpertWeights((self.num_experts, self.dim_layers[i + 1], self.dim_layers[i]))
            experts.append(expert)
        return experts

    def forward(self, input, weight_blend):
        # TODO(cyrus.huang): Add FiLM back.
        batch_size = weight_blend.shape[1]
        device = weight_blend.device
        H = input.unsqueeze(-1)  # ?*in -> ?*in*1
        H = torch.nn.Dropout(self.dropout_ratio)(H)
        for i in range(self.num_layers - 1):
            w = self.experts[i].get_NNweight(weight_blend, batch_size, device)
            b = self.experts[i].get_NNbias(weight_blend, batch_size, device)

            H = torch.matmul(w, H) + b  # ?*out*in mul ?*in*1 + ?*out*1 = ?*out*1

            acti = self.activation[i]
            if acti != 0:
                if acti == torch.nn.functional.softmax:
                    H = acti(H, axis=1)
                else:
                    H = acti(H)
            H = torch.nn.Dropout(self.dropout_ratio)(H)

        w = self.experts[self.num_layers - 1].get_NNweight(weight_blend, batch_size, device)
        b = self.experts[self.num_layers - 1].get_NNbias(weight_blend, batch_size, device)
        H = torch.matmul(w, H) + b
        H = torch.squeeze(H, -1)  # ?*out*1 ->?*out
        acti = self.activation[self.num_layers - 1]
        if acti != 0:
            if acti == torch.nn.functional.softmax:
                H = acti(H, dim=1)
            else:
                H = acti(H)
        return H


class NSMDecoder(nn.Module):

    def __init__(self, n_embed):
        super(NSMDecoder, self).__init__()
        hidden_size = n_embed

        self.expert_components = [1, 10]
        self.act_components = [
            [nn.ReLU(), nn.ReLU(), torch.nn.functional.softmax],
            [nn.ReLU(), nn.ReLU(), 0]
        ]
        start_gating = 64
        self.dim_gating = 64
        self.input_components = [
            np.arange(start_gating, start_gating + self.dim_gating),
            []
        ]
        self.dim_components = [
            [hidden_size, hidden_size],
            [hidden_size, hidden_size]
        ]
        num_components = len(self.expert_components)

        # Define gating network.
        self.comp_first = ComponentNN(self.expert_components[0],
                                 [len(self.input_components[0])] + self.dim_components[0] + [self.expert_components[1]],
                                 self.act_components[0],
                                 0.1)
        # Define motion network.
        self.comp_final = ComponentNN(self.expert_components[num_components - 1],
                                 [hidden_size] + self.dim_components[-1] + [hidden_size],
                                 self.act_components[num_components - 1], 0.1)


    def forward(self, hidden_states: Tensor, gating_input: Tensor):
        """
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        weight_blend = torch.ones((self.expert_components[0], batch_size)).to(device)
        # TODO(cyrushx): Add gating input.
        gating_input = torch.ones((batch_size, self.dim_gating)).to(device)
        weight_blend = self.comp_first(gating_input, weight_blend).T
        output_embedding = self.comp_final(hidden_states, weight_blend)
        # TODO(cyrushx): Generate trajectories from output embedding.s
        return output_embedding