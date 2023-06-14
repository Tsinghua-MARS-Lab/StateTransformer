import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return x


class SimpleTrajectoryDecoder(nn.Module):
    
    def __init__(self, embed_dim:int=256) -> None:
        super().__init__()
        
        self.coord_mlp = MLP(embed_dim,embed_dim,2)
        self.conf_mlp = MLP(embed_dim,embed_dim,1)
        
    def forward(self, hs):
        '''
            Args:
                hs: torch.Tensor [batch size, num query, hidden size]

            Returns:
                outputs_coord: [batch size, 30, 2]
                outputs_conf: [batch size]
        '''
        
        outputs_coord = self.coord_mlp(hs[:,:-1])
        outputs_conf = self.conf_mlp(hs[:,-1]).squeeze(-1)
        outputs_conf = F.softmax(outputs_conf,dim=-1)

        return outputs_coord, outputs_conf
