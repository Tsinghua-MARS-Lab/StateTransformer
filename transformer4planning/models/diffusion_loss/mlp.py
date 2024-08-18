import torch
import torch.nn as nn
import torch.optim as optim

# 定义MLP模型
class SimpleMLP(nn.Module):
    def __init__(self, cfg):
        super(SimpleMLP, self).__init__()
        # 定义模型的第一层，即输入层到隐藏层
        self.fc1 = nn.Linear(40, 256)
        # 定义模型的第二层，即隐藏层到输出层
        self.fc2 = nn.Linear(256, 40)

    def forward(self, label, transition_info, trajectory_prior, maps_info):
        x = torch.relu(self.fc1(trajectory_prior))
        x = self.fc2(x)
        loss = torch.nn.functional.mse_loss(x, label)
        return loss, x
    
    @torch.no_grad()
    def generate(self, label, transition_info, trajectory_prior, maps_info):
        x = torch.relu(self.fc1(trajectory_prior))
        x = self.fc2(x)
        return x
