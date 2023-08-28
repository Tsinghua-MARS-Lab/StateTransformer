import torch

def angle_to_range(angle: torch.Tensor):
    # change angle to (-pi, pi]
    while (angle>torch.pi).any().item():
        index = (angle>torch.pi)
        angle[index] = angle[index] - 2*torch.pi
    while (angle<=-torch.pi).any().item():
        index = (angle<=-torch.pi)
        angle[index] = angle[index] + 2*torch.pi
    return angle