import datasets
import numpy as np
import math
import torch
import torch.nn as nn
from transformers import HfArgumentParser
from transformer4planning.utils import ModelArguments
from transformer4planning.models.model import build_models
from dataclasses import dataclass, field

@dataclass
class TestArguments:
    dataset: str = field(
        default="/media/shiduozhang/My Passport/nuplan/boston_byscenario_autoregressive"
    )

def autoregressive_unittest():
    parser = HfArgumentParser((ModelArguments, TestArguments))

    model_args, test_args = parser.parse_args_into_dataclasses()
    model_args.model_name = "scratch-gpt"
    model_args.task = "nuplan"
    model_args.with_traffic_light = True
    model_args.autoregressive = True
    model_args.recover_obs = True
    model = build_models(model_args)

    def compute_world_points(pred_traj, yaw=0):
        ego_trajectory = np.zeros((1, 3))
        ego_trajectory[-1] = yaw
        next_world_coor_trajectories = list()
        for idx in range(0, pred_traj.shape[0]):
            cos_, sin_ = math.cos(-ego_trajectory[-1][2]), math.sin(-ego_trajectory[-1][2])
            offset_x = pred_traj[idx, 0] * cos_ + pred_traj[idx, 1] * sin_
            offset_y = pred_traj[idx, 1] * cos_ - pred_traj[idx, 0] * sin_
            next_ego_traj = [ego_trajectory[-1][0] + offset_x,
                            ego_trajectory[-1][1] + offset_y,
                            ego_trajectory[-1][2] + pred_traj[idx, -1]]
            ego_trajectory = np.concatenate((ego_trajectory, np.array(next_ego_traj.copy()).reshape(1, -1)), axis=0)
            next_world_coor_trajectories.append(next_ego_traj)

        next_world_coor_trajectories = np.array(next_world_coor_trajectories)
        next_world_coor_points = next_world_coor_trajectories[::2]
        next_world_coor_x = next_world_coor_trajectories[:,0]
        next_world_coor_y = next_world_coor_trajectories[:,1]
        return np.array([next_world_coor_x - yaw, next_world_coor_y - yaw]).transpose(1, 0)
    
    dataset = datasets.load_from_disk(test_args.dataset)
    # dataset = datasets.load_from_disk("/localdata_ssd/nuplan/boston_byscenario_autoregressive_interval100")
    for i in range(100):
        example = dataset[i]

        model.eval()
        model.clf_metrics = dict()
        output = model(
            trajectory = example['trajectory'].unsqueeze(0),
            high_res_raster=example['high_res_raster'].unsqueeze(0),
            low_res_raster=example['low_res_raster'].unsqueeze(0),
            return_dict=True,
        )
        pred_label = output.logits
        pred_traj = model.generate(
            trajectory = example['trajectory'].unsqueeze(0),
            high_res_raster=example['high_res_raster'].unsqueeze(0),
            low_res_raster=example['low_res_raster'].unsqueeze(0),
            return_dict=True,
            past_length=11,
            seq_length=40
        )
        # CE loss compute
        labels = model.tokenize(example["trajectory"])
        ce_loss_fn = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fn(pred_label.squeeze(0), labels.long())

        # MSE loss compute
        mse_loss_fn = nn.MSELoss()
        gt_traj = compute_world_points(example['trajectory'][11:].detach().cpu().numpy())
        pred_traj = compute_world_points(pred_traj.squeeze(0).detach().cpu().numpy())
        mse_loss = mse_loss_fn(torch.tensor(pred_traj), torch.tensor(gt_traj))

        print('MSE LOSS:', mse_loss.item(), "\nCE LOSS:", ce_loss.item())


if __name__ == "__main__":
    autoregressive_unittest()
