import torch
from torch import nn
from typing import Dict
from transformer4planning.libs.mlp import DecoderResCat
    
class TrajectoryDecoder(nn.Module):
    def __init__(self, model_args, config):
        super().__init__()
        self.model_args = model_args
        out_features = 4 if self.model_args.predict_yaw else 2
        self.model = DecoderResCat(config.n_inner, 
                                   config.n_embd, 
                                   out_features=out_features)
        
        if 'mse' in self.model_args.loss_fn:
            self.loss_fct = nn.MSELoss(reduction="mean")
        elif 'l1' in self.model_args.loss_fn:
            self.loss_fct = nn.SmoothL1Loss()
    
    def compute_traj_loss(self, 
                          hidden_output,
                          label, 
                          info_dict,
                          device=None):
        pred_length = info_dict.get("pred_length", label.shape[1])
        traj_hidden_state = hidden_output[:, -pred_length -1:-1, :]
        if device is None:
            device = traj_hidden_state.device
        # compute trajectory loss conditioned on gt keypoints
        if not self.model_args.pred_key_points_only:
            traj_logits = self.model(traj_hidden_state)
            if self.model_args.task == "waymo":
                trajectory_label_mask = info_dict.get("trajectory_label_mask", None)
                assert trajectory_label_mask is not None, "trajectory_label_mask is None"
                traj_loss = (self.loss_fct(traj_logits[..., :2], label[..., :2].to(device)) * trajectory_label_mask).sum() / (
                    trajectory_label_mask.sum() + 1e-7)
            elif self.model_args.task == "nuplan":
                traj_loss = self.loss_fct(traj_logits, label.to(device)) if self.model_args.predict_yaw else \
                            self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))   
            # Modification here: both waymo&nuplan use the same loss scale and loss fn
            traj_loss *= self.model_args.trajectory_loss_rescale
        else:
            traj_logits = torch.zeros_like(label[..., :2])
            traj_loss = None
        
        return traj_loss, traj_logits

class KeyPointMLPDeocder(nn.Module):
    def __init__(self, model_args, config):
        super().__init__()
        self.model_args = model_args
        self.k = model_args.k
        out_features = 4 if self.model_args.predict_yaw else 2
        self.model = DecoderResCat(config.n_inner, 
                                    config.n_embd, 
                                    out_features=out_features * self.k)
        if 'mse' in self.model_args.loss_fn:
            self.loss_fct = nn.MSELoss(reduction="mean")
        elif 'l1' in self.model_args.loss_fn:
            self.loss_fct = nn.SmoothL1Loss()
    
    def compute_keypoint_loss(self,
                        hidden_output,
                        info_dict:Dict=None,
                        device=None
                        ):
        """
        
        """
        if device is None:
            device = hidden_output.device
        context_length = info_dict.get("context_length", None)
        if context_length is None: # pdm encoder
           input_length = info_dict.get("input_length", None)

        future_key_points = info_dict["future_key_points"]
        scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
        kp_start_index = scenario_type_len + context_length * 2 - 1 if context_length is not None \
            else scenario_type_len + input_length
        future_key_points_hidden_state = hidden_output[:, kp_start_index:kp_start_index + future_key_points.shape[1], :]
        key_points_logits = self.model(future_key_points_hidden_state)  # b, s, 4/2*k
        
        if self.k == 1:
            assert self.model_args.task == "nuplan", "k=1 case only support nuplan task"
            kp_loss = self.loss_fct(key_points_logits, future_key_points.to(device)) if self.model_args.predict_yaw else \
                            self.loss_fct(key_points_logits[..., :2], future_key_points[..., :2].to(device))

        else:
            assert self.model_args.task == "waymo", "k>1 case only support waymo task"
            b, s, _ = future_key_points.shape
            k_result = key_points_logits.rehsape(b, s, self.k, -1)

            # get loss of minimal loss from k results
            k_future_key_points = future_key_points.unsqueeze(2).repeat(1, 1, self.k, 1).reshape(b, s, self.k, -1)
            kp_loss_candidate = self.loss_fct(k_result, k_future_key_points.to(device)) if self.model_args.predict_yaw else \
                        self.loss_fct(k_result[..., :2], k_future_key_points[..., :2].to(device))
        
            kp_loss, min_loss_indices = torch.min(kp_loss_candidate.sum(dim=-1), dim=2)
    
            future_key_points_gt_mask = info_dict["future_key_points_gt_mask"]
            kp_loss = (kp_loss.unsqueeze(-1) * future_key_points_gt_mask).sum() / (future_key_points_gt_mask.sum() + 1e-7)
        
        return kp_loss, key_points_logits

    def generate_keypoints(self, hidden_state, info_dict:Dict=None):
        batch_size = hidden_state.shape[0]
        key_points_logit = self.model(hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2*k
        return key_points_logit, None
