import torch
from torch import nn
from typing import Dict
from transformer4planning.libs.mlp import DecoderResCat
from torch.nn import CrossEntropyLoss, MSELoss
import os


def mean_circular_error(y_pred, y_true):
    """
    Calculate Mean Circular Error for predicted and true angles.

    Args:
    y_pred (torch.Tensor): Predicted angles (in radians).
    y_true (torch.Tensor): True angles (in radians).

    Returns:
    torch.Tensor: Mean circular error.
    """
    # Ensure the angles are in the range -pi to pi
    y_pred = torch.atan2(torch.sin(y_pred), torch.cos(y_pred))
    y_true = torch.atan2(torch.sin(y_true), torch.cos(y_true))

    # Calculate the angular difference
    angular_difference = y_true - y_pred

    # Adjust differences to be in the range -pi to pi for circular continuity
    angular_difference = torch.atan2(torch.sin(angular_difference), torch.cos(angular_difference))

    # Compute the mean squared error of the angular differences
    loss = torch.mean(torch.square(angular_difference))

    return loss


class TrajectoryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        out_features = 4 if self.config.predict_yaw else 2
        self.model = DecoderResCat(config.n_inner, 
                                   config.n_embd, 
                                   out_features=out_features)
        
        if self.config.task == "waymo": loss_reduction = "none"
        else: loss_reduction = "mean"

        if 'mse' in self.config.loss_fn:
            self.loss_fct = nn.MSELoss(reduction=loss_reduction)
        elif 'l1' in self.config.loss_fn:
            self.loss_fct = nn.SmoothL1Loss(reduction=loss_reduction)
        else:
            print(self.config.loss_fn)
            assert False, "loss fn not supported"

    
    def compute_traj_loss(self, 
                          hidden_output,
                          label, 
                          info_dict,
                          device=None):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        pred_length = info_dict.get("pred_length", label.shape[1])
        traj_hidden_state = hidden_output[:, -pred_length-1:-1, :]
        if device is None:
            device = traj_hidden_state.device
        # compute trajectory loss conditioned on gt keypoints
        if not self.config.pred_key_points_only:
            traj_logits = self.model(traj_hidden_state)
            if self.config.task == "waymo":
                trajectory_label_mask = info_dict.get("trajectory_label_mask", None)
                assert trajectory_label_mask is not None, "trajectory_label_mask is None"
                traj_loss = (self.loss_fct(traj_logits[..., :2], label[..., :2].to(device)) * trajectory_label_mask).sum() / (
                    trajectory_label_mask.sum() + 1e-7)
            elif self.config.task == "nuplan":
                aug_current = 1 - info_dict['aug_current']
                # expand aug_current to match traj_logits shape
                aug_current = aug_current.unsqueeze(-1).unsqueeze(-1).expand_as(traj_logits)
                # set traj_logits equal to label where aug_current is 0
                traj_logits = traj_logits * aug_current + label.to(device) * (1 - aug_current) if self.config.predict_yaw else \
                    traj_logits[..., :2] * aug_current + label[..., :2].to(device) * (1 - aug_current)
                if self.config.mean_circular_loss:
                    assert self.config.predict_yaw, "mean_circular_loss only works for yaw prediction"
                    traj_loss = self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
                    traj_loss += mean_circular_error(traj_logits[..., -1], label[..., -1]).to(device)
                else:
                    traj_loss = self.loss_fct(traj_logits, label.to(device)) if self.config.predict_yaw else \
                                self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
            traj_loss *= self.config.trajectory_loss_rescale
        else:
            traj_logits = torch.zeros_like(label[..., :2])
            traj_loss = None
        
        return traj_loss, traj_logits
    
    def generate_trajs(self, hidden_output, info_dict):
        pred_length = info_dict.get("pred_length", 0)
        assert pred_length > 0
        traj_hidden_state = hidden_output[:, -pred_length-1:-1, :]
        traj_logits = self.model(traj_hidden_state)

        return traj_logits
    
class ProposalDecoder(nn.Module):
    def __init__(self, config, proposal_num=64):
        super().__init__()
        self.config = config
        self.proposal_num = proposal_num
        self.proposal_cls_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.proposal_num)
        self.proposal_logits_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=2 * self.proposal_num)

        self.cls_proposal_loss = CrossEntropyLoss(reduction="none")
        self.logits_proposal_loss = MSELoss(reduction="none")

    def compute_proposal_loss(self, 
                          hidden_output,
                          info_dict):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        gt_proposal_cls= info_dict["gt_proposal_cls"]
        gt_proposal_mask = info_dict["gt_proposal_mask"]
        gt_proposal_logits= info_dict["gt_proposal_logits"]

        context_length = info_dict["context_length"]
        pred_proposal_embed = hidden_output[:, context_length-1:context_length-1+1, :] # (bs, 1, n_embed)

        pred_proposal_cls = self.proposal_cls_decoder(pred_proposal_embed) # (bs, 1, 64)
        pred_proposal_logits = self.proposal_logits_decoder(pred_proposal_embed)

        loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(torch.float64), gt_proposal_cls.reshape(-1).long())
        loss_proposal = (loss_proposal * gt_proposal_mask.view(-1)).sum()/ (gt_proposal_mask.sum()+1e-7)
        
        bs = gt_proposal_cls.shape[0]
        pred_proposal_logits = pred_proposal_logits.view(bs, self.proposal_num, 2)
        
        pred_pos_proposal_logits = pred_proposal_logits[torch.arange(bs), gt_proposal_cls, :] # (bs, 2)            
        loss_proposal_logits = self.logits_proposal_loss(pred_pos_proposal_logits, gt_proposal_logits)
        loss_proposal_logits = (loss_proposal_logits * gt_proposal_mask).sum() / (gt_proposal_mask.sum() + 1e-7)
        
        return loss_proposal, loss_proposal_logits


class ProposalDecoderCLS(nn.Module):
    def __init__(self, config, proposal_num=5):
        super().__init__()
        self.config = config
        self.proposal_num = proposal_num
        self.proposal_cls_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.proposal_num)
        self.cls_proposal_loss = CrossEntropyLoss(reduction="none")

    def compute_proposal_loss(self, hidden_output, info_dict):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc.
        """
        assert self.config.use_proposal, 'must set use_proposal to true to compute proposal loss'

        if self.config.autoregressive_proposals:
            if 'intentions' not in info_dict:
                print('WARNING: no intentions in info_dict')
                return torch.tensor(0.0, device=hidden_output.device), torch.tensor([0] * int(self.config.proposal_num), device=hidden_output.device)
            gt_proposal_cls = info_dict["intentions"]
            context_length = info_dict["context_length"]
            pred_proposal_embed = hidden_output[:, context_length - 1:context_length - 1 + int(self.config.proposal_num), :]  # (bs, 16, n_embed)
            pred_proposal_cls = self.proposal_cls_decoder(pred_proposal_embed)  # (bs, 16, 5)
            loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(torch.float64), gt_proposal_cls.reshape(-1).long())  # (bs * 16)
            return loss_proposal.mean(), pred_proposal_cls
        else:
            if 'halfs_intention' in info_dict:
                gt_proposal_cls = info_dict["halfs_intention"]
            elif 'intentions' in info_dict:
                gt_proposal_cls = info_dict["intentions"][0]
            else:
                print('WARNING: no halfs_intention or intentions in info_dict ', list(info_dict.keys()))
                return torch.tensor(0.0, device=hidden_output.device), torch.tensor(0, device=hidden_output.device)
            context_length = info_dict["context_length"]
            pred_proposal_embed = hidden_output[:, context_length - 1:context_length - 1 + 1, :]  # (bs, 1, n_embed)
            pred_proposal_cls = self.proposal_cls_decoder(pred_proposal_embed)  # (bs, 1, 5)
            loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(torch.float64), gt_proposal_cls.reshape(-1).long())
            return loss_proposal.mean(), pred_proposal_cls

        # if 'halfs_intention' not in info_dict and self.config.use_proposal:
        #     print('WARNING: no halfs_intention in info_dict')
        #     return torch.tensor(0.0, device=hidden_output.device)
        # gt_proposal_cls = info_dict["halfs_intention"]
        # context_length = info_dict["context_length"]
        # pred_proposal_embed = hidden_output[:, context_length - 1:context_length - 1 + 1, :]  # (bs, 1, n_embed)
        #
        # pred_proposal_cls = self.proposal_cls_decoder(pred_proposal_embed)  # (bs, 1, 5)
        # loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(torch.float64), gt_proposal_cls.reshape(-1).long())
        # return loss_proposal.mean(), pred_proposal_cls

class KeyPointMLPDeocder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        out_features = 4 if self.config.predict_yaw else 2
        if 'denoise_kp' in self.config.use_key_points:
            out_features = 2
        self.model = DecoderResCat(config.n_inner,
                                   config.n_embd,
                                   out_features=out_features)
        
        if self.config.task == "waymo": loss_reduction = "none"
        else: loss_reduction = "mean"

        if 'mse' in self.config.loss_fn:
            self.loss_fct = nn.MSELoss(reduction=loss_reduction)
        elif 'l1' in self.config.loss_fn:
            self.loss_fct = nn.SmoothL1Loss(reduction=loss_reduction)
        else:
            print(self.config.loss_fn)
            assert False, "loss fn not supported"
        if self.config.generate_diffusion_dataset_for_key_points_decoder:
            self.save_training_diffusion_feature_dir = os.path.join(self.config.diffusion_feature_save_dir,'train/')
            self.save_testing_diffusion_feature_dir  = os.path.join(self.config.diffusion_feature_save_dir,'val/')
            self.save_test_diffusion_feature_dir = os.path.join(self.config.diffusion_feature_save_dir,'test/')
            if not os.path.exists(self.save_training_diffusion_feature_dir):
                os.makedirs(self.save_training_diffusion_feature_dir)
            if not os.path.exists(self.save_testing_diffusion_feature_dir):
                os.makedirs(self.save_testing_diffusion_feature_dir)
            if not os.path.exists(self.save_test_diffusion_feature_dir):
                os.makedirs(self.save_test_diffusion_feature_dir)
            self.current_idx = 0
            self.gpu_device_count = torch.cuda.device_count()
            # Notice that although we check and create two directories (train/ and test/) here, in the forward method we only save features in eval loops.
            # This is because evaluation is way faster than training (since there are no backward propagation), and after saving features for evaluation, we just change our test set to training set and then run the evaluation loop again.
            # The related code can be found in runner.py at around line 511.
            self.current_idx = 0
            
    def compute_keypoint_loss(self,
                              hidden_output,
                              info_dict: Dict = None,
                              device=None):
        """
        pred the next key point conditioned on all the previous points are ground truth, and then compute the correspond loss
        param:
            hidden_output: the whole hidden_state output from transformer backbone
            info_dict: dict contains additional information, such as context length/input length, pred length, etc.
        """
        if device is None:
            device = hidden_output.device

        context_length = info_dict.get("context_length", None)
        future_key_points = info_dict["future_key_points"]
        
        kp_start_index = context_length - 1
        if self.config.use_proposal:
            if self.config.autoregressive_proposals:
                kp_start_index += int(self.config.proposal_num)
            else:
                kp_start_index += 1
        future_key_points_hidden_state = hidden_output[:, kp_start_index:kp_start_index + future_key_points.shape[1], :]
        key_points_logits = self.model(future_key_points_hidden_state)  # b, s, 4/2*k

        if self.config.task == "nuplan":
            kp_loss = self.loss_fct(key_points_logits, future_key_points.to(device)) if self.config.predict_yaw else \
                        self.loss_fct(key_points_logits[..., :2], future_key_points[..., :2].to(device))
        elif self.config.task == "waymo":
            kp_mask = info_dict["key_points_mask"]
            assert kp_mask is not None, "key_points_mask is None"
            kp_loss = (self.loss_fct(key_points_logits[..., :2], future_key_points[..., :2].to(device)) * kp_mask).sum() / (
                kp_mask.sum() + 1e-7)
        else:
            raise NotImplementedError
        
        return kp_loss, key_points_logits

    def generate_keypoints(self, hidden_state, info_dict:Dict=None):
        batch_size = hidden_state.shape[0]
        key_points_logit = self.model(hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2*k
        return key_points_logit, None
    
    def save_features(self,input_embeds,context_length,info_dict,future_key_points,transformer_outputs_hidden_state):
        # print("hidden_state shape: ",transformer_outputs_hidden_state.shape)
        current_device_idx = int(str(input_embeds.device)[-1])
        context_length = info_dict.get("context_length", None)
        assert context_length is not None, "context length can not be None"
        if context_length is None: # pdm encoder
            input_length = info_dict.get("input_length", None)
        future_key_points = info_dict["future_key_points"]
        key_points_num = future_key_points.shape[-2]

        # hidden state to predict future kp is different from mlp decoder
        kp_end_index = context_length
        if self.config.use_proposal:
            kp_end_index += 1
        # print("kp_end_index: ",kp_end_index)
        save_id = (self.gpu_device_count * self.current_idx + current_device_idx)*key_points_num
        for key_point_idx in range(key_points_num):
            current_save_id = save_id + key_point_idx
            torch.save(transformer_outputs_hidden_state[:,kp_end_index-1+key_point_idx:kp_end_index-1+key_point_idx+1,:].detach().cpu(), os.path.join(self.save_testing_diffusion_feature_dir, f'future_key_points_hidden_state_{current_save_id}.pth'), )
            torch.save(info_dict['future_key_points'][...,key_point_idx:key_point_idx+1,:].detach().cpu(), os.path.join(self.save_testing_diffusion_feature_dir, f'future_key_points_{current_save_id}.pth'), )
        self.current_idx += 1