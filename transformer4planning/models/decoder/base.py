import torch
from torch import nn
from typing import Dict
from transformer4planning.libs.mlp import DecoderResCat
from torch.nn import CrossEntropyLoss, MSELoss
    
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
        else:
            print(self.model_args.loss_fn)
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
                aug_current = 1 - info_dict['aug_current']
                # expand aug_current to match traj_logits shape
                aug_current = aug_current.unsqueeze(-1).unsqueeze(-1).expand_as(traj_logits)
                # set traj_logits equal to label where aug_current is 0
                traj_logits = traj_logits * aug_current + label.to(device) * (1 - aug_current) if self.model_args.predict_yaw else \
                    traj_logits[..., :2] * aug_current + label[..., :2].to(device) * (1 - aug_current)

                traj_loss = self.loss_fct(traj_logits, label.to(device)) if self.model_args.predict_yaw else \
                            self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
            traj_loss *= self.model_args.trajectory_loss_rescale
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
    def __init__(self, model_args, config, proposal_num=64):
        super().__init__()
        self.model_args = model_args
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
        pred_proposal_embed = hidden_output[:, context_length-1:context_length-1+self.model_args.proposal_length, :] # (bs, proposal_length, n_embed)

        pred_proposal_cls = self.proposal_cls_decoder(pred_proposal_embed) # (bs, proposal_length, 64)
        pred_proposal_logits = self.proposal_logits_decoder(pred_proposal_embed)

        loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(torch.float64), gt_proposal_cls.reshape(-1).long())
        loss_proposal = (loss_proposal * gt_proposal_mask.view(-1)).sum()/ (gt_proposal_mask.sum()+1e-7)
        
        bs = gt_proposal_cls.shape[0]
        pred_proposal_logits = pred_proposal_logits.view(bs, self.proposal_num, 2)
        
        pred_pos_proposal_logits = pred_proposal_logits[torch.arange(bs), gt_proposal_cls, :] # (bs, 2)            
        loss_proposal_logits = self.logits_proposal_loss(pred_pos_proposal_logits, gt_proposal_logits)
        loss_proposal_logits = (loss_proposal_logits * gt_proposal_mask).sum() / (gt_proposal_mask.sum() + 1e-7)
        
        return loss_proposal, loss_proposal_logits

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
        else:
            print(self.model_args.loss_fn)
            assert False, "loss fn not supported"

    def compute_keypoint_loss(self,
                              hidden_output,
                              info_dict: Dict = None,
                              device=None):
        """
        pred the next key point conditioned on all the previous points are ground truth, and then compute the correspond loss
        param:
            hidden_output: the whole hidden_state output from transformer backbone
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        if device is None:
            device = hidden_output.device

        context_length = info_dict.get("context_length", None)

        future_key_points = info_dict["future_key_points"]
        
        kp_start_index = context_length + self.model_args.proposal_length - 1
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
