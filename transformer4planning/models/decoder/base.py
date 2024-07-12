import torch
from torch import nn
from typing import Dict
from transformer4planning.libs.mlp import DecoderResCat
from torch.nn import CrossEntropyLoss, MSELoss
import os


def mean_circular_error(y_pred, y_true, mean=True):
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
    loss = torch.square(angular_difference)
    if mean:
        loss = torch.mean(loss)

    return loss


def normalize_angles(angles):
    return torch.atan2(torch.sin(angles), torch.cos(angles))


class TrajectoryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        out_features = 4 if self.config.predict_yaw else 2

        if self.config.trajectory_decoder_type == 'linear':
            if self.config.traj_dropout:
                self.dropout = nn.Dropout(self.config.traj_dropout)
            self.model = nn.Linear(config.n_embd, out_features)
        else:
            self.model = DecoderResCat(config.n_inner,
                                       config.n_embd,
                                       out_features=out_features,
                                       dropout=self.config.traj_dropout)
        
        if self.config.task == "waymo": loss_reduction = "none"
        else: loss_reduction = "mean"

        if 'mse' in self.config.loss_fn:
            self.loss_fct = nn.MSELoss(reduction=loss_reduction)
        elif 'l1' in self.config.loss_fn:
            self.loss_fct = nn.SmoothL1Loss(reduction=loss_reduction)
        else:
            print(self.config.loss_fn)
            assert False, "loss fn not supported"

    def compute_traj_loss_autoregressive(self,
                                         hidden_output,
                                         label,
                                         info_dict,
                                         no_yaw_loss=False,
                                         device=None):
        pred_legth = info_dict.get("pred_length", label.shape[1])
        raster_seq_len = info_dict.get("sequence_length")
        traj_hidden_state = hidden_output[:, -2-(raster_seq_len+1)*(pred_legth-1):-1:(raster_seq_len+1), :]
        assert traj_hidden_state.shape[1] == pred_legth, f"trajectory hidden state shape is {traj_hidden_state.shape} but pred_legth is {pred_legth}"
        if device is None:
            device = traj_hidden_state.device
        # compute trajectory loss
        if self.config.trajectory_decoder_type == 'linear' and self.config.traj_dropout and self.training:
            traj_hidden_state = self.dropout(traj_hidden_state)
        traj_logits = self.model(traj_hidden_state)
        assert self.config.task == "nuplan", "only support nuplan task"
        assert self.config.mean_circular_loss, "only support mean circular loss"
        assert self.config.predict_yaw, "only support yaw prediction"
        traj_loss = self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
        if not no_yaw_loss:
            traj_loss += mean_circular_error(traj_logits[..., 3], label[..., 3]).to(device)
        return traj_loss, traj_logits


    def compute_traj_loss(self, 
                          hidden_output,
                          label, 
                          info_dict,
                          no_yaw_loss=False,
                          device=None):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        pred_length = info_dict.get("pred_length", label.shape[1])

        context_length = info_dict.get("context_length", None)
        selected_indices = info_dict.get("selected_indices", None)

        if context_length is not None:
            if selected_indices is None:
                traj_hidden_state = hidden_output[:, context_length - 1:context_length + pred_length - 1, :]
            else:
                traj_hidden_state = hidden_output[:, context_length - 1 + len(selected_indices):context_length + pred_length - 1 + len(selected_indices), :]
        else:
            traj_hidden_state = hidden_output[:, -pred_length-1:-1, :]
        if device is None:
            device = traj_hidden_state.device
        # compute trajectory loss conditioned on gt keypoints
        if not self.config.pred_key_points_only:
            if self.config.trajectory_decoder_type == 'linear' and self.config.traj_dropout and self.training:
                traj_hidden_state = self.dropout(traj_hidden_state)
            traj_logits = self.model(traj_hidden_state)
            if self.config.task == "waymo":
                trajectory_label_mask = info_dict.get("trajectory_label_mask", None)
                assert trajectory_label_mask is not None, "trajectory_label_mask is None"
                traj_loss = (self.loss_fct(traj_logits[..., :2], label[..., :2].to(device)) * trajectory_label_mask).sum() / (
                    trajectory_label_mask.sum() + 1e-7)
            elif self.config.task == "nuplan":
                # aug_current = 1 - info_dict['aug_current']
                # # expand aug_current to match traj_logits shape
                # aug_current = aug_current.unsqueeze(-1).unsqueeze(-1).expand_as(traj_logits)
                # # set traj_logits equal to label where aug_current is 0
                # traj_logits = traj_logits * aug_current + label.to(device) * (1 - aug_current) if self.config.predict_yaw else \
                #     traj_logits[..., :2] * aug_current + label[..., :2].to(device) * (1 - aug_current)

                if self.config.mean_circular_loss:
                    assert self.config.predict_yaw, "mean_circular_loss only works for yaw prediction"
                    if self.config.trajectory_prediction_mode == 'start_frame':
                        traj_loss = self.loss_fct(traj_logits[..., 0, :2], label[..., 0, :2].to(device)) * self.config.first_frame_scele
                        if not no_yaw_loss:
                            traj_loss += mean_circular_error(traj_logits[..., 0, -1], label[..., 0, -1]).to(device) * self.config.first_frame_scele
                    elif self.config.trajectory_prediction_mode == 'both':
                        assert not no_yaw_loss, "yaw loss must be included"
                        traj_loss = self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
                        traj_loss += mean_circular_error(traj_logits[..., -1], label[..., -1]).to(device)
                        traj_loss += self.loss_fct(traj_logits[..., 0, :2], label[..., 0, :2].to(device)) * self.config.first_frame_scele
                        traj_loss += mean_circular_error(traj_logits[..., 0, -1], label[..., 0, -1]).to(device) * self.config.first_frame_scele
                        traj_loss += mean_circular_error(traj_logits[..., 0, -1], label[..., 0, -1]).to(device) * self.config.first_frame_scele
                    elif self.config.trajectory_prediction_mode == 'rebalance':
                        total_points = traj_logits.shape[1]
                        # rebalance each point loss and then add them together
                        point_losses = []
                        # generate a linearly decreasing weight for each point from 1000 to 1 with length of 80
                        weights = torch.linspace(1000, 1, total_points, device=device)
                        for i in range(total_points):
                            point_loss = self.loss_fct(traj_logits[..., i, :2], label[..., i, :2].to(device))
                            if not no_yaw_loss:
                                point_loss += mean_circular_error(traj_logits[..., i, -1], label[..., i, -1]).to(device)
                            point_losses.append(point_loss * weights[i])
                        # scale each point loss based on the last point loss
                        point_losses = torch.stack(point_losses)
                        traj_loss = point_losses.mean()
                    else:
                        traj_loss = self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
                        if not no_yaw_loss:
                            traj_loss += mean_circular_error(traj_logits[..., -1], label[..., -1]).to(device)
                else:
                    traj_loss = self.loss_fct(traj_logits, label.to(device)) if self.config.predict_yaw else \
                                self.loss_fct(traj_logits[..., :2], label[..., :2].to(device))
            traj_loss *= self.config.trajectory_loss_rescale
        else:
            last_dimension = 4 if self.config.predict_yaw else 2
            traj_logits = torch.zeros_like(label[..., :last_dimension])
            traj_loss = torch.tensor(0, dtype=hidden_output.dtype, device=hidden_output.device)
        
        return traj_loss, traj_logits
    
    def generate_trajs(self, hidden_output, info_dict):
        pred_length = info_dict.get("pred_length", 0)
        assert pred_length > 0
        traj_hidden_state = hidden_output[:, -pred_length-1:-1, :]
        if self.config.trajectory_decoder_type == 'linear' and self.config.traj_dropout and self.training:
            traj_hidden_state = self.dropout(traj_hidden_state)
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

        loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(hidden_output.dtype), gt_proposal_cls.reshape(-1).long())
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
            loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(hidden_output.dtype), gt_proposal_cls.reshape(-1).long())  # (bs * 16)
            return loss_proposal.mean(), pred_proposal_cls
        else:
            if 'intentions' in info_dict:
                gt_proposal_cls = info_dict["intentions"][0]
            else:
                print('WARNING: no intentions in info_dict ', list(info_dict.keys()))
                return torch.tensor(0.0, device=hidden_output.device), torch.tensor(0, device=hidden_output.device)
            context_length = info_dict["context_length"]
            pred_proposal_embed = hidden_output[:, context_length - 1:context_length - 1 + 1, :]  # (bs, 1, n_embed)
            pred_proposal_cls = self.proposal_cls_decoder(pred_proposal_embed)  # (bs, 1, 5)
            loss_proposal = self.cls_proposal_loss(pred_proposal_cls.reshape(-1, self.proposal_num).to(hidden_output.dtype), gt_proposal_cls.reshape(-1).long())
            return loss_proposal.mean(), pred_proposal_cls


class KeyPointDecoderCLS(nn.Module):
    def __init__(self, config, proposal_num):
        super().__init__()
        self.config = config
        self.kp_tokenizer = None
        self.proposal_num = proposal_num
        self.proposal_cls_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.proposal_num, dropout=self.config.kp_dropout)
        self.cls_proposal_loss = CrossEntropyLoss(reduction="none")
        if self.config.add_regression_loss:
            self.regression_loss = MSELoss(reduction="none")

    def compute_keypoint_loss(self, hidden_output, info_dict, i):
        """
        pred future 8-s trajectory and compute loss(l2 loss or smooth l1)
        params:
            hidden_output: whole hidden state output from transformer backbone
            label: ground truth trajectory in future 8-s
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc.
        """
        if 'future_key_points_ids' in info_dict:
            gt_proposal_cls = info_dict["future_key_points_ids"][:, i]
        else:
            assert False, f'no future_key_points_ids in info_dict {list(info_dict.keys())}'
        context_length = info_dict["context_length"]
        pred_proposal_embed = hidden_output[:, context_length - 1 + i:context_length - 1 + 1 + i, :]  # (bs, 1, n_embed)
        pred_proposal_scores = self.proposal_cls_decoder(pred_proposal_embed)  # (bs, 1, n)
        pred_proposal_scores = nn.Softmax(dim=-1)(pred_proposal_scores)
        loss_proposal = self.cls_proposal_loss(pred_proposal_scores.reshape(-1, self.proposal_num).to(hidden_output.dtype), gt_proposal_cls.reshape(-1).long())
        pred_proposal_cls = pred_proposal_scores.argmax(dim=-1).squeeze(-1)
        if self.config.add_regression_loss:
            gt_proposal_logits = info_dict["future_key_points_after"][:, i]
            pred_proposal_logits = self.kp_tokenizer.decode(pred_proposal_cls, dtype=hidden_output.dtype, device=hidden_output.device)
            loss_proposal_logits = self.regression_loss(pred_proposal_logits, gt_proposal_logits)
            final_loss = loss_proposal.mean() + loss_proposal_logits.mean()
            return final_loss, pred_proposal_cls
        else:
            return loss_proposal.mean(), pred_proposal_cls

    def generate_keypoints(self, hidden_state):
        batch_size, seq_len, hidden_dim = hidden_state.shape
        key_point_id_scores = self.proposal_cls_decoder(hidden_state)  # b, s=1, n
        key_point_id_scores = nn.Softmax(dim=-1)(key_point_id_scores)  # b, s=1, n
        key_point_ids = key_point_id_scores.argmax(dim=-1).squeeze(-1)  # b, s=1
        # key_point_logits = self.tokenizer.decode(key_point_id)  # b, s=1, 2
        return key_point_ids, key_point_id_scores


class KeyPointMLPDeocder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.k = int(self.config.k)
        # out_features = 4 if self.config.predict_yaw else 2
        out_features = 2
        if self.config.task == "waymo": loss_reduction = "none"
        else: loss_reduction = "mean"

        if self.k > 1:
            out_features *= self.k
            self.logits_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=out_features, dropout=self.config.kp_dropout)
            self.score_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.k, dropout=self.config.kp_dropout)
            self.cls_loss_fct = CrossEntropyLoss(reduction="none")
            self.softmax = nn.Softmax(dim=-1)
            loss_reduction = "none"
        else:
            self.model = DecoderResCat(config.n_inner, config.n_embd, out_features=out_features, dropout=self.config.kp_dropout)

        if 'mse' in self.config.loss_fn:
            self.loss_fct = nn.MSELoss(reduction=loss_reduction)
        elif 'l1' in self.config.loss_fn:
            self.loss_fct = nn.SmoothL1Loss(reduction=loss_reduction)
        else:
            assert False, f"loss fn not supported {self.config.loss_fn}"
        if self.config.generate_diffusion_dataset_for_key_points_decoder:
            self.save_training_diffusion_feature_dir = os.path.join(self.config.diffusion_feature_save_dir,'train/')
            self.save_testing_diffusion_feature_dir = os.path.join(self.config.diffusion_feature_save_dir,'val/')
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

        if self.k > 1:
            hidden_dim = future_key_points_hidden_state.shape[-1]
            key_points_logits = self.logits_decoder(future_key_points_hidden_state)  # b, s, 4*k
            key_points_scores = self.score_decoder(future_key_points_hidden_state)  # b, s, k
        else:
            key_points_logits = self.model(future_key_points_hidden_state)  # b, s, 4/2*k

        loss_per_kp = None

        if self.config.task == "nuplan":
            # # normalize each selected key point loss
            if self.k > 1:
                batch_size, seq_len, _ = key_points_logits.shape
                key_points_logits = key_points_logits.view(batch_size, seq_len, self.k, -1)
                loss_per_kp = []
                select_logtis = []
                for i in range(future_key_points.shape[1]):
                    # repeat gt k times
                    gt = future_key_points[:, i, :].unsqueeze(1).repeat(1, self.k, 1)
                    # compute loss for k key points
                    loss_kp_i = self.loss_fct(key_points_logits[:, i, :, :2], gt[..., :2].to(device))  # b, k, 2
                    # if self.config.pred_key_point_yaw:
                    #     if self.config.kp_mean_circular_loss:
                    #         loss_kp_yaw = mean_circular_error(key_points_logits[:, i, :, -1], gt[..., -1].to(device), mean=False).unsqueeze(-1)
                    #     else:
                    #         loss_kp_yaw = self.loss_fct(key_points_logits[:, i, :, -1], gt[..., -1].to(device)).unsqueeze(-1)
                    #     loss_kp_i = torch.cat([loss_kp_i, loss_kp_yaw], dim=-1)
                    # sum last dimension
                    loss_kp_i = loss_kp_i.sum(dim=-1)  # b, k
                    # get minimal index of k predictions at last dimension
                    # min_loss_idx = loss_kp_i.argmin(dim=-1)  # b, 1
                    # min_loss = loss_kp_i.min(dim=-1)[0]
                    min_loss, min_loss_idx = loss_kp_i.min(dim=-1)  # b, 1; b, 1
                    # compute cross entropy loss
                    score_hidden = self.softmax(key_points_scores[:, i, :])
                    loss_per_kp_i = self.cls_loss_fct(score_hidden, min_loss_idx)
                    # add loss (regression+classification) for each key point
                    loss_per_kp.append(min_loss.mean() * 0.1**(4-i) + loss_per_kp_i.mean())
                    selected_logits_at_i = []
                    for j in range(batch_size):
                        selected_logits_at_i.append(key_points_logits[j, i, int(min_loss_idx[j]), :].unsqueeze(0))
                    stacked_selected_logits_at_i = torch.stack(selected_logits_at_i, dim=0)  # [batch_size, 1, 4/2]
                    select_logtis.append(stacked_selected_logits_at_i)
                loss_per_kp = torch.stack(loss_per_kp)  # [seq_len]
                # stack the list of selected logits, each has a shape of [batch_size, 1, 4/2]
                select_logtis_all = torch.stack(select_logtis, dim=1)[:, :, 0, :]  # [batch_size, seq_len, 4/2]
                # pack key points logits and scores
                key_points_logits = {
                    "logits": key_points_logits,
                    "scores": key_points_scores,
                    "selected_logits": select_logtis_all,
                }
            else:
                loss_per_kp = []
                for i in range(future_key_points.shape[1]):
                    loss_this_kp = self.loss_fct(key_points_logits[:, i, :2], future_key_points[:, i, :2].to(device))
                    # if self.config.pred_key_point_yaw:
                    #     if self.config.kp_mean_circular_loss:
                    #         loss_this_kp_yaw = mean_circular_error(key_points_logits[:, i, -1], future_key_points[:, i, -1].to(device), mean=False).unsqueeze(-1)
                    #     else:
                    #         loss_this_kp_yaw = self.loss_fct(key_points_logits[:, i, -1], future_key_points[:, i, -1].to(device))
                    #     loss_this_kp = torch.cat([loss_this_kp, loss_this_kp_yaw], dim=-1)
                    loss_this_kp = loss_this_kp.sum(dim=-1)
                    # if i == 0 and self.training:
                    #     loss_per_kp *= 0
                    loss_per_kp.append(loss_this_kp)
                loss_per_kp = torch.stack(loss_per_kp)  # [seq_len]
            kp_loss = loss_per_kp.mean() * self.config.kp_loss_rescale
            # kp_loss = self.loss_fct(key_points_logits, future_key_points.to(device)) if self.config.predict_yaw else \
            #             self.loss_fct(key_points_logits[..., :2], future_key_points[..., :2].to(device)) * self.config.kp_loss_rescale
        elif self.config.task == "waymo":
            kp_mask = info_dict["key_points_mask"]
            assert kp_mask is not None, "key_points_mask is None"
            kp_loss = (self.loss_fct(key_points_logits[..., :2], future_key_points[..., :2].to(device)) * kp_mask).sum() / (
                kp_mask.sum() + 1e-7)
        else:
            raise NotImplementedError

        return kp_loss, key_points_logits, loss_per_kp

    def generate_keypoints(self, hidden_state, info_dict:Dict=None):
        batch_size, seq_len, hidden_dim = hidden_state.shape
        assert hidden_state.shape[1] == 1
        if self.k > 1:
            key_points_logit = self.logits_decoder(hidden_state).view(batch_size, seq_len, self.k, -1)  # b, s, k, 4/2
            key_points_score = self.softmax(self.score_decoder(hidden_state))  # b, s, k
            # if self.config.predict_yaw:
            #     key_points_logit[:, :, :, -1] = normalize_angles(key_points_logit[:, :, :, -1])
            key_points_logit = {
                "logits": key_points_logit,
                "scores": key_points_score,
            }
        else:
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


class KeyPointLinearDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # out_features = 4 if self.config.predict_yaw else 2
        out_features = 2
        loss_reduction = "mean"
        if self.config.kp_dropout:
            self.kp_dropout = nn.Dropout(self.config.kp_dropout)
        self.model = nn.Linear(self.config.n_embd, out_features)
        if 'mse' in self.config.loss_fn:
            self.loss_fct = nn.MSELoss(reduction=loss_reduction)
        elif 'l1' in self.config.loss_fn:
            self.loss_fct = nn.SmoothL1Loss(reduction=loss_reduction)
        else:
            assert False, f"loss fn not supported {self.config.loss_fn}"

    def compute_keypoint_loss(self,
                              hidden_output,
                              info_dict: Dict = None,
                              device=None):
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
        if self.config.kp_dropout and self.training:
            future_key_points_hidden_state = self.kp_dropout(future_key_points_hidden_state)
        key_points_logits = self.model(future_key_points_hidden_state)  # b, s, 2
        kp_loss = self.loss_fct(key_points_logits, future_key_points[..., :2].to(device)).mean()
        return kp_loss, key_points_logits

    def generate_keypoints(self, hidden_state, info_dict: Dict = None):
        batch_size, seq_len, hidden_dim = hidden_state.shape
        assert hidden_state.shape[1] == 1
        if self.config.kp_dropout and self.training:
            hidden_state = self.kp_dropout(hidden_state)
        key_points_logit = self.model(hidden_state).reshape(batch_size, 1, -1)  # b, 1, 2
        return key_points_logit, None
