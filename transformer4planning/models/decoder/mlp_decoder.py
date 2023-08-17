import torch
from typing import Dict
from transformer4planning.models.decoder.base import TrajectoryDecoder
from transformer4planning.libs.mlp import DecoderResCat

class MultiTrajDecoder(TrajectoryDecoder):
    """
    MultiTrajDecoder is a decoder that support predicting multiple trajectories.
    Also, it can be initialized with keypoint decoder to generate key points of trajectories,
    and a scorer decoder to judge different trajectories.
    """
    def __init__(self, model_args, config):
        super().__init__(model_args, config)
        if self.ar_future_interval > 0:
            self.key_points_decoder = DecoderResCat(config.n_inner, config.n_embd, out_features=self.out_features * self.k) 
    
    def forward(self, 
                hidden_output,
                label, 
                info_dict:Dict=None):
        """
        
        """
        device = hidden_output.device
        pred_length = info_dict.get("pred_length", label.shape[1])
        context_length = info_dict.get("context_length", None)
        assert context_length is not None, "context length can not be None"
        
        traj_hidden_state = hidden_output[:, -pred_length -1:-1, :]
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        if not self.model_args.pred_key_points_only:
            traj_logits = self.traj_decoder(traj_hidden_state)
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
            loss += traj_loss
        else:
            traj_logits = torch.zeros_like(label[..., :2])
            traj_loss = None
        
        kp_loss, cls_loss = None, None
        if self.ar_future_interval > 0:
            future_key_points = info_dict["future_key_points"]
            scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
            future_key_points_hidden_state = hidden_output[:, scenario_type_len + context_length * 2 - 1:scenario_type_len + context_length * 2 + future_key_points.shape[1] - 1, :]
            key_points_logits = self.key_points_decoder(future_key_points_hidden_state)  # b, s, 4/2*k

            if self.k == 1:
                kp_loss = self.loss_fct(key_points_logits, future_key_points.to(device)) if self.model_args.predict_yaw else \
                            self.loss_fct(key_points_logits[..., :2], future_key_points[..., :2].to(device))
                if self.model_args.task == "waymo":
                    future_key_points_gt_mask = info_dict["future_key_points_gt_mask"]
                    kp_loss = (kp_loss* future_key_points_gt_mask).sum() / (future_key_points_gt_mask.sum() + 1e-7)
                loss += kp_loss
                traj_logits = torch.cat([key_points_logits, traj_logits], dim=1)
            else:
                b, s, _ = future_key_points.shape
                k_result = key_points_logits.rehsape(b, s, self.k, -1)

                # get loss of minimal loss from k results
                k_future_key_points = future_key_points.unsqueeze(2).repeat(1, 1, self.k, 1).reshape(b, s, self.k, -1)
                kp_loss_candidate = self.loss_fct(k_result, k_future_key_points.to(device)) if self.model_args.predict_yaw else \
                            self.loss_fct(k_result[..., :2], k_future_key_points[..., :2].to(device))
            
                kp_loss, min_loss_indices = torch.min(kp_loss_candidate.sum(dim=-1), dim=2)
                if self.model_args.task == "waymo":
                    future_key_points_gt_mask = info_dict["future_key_points_gt_mask"]
                    kp_loss = (kp_loss.unsqueeze(-1) * future_key_points_gt_mask).sum() / (future_key_points_gt_mask.sum() + 1e-7)
                else:
                    kp_loss = kp_loss.mean()
                loss += kp_loss


                if self.next_token_scorer_decoder is not None:
                    pred_logits = self.next_token_scorer_decoder(future_key_points_hidden_state.to(device))
                    cls_loss_fn = torch.nn.CrossEntropyLoss()
                    cls_loss = cls_loss_fn(pred_logits.reshape(b * s, self.k).to(torch.float64), min_loss_indices.reshape(-1).long())
                    loss += cls_loss
                    if self.training:
                        # concatenate the key points with predicted trajectory for evaluation
                        selected_key_points = key_points_logits.reshape(b * s, self.k, -1)[torch.arange(b * s),
                                              min_loss_indices.reshape(-1), :].reshape(b, s, -1)
                    else:
                        # concatenate the key points with predicted trajectory selected from the classifier for evaluation
                        selected_key_points = key_points_logits.reshape(b * s, self.k, -1)[torch.arange(b * s),
                                              pred_logits.argmax(dim=-1).reshape(-1), :].reshape(b, s, -1)
                    traj_logits = torch.cat([selected_key_points, traj_logits], dim=1)
                else:
                    print('WARNING: Randomly select key points for evaluation, try to use next_token_scorer_decoder')
                    traj_logits = torch.cat([key_points_logits[0].reshape(b, s, -1), traj_logits], dim=1)
        
        loss_items = dict(
            trajectory_loss=traj_loss,
            key_points_loss=kp_loss,
            scorer_loss=cls_loss,
        )
        return traj_logits, loss, loss_items

    def generate_keypoints(self, hidden_state, info_dict:Dict=None):
        batch_size = hidden_state.shape[0]
        key_points_logit = self.key_points_decoder(hidden_state).reshape(batch_size, 1, -1)  # b, 1, 4/2*k
        pred_logits = self.next_token_scorer_decoder(hidden_state).reshape(batch_size, 1, -1) if self.k > 1 else None # b, 1, k
        return key_points_logit, pred_logits