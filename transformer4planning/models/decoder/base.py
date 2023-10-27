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
        if self.model_args.generate_diffusion_dataset_for_key_points_decoder:
            self.save_training_diffusion_feature_dir = os.path.join(self.model_args.diffusion_feature_save_dir,'train/')
            self.save_testing_diffusion_feature_dir  = os.path.join(self.model_args.diffusion_feature_save_dir,'val/')
            self.save_test_diffusion_feature_dir = os.path.join(self.model_args.diffusion_feature_save_dir,'test/')
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
            info_dict: dict contains additional infomation, such as context length/input length, pred length, etc. 
        """
        if device is None:
            device = hidden_output.device
        context_length = info_dict.get("context_length", None)
        if context_length is None: # pdm encoder
           input_length = info_dict.get("input_length", None)

        future_key_points = info_dict["future_key_points"]
        scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
        kp_start_index = scenario_type_len + context_length * 2 - 1 if context_length is not None \
            else scenario_type_len + input_length - 1
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
    def save_features(self,input_embeds,context_length,info_dict,future_key_points,transformer_outputs_hidden_state):
        current_device_idx = int(str(input_embeds.device)[-1])
        context_length = info_dict.get("context_length", None)
        assert context_length is not None, "context length can not be None"
        if context_length is None: # pdm encoder
            input_length = info_dict.get("input_length", None)
        future_key_points = info_dict["future_key_points"]
        key_points_num = future_key_points.shape[-2]
        scenario_type_len = self.model_args.max_token_len if self.model_args.token_scenario_tag else 0
        # hidden state to predict future kp is different from mlp decoder
        kp_end_index = scenario_type_len + context_length * 2 if context_length is not None \
                    else scenario_type_len + input_length
        save_id = (self.gpu_device_count * self.current_idx + current_device_idx)*key_points_num
        for key_point_idx in range(key_points_num):
            current_save_id = save_id + key_point_idx
            torch.save(transformer_outputs_hidden_state[:,kp_end_index-1+key_point_idx:kp_end_index-1+key_point_idx+1,:].detach().cpu(), os.path.join(self.save_testing_diffusion_feature_dir, f'future_key_points_hidden_state_{current_save_id}.pth'), )
            torch.save(info_dict['future_key_points'][...,key_point_idx:key_point_idx+1,:].detach().cpu(), os.path.join(self.save_testing_diffusion_feature_dir, f'future_key_points_{current_save_id}.pth'), )
        self.current_idx += 1