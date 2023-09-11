import torch
import pytorch_lightning as pl
from transformer4planning.models.decoder.diffusion_decoder import DiffusionDecoderTFBased, DiffusionDecoderTFBasedForKeyPoints
from tqdm import tqdm
from torch.utils.data import Sampler, DataLoader
import datetime
import time
import os
import wandb
# wandb.login(key='3cb4a5ee4aefb4f4e25dae6a16db1f59568ac603')
import argparse
import torch.nn as nn
import copy
import numpy as np
from modify_traj_utils import modify_func
class PL_MODEL_WRAPPER(pl.LightningModule):
    def __init__(self,
                 model,
                 lr,
                 weight_decay):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
    def forward(self,x):
        return self.model(x)
    def training_step(self,batch,batch_idx):
        self.model.train()
        traj,cond = batch['traj'],batch['cond']
        train_loss = self.model.train_forward(cond,traj)
        self.log_dict({"train_loss":train_loss})
        return {"loss":train_loss}
    def validation_step(self,batch,batch_idx):
        self.model.eval()
        # with torch.no_grad():
        #     traj,cond = batch['traj'],batch['cond']
        #     traj_pred, cls_pred = self.model.sample_forward(cond,verbose=False, cal_elbo=False,return_diffusion=False,mc_num=1,determin=True)
        #     # calculate mean ADE and FDE between traj and traj_pred.
        #     error = traj - traj_pred if traj_pred.shape[-1] == 4 else traj[...,:2] - traj_pred
        #     # batchsize * length * 4
        #     error = error[...,:2] # batchsize * length * 2
        #     ADE = torch.mean(torch.sqrt(torch.sum(error**2,dim=-1)),dim=-1) # batchsize
        #     FDE = torch.sqrt(torch.sum(error[...,-1]**2,dim=-1)) # batchsize
        #     # use ade as validation loss.
        #     val_loss = ADE.mean()
        with torch.no_grad():
            traj,cond = batch['traj'],batch['cond']
            traj_pred, cls_pred = self.model.sample_forward(cond,verbose=False, cal_elbo=False,return_diffusion=False,mc_num=50,determin=False)
            
            out_dict = modify_func(
                output = dict(
                    reg = [traj_p for traj_p in traj_pred.detach().unsqueeze(1)],
                    cls = [cls for cls in cls_pred.detach().unsqueeze(1)],
                ),
                num_mods_out=6
            )
            traj_pred = torch.cat(out_dict['reg'],dim=0)
            cls_pred = torch.cat(out_dict['cls'],dim=0)
            
            # calculate mean ADE and FDE between traj and traj_pred.
            # print(traj.shape)
            # print(traj_pred.shape)
            error = traj.unsqueeze(1) - traj_pred if traj_pred.shape[-1] == 4 else traj[...,:2].unsqueeze(1) - traj_pred
            # batchsize * 6 * length * 2/4
            error = error[...,:2] # batchsize * 6 * length * 2
            ADE = torch.mean(torch.sqrt(torch.sum(error**2,dim=-1)),dim=-1) # batchsize * 6
            FDE = torch.sqrt(torch.sum(error[...,-1]**2,dim=-1)) # batchsize * 6
            # use ade as validation loss.
            val_loss = (ADE.min(dim=-1)[0]).mean()
        self.log_dict({"val_loss":val_loss})
        return {"val_loss":val_loss}
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
def obtain_valid_index(mask, threshold):
    """
    Obtain valid indices where the sum along the trajectory is >= 2.
    """
    sums = mask.sum(dim=1)
    valid_indices = (sums >= int(threshold)).squeeze().nonzero(as_tuple=True)[0]
    return valid_indices

def get_range_indices(tensor):
        # Find the indices of the first and last occurrences of 1s along the rows for each column
    min_indices = (tensor == 1).int().argmax(dim=0)
    max_indices = (tensor == 1).int().flip(dims=[0]).argmax(dim=0)
    max_indices = tensor.size(0) - 1 - max_indices

    # Handle the case where there are no 1s in the column
    no_ones = (tensor.sum(dim=0) == 0)
    min_indices[no_ones] = -1
    max_indices[no_ones] = -1

    # Stack the indices together to form the pairs
    result = torch.stack([min_indices, max_indices], dim=1)

    return result

def interpolate_with_exp_interval(traj, mask):
    mask = mask.int()
    valid_index = obtain_valid_index(mask, 2)
    mask = mask[valid_index]
    traj = traj[valid_index]
    x_dist = torch.tensor([79, 39, 19, 9, 4]).float().to(traj.device).unsqueeze(0).unsqueeze(2)  # (1, l, 1)
    b, l, d = traj.shape
    # print(mask[:,:,0])
    valid_pts = mask[:, :, 0].nonzero()  # Returns indices where mask is non-zero
    # print(valid_pts)
    max_pairs = get_range_indices(mask[:,:,0].t())
    # print(max_pairs)
    first_valid_idx = max_pairs[:,0]
    last_valid_idx = max_pairs[:,1]

    # print(first_valid_idx)
    # print(last_valid_idx)
    # print("sadlkkfnwoeu")
    start = torch.gather(traj, 1, first_valid_idx.unsqueeze(1).unsqueeze(2).expand(b, 1, d))
    end = torch.gather(traj, 1, last_valid_idx.unsqueeze(1).unsqueeze(2).expand(b, 1, d))
    delta_x = (x_dist[:, last_valid_idx, :] - x_dist[:, first_valid_idx, :]).squeeze(0).unsqueeze(-1)
    # print(start.shape)
    # print(end.shape)
    # print(delta_x.shape)
    # print(x_dist[0, first_valid_idx, :].unsqueeze(-1).shape)
    _k = (end - start) / delta_x
    _b = start - _k * x_dist[0, first_valid_idx, :].unsqueeze(-1)
    # print(_k.shape)
    # print(_b.shape)
    # print("dslivbuquw9oefhdjnkm")
    interpolated_traj = _k * x_dist + _b
    # print(interpolated_traj)
    # print(interpolated_traj.shape)

    # Use mask to replace known points from the original traj
    mask_expanded = mask.expand(b, l, d)
    interpolated_traj = torch.where(mask_expanded.bool(), traj, interpolated_traj)

    return interpolated_traj


class Tensor_Dataset(torch.utils.data.Dataset):
    def __init__(self,pth_dir_path,split="train",start_idx = 0, finish_idx = 100, traj_or_keypoints = "traj",predict_yaw = True):
        super().__init__()
        self.pth_dir_path = pth_dir_path
        # self.ith_file_list = torch.load(self.pth_dir_path + 'ith_file_list.pth')
        # self.relative_idx_list = torch.load(self.pth_dir_path + 'relative_idx_list.pth')
        # self.length = 
        self.traj_filename_list = []
        self.cond_filename_list = []
        self.mask_filename_list = []
        traj_label_name = 'traj_label_' if traj_or_keypoints == "traj" else 'future_key_points_'
        traj_hidden_state_name = 'traj_hidden_state_' if traj_or_keypoints == "traj" else 'future_key_points_hidden_state_'
        traj_mask_name = 'traj_gt_mask_' if traj_or_keypoints == 'traj' else 'future_key_points_gt_mask_'
        for i in range(start_idx,finish_idx+1):
            self.traj_filename_list.append(self.pth_dir_path + traj_label_name+str(i)+'.pth')
            self.cond_filename_list.append(self.pth_dir_path + traj_hidden_state_name+str(i)+'.pth')
            self.mask_filename_list.append(self.pth_dir_path + traj_mask_name+str(i)+'.pth')
        print("Now checking if all files exist for {} set...".format(split))
        
        for traj_file, cond_file, mask_file in zip(self.traj_filename_list, self.cond_filename_list, self.mask_filename_list):
            assert os.path.exists(traj_file)
            assert os.path.exists(cond_file)
            assert os.path.exists(mask_file)
        self.file_length = len(self.traj_filename_list)
        self.predict_yaw = predict_yaw

    def __getitem__(self,idx):
        traj_filename = self.traj_filename_list[idx]
        cond_filename = self.cond_filename_list[idx]
        mask_filename = self.mask_filename_list[idx]
        traj = torch.load(traj_filename)
        cond = torch.load(cond_filename)
        mask = torch.load(mask_filename)
        return dict(
            traj = traj if self.predict_yaw else traj[...,:2],
            cond = cond,
            mask = mask,
        )
    def __len__(self):
        return self.file_length

def do_eval(dataloader, model, param_dict):
    model.eval()
    counter = 0
    current_sum = 0.0
    for batch in tqdm(dataloader):
        counter += 1
        with torch.no_grad():
            traj,cond = batch['traj'].cuda(),batch['cond'].cuda()
            traj_pred, cls_pred = model.sample_forward(cond,verbose=False, cal_elbo=False,return_diffusion=False,mc_num=param_dict["mc_num"],determin=False)
            out_dict = modify_func(
                    output = dict(
                        reg = [traj_p for traj_p in traj_pred.detach().unsqueeze(1)],
                        cls = [cls for cls in cls_pred.detach().unsqueeze(1)],
                    ),
                    num_mods_out=6,
                    EM_Iter = param_dict['EM_Iter'],
                )
            traj_pred = torch.cat(out_dict['reg'],dim=0)
            cls_pred = torch.cat(out_dict['cls'],dim=0)
            
            # calculate mean ADE and FDE between traj and traj_pred.
            # print(traj.shape)
            # print(traj_pred.shape)
            error = traj.unsqueeze(1) - traj_pred if traj_pred.shape[-1] == 4 else traj[...,:2].unsqueeze(1) - traj_pred
            # batchsize * 6 * length * 2/4
            error = error[...,:2] # batchsize * 6 * length * 2
            ADE = torch.mean(torch.sqrt(torch.sum(error**2,dim=-1)),dim=-1) # batchsize * 6
            FDE = torch.sqrt(torch.sum(error[...,-1]**2,dim=-1)) # batchsize * 6
            # use ade as validation loss.
            val_loss = (ADE.min(dim=-1)[0]).mean()
        current_sum += val_loss
    mean_ade6 = current_sum / float(counter)
    return mean_ade6
def main():
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument('--mc_num',type=int,default=None)
    parser.add_argument('--EM_Iter',type=int,default=None)
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--gpu_id', type=int ,default=1)
    import os
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    pl.seed_everything(args.seed)
    
    diffusionDecoder = DiffusionDecoderTFBasedForKeyPoints(1024,256,out_features=2, feat_dim=256, input_feature_seq_lenth=22, num_key_points=5, specified_key_points=True, forward_specified_key_points=False)
    plmodel = PL_MODEL_WRAPPER.load_from_checkpoint(checkpoint_path='/localdata_ssd/waymo_1/waymo_diff_new_keypoints_decoderTFBased_saving_dir/Waymo6001602-tsuzuki-allvalid-256FeatDim-run-LargeTFBased-keypoints_AllTrainAllTest-ade6valloss/2023_08_24___04_39_58/epoch=37-step=17859.ckpt',model=diffusionDecoder,lr=1e-4,weight_decay=1e-5)
    model = copy.deepcopy(plmodel.model).eval().cuda()
    TEST_INIT_IDX = 0
    TEST_FINI_IDX = 1360
    TEST_DIR = '/localdata_ssd/waymo_1/diffusion_dataset/test/'
    TRAJ_OR_KEYPOINTS = 'keypoints'
    test_dataset =  Tensor_Dataset(TEST_DIR,split="test",start_idx = TEST_INIT_IDX, finish_idx = TEST_FINI_IDX, traj_or_keypoints = TRAJ_OR_KEYPOINTS)
    PREDICT_YAW = False
    NUM_KEY_POINTS = 5
    def collate_fn(current_mode, batch):
        if PREDICT_YAW:
            traj = torch.cat([item['traj'] for item in batch], dim=0)
        else:
            traj = torch.cat([item['traj'] for item in batch], dim=0)[...,:2]
        cond = torch.cat([item['cond'] for item in batch], dim=0)
        mask = torch.cat([item['mask'] for item in batch], dim=0)
        
        if current_mode == 'train':
            valid_index = obtain_valid_index(mask, threshold = 2)
            cond = cond[valid_index]
            traj = interpolate_with_exp_interval(traj,mask)
        else:
            valid_index = obtain_valid_index(mask, threshold = NUM_KEY_POINTS)
            cond = cond[valid_index]
            traj = traj[valid_index]
        return dict(
            traj = traj,
            cond = cond,
        )
    def collate_fn_eval(batch):
        return collate_fn('test',batch)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 4,
        num_workers = 10,
        pin_memory = True,
        collate_fn = collate_fn_eval,
        shuffle = False,
    )
    mean_ade6 = do_eval(test_dataloader, model, param_dict = dict(mc_num=args.mc_num,EM_Iter=args.EM_Iter))
    print("mean_min_ade6: {}".format(mean_ade6))

if __name__ == "__main__":
    main()