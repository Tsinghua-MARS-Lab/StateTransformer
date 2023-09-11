# This is used to train the Diffusion Key Point Decoder separately.
# Need to generate the training set and testing set for DiffusionKeyPointDecoder using runner.py
    #   by setting generate_diffusion_dataset_for_key_points_decoder = True and specify diffusion_dataset_save_dir first.
# After training the Diffusion Key Point Decoder using this file, one can use the trained Diffusion Key Point Decoder as key_point_decoder for 
    # Model type TrajectoryGPTDiffusionKPDecoder by setting the model-name to be pretrain/scratch-diffusion_KP_decoder_gpt 
                                                    # and set the key_points_diffusion_decoder_load_from to be the best_model.pth file that is generated and saved by this runner_diffusionKPdecoder.py program.
import pickle
import logging
import torch
import pytorch_lightning as pl
from transformer4planning.models.decoder.diffusion_decoder_old import DiffusionDecoderTFBased, DiffusionDecoderTFBasedForKeyPoints
from tqdm import tqdm
from torch.utils.data import Sampler, DataLoader
import datetime
import time
import os
# import wandb
# wandb.login(key='3cb4a5ee4aefb4f4e25dae6a16db1f59568ac603')
import argparse
import torch.nn as nn
import copy
import numpy as np
from transformer4planning.utils.modify_traj_utils import modify_func
from dataset_gen.waymo.config import cfg_from_yaml_file, cfg

# os.environ["WANDB_DISABLED"] = "true"

cfg_from_yaml_file("/home/QJ00367/danjiao/dlnets/transformer4planning/config/config_gpt2_small.yaml", cfg)

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
    valid_index = obtain_valid_index(mask, 1)
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
        self.saved_dataset_folder = pth_dir_path
        self.mode = split
        # self.ith_file_list = torch.load(self.pth_dir_path + 'ith_file_list.pth')
        # self.relative_idx_list = torch.load(self.pth_dir_path + 'relative_idx_list.pth')
        # self.length = 
        
        self.logger = logging.getLogger(__name__)
        
        self.dataset_cfg = cfg.DATA_CONFIG
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]
        self.infos = self.get_all_infos(self.data_root / self.dataset_cfg.INFO_FILE[self.mode])
        self.logger.info(f'Total scenes after filters: {len(self.infos)}')
        
        
        # self.traj_filename_list = []
        # self.cond_filename_list = []
        # self.mask_filename_list = []
        # traj_label_name = 'traj_label_' if traj_or_keypoints == "traj" else 'future_key_points_'
        # traj_hidden_state_name = 'traj_hidden_state_' if traj_or_keypoints == "traj" else 'future_key_points_hidden_state_'
        # traj_mask_name = 'traj_gt_mask_' if traj_or_keypoints == 'traj' else 'future_key_points_gt_mask_'
        # for i in range(start_idx,finish_idx+1):
        #     self.traj_filename_list.append(self.pth_dir_path + traj_label_name+str(i)+'.pth')
        #     self.cond_filename_list.append(self.pth_dir_path + traj_hidden_state_name+str(i)+'.pth')
        #     self.mask_filename_list.append(self.pth_dir_path + traj_mask_name+str(i)+'.pth')
        # print("Now checking if all files exist for {} set...".format(split))
        
        # for traj_file, cond_file, mask_file in zip(self.traj_filename_list, self.cond_filename_list, self.mask_filename_list):
        #     assert os.path.exists(traj_file)
        #     assert os.path.exists(cond_file)
        #     assert os.path.exists(mask_file)
        # self.file_length = len(self.traj_filename_list)
        # self.predict_yaw = predict_yaw
        
    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        self.logger.info(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos
    
    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    # def __getitem__(self,idx):
    #     traj_filename = self.traj_filename_list[idx]
    #     cond_filename = self.cond_filename_list[idx]
    #     mask_filename = self.mask_filename_list[idx]
    #     traj = torch.load(traj_filename)
    #     cond = torch.load(cond_filename)
    #     mask = torch.load(mask_filename)
    #     return dict(
    #         traj = traj if self.predict_yaw else traj[...,:2],
    #         cond = cond,
    #         mask = mask,
    #     )
        
    def __getitem__(self, index):
        ret_infos = self.load_feature_data(index)
    
        return ret_infos

    def load_feature_data(self, index):
        """
        Args:
            index (index):

        Returns:

        """
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(self.saved_dataset_folder + f'/opt_res_{scene_id}.pkl', 'rb') as f:
            feature_data = pickle.load(f)
            
        info['cond'] = torch.from_numpy(feature_data['hidden_state'])
        info['traj'] = torch.from_numpy(feature_data['label']).unsqueeze(1)
        info['label_cls'] = torch.from_numpy(feature_data['label_cls']).unsqueeze(1)
        info['mask'] = torch.from_numpy(feature_data['label_mask']).unsqueeze(-1)
        
        return info
    
    def __len__(self):
        return len(self.infos)


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


def main():
    # Define default hyperparameters
    NAME = 'diffusion_decoder'
    HYPERPARAMS = {
        "NAME": NAME,
        "SEED": 42,
        "N_INNER": 1024,
        "N_EMBED": 256,
        "FEAT_DIM": 256,
        "BATCH_SIZE": 16,
        "NUM_WORKERS": 10,
        "LR": 8e-5,
        "WEIGHT_DECAY": 2e-5,
        "TRAIN_DIR": '/data/madanjiao/model_res/z_gptS_vehicle_k1_KP0_anchored1_e100_finetuneWithAnchorClsOnly/training_results/checkpoint-430000/eval_output/feature_out_train/',
        "TEST_DIR": '/data/madanjiao/model_res/z_gptS_vehicle_k1_KP0_anchored1_e100_finetuneWithAnchorClsOnly/training_results/checkpoint-430000/eval_output/feature_out_val/',
        "SAVING_K": 2,
        "SAVING_DIR": f"/data/madanjiao/model_res/diffusion_decoder/{NAME}",
        "WANDB_PROJECT": f"diffusion_decoder_TFBased_{NAME}",
        "WANDB_ENTITY": "madanjiao1993",
        "MAX_EPOCHS": 300,
        "PRECISION": 32,
        "TRAIN_INIT_IDX": 6999,
        "TRAIN_FINI_IDX": 73999,
        "TEST_INIT_IDX":  0,
        "TEST_FINI_IDX":  5999,
        "TRAJ_OR_KEYPOINTS": 'keypoints',
        "NUM_KEY_POINTS": 1,
        "FEATURE_SEQ_LENTH": 22,
        "PREDICT_YAW": False,
        "SPECIFIED_KEY_POINTS": True,
        "FORWARD_SPECIFIED_KEY_POINTS": False,
        "NUM_GPU":7,
        "LOAD_FROM":None,
    }

    # Validate that TRAJ_OR_KEYPOINTS has an acceptable value


    # Initialize argument parser
    parser = argparse.ArgumentParser(description="parser")
    for key, value in HYPERPARAMS.items():
        if isinstance(value, str):
            parser.add_argument(f'--{key.lower()}', type=str, default=value)
        elif isinstance(value, int):
            parser.add_argument(f'--{key.lower()}', type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(f'--{key.lower()}', type=float, default=value)
        elif isinstance(value, bool):
            parser.add_argument(f'--{key.lower()}',type=bool, default=value)
        elif value is None:
            parser.add_argument(f'--{key.lower()}',type=str,default=None)
        else:
            raise NotImplementedError(f"Type of {key} is not supported yet.")

    # Print hyperparameters
    # Parse arguments
    args = parser.parse_args()

    

    # Overwrite default hyperparameters with command line arguments
    for key in HYPERPARAMS:
        HYPERPARAMS[key] = getattr(args, key.lower())

    assert HYPERPARAMS["TRAJ_OR_KEYPOINTS"] in ['traj', 'keypoints']

    # Print effective hyperparameters
    print("We use the following hyper-parameters:")
    for key, value in HYPERPARAMS.items():
        print(f"{key:<20} : {value}")

    # You can later parse the arguments using
    # args = parser.parse_args()

    NAME = HYPERPARAMS["NAME"]
    SEED = HYPERPARAMS["SEED"]
    N_INNER = HYPERPARAMS["N_INNER"]
    N_EMBED = HYPERPARAMS["N_EMBED"]
    FEAT_DIM = HYPERPARAMS["FEAT_DIM"]
    BATCH_SIZE = HYPERPARAMS["BATCH_SIZE"]
    NUM_WORKERS = HYPERPARAMS["NUM_WORKERS"]
    LR = HYPERPARAMS["LR"]
    WEIGHT_DECAY = HYPERPARAMS["WEIGHT_DECAY"]
    TRAIN_DIR = HYPERPARAMS["TRAIN_DIR"]
    TEST_DIR = HYPERPARAMS["TEST_DIR"]
    SAVING_K = HYPERPARAMS["SAVING_K"]
    SAVING_DIR = HYPERPARAMS["SAVING_DIR"]
    WANDB_PROJECT = HYPERPARAMS["WANDB_PROJECT"]
    WANDB_ENTITY = HYPERPARAMS["WANDB_ENTITY"]
    MAX_EPOCHS = HYPERPARAMS["MAX_EPOCHS"]
    PRECISION = HYPERPARAMS["PRECISION"]
    TRAIN_INIT_IDX = HYPERPARAMS["TRAIN_INIT_IDX"]
    TRAIN_FINI_IDX = HYPERPARAMS["TRAIN_FINI_IDX"]
    TEST_INIT_IDX = HYPERPARAMS["TEST_INIT_IDX"]
    TEST_FINI_IDX = HYPERPARAMS["TEST_FINI_IDX"]
    TRAJ_OR_KEYPOINTS = HYPERPARAMS["TRAJ_OR_KEYPOINTS"]
    NUM_KEY_POINTS = HYPERPARAMS["NUM_KEY_POINTS"]
    FEATURE_SEQ_LENTH = HYPERPARAMS["FEATURE_SEQ_LENTH"]
    PREDICT_YAW = HYPERPARAMS["PREDICT_YAW"]
    SPECIFIED_KEY_POINTS = HYPERPARAMS["SPECIFIED_KEY_POINTS"]
    FORWARD_SPECIFIED_KEY_POINTS = HYPERPARAMS["FORWARD_SPECIFIED_KEY_POINTS"]
    NUM_GPU = HYPERPARAMS["NUM_GPU"]
    LOAD_FROM=HYPERPARAMS["LOAD_FROM"]
    
    pl.seed_everything(SEED)
    diffusionDecoder = DiffusionDecoderTFBased(N_INNER,N_EMBED,out_features = 4 if PREDICT_YAW else 2,feat_dim=FEAT_DIM) if TRAJ_OR_KEYPOINTS == 'traj' else DiffusionDecoderTFBasedForKeyPoints(N_INNER,N_EMBED,out_features = 4 if PREDICT_YAW else 2,feat_dim=FEAT_DIM,input_feature_seq_lenth=FEATURE_SEQ_LENTH, num_key_points = NUM_KEY_POINTS, specified_key_points = SPECIFIED_KEY_POINTS, forward_specified_key_points = FORWARD_SPECIFIED_KEY_POINTS)
    model_parameters = filter(lambda p: p.requires_grad, diffusionDecoder.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters == {params}")

    model = diffusionDecoder

    train_dataset = Tensor_Dataset(TRAIN_DIR,split="train",start_idx = TRAIN_INIT_IDX, finish_idx = TRAIN_FINI_IDX, traj_or_keypoints = TRAJ_OR_KEYPOINTS)
    test_dataset =  Tensor_Dataset(TEST_DIR,split="test",start_idx = TEST_INIT_IDX, finish_idx = TEST_FINI_IDX, traj_or_keypoints = TRAJ_OR_KEYPOINTS)
    
    def collate_fn(current_mode, batch):
        if PREDICT_YAW:
            traj = torch.cat([item['traj'] for item in batch], dim=0)
        else:
            traj = torch.cat([item['traj'] for item in batch], dim=0)[...,:2]
        cond = torch.cat([item['cond'] for item in batch], dim=0)
        mask = torch.cat([item['mask'] for item in batch], dim=0)
        
        if current_mode == 'train':
            valid_index = obtain_valid_index(mask, threshold = NUM_KEY_POINTS)
            cond = cond[valid_index]
            # traj = interpolate_with_exp_interval(traj,mask)
            traj = traj[mask.squeeze(-1).to(torch.bool)].unsqueeze(1)
        else:
            valid_index = obtain_valid_index(mask, threshold = NUM_KEY_POINTS)
            cond = cond[valid_index]
            traj = traj[valid_index]
        return dict(
            traj = traj,
            cond = cond,
        )
    def collate_fn_train(batch):
        return collate_fn('train',batch)
    def collate_fn_eval(batch):
        return collate_fn('test',batch)
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = True,
        collate_fn = collate_fn_train,
        shuffle = True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = True,
        collate_fn = collate_fn_eval,
        shuffle = False,
    )

    now=datetime.datetime.now()
    timestr=now.strftime("%Y_%m_%d___%H_%M_%S")
    checkpoint_callback=pl.callbacks.ModelCheckpoint(
        monitor = "val_loss",
        mode="min",
        #to minimize valid_loss or to maximize valid_accuracy
        dirpath=SAVING_DIR+'/'+timestr+'/',
        save_top_k=SAVING_K,
    )
    


    
    if LOAD_FROM is not None:
        plmodel = PL_MODEL_WRAPPER.load_from_checkpoint(checkpoint_path=LOAD_FROM,model=model,lr=LR,weight_decay=WEIGHT_DECAY)
    else:
        plmodel = PL_MODEL_WRAPPER(model,LR,WEIGHT_DECAY)
    
    wandb_logger = pl.loggers.WandbLogger(
        project = WANDB_PROJECT,
        name = '_runtime:'+timestr,
        entity = WANDB_ENTITY)
    wandb_logger.watch(plmodel,log="all")
    
    trainer=pl.Trainer(callbacks=[checkpoint_callback],logger=wandb_logger,
                       accelerator = 'ddp',
                       gpus=list(range(NUM_GPU)),
                       max_epochs=MAX_EPOCHS,precision=PRECISION)
    
    trainer.fit(plmodel,train_dataloader,test_dataloader)


if __name__ == '__main__':
    main()