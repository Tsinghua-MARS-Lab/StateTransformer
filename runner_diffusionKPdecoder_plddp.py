# This is used to train the Diffusion Key Point Decoder separately.
# Need to generate the training set and testing set for DiffusionKeyPointDecoder using runner.py
    #   by setting generate_diffusion_dataset_for_key_points_decoder = True and specify diffusion_dataset_save_dir first.
# After training the Diffusion Key Point Decoder using this file, one can use the trained Diffusion Key Point Decoder as key_point_decoder for 
    # Model type TrajectoryGPTDiffusionKPDecoder by setting the model-name to be pretrain/scratch-diffusion_KP_decoder_gpt 
                                                    # and set the key_points_diffusion_decoder_load_from to be the best_model.pth file that is generated and saved by this runner_diffusionKPdecoder.py program.
import torch
import pytorch_lightning as pl
from transformer4planning.models.diffusion_decoders import DiffusionDecoderTFBased, DiffusionDecoderTFBasedForKeyPoints
from tqdm import tqdm
from torch.utils.data import Sampler, DataLoader
import datetime
import time
import os
import wandb
wandb.login(key='3cb4a5ee4aefb4f4e25dae6a16db1f59568ac603')
import argparse
import torch.nn as nn
import copy
import numpy as np
class Tensor_Dataset(torch.utils.data.Dataset):
    def __init__(self,pth_dir_path,split="train",start_idx = 0, finish_idx = 100, traj_or_keypoints = "traj",predict_yaw = True):
        super().__init__()
        self.pth_dir_path = pth_dir_path
        # self.ith_file_list = torch.load(self.pth_dir_path + 'ith_file_list.pth')
        # self.relative_idx_list = torch.load(self.pth_dir_path + 'relative_idx_list.pth')
        # self.length = 
        self.traj_filename_list = []
        self.cond_filename_list = []
        traj_label_name = 'traj_label_' if traj_or_keypoints == "traj" else 'future_key_points_'
        traj_hidden_state_name = 'traj_hidden_state_' if traj_or_keypoints == "traj" else 'future_key_points_hidden_state_'
        for i in range(start_idx,finish_idx+1):
            self.traj_filename_list.append(self.pth_dir_path + traj_label_name+str(i)+'.pth')
            self.cond_filename_list.append(self.pth_dir_path + traj_hidden_state_name+str(i)+'.pth')
        print("Now checking if all files exist for {} set...".format(split))
        
        for traj_file, cond_file in zip(self.traj_filename_list, self.cond_filename_list):
            assert os.path.exists(traj_file), f'{traj_file} does not exist!'
            assert os.path.exists(cond_file), f'{cond_file} does not exist!'
        self.file_length = len(self.traj_filename_list)
        self.predict_yaw = predict_yaw

    def __getitem__(self,idx):
        traj_filename = self.traj_filename_list[idx]
        cond_filename = self.cond_filename_list[idx]
        traj = torch.load(traj_filename)
        cond = torch.load(cond_filename)
        return dict(
            traj = traj if self.predict_yaw else traj[...,:2],
            cond = cond if self.predict_yaw else traj[...,:2],
        )
    def __len__(self):
        return self.file_length
def collate_fn(batch):
    return dict(
        traj = torch.cat([item['traj'] for item in batch], dim=0),
        cond = torch.cat([item['cond'] for item in batch], dim=0),
    )

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
        with torch.no_grad():
            traj,cond = batch['traj'],batch['cond']
            traj_pred, cls_pred = self.model.sample_forward(cond,verbose=False, cal_elbo=False,return_diffusion=False,mc_num=1,determin=True)
            # calculate mean ADE and FDE between traj and traj_pred.
            error = traj - traj_pred
            # batchsize * length * 4
            error = error[...,:2] # batchsize * length * 2
            ADE = torch.mean(torch.sqrt(torch.sum(error**2,dim=-1)),dim=-1) # batchsize
            FDE = torch.sqrt(torch.sum(error[...,-1]**2,dim=-1)) # batchsize
            # use ade as validation loss.
            val_loss = ADE.mean()
        self.log_dict({"val_loss":val_loss})
        return {"val_loss":val_loss}
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)


def main():
    # Define default hyperparameters
    NAME = '30009-1024FeatDim-run-LargeTFBased-keypoints_0.67Train0.40Test'
    HYPERPARAMS = {
        "NAME": NAME,
        "SEED": 42,
        "N_INNER": 6400,
        "N_EMBED": 1600,
        "FEAT_DIM": 1024,
        "BATCH_SIZE": 125,
        "NUM_WORKERS": 5,
        "LR": 8e-5,
        "WEIGHT_DECAY": 2e-5,
        "TRAIN_DIR": '/localdata_ssd/nuplan_3_diff_dataset_new/gen240_GenDiffFeat_trytry/train/',
        "TEST_DIR": '/localdata_ssd/nuplan_3_diff_dataset_new/gen240_GenDiffFeat_trytry/test/',
        "SAVING_K": 2,
        "SAVING_DIR": f"/localdata_ssd/nuplan_3/nuplan_diff_new_keypoints_decoderTFBased_saving_dir/{NAME}",
        "WANDB_PROJECT": f"diffusion_decoder_TFBased_{NAME}",
        "WANDB_ENTITY": "jingzheshi",
        "MAX_EPOCHS": 100,
        "PRECISION": 32,
        "TRAIN_INIT_IDX": 0      ,
        "TRAIN_FINI_IDX": 1245000,
        "TEST_INIT_IDX":  1255000,
        "TEST_FINI_IDX":  1330000,
        "TRAJ_OR_KEYPOINTS": 'keypoints',
        "NUM_KEY_POINTS": 5,
        "FEATURE_SEQ_LENTH": 16,
        "PREDICT_YAW": True,
        "SPECIFIED_KEY_POINTS": True,
        "FORWARD_SPECIFIED_KEY_POINTS": False,
        "NUM_GPU":7,
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
    
    pl.seed_everything(SEED)
    diffusionDecoder = DiffusionDecoderTFBased(N_INNER,N_EMBED,out_features = 4 if PREDICT_YAW else 2,feat_dim=FEAT_DIM) if TRAJ_OR_KEYPOINTS == 'traj' else DiffusionDecoderTFBasedForKeyPoints(N_INNER,N_EMBED,out_features = 4 if PREDICT_YAW else 2,feat_dim=FEAT_DIM,input_feature_seq_lenth=FEATURE_SEQ_LENTH, num_key_points = NUM_KEY_POINTS, specified_key_points = SPECIFIED_KEY_POINTS, forward_specified_key_points = FORWARD_SPECIFIED_KEY_POINTS)
    model_parameters = filter(lambda p: p.requires_grad, diffusionDecoder.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters == {params}")

    model = diffusionDecoder

    train_dataset = Tensor_Dataset(TRAIN_DIR,split="train",start_idx = TRAIN_INIT_IDX, finish_idx = TRAIN_FINI_IDX, traj_or_keypoints = TRAJ_OR_KEYPOINTS)
    test_dataset =  Tensor_Dataset(TEST_DIR,split="test",start_idx = TEST_INIT_IDX, finish_idx = TEST_FINI_IDX, traj_or_keypoints = TRAJ_OR_KEYPOINTS)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = True,
        collate_fn = collate_fn,
        shuffle = True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = True,
        collate_fn = collate_fn,
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
    

    plmodel = PL_MODEL_WRAPPER(model,lr=LR,weight_decay=WEIGHT_DECAY)
    
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