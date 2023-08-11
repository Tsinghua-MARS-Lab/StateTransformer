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
import argparse
class Tensor_Dataset(torch.utils.data.Dataset):
    def __init__(self,pth_dir_path,split="train",start_idx = 0, finish_idx = 100, traj_or_keypoints = "traj"):
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
        
        for traj_file, cond_file in tqdm(zip(self.traj_filename_list, self.cond_filename_list)):
            assert os.path.exists(traj_file), f'{traj_file} does not exist!'
            assert os.path.exists(cond_file), f'{cond_file} does not exist!'
        self.file_length = len(self.traj_filename_list)

    def __getitem__(self,idx):
        traj_filename = self.traj_filename_list[idx]
        cond_filename = self.cond_filename_list[idx]
        traj = torch.load(traj_filename)
        cond = torch.load(cond_filename)
        return dict(
            traj = traj,
            cond = cond,
        )
    def __len__(self):
        return self.file_length
def collate_fn(batch):
    return dict(
        traj = torch.cat([item['traj'] for item in batch], dim=0),
        cond = torch.cat([item['cond'] for item in batch], dim=0),
    )


def do_test(model, dataloader, current_min_loss, saving_dir):
    sum_val_loss = torch.tensor(0.0).cuda()
    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        model.eval()
        traj,cond = batch['traj'][...,:2].cuda(),batch['cond'].cuda()
        # print(traj.shape)
        # print(cond.shape)
        with torch.no_grad():
            traj_pred, cls_pred = model.sample_forward(cond,verbose=False, cal_elbo=False,return_diffusion=False,mc_num=1,determin=True)
            # calculate mean ADE and FDE between traj and traj_pred.
            error = traj - traj_pred
            # batchsize * length * 4
            error = error[...,:2] # batchsize * length * 2
            ADE = torch.mean(torch.sqrt(torch.sum(error**2,dim=-1)),dim=-1) # batchsize
            FDE = torch.sqrt(torch.sum(error[...,-1]**2,dim=-1)) # batchsize
            # use ade as validation loss.
            val_loss = ADE.mean()
        wandb.log({"val_loss":val_loss.item()})
        sum_val_loss += val_loss
    avg_val_loss = sum_val_loss / counter
    # print("Current eval ADE: ", avg_val_loss.item())
    if avg_val_loss < current_min_loss:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        torch.save(model, saving_dir + 'best_model.pth')
        print("Best model saved: val loss decreased from {} to {}".format(current_min_loss, avg_val_loss))
    wandb.log({"avg_val_loss":avg_val_loss.item()})
    return avg_val_loss

def do_train(model, train_dataloader, test_dataloader,  saving_dir, max_epoch, per_eval_step, optimizer):

    steps_counter = 0
    current_min_loss = float("inf")
    for epoch_num in range(max_epoch):
        for batch in tqdm(train_dataloader):
            steps_counter += 1
            if steps_counter % per_eval_step == 1:
                new_val_loss = do_test(model,test_dataloader,current_min_loss,saving_dir)
                current_min_loss = new_val_loss if new_val_loss < current_min_loss else current_min_loss
            model.train()
            optimizer.zero_grad()
            traj,cond = batch['traj'][...,:2].cuda(),batch['cond'].cuda()
            # print(traj.shape)
            # print(cond.shape)
            train_loss = model.train_forward(cond,traj)
            # print(f"Currently: Epoch {epoch_num} Step {steps_counter}, Train Loss {train_loss.item()}.")
            wandb.log({"Train loss": train_loss.item()})
            train_loss.backward()
            optimizer.step()


def main():
    # Define default hyperparameters
    NAME = '200032-256FeatDim-run-LargeTFBased-keypoints_0.65TrainAllTest'
    HYPERPARAMS = {
        "NAME": NAME,
        "SEED": 114514,
        "FEAT_DIM": 256,
        "BATCH_SIZE": 512,
        "NUM_WORKERS": 5,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-5,
        "TRAIN_DIR": '/localdata_ssd/nuplan_2_diff_dataset_new/gen_164_/train/',
        "TEST_DIR": '/localdata_ssd/nuplan_2_diff_dataset_new/gen_164_/test/',
        "SAVING_K": 1,
        "SAVING_DIR": f"/localdata_ssd/nuplan_2/nuplan_diff_new_keypoints_decoderTFBased_saving_dir/{NAME}",
        "WANDB_PROJECT": f"diffusion_decoder_TFBased_{NAME}",
        "WANDB_ENTITY": "jingzheshi",
        "VAL_STEP": 700,
        "MAX_EPOCHS": 5,
        "PRECISION": 32,
        "TRAIN_INIT_IDX": 200000,# min: 195872?
        "TRAIN_FINI_IDX": 1400000, # max: 2136264
        "TEST_INIT_IDX": 0,
        "TEST_FINI_IDX": 195000,# max: 195871
        "TRAJ_OR_KEYPOINTS": 'keypoints',
        "NUM_KEY_POINTS": 5,
        "FEATURE_SEQ_LENTH": 16,
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
    VAL_STEP = HYPERPARAMS["VAL_STEP"]
    MAX_EPOCHS = HYPERPARAMS["MAX_EPOCHS"]
    PRECISION = HYPERPARAMS["PRECISION"]
    TRAIN_INIT_IDX = HYPERPARAMS["TRAIN_INIT_IDX"]
    TRAIN_FINI_IDX = HYPERPARAMS["TRAIN_FINI_IDX"]
    TEST_INIT_IDX = HYPERPARAMS["TEST_INIT_IDX"]
    TEST_FINI_IDX = HYPERPARAMS["TEST_FINI_IDX"]
    TRAJ_OR_KEYPOINTS = HYPERPARAMS["TRAJ_OR_KEYPOINTS"]
    NUM_KEY_POINTS = HYPERPARAMS["NUM_KEY_POINTS"]
    FEATURE_SEQ_LENTH = HYPERPARAMS["FEATURE_SEQ_LENTH"]
    assert SAVING_K == 1, ''
    pl.seed_everything(SEED)
    diffusionDecoder = DiffusionDecoderTFBased(1024,256,2,feat_dim=FEAT_DIM) if TRAJ_OR_KEYPOINTS == 'traj' else DiffusionDecoderTFBasedForKeyPoints(1024,256,2,feat_dim=FEAT_DIM,input_feature_seq_lenth=FEATURE_SEQ_LENTH, num_key_points = NUM_KEY_POINTS)
    diffusionDecoder.to("cuda")

    model = diffusionDecoder
    optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)

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


    saving_dir_path = SAVING_DIR + '/'+timestr+'/'
    wandb.init(project = WANDB_PROJECT, entity = WANDB_ENTITY, name = NAME)
    wandb.watch(model,log_freq = 100)
    do_train(model,train_dataloader, test_dataloader, saving_dir_path, MAX_EPOCHS, VAL_STEP, optimizer)










if __name__ == '__main__':
    main()