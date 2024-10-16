## Training 
For example, train with 8 GPUs: 
```

bash dist_train.sh 1 --cfg_file configs/config_v1.yaml --batch_size 8 --epochs 20 --extra_tag my_first_exp
bash dist_train.sh 8 --cfg_file configs/config_v1.yaml --batch_size 512 --epochs 20 --extra_tag my_first_exp
```
Actually, during the training process, the evaluation results will be logged to the log file under `output/waymo/mtr+100_percent_data/my_first_exp/log_train_xxxx.txt`

## Testing
For example, test with 8 GPUs: 
```
cd tools
bash scripts/dist_test.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --ckpt ../output/waymo/mtr+100_percent_data/my_first_exp/ckpt/checkpoint_epoch_30.pth --extra_tag my_first_exp --batch_size 80 
```
