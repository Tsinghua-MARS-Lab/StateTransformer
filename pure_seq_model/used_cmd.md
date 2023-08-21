bash dist_train.sh 8 --cfg_file configs/config_v1.yaml --batch_size 512 --epochs 20 --extra_tag normed_data
bash dist_train.sh 1 --cfg_file configs/config_v1.yaml --batch_size 8 --epochs 20 --extra_tag normed_data

bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag first_exp
bash dist_train.sh 1 --cfg_file configs/config_v1_aug.yaml --batch_size 8 --epochs 20 --extra_tag vel_test
bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag vel_test

bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag update_pos_embed