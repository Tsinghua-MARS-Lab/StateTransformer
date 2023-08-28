bash dist_train.sh 8 --cfg_file configs/config_v1.yaml --batch_size 512 --epochs 20 --extra_tag normed_data
bash dist_train.sh 1 --cfg_file configs/config_v1.yaml --batch_size 8 --epochs 20 --extra_tag normed_data

bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag first_exp
bash dist_train.sh 1 --cfg_file configs/config_v1_aug.yaml --batch_size 8 --epochs 20 --extra_tag vel_test
bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag vel_test

bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag update_pos_embed
bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag valid_index_bug_fixed
bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag l1_norm
bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag lr1e3

bash dist_train.sh 8 --cfg_file configs/config_v1_aug.yaml --batch_size 512 --epochs 20 --extra_tag base_weight

bash dist_test.sh 1 --cfg_file configs/config_v1_aug.yaml --batch_size 8 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_v1_aug/valid_index_bug_fixed/ckpt/checkpoint_epoch_20.pth --extra_tag first_test


bash dist_train.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --epochs 20 --extra_tag base_trial
bash dist_train.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --epochs 30 --extra_tag fix_head_bug
bash dist_train.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --epochs 30 --extra_tag 1e4
bash dist_train.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --epochs 30 --extra_tag 3e5

bash dist_test.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_v1_no_vel/fix_head_bug/ckpt/checkpoint_epoch_30.pth --extra_tag fix_head_bug

bash dist_train.sh 1 --cfg_file configs/config_v1_no_vel.yaml --batch_size 8 --epochs 40 --extra_tag fix_head_bug