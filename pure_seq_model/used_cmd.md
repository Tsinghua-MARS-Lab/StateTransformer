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

bash dist_test.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_v1_no_vel/head_weight_100/ckpt/checkpoint_epoch_30.pth --extra_tag head_weight_100

bash dist_train.sh 1 --cfg_file configs/config_v1_no_vel.yaml --batch_size 8 --epochs 40 --extra_tag fix_head_bug

bash dist_train.sh 8 --cfg_file configs/config_v1_no_vel.yaml --batch_size 512 --epochs 30 --extra_tag head_weight_10

bash dist_train.sh 8 --cfg_file configs/config_v2.yaml --batch_size 256 --epochs 30 --extra_tag base_trial
bash dist_train.sh 8 --cfg_file configs/config_v2.yaml --batch_size 256 --epochs 30 --extra_tag classification_5


bash dist_test.sh 8 --cfg_file configs/config_v2.yaml --batch_size 256 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_v2/base_trial/ckpt/checkpoint_epoch_30.pth --extra_tag base_trial_top_1


bash dist_train.sh 1 --cfg_file configs/config_base_encoder_decoder.yaml --batch_size 8 --epochs 30 --extra_tag code_test
bash dist_train.sh 8 --cfg_file configs/config_base_encoder_decoder.yaml --batch_size 128 --epochs 30 --extra_tag base_trial
bash dist_train.sh 8 --cfg_file configs/config_base_encoder_decoder.yaml --batch_size 128 --epochs 20 --extra_tag fix_data
bash dist_train.sh 8 --cfg_file configs/config_base_encoder_decoder.yaml --batch_size 128 --epochs 20 --extra_tag l2_norm

bash dist_test.sh 1 --cfg_file configs/config_base_encoder_decoder.yaml --batch_size 8 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_base_encoder_decoder/fix_data_3/ckpt/checkpoint_epoch_20.pth --extra_tag fix_data_3

bash dist_test.sh 8 --cfg_file configs/config_base_encoder_decoder.yaml --batch_size 64 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_base_encoder_decoder/l2_norm_filtered/ckpt/checkpoint_epoch_20.pth --extra_tag l2_norm_filtered

bash dist_train.sh 8 --cfg_file configs/config_learned_anchor.yaml --batch_size 128 --epochs 20 --extra_tag base_trial
bash dist_train.sh 1 --cfg_file configs/config_learned_anchor.yaml --batch_size 1 --epochs 20 --extra_tag base_trial

bash dist_train.sh 8 --cfg_file configs/config_learned_anchor.yaml --batch_size 128 --epochs 20 --extra_tag mod_normalization_para
bash dist_train.sh 8 --cfg_file configs/config_learned_anchor.yaml --batch_size 128 --epochs 20 --extra_tag output_for_classification

bash dist_test.sh 8 --cfg_file configs/config_learned_anchor.yaml --batch_size 64 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_learned_anchor/output_for_classification/ckpt/checkpoint_epoch_20.pth --extra_tag output_for_classification


bash dist_train.sh 1 --cfg_file configs/config_wayformer.yaml --batch_size 1 --epochs 20 --extra_tag code_test
bash dist_train.sh 8 --cfg_file configs/config_wayformer.yaml --batch_size 64 --epochs 20 --extra_tag code_test
bash dist_train.sh 8 --cfg_file configs/config_wayformer.yaml --batch_size 64 --epochs 20 --extra_tag type_vehicle
bash dist_train.sh 8 --cfg_file configs/config_wayformer.yaml --batch_size 64 --epochs 20 --extra_tag type_vehicle_fix

bash dist_test.sh 8 --cfg_file configs/config_wayformer_gmm.yaml --batch_size 32 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_wayformer_gmm/base_test/ckpt/checkpoint_epoch_20.pth --extra_tag base_test


bash dist_train.sh 1 --cfg_file configs/config_wayformer_gmm.yaml --batch_size 1 --epochs 20 --extra_tag code_test
bash dist_train.sh 8 --cfg_file configs/config_wayformer_gmm.yaml --batch_size 32 --epochs 20 --extra_tag free_anchor_no_dropout

bash dist_test.sh 8 --cfg_file configs/config_wayformer_gmm.yaml --batch_size 32 --ckpt /home/xiweitao/study/transformer4planning/pure_seq_model/output/config_wayformer_gmm/free_anchor_new/ckpt/checkpoint_epoch_20.pth --extra_tag free_anchor_new
