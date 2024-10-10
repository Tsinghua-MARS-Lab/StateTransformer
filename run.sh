#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python dummy_gpu_task.py --size 5000 --gpus 8 --interval 0.001

#------------------------DEBUG for devspace-----------------------------
export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=True
LPAI_CODE_DIR_0=${LPAI_CODE_DIR_0:-"."}

#------------------------For component------------------------------
export PYTHONPATH=${LPAI_CODE_DIR_0}:${PYTHONPATH};
cd ${LPAI_CODE_DIR_0};

MODEL_BASE=scratch-mixtral-medium
RUN_NAME=Train_li3.5m_nuplan3.5m_Val_nuplan0.017m_MixtralM_CS_S6_bf16
OUTPUT_DIR=${LPAI_MODEL_DIR}/${MODEL_BASE}-${RUN_NAME}

# liauto data
# /lpai/dataset/str-liauto-dataset/0-1-0/Liauto_dataset_0416_140w
LIAUTO_DATA_FOLDER=/lpai/volumes/lpai-autopilot-root-muses/wanghuimin/33m-data 
LIAUTO_PKL_FOLDER=/lpai/dataset/li-str-dataset/1-0-0/data

# nuplan data
NUPLAN_DATA_FOLDER=/lpai/dataset/nuplan-str-0327/0-1-1/online_s6
NUPLAN_PKL_FOLDER=/lpai/dataset/nuplan-str-0327/0-1-1/online_s6

torchrun \
    --nproc_per_node=${GPU_NUM:-1} \
    --nnodes=${NODE_NUM:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-"127.0.0.1"} \
    --master_port=${MASTER_PORT:-29500} \
    runner.py \
    --model_name ${MODEL_BASE} \
    --saved_dataset_folder ${LIAUTO_DATA_FOLDER} \
    --saved_pkl_folder ${LIAUTO_PKL_FOLDER} \
    --dataset_scale 1.0 \
    --output_dir ${OUTPUT_DIR}/training_results \
    --logging_dir ${OUTPUT_DIR}/training_logs \
    --run_name ${RUN_NAME} \
    --num_train_epochs 50 \
    --per_device_train_batch_size 16 \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 9000 \
    --dataloader_num_workers 64 \
    --dataloader_drop_last True \
    --save_total_limit 5 \
    --do_train \
    --task lpai \
    --remove_unused_columns False \
    --do_eval \
    --val_on_selected_data nuplan \
    --evaluation_strategy steps \
    --eval_steps 9000 \
    --per_device_eval_batch_size 32 \
    --predict_yaw True \
    --use_full_training_set False \
    --use_proposal 0 \
    --selected_exponential_past True \
    --mean_circular_loss True \
    --raster_channels 51 \
    --use_mission_goal False \
    --raster_encoder_type vit \
    --vit_intermediate_size 768 \
    --lr_scheduler_type cosine \
    --use_speed \
    --use_key_points no \
    --bf16 True \
    --attn_implementation flash_attention_2 \
    --sync_norm True \
    --overwrite_output_dir \
    --nuplan_data_path_to_merge ${NUPLAN_DATA_FOLDER} \
    --max_lpai_train_samples 3500001 \
    --max_nuplan_train_samples 3500000
    # --resume_from_checkpoint
    # --val_on_selected_data nuplan+lpai
    #--max_train_samples
