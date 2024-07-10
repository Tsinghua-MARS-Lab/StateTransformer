# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

#------------------------DEBUG for devspace-----------------------------
export WANDB_DISABLED=True
# export WANDB_API_KEY=0dd71dd96b21456225d26cceb6583465fe70254a
export CUDA_VISIBLE_DEVICES=6
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_NUM=1
LPAI_CODE_DIR_0=${LPAI_CODE_DIR_0:-"."}

#------------------------For component------------------------------
export PYTHONPATH=${LPAI_CODE_DIR_0}:${PYTHONPATH};
cd ${LPAI_CODE_DIR_0};

# export TMP=/lpai/

NUPLAN_DATA_FOLDER=/lpai/dataset/nuplan-str-0327/0-1-0/online_s6

#MODEL_BASE=scratch-mixtral-medium
#RUN_NAME=CKS-Medium-Uniform

#MODEL_BASE=scratch-mixtral-medium
#RUN_NAME=CKS-Medium-Cluster-1024

#MODEL_BASE=scratch-mixtral-medium
#RUN_NAME=CKS-Medium-Cluster-512

#MODEL_BASE=scratch-mixtral-small
#RUN_NAME=CKS-Small-Uniform

MODEL_BASE=scratch-mixtral-small
RUN_NAME=CKS-Small-Cluster-1024

#MODEL_BASE=scratch-mixtral-small
#RUN_NAME=CKS-Small-Cluster-512


OUTPUT_DIR=${LPAI_MODEL_DIR:-"."}/${MODEL_BASE}-${RUN_NAME}

torchrun \
    --nproc_per_node=${GPU_NUM:-1} \
    --nnodes=${NODE_NUM:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-"127.0.0.1"} \
    --master_port=${MASTER_PORT:-29500} \
    runner.py \
    --model_name ${MODEL_BASE} \
    --model_pretrain_name_or_path None \
    --saved_dataset_folder ${NUPLAN_DATA_FOLDER} \
    --output_dir ${OUTPUT_DIR}/training_results \
    --logging_dir ${OUTPUT_DIR}/training_logs \
    --run_name ${RUN_NAME} \
    --num_train_epochs 50 \
    --per_device_train_batch_size 16 \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 3000 \
    --dataloader_num_workers 96 \
    --dataloader_drop_last True \
    --save_total_limit 5 \
    --do_train \
    --task nuplan \
    --remove_unused_columns False \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 9000 \
    --per_device_eval_batch_size 8 \
    --predict_yaw True \
    --use_proposal 0 \
    --selected_exponential_past True \
    --mean_circular_loss True \
    --raster_channels 34 \
    --use_mission_goal False \
    --raster_encoder_type vit \
    --vit_intermediate_size 768 \
    --lr_scheduler_type cosine \
    --use_speed \
    --use_key_points specified_4s \
    --augment_index 5 \
    --attn_implementation flash_attention_2 \
    --sync_norm True \
    --bf16 True \
    --inspect_kp_loss \
    --overwrite_output_dir \
    --kp_tokenizer cluster \
    --kp_cluster_files "/lpai/volumes/lpai-autopilot-root-muses/xuleimeng/nuplan/cluster/kmeans_points_8s_1024.csv,/lpai/volumes/lpai-autopilot-root-muses/xuleimeng/nuplan/cluster/kmeans_points_4s_1024.csv,/lpai/volumes/lpai-autopilot-root-muses/xuleimeng/nuplan/cluster/kmeans_points_2s_1024.csv,/lpai/volumes/lpai-autopilot-root-muses/xuleimeng/nuplan/cluster/kmeans_points_1s_1024.csv,/lpai/volumes/lpai-autopilot-root-muses/xuleimeng/nuplan/cluster/kmeans_points_0_5s_1024.csv"

    # --kp_cluster_files "/lpai/volumes/lpai-autopilot-root-muses/xuleimeng/nuplan/cluster/kmeans_points_4s_1024.csv"

    #--model_pretrain_name_or_path ""

    #--kp_tokenizer uniform

    #--use_key_points specified_backward \
