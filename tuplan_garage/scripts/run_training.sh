TRAIN_EPOCHS=100
TRAIN_LR=1e-4
TRAIN_LR_MILESTONES=[50,75]
TRAIN_LR_DECAY=0.1
BATCH_SIZE=16
SEED=0

JOB_NAME=training_pdm_offset_model_without_kpdecoder
CACHE_PATH=/storage/Cache/
USE_CACHE_WITHOUT_DATASET=True

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=train \
+training=training_pdm_offset_model \
job_name=$JOB_NAME \
scenario_builder=nuplan \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
lightning.trainer.params.max_time=00:10000:00:00 \
lightning.trainer.checkpoint.resume_training=True \
data_loader.params.batch_size=$BATCH_SIZE \
optimizer.lr=$TRAIN_LR \
lr_scheduler=multistep_lr \
lr_scheduler.milestones=$TRAIN_LR_MILESTONES \
lr_scheduler.gamma=$TRAIN_LR_DECAY \
worker=single_machine_thread_pool \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
# worker=single_machine_thread_pool \\
#worker.threads_per_node=24 
