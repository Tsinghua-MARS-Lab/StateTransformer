SEED=0

JOB_NAME=caching_dataset
CACHE_PATH=/storage/Cache/

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=cache \
+training=training_pdm_offset_model \
job_name=$JOB_NAME \
scenario_builder=nuplan \
cache.cache_path=$CACHE_PATH \
cache.force_feature_computation=True \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"