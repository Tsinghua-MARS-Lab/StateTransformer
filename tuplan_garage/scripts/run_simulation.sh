
SPLIT=test14_hard # test14_hard, val14_split
CHALLENGE=open_loop_boxes # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT=/cephfs/zhanjh/PDM_EXP/EXP/exp/training_pdm_ref_offset_model/training_pdm_ref_offset_model_lr_warmup/2024.08.31.11.36.45/checkpoints/epoch_130.ckpt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_hybrid_ref_planner \
planner.pdm_hybrid_planner.checkpoint_path=$CHECKPOINT \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
scenario_builder.data_root=/cephfs/shared/test \
worker.threads_per_node=16 \
experiment_uid=pdm_str_warmup_130 \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
# worker=single_machine_thread_pool \

# scenario_builder.data_root=$NUPLAN_DATA_ROOT/nuplan-v1.1/test \