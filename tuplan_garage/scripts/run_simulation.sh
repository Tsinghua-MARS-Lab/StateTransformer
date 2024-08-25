
SPLIT=val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT=/cephfs/zhanjh/PDM_EXP/EXP/exp/training_pdm_offset_model/training_pdm_offset_model/2024.08.25.14.19.13/best_model/epoch_1-step_2539.ckpt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_hybrid_planner \
planner.pdm_hybrid_planner.checkpoint_path=$CHECKPOINT \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
worker=single_machine_thread_pool \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"