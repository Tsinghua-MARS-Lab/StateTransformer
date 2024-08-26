
SPLIT=val14_split
CHALLENGE=closed_loop_nonreactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT=/cephfs/zhanjh/checkpoint/epoch14-step19049_pdm_str.ckpt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_hybrid_planner \
planner.pdm_hybrid_planner.checkpoint_path=$CHECKPOINT \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
worker.threads_per_node=16 \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
# worker=single_machine_thread_pool \