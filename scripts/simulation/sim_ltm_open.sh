
SPLIT=val14_split
CHALLENGE=open_loop_boxes # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

NUPLAN_DEVKIT_ROOT="$HOME/nuplan-devkit-v1.2/"
NUPLAN_DATA_ROOT="/public/MARS/datasets/nuPlan"
NUPLAN_MAPS_ROOT="/public/MARS/datasets/nuPlan/nuplan-maps-v1.0"
NUPLAN_DB_FILES="/public/MARS/datasets/nuPlan/nuplan-v1.1/trainval"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=control_tf_planner \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
hydra.searchpath="[pkg://nuplan_garage.planning.script.config.common, pkg://nuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
