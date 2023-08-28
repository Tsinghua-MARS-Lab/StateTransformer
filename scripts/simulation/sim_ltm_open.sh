
#SPLIT=val14_split
#SPLIT=reduced_val14_split
SPLIT=val14_split
CHALLENGE=open_loop_boxes # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

NUPLAN_DEVKIT_ROOT="$HOME/nuplan-devkit-v1.2/"
NUPLAN_DATA_ROOT="/public/MARS/datasets/nuPlan"
NUPLAN_MAPS_ROOT="/public/MARS/datasets/nuPlan/nuplan-maps-v1.0"
#NUPLAN_DB_FILES="/public/MARS/datasets/nuPlan/nuplan-v1.1/trainval"
NUPLAN_DB_FILES="/localdata_ssd/nuplan/dataset/nuplan-v1.1/val"
#CHECKPOINT="/public/MARS/datasets/nuPlanCache/checkpoint/GPT30mMC_arfi20_PI5FI2_SpecifiedKP_K1/training_results/checkpoint-263000-encoder"
#CHECKPOINT="/public/MARS/t4p/checkpoints/GPTL_SKPY_loss1_K1_data1_SgFix"
CHECKPOINT="/public/MARS/t4p/checkpoints/GPTS_SKP_loss1_K1_data1_SgFix/training_results/checkpoint-275000"

#NUPLAN_DEVKIT_ROOT="$HOME/Documents/codes_on_git/nuplan-devkit/"
#NUPLAN_DATA_ROOT="/Volumes/Elements SE/nuPlan/"
#NUPLAN_MAPS_ROOT="/Volumes/Elements SE/nuPlan/maps/"
#NUPLAN_DB_FILES="/Volumes/Elements SE/nuPlan/nuplan-v1.1/mini"
#CHECKPOINT="/Users/qiaosun/Documents/codes_on_git/InterSim-Dev_Opt/saved_model/gpt30m_kp"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=control_tf_planner \
planner.control_tf_planner.checkpoint_path=$CHECKPOINT \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
worker=ray_distributed \
hydra.searchpath="[pkg://nuplan_garage.planning.script.config.common, pkg://nuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
