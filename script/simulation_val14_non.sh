python run_simulation.py  \
--test_type closed_loop_nonreactive_agents  \
--data_path /cephfs/shared/nuplan-v1.1/trainval  \
--map_path /cephfs/shared/nuplan-v1.1/maps  \
--model_path /cephfs/zhanjh/checkpoint-27000 \
--split_filter_yaml /cephfs/zhanjh/str1/StateTransformer/nuplan_simulation/val14_split.yaml \
--max_scenario_num 1000000 \
--max_scenario_num 1000000 \
--batch_size 4  \
--device cuda  \
--exp_folder str124m_val14_non  \
--processes-repetition 8 \
# open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
exp
