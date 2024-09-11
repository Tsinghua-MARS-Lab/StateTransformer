python run_simulation.py  \
--test_type closed_loop_nonreactive_agents  \
--data_path /cephfs/shared/nuplan-v1.1/test  \
--map_path /cephfs/shared/nuplan-v1.1/maps  \
--model_path /cephfs/zhanjh/checkpoint-27000 \
--split_filter_yaml /cephfs/zhanjh/str1/StateTransformer/nuplan_simulation/test14_hard.yaml \
--max_scenario_num 1000000 \
--max_scenario_num 1000000 \
--batch_size 4  \
--device cuda  \
--exp_folder str124m_testhard_non  \
--processes-repetition 8 \
# open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

experimental: 27.59