python run_simulation.py  \
--test_type closed_loop_nonreactive_agents  \
--data_path /cephfs/shared/nuplan-v1.1/test  \
--map_path /cephfs/shared/nuplan-v1.1/maps  \
--model_path /cephfs/shared/Medium_SKPY_PI2_x10_auXYd1_VAL/checkpoint-407000 \
--split_filter_yaml /cephfs/zhanjh/str1/StateTransformer/nuplan_simulation/test14_hard.yaml \
--max_scenario_num 100000 \
--batch_size 4  \
--device cuda  \
--exp_folder str124m_testhard_non_new  \
--processes-repetition 8 \
# open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
