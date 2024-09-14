python run_simulation.py  \
--test_type closed_loop_reactive_agents  \
--data_path /cephfs/shared/nuplan-v1.1/trainval  \
--map_path /cephfs/shared/nuplan-v1.1/maps  \
--model_path /cephfs/shared/Medium_SKPY_PI2_x10_auXYd1_VAL/checkpoint-407000 \
--split_filter_yaml /cephfs/zhanjh/str1/StateTransformer/nuplan_simulation/val_whole.yaml \
--max_scenario_num 1000000 \
--max_scenario_num 1000000 \
--batch_size 10  \
--device cuda  \
--exp_folder str124m_val_re  \
--processes-repetition 8 \
# open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents