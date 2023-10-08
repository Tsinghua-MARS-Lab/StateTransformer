import torch
import math


def build_anchor_tensor(speed_list, angle_list, step_time, step_num):
    # total anchor num is len(speed_list)*len(angle_list)+1. The additional one anchor is the zeros anchor with corresponding to stational agent
    
    anchor_tensor = torch.zeros(len(angle_list)*len(speed_list)+1, step_num, 3)
    for a_index in range(len(angle_list)):
        for s_index in range(len(speed_list)):
            angle = angle_list[a_index]
            distance = speed_list[s_index]*step_time
            current_x = 0
            current_y = 0
            current_yaw = 0
            for i in range(step_num):
                current_yaw = (i+1)*angle/step_num
                current_x += distance*math.cos(current_yaw)
                current_y += distance*math.sin(current_yaw)
                anchor_tensor[1+a_index*len(speed_list)+s_index, step_num-i-1, 0] = current_x
                anchor_tensor[1+a_index*len(speed_list)+s_index, step_num-i-1, 1] = current_y
                anchor_tensor[1+a_index*len(speed_list)+s_index, step_num-i-1, 2] = current_yaw
    return anchor_tensor.view(len(angle_list)*len(speed_list)+1, step_num*3)