import numpy as np
import os
import cv2

def save_raster(result_dic, debug_raster_path, agent_type_num, past_frames_num, image_file_name, split,
                high_scale, low_scale):
    # save rasters
    path_to_save = debug_raster_path
    # check if path not exist, create
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        file_number = 0
    else:
        file_number = len(os.listdir(path_to_save))
        if file_number > 200:
            return False
    image_shape = None
    for each_key in ['high_res_raster', 'low_res_raster']:
        """
        # channels:
        # 0: route raster
        # 1-20: road raster
        # 21-24: traffic raster
        # 25-56: agent raster (32=8 (agent_types) * 4 (sample_frames_in_past))
        
        # 25-89: agent raster (64=8 (agent_types) * 8 (sample_frames_in_past))
        """
        each_img = result_dic[each_key]
        goal = each_img[:, :, 0]
        road = each_img[:, :, :21]
        traffic_lights = each_img[:, :, 21:25]
        agent = each_img[:, :, 25:]
        # generate a color pallet of 20 in RGB space
        color_pallet = np.random.randint(0, 255, size=(21, 3)) * 0.5
        target_image = np.zeros([each_img.shape[0], each_img.shape[1], 3], dtype=np.float32)
        image_shape = target_image.shape
        for i in list(range(21))[::-1]:
            road_per_channel = road[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(road_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][road_per_channel == 1] = color_pallet[i, k]
        for i in range(3):
            traffic_light_per_channel = traffic_lights[:, :, i].copy()
            # repeat on the third dimension into RGB space
            # replace the road channel with the color pallet
            if np.sum(traffic_light_per_channel) > 0:
                for k in range(3):
                    target_image[:, :, k][traffic_light_per_channel == 1] = color_pallet[i, k]
        target_image[:, :, 0][goal == 1] = 255
        # generate 9 values interpolated from 0 to 1
        agent_colors = np.array([[0.01 * 255] * past_frames_num,
                                 np.linspace(0, 255, past_frames_num),
                                 np.linspace(255, 0, past_frames_num)]).transpose()

        # print('test: ', past_frames_num, agent_type_num, agent.shape)
        for i in range(past_frames_num):
            for j in range(agent_type_num):
                # if j == 7:
                #     print('debug', np.sum(agent[:, :, j * 9 + i]), agent[:, :, j * 9 + i])
                agent_per_channel = agent[:, :, j * past_frames_num + i].copy()
                # agent_per_channel = agent_per_channel[:, :, None].repeat(3, axis=2)
                if np.sum(agent_per_channel) > 0:
                    for k in range(3):
                        target_image[:, :, k][agent_per_channel == 1] = agent_colors[i, k]
        cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '.png'), target_image)
    for each_key in ['context_actions', 'trajectory_label']:
        pts = result_dic[each_key]
        for scale in [high_scale, low_scale]:
            target_image = np.zeros(image_shape, dtype=np.float32)
            for i in range(pts.shape[0]):
                x = int(pts[i, 0] * scale) + target_image.shape[0] // 2
                y = int(pts[i, 1] * scale) + target_image.shape[1] // 2
                if x < target_image.shape[0] and y < target_image.shape[1]:
                    target_image[x, y, :] = [255, 255, 255]
            cv2.imwrite(os.path.join(path_to_save, split + '_' + image_file_name + '_' + str(each_key) + '_' + str(scale) +'.png'), target_image)
    print('length of action and labels: ', result_dic['context_actions'].shape, result_dic['trajectory_label'].shape)
    print('debug images saved to: ', path_to_save, file_number, each_img.shape)
    return True
