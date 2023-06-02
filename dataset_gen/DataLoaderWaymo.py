import os
import numpy as np
import pickle
import math

TYPES = {
    "TYPE_UNSET": 0,  
    "TYPE_VEHICLE": 1,
    "TYPE_PEDESTRIAN": 2,
    "TYPE_CYCLIST": 3,
    "TYPE_OTHER": 4,
    "EGO": 5,
}


def rotate_points_along_z(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)

    rot_matrix = np.array((
        cosa,  sina,
        -sina, cosa
    )).reshape(2, 2)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot

class WaymoDL:
    def __init__(self, data_path=None, mode="train"):
        
        assert data_path is not None and mode in ["train", "test"]

        self.data_root = data_path["WAYMO_DATA_ROOT"]
        self.data_path = os.path.join(self.data_root, data_path["SPLIT_DIR"][mode])

        self.global_file_names = sorted([os.path.join(self.data_path, each_path) for each_path in os.listdir(self.data_path) if each_path[0] != '.'])
        self.total_file_num = len(self.global_file_names)

    def get_next_file(self, specify_file_index=None):
        if specify_file_index is None:
            self.current_file_index += 1
            file_index = self.current_file_index
        else:
            # for non-sequential parallel loading
            file_index = specify_file_index
        if not file_index < self.total_file_num:
            print('index exceed total file number', file_index, self.total_file_num)
            self.end = True
            return None

        if os.path.getsize(self.global_file_names[file_index]) > 0:
            with open(self.global_file_names[file_index], 'rb') as f:
                info = pickle.load(f)
        else:
            print('empty file', self.global_file_names[file_index])
            return None
            
        track_infos = info['track_infos']
        trajs = track_infos['trajs']
        obj_types = track_infos['object_type']
        A, T, _ = trajs.shape
        curr_frame_index = info["current_time_index"]
        data = []
        # process scenario for each track_to_predict agent as ego
        for ego_index in info['tracks_to_predict']['track_index']:
            assert trajs[ego_index, curr_frame_index, -1] > 0
            # init agent dic
            agent_dic = {}
            ego_frame = trajs[ego_index, curr_frame_index].copy()
            # transforme to the ego cooridinate
            trajs_at_ego = self.transform_trajs_to_ego(trajs.copy(), ego_frame[0:2], ego_frame[6])
            # pack ego
            agent_dic["ego"] = {
                    'pose': trajs_at_ego[ego_index, :, [0,1,2,6]].transpose(1,0),
                    'shape': trajs_at_ego[ego_index, :, 3:6],
                    'speed': trajs_at_ego[ego_index, :, 7:9],
                    'type': TYPES["EGO"],
                    'is_sdc': ego_index==info['sdc_track_index'], 
                    'to_predict': 1,
                    }
            
            for agent_index in range(A):
                if agent_index == ego_index: continue

                agent_dic[agent_index] = {
                    'pose': trajs_at_ego[agent_index, :, [0,1,2,6]].transpose(1,0),
                    'shape': trajs_at_ego[agent_index, :, 3:6],
                    'speed': trajs_at_ego[agent_index, :, 7:9],
                    'type': TYPES[obj_types[agent_index]],
                    'is_sdc': ego_index==info['sdc_track_index'], 
                    'to_predict': 0,
                    }
            
            map_points = self.create_map_data_for_ego(ego_frame, info["map_infos"])

            road_dic = {
                    "xyz": map_points[:, :3],
                    "type": map_points[:, 6],
            }
            data_ego = {
                "road": road_dic,
                "agent": agent_dic,
                "total_frames": T,
                "scenario_id": info["scenario_id"]
            }
            data.append(data_ego)
        return data
    
    def transform_trajs_to_ego(self, obj_trajs, center_xyz, center_heading, heading_index=6, rot_vel_index=[7, 8]):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (2): [x, y]
            center_heading:
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, _ = obj_trajs.shape
        obj_trajs[:, :, 0:center_xyz.shape[0]] -= center_xyz[None, None, :]
        obj_trajs[:, :, 0:2] = rotate_points_along_z(
            points=obj_trajs[:, :, 0:2].reshape(-1, 2),
            angle=-center_heading
        ).reshape(num_objects, num_timestamps, 2)

        obj_trajs[:, :, heading_index] -= center_heading

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, rot_vel_index] = rotate_points_along_z(
                points=obj_trajs[:, :, rot_vel_index].reshape(-1, 2),
                angle=-center_heading
            ).reshape(num_objects, num_timestamps, 2)

        return obj_trajs
    
    def create_map_data_for_ego(self, center_object, map_infos):
        """
        Args:
            center_object (10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        Returns:
            map (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        """
        # transform object coordinates by center objects
        def transform_to_center_coordinates(map):
            map[:, 0:3] -= center_object[None, 0:3]
            map[:, 0:2] = rotate_points_along_z(
                points=map[:, 0:2],
                angle=-center_object[6] + math.pi / 2
            )
            # map[:, 3:5] = rotate_points_along_z(
            #     points=map[:, 3:5],
            #     angle=-center_object[6] - math.pi/2
            # )

            return map

        map_points = map_infos['all_polylines'].copy()
        map_points = transform_to_center_coordinates(map_points)

        return map_points