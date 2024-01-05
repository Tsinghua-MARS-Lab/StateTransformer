import os
import gc
import pickle
import numpy as np
import torch
import math
from copy import deepcopy
from typing import List
from shapely.geometry import Polygon, Point
from functools import partial
from torch.utils.data._utils.collate import default_collate
from nuplan.common.actor_state.state_representation import StateSE2, Point2D, StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from transformer4planning.preprocess.utils import compute_derivative, route_roadblock_correction
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan_garage.planning.training.preprocessing.features.pdm_feature import PDMFeature
from nuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)

DRIVABLE_MAP_LAYERS = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.CARPARK_AREA,
]

def nuplan_vector_collate_func(batch, dic_path=None, map_api=dict(), **kwargs):
    use_centerline = kwargs.get("use_centerline", False)
    expected_padding_keys = ["road_ids", "route_ids", "traffic_ids"]
    agent_id_lengths = list()
    for i, d in enumerate(batch):
        agent_id_lengths.append(len(d["agent_ids"]))
    max_agent_id_length = max(agent_id_lengths)
    for i, d in enumerate(batch):
        agent_ids = d["agent_ids"]
        agent_ids.extend(["null"] * (max_agent_id_length - len(agent_ids)))
        batch[i]["agent_ids"] = agent_ids
    padded_tensors = dict()
    for key in expected_padding_keys:
        tensors = [data[key] for data in batch]
        padded_tensors[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
        for i, _ in enumerate(batch):
            batch[i][key] = padded_tensors[key][i]
    
    map_func = partial(pdm_vectorize, data_path=dic_path, map_api=map_api, use_centerline=use_centerline)
    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)

    if len(new_batch) == 0:
        return None
    
    # process as data dictionary
    result = dict()
    for key in new_batch[0].keys():
        try:
            result[key] = default_collate([d[key] for d in new_batch])
        except:
            print(f"{key} is invalid")
    return result

def get_drivable_area_map(map_api, ego_position, map_radius=50):
    position = Point2D(ego_position[0], ego_position[1])
    drivable_area = map_api.get_proximal_map_objects(
        position, map_radius, DRIVABLE_MAP_LAYERS
    )

    # collect lane polygons in list, save on-route indices
    drivable_polygons: List[Polygon] = []
    drivable_polygon_ids: List[str] = []
    
    for type in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for roadblock in drivable_area[type]:
            for lane in roadblock.interior_edges:
                drivable_polygons.append(lane.polygon)
                drivable_polygon_ids.append(lane.id)

    for carpark in drivable_area[SemanticMapLayer.CARPARK_AREA]:
        drivable_polygons.append(carpark.polygon)
        drivable_polygon_ids.append(carpark.id)

    # create occupancy map with lane polygons
    drivable_area_map = PDMOccupancyMap(drivable_polygon_ids, drivable_polygons)

    return drivable_area_map

def load_route_dicts(route_roadblock_ids: List[str], map_api) -> None:
    """
    Loads roadblock and lane dictionaries of the target route from the map-api.
    :param route_roadblock_ids: ID's of on-route roadblocks
    """
    # remove repeated ids while remaining order in list
    route_roadblock_ids = [str(id) for id in route_roadblock_ids]
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

    route_roadblock_dict = {}
    route_lane_dict = {}

    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(
            id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        if block is None:
            continue
        # assert block is not None, f"Roadblock {route_roadblock_ids} not found in map-api."
        route_roadblock_dict[block.id] = block

        for lane in block.interior_edges:
            route_lane_dict[lane.id] = lane
    return route_lane_dict, route_roadblock_dict

def build_geometry(ego_position, ego_shape):
    from nuplan.common.geometry.transform import translate_longitudinally_and_laterally
    half_width = ego_shape[0] / 2
    half_length = ego_shape[1] / 2
    corners = [
        tuple(translate_longitudinally_and_laterally(StateSE2(ego_position[0], ego_position[1], ego_position[-1]), half_length, half_width).point),
        tuple(translate_longitudinally_and_laterally(StateSE2(ego_position[0], ego_position[1], ego_position[-1]), -half_length, half_width).point),
        tuple(translate_longitudinally_and_laterally(StateSE2(ego_position[0], ego_position[1], ego_position[-1]), -half_length, -half_width).point),
        tuple(translate_longitudinally_and_laterally(StateSE2(ego_position[0], ego_position[1], ego_position[-1]), half_length, -half_width).point),
    ]
    return Polygon(corners)

def get_starting_lane(ego_position, driviable_area_map, route_lane_dict, ego_shape):
    """
    Returns the most suitable starting lane, in ego's vicinity.
    :param ego_state: state of ego-vehicle
    :return: lane object (on-route)
    """
    starting_lane = None
    on_route_lanes, heading_error = get_intersecting_lanes(ego_position, driviable_area_map, route_lane_dict)

    if on_route_lanes:
        # 1. Option: find lanes from lane occupancy-map
        # select lane with lowest heading error
        starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
        return starting_lane
    else:
        # 2. Option: find any intersecting or close lane on-route
        closest_distance = np.inf
        for edge in route_lane_dict.values():
            if edge.contains_point(Point(*ego_position[:2])):
                starting_lane = edge
                break
            # TODO: ego_state.car_footprint.geometry mapping
            geometry = build_geometry(ego_shape, ego_position)
            distance = edge.polygon.distance(geometry)
            if distance < closest_distance:
                starting_lane = edge
                closest_distance = distance
    return starting_lane

def get_intersecting_lanes(ego_position, drivible_area_map, routd_ids):
    """
    Returns on-route lanes and heading errors where ego-vehicle intersects.
    :param ego_state: state of ego-vehicle
    :return: tuple of lists with lane objects and heading errors [rad].
    """
    ego_position = np.array(ego_position[:2])
    ego_position_point = Point(*ego_position)
    ego_heading = ego_position[-1]
    intersecting_lanes =drivible_area_map.intersects(ego_position_point)

    on_route_lanes, on_route_heading_errors = [], []
    for lane_id in intersecting_lanes:
        if lane_id in routd_ids.keys():
            lane_object = routd_ids[lane_id]
            lane_discrete_path = lane_object.baseline_path.discrete_path
            lane_state_se2_array = np.array(
                [state.array for state in lane_discrete_path], dtype=np.float64
            )
            # take absolute position of lane-distance, calculate nearest state on baseline
            lane_distances = (ego_position[None, ...] - lane_state_se2_array)**2
            lane_distances = lane_distances.sum(axis=-1)**0.5

            # calculate heading error
            heading_error = (
                lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
            )
            heading_error = np.abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))

            # add lane to candidates
            on_route_lanes.append(lane_object)
            on_route_heading_errors.append(heading_error)

    return on_route_lanes, on_route_heading_errors

def get_discrete_centerline(current_lane, route_block_dict, route_lane_dict, search_depth=30):
    """
    Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
    :param current_lane: lane object of starting lane.
    :param search_depth: depth of search (for runtime), defaults to 30
    :return: list of discrete states on centerline (x,y,θ)
    """
    roadblocks = list(route_block_dict.values())
    roadblock_ids = list(route_block_dict.keys())

    start_idx = np.argmax(
        np.array(roadblock_ids) == current_lane.get_roadblock_id()
    )
    roadblock_window = roadblocks[start_idx : start_idx + search_depth]
    
    graph_search = Dijkstra(current_lane, list(route_lane_dict.keys()))
    route_plan, path_found = graph_search.search(roadblock_window[-1])

    centerline_discrete_path: List[StateSE2] = []
    for lane in route_plan:
        centerline_discrete_path.extend(lane.baseline_path.discrete_path)

    return centerline_discrete_path


def pdm_vectorize(sample, data_path, map_api=None, map_radius=50, 
                  centerline_samples=120, centerline_interval=1.0, 
                  frame_rate=20, past_seconds=2, use_centerline=False):
    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]
    frame_id = sample["frame_id"]
    scenario_type = sample.get("scenario_type", "unknown")
    map_api = map_api[map]
    route_ids = sample["route_ids"].tolist()
    assert len(route_ids) > 0
    pickle_path = os.path.join(data_path, f"{split}", f"{map}", f"{filename}.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            data_dic = pickle.load(f)
            if 'agent_dic' in data_dic:
                agent_dic = data_dic["agent_dic"]
            elif 'agent' in data_dic:
                agent_dic = data_dic['agent']
            else:
                raise ValueError(f'cannot find agent_dic or agent in pickle file, keys: {data_dic.keys()}')
    else:
        print(f"Error: cannot load {filename} from {data_path} with {map}")
        return None
    # convert ego poses to nuplan format (x, y, heading)
    ego_poses = deepcopy(agent_dic["ego"]["pose"][(frame_id - past_seconds * frame_rate)//2:frame_id//2, :])
    ego_shape = agent_dic["ego"]["shape"][0]
    nuplan_ego_poses = [StateSE2(x=ego_pose[0], y=ego_pose[1], heading=ego_pose[-1]) for ego_pose in ego_poses]
    anchor_ego_pose = nuplan_ego_poses[-1]
    ego_position = convert_absolute_to_relative_poses(deepcopy(anchor_ego_pose), nuplan_ego_poses)
    ego_velocities = compute_derivative(deepcopy(ego_position), interval=0.1, drop_z_axis=True) # (bsz, 4) -> (bsz, 3)
    ego_accelerations = compute_derivative(deepcopy(ego_velocities), interval=0.1, drop_z_axis=False) # (bsz, 3) -> (bsz, 3)
    
    cos_, sin_ = math.cos(-ego_poses[-1][-1]), math.sin(-ego_poses[-1][-1])
    absolute_traj = deepcopy(agent_dic["ego"]["pose"][frame_id // 2 : frame_id // 2 + 80, :])
    absolute_traj -= ego_poses[-1]
    traj_x = deepcopy(absolute_traj[:, 0])
    traj_y = deepcopy(absolute_traj[:, 1])
    trajectory_label = np.zeros((absolute_traj.shape[0], 4), dtype=np.float32)
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_
    trajectory_label[:, -1] = absolute_traj[:, -1]
    if trajectory_label.shape[0] < 80:
        trajectory_label = np.concatenate([trajectory_label, np.zeros((80 - trajectory_label.shape[0], 4), dtype=np.float32)], 
                                        axis=0)
    if not use_centerline:
        return dict(
            ego_position=ego_position,
            ego_velocity=ego_velocities,
            ego_acceleration=ego_accelerations,
            scenario_type=scenario_type,
            trajectory_label=trajectory_label.astype(np.float32),
            map=map,
            file_name=filename,
            frame_id=frame_id
        )
    # build drivable area map and extract centerline
    divable_area_map = get_drivable_area_map(map_api, ego_poses[-1], map_radius=map_radius)
    
    # compute centerlines
    _, init_route_dict = load_route_dicts(route_ids, map_api)
    gc.collect()
    gc.disable()
    route_ids = route_roadblock_correction(ego_poses[-1], map_api, init_route_dict)
    route_lane_dict, route_block_dict = load_route_dicts(route_ids, map_api)
    current_lane = get_starting_lane(ego_poses[-1], divable_area_map, route_lane_dict, ego_shape)
    centerline = PDMPath(get_discrete_centerline(current_lane, route_block_dict, route_lane_dict))
    current_progress = centerline.project(Point(*anchor_ego_pose.array))
    centerline_progress_values = (
        np.arange(centerline_samples, dtype=np.float64) * centerline_interval + current_progress
    )
    planner_centerline = convert_absolute_to_relative_se2_array(
        anchor_ego_pose,
        centerline.interpolate(centerline_progress_values, as_array=True),
    )

    return dict(
        ego_position=ego_position,
        ego_velocity=ego_velocities,
        ego_acceleration=ego_accelerations,
        planner_centerline=planner_centerline,
        scenario_type=scenario_type,
        trajectory_label=trajectory_label,
        map=map,
        file_name=filename,
        frame_id=frame_id
    )
