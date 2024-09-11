import scipy
import numpy as np
import matplotlib.pyplot as plt
from nuplan_simulation.common_utils import *

from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType, STATIC_OBJECT_TYPES
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.metrics.utils.expert_comparisons import principal_value


def check_path(path):
    refine_path = [path[0]]

    for i in range(1, path.shape[0]):
        if np.linalg.norm(path[i] - path[i - 1]) < 0.1:
            continue
        else:
            refine_path.append(path[i])

    line = np.array(refine_path)

    return line


def calculate_path_heading(path):
    heading = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
    heading = np.append(heading, heading[-1])

    return heading


def trajectory_smoothing(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    h = trajectory[:, 2]

    window_length = 40
    x = scipy.signal.savgol_filter(x, window_length=window_length, polyorder=3)
    y = scipy.signal.savgol_filter(y, window_length=window_length, polyorder=3)
    h = scipy.signal.savgol_filter(h, window_length=window_length, polyorder=3)

    return np.column_stack([x, y, h])


def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
