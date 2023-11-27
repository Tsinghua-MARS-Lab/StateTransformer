import copy
import math
import multiprocessing as mp
import streamlit as st
from planning_map import planning_map
import argparse
import pickle
import numpy as np
# Load data from dataset folder, this logic is identical to the one in `runner.py`.
# We use the datasets library to load the data, which is a wrapper around the HuggingFace datasets library.
import os
from datasets import disable_caching
import sys
from runner import load_dataset
from transformer4planning.trainer import convert_names_to_ids

css = """
<style>
iframe {
    width: 100% !important;
    height: 80vh !important;
}
</style>
"""

def parse_args():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('-f', '--saved_dataset_folder', help='Dataset Directory', required=True)
    parser.add_argument('--task', help='nuplan or waymo', default='nuplan')
    parser.add_argument('--agent_type', help='agent type filer for waymo', default='all')
    parser.add_argument('-c', '--model_checkpoint_path', help='path of the model checkpoint', default=None)
    parser.add_argument('-l', '--training_log_path', help='path of the log', default=None)
    parser.add_argument('-m', '--mode', help='choose from {scenario}, {file}', default='scenario')
    return parser.parse_args()


class LoadedModelArguments(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def np_to_tensor(dic):
    for key in dic:
        if isinstance(dic[key], dict):
            dic[key] = np_to_tensor(dic[key])
        elif isinstance(dic[key], np.ndarray):
            dic[key] = torch.tensor(dic[key]).unsqueeze(0)
    return dic

sys.path.append('../')
args = parse_args()
disable_caching()
# loop all datasets
assert os.path.isdir(args.saved_dataset_folder), "Dataset folder {} does not exist".format(args.saved_dataset_folder)
assert args.task in ["nuplan", "waymo"], "Task must be either nuplan or waymo"
print("Loading full set of datasets from {}".format(args.saved_dataset_folder))
index_root = os.path.join(args.saved_dataset_folder, 'index')
root_folders = os.listdir(index_root)

map_name_dic = {
    'us-pa-pittsburgh-hazelwood': 'Pittsburgh',
    'us-nv-las-vegas-strip': 'Las Vegas',
    'us-ma-boston': 'Boston',
    'sg-one-north': 'Singapore',
}

# Set basic page config and style
st.set_page_config(layout="wide")
st.markdown(css, unsafe_allow_html=True)

# initialization of the python session cache data
if 'all_map_dic' not in st.session_state:
    st.session_state.all_map_dic = {}
if 'road_dic' not in st.session_state:
    st.session_state.road_dic = {}
if 'agent_dic' not in st.session_state:
    st.session_state.agent_dic = {}
if 'route_ids' not in st.session_state:
    st.session_state.route_ids = []
if 'data' not in st.session_state:
    st.session_state.data = {
        'selected_index': 0,
        'current_file_name': 'initializating',
        'frame_index': 0,
        'current_scenario_id': 'initializating',
    }
if 'scenario_ids_by_file' not in st.session_state:
    st.session_state.scenario_ids_by_file = {}
if 'scenario_ids_current_file' not in st.session_state:
    st.session_state.scenario_ids_current_file = []
if 'scenario_ids_by_file_readable' not in st.session_state:
    st.session_state.scenario_ids_by_file_readable = []
if 'scenario_ids_current_file_readable' not in st.session_state:
    st.session_state.scenario_ids_current_file_readable = []
if 'running_conditions' not in st.session_state:
    st.session_state.running_conditions = 'initializating'
if 'dataset' not in st.session_state:
    st.session_state.dataset = 'undefined'


def nested_numpy_to_list_for_json(dic):
    """Convert numpy array to list for json dump. Streamlit use json to communicate, so this is compulsory."""
    for each_key in dic:
        if isinstance(dic[each_key], dict):
            nested_numpy_to_list_for_json(dic[each_key])
        elif isinstance(dic[each_key], list):
            for each_obj in dic[each_key]:
                if isinstance(each_obj, dict):
                    nested_numpy_to_list_for_json(each_obj)
                elif isinstance(each_obj, np.ndarray):
                    dic[each_key] = dic[each_key].tolist()
        elif isinstance(dic[each_key], np.ndarray):
            dic[each_key] = dic[each_key].tolist()
    return dic

@st.cache_data
def load_data(root, split='train', dataset_scale=1, agent_type="all", select=False):
    """Load data (the indices) from dataset folder."""
    dataset = load_dataset(root=root, split=split, dataset_scale=dataset_scale, agent_type=agent_type, select=select)
    print('dataset sample keys: ', dataset[0].keys())
    return dataset

# @st.cache_data
def load_dictionary_for_file(root_path, split, _sample):
    print('Processing Data Dic')
    file_name = _sample['file_name']
    map = _sample['map']
    if map not in st.session_state.all_map_dic:
        if os.path.exists(os.path.join(root_path, "map", f"{map}.pkl")):
            with open(os.path.join(root_path, "map", f"{map}.pkl"), "rb") as f:
                road_dic = pickle.load(f)
            st.session_state.all_map_dic[map] = road_dic
        else:
            print(f"Error: cannot load map {map} from {root_path}")
            return None
    else:
        road_dic = st.session_state.all_map_dic[map]
    if split == 'val':
        map = 'all_cities'
    if file_name is not None:
        pickle_path = os.path.join(root_path, f"{split}", f"{map}", f"{file_name}.pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data_dic = pickle.load(f)
                if 'agent_dic' in data_dic:
                    agent_dic = data_dic["agent_dic"]
                elif 'agent' in data_dic:
                    agent_dic = data_dic['agent']
                else:
                    raise ValueError(f'cannot find agent_dic or agent in pickle file, keys: {data_dic.keys()}')
                # agent_dic = add_intention_to_agent_dic(agent_dic)
        else:
            print(f"Error: cannot load {pickle_path} from {root_path} with {map}")
            return None
    else:
        assert False, 'either filename or agent_dic should be provided for online process'
    if 'data_dic' not in st.session_state:
        st.session_state.data_dic = {}
    st.session_state.road_dic = nested_numpy_to_list_for_json(copy.deepcopy(road_dic))
    st.session_state.agent_dic = nested_numpy_to_list_for_json(copy.deepcopy(agent_dic))
    # init by default scenario
    st.session_state.route_ids = _sample['route_ids'].tolist()
    st.session_state.data['road_dic'] = st.session_state.road_dic
    st.session_state.data['agent_dic'] = st.session_state.agent_dic
    st.session_state.data['route_ids'] = st.session_state.route_ids
    st.session_state.data['current_file_name'] = file_name
    st.session_state.data['frame_index'] = int(_sample['frame_id'])  # in 20hz
    st.session_state.data['current_scenario_id'] = str(_sample['scenario_id'])
    print('Data Dic Process Done', list(st.session_state.keys()), list(agent_dic['ego'].keys()), type(agent_dic['ego']['pose']))
    return agent_dic, road_dic

def turn_file_scenarios_to_readable_strings(dictionary):
    list_to_return = []
    for each_scenario_dic in dictionary:
        # each_scenario_dic['scenario_id'][:12] + ' - ' +
        list_to_return.append(each_scenario_dic['scenario_type'] + ' - ' + str(each_scenario_dic['frame_id']))
    return list_to_return

def turn_file_name_to_readable_string(dictionary):
    list_to_return = []
    for file_name in dictionary:
        each_dic_list = dictionary[file_name]
        map_name = each_dic_list[0]['map']
        if map_name not in map_name_dic:
            print('Map name not found: ', map_name)
        list_to_return.append(file_name.split('_')[0] + ' - ' + map_name_dic[map_name])
    return list_to_return

def ego_to_global(poses, ego_pose, y_reverse=1):
    poses = copy.deepcopy(poses)
    poses[:, 1] *= y_reverse
    rotated_poses = np.zeros_like(poses)
    cos_, sin_ = math.cos(ego_pose[3]), math.sin(ego_pose[3])
    rotated_poses[:, 0] = poses[:, 0] * cos_ - poses[:, 1] * sin_
    rotated_poses[:, 1] = poses[:, 0] * sin_ + poses[:, 1] * cos_
    rotated_poses[:, 0] += ego_pose[0]
    rotated_poses[:, 1] += ego_pose[1]
    rotated_poses[:, 3] += -ego_pose[3]
    return rotated_poses

def add_intention_to_agent_dic(agent_dic):
    total_frames = agent_dic['ego']['pose'].shape[0]
    intentions = np.zeros((total_frames, 1))
    intentions[:40] = 0
    intentions[-40:] = 0
    for i in range(40, total_frames - 40):
        ego_poses = copy.deepcopy(agent_dic['ego']['pose'])
        current_pose = ego_poses[i]
        # normalize at current pose
        cos_, sin_ = math.cos(-current_pose[3]), math.sin(-current_pose[3])
        ego_poses -= current_pose
        rotated_poses = [ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                         ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_]
        rotated_poses = np.stack(rotated_poses, axis=1)
        assert rotated_poses[i, 0] == 0 and rotated_poses[i, 1] == 0, f'rotated pose not zero at 40: {rotated_poses[i]}'
        # yaw in 1 s
        future_yaw = np.mean(rotated_poses[i + 5: i + 15, -1])
        # normalize yaw angle to [-pi, pi]
        if future_yaw > math.pi:
            future_yaw -= 2 * math.pi
        elif future_yaw < -math.pi:
            future_yaw += 2 * math.pi
        yaw_threshold = 0.5

        velocity = rotated_poses[i - 10:i + 10, :2] - rotated_poses[i - 40:i - 20,:2]
        estimated_pose = rotated_poses[i - 10:i + 10, :2] + velocity
        delta_pose = np.mean(estimated_pose - rotated_poses[i + 20:i + 40, :2], axis=0)
        # y_threshold = 4
        x_threshold = 5

        if future_yaw > yaw_threshold:
            intentions[i] = 0  # left
        elif future_yaw < -yaw_threshold:
            intentions[i] = 1  # right
        # if delta_pose[1] > y_threshold:
        #     intentions[i] = 0  # left
        # elif delta_pose[1] < -y_threshold:
        #     intentions[i] = 1  # right
        elif delta_pose[0] > x_threshold:
            intentions[i] = 3  # accelerate
        elif delta_pose[0] < -x_threshold:
            intentions[i] = 2  # decelerate
        else:
            intentions[i] = 4  # keep
    agent_dic['ego']['intention'] = intentions
    return agent_dic

# if 'test' in root_folders:
#     test_dataset = load_data(index_root, "test", data_args.agent_type, False)

# Fetch new data for given timestamp and update the app state
# Sidebar content
st.sidebar.title("STR Scenario Visualization")

# Initialize data in the app state
if not st.session_state.scenario_ids_by_file:
    st.session_state.new_frame = 'Initializating...'
    if 'val' in root_folders:
        st.session_state.dataset = load_data(root=index_root, split='val', agent_type=args.agent_type, select=False)
        if args.training_log_path is not None:
            # load training log from pickle
            with open(args.training_log_path, 'rb') as f:
                training_log = pickle.load(f)
            # st.session_state.dataset = st.session_state.dataset.filter(lambda example: convert_names_to_ids(
            #     file_names=[example['file_name']],t0_frame_ids=[example['frame_id']]
            # )[0] in training_log['miss_scenarios'], num_proc=mp.cpu_count())
            print('after filter: ', len(st.session_state.dataset))
        for i, each_sample in enumerate(st.session_state.dataset):
            file_name = each_sample['file_name']
            if args.training_log_path is not None:
                # filter
                converted_id = convert_names_to_ids(file_names=[file_name], t0_frame_ids=[each_sample['frame_id']])[0]
                if converted_id not in training_log['miss_scenarios']:
                    continue
            if file_name not in st.session_state.scenario_ids_by_file:
                st.session_state.scenario_ids_by_file[file_name] = []
            st.session_state.scenario_ids_by_file[file_name].append({
                'scenario_id': each_sample['scenario_id'],
                'map': each_sample['map'],
                'frame_id': int(each_sample['frame_id']),
                'scenario_type': each_sample['scenario_type'],
                'route_ids': each_sample['route_ids'],
                'dataset_index': i
            })
    print('Updating data', st.session_state.running_conditions)

agent_dic = None
road_dic = None

if st.session_state.scenario_ids_by_file:
    if not st.session_state.scenario_ids_by_file_readable:
        st.session_state.scenario_ids_by_file_readable = turn_file_name_to_readable_string(st.session_state.scenario_ids_by_file)
    st.sidebar.write('Current Scenario: ', st.session_state.data['current_scenario_id'])
    selected_file_name_readable = st.sidebar.selectbox('Select a file', st.session_state.scenario_ids_by_file_readable)
    selected_file_name_index = st.session_state.scenario_ids_by_file_readable.index(selected_file_name_readable)
    selected_file_name = list(st.session_state.scenario_ids_by_file.keys())[selected_file_name_index]

    st.session_state.scenario_ids_current_file_readable = turn_file_scenarios_to_readable_strings(st.session_state.scenario_ids_by_file[selected_file_name])
    selected_scenario_readable = st.sidebar.selectbox('Select a scenario', st.session_state.scenario_ids_current_file_readable)
    selected_scenario_index = st.session_state.scenario_ids_current_file_readable.index(selected_scenario_readable)
    selected_scenario_dic = st.session_state.scenario_ids_by_file[selected_file_name][selected_scenario_index]
    # random_scenario_dic = st.session_state.scenario_ids_by_file[selected_file_name][0]
    # st.session_state.data['selected_index'] = random_scenario_dic['dataset_index']

    st.session_state.data['selected_index'] = selected_scenario_dic['dataset_index']
    st.session_state.data['current_scenario_id'] = st.session_state.dataset[st.session_state.data['selected_index']]['scenario_id']
    agent_dic, road_dic = load_dictionary_for_file(args.saved_dataset_folder, 'val', st.session_state.dataset[st.session_state.data['selected_index']])
else:
    st.sidebar.write('Current Scenario: ', st.session_state.data['current_scenario_id'])


make_prediction = st.sidebar.button('Make Prediction')
st.sidebar.write('Prediction frame: ', st.session_state.new_frame)

if make_prediction:
    print('making prediction')
    import json, torch
    from transformer4planning.models.model import build_models
    from transformer4planning.preprocess.nuplan_rasterize import static_coor_rasterize
    if args.model_checkpoint_path is None:
        st.sidebar.write('No model path given. Please specify a model checkpoint path and rerun the script')
    else:
        # get config.json from checkpoint folder
        config_path = os.path.join(args.model_checkpoint_path, 'config.json')
        with open(config_path) as f:
            model_args = json.load(f)
        print('loaded model args: ', model_args)
        model_args = LoadedModelArguments(model_args)
        model_args.model_name = model_args.model_name.replace('scratch', 'pretrained')
        model_args.model_pretrain_name_or_path = args.model_checkpoint_path
        model = build_models(model_args)
        current_data = st.session_state.dataset[st.session_state.data['selected_index']]
        # update frame index
        if st.session_state.new_frame is not None:
            current_data['frame_id'] = st.session_state.new_frame * 2
        current_frame = int(current_data['frame_id'])
        map_name = current_data['map']
        prepared_data = static_coor_rasterize(sample=current_data,
                                              data_path=args.saved_dataset_folder,
                                              agent_dic=agent_dic,
                                              all_maps_dic={map_name: road_dic})
        prepared_data = np_to_tensor(prepared_data)

        with torch.no_grad():
            prediction_generation = model.generate(**prepared_data)
            print('prediction_generation: ', list(prediction_generation.keys()), st.session_state.data['selected_index'], current_frame)
            from transformer4planning.trainer import save_raster
            image_dictionary = save_raster(inputs=prepared_data, sample_index=0,
                                           prediction_trajectory_by_gen=prediction_generation['traj_logits'][0],
                                           prediction_key_point_by_gen=prediction_generation['key_points_logits'][0])
            with st.sidebar.expander('raster_visualization', expanded=True):
                for each_key in image_dictionary:
                    image = image_dictionary[each_key]
                    st.image(image/255, caption=each_key, use_column_width=True, clamp=True, channels='BGR')
            st.session_state.data['prediction_generation'] = ego_to_global(
                prediction_generation['traj_logits'][0].numpy(),
                agent_dic['ego']['pose'][current_frame // 2], y_reverse=-1 if map_name == 'sg-one-north' else 1).tolist()
            st.session_state.data['pred_kp_generation'] = ego_to_global(
                prediction_generation['key_points_logits'][0].numpy(),
                agent_dic['ego']['pose'][current_frame // 2], y_reverse=-1 if map_name == 'sg-one-north' else 1).tolist()

# Render the component with given data, and it will return a new timestamp if animating
st.session_state.new_frame = planning_map(data=st.session_state.data, key="planning_map")  # in 10Hz or None
print('loop finished: ', st.session_state.new_frame, list(st.session_state.data.keys()), st.session_state.data['current_scenario_id'])