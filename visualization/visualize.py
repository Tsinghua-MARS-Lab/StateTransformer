import streamlit as st
from planning_map import planning_map
import argparse
import pickle
import numpy as np

css = """
<style>
iframe {
    width: 100% !important;
    height: 80vh !important;
}
</style>
"""

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

def parse_args():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('-f', '--saved_dataset_folder', help='Dataset Directory', required=True)
    parser.add_argument('--task', help='nuplan or waymo', default='nuplan')
    parser.add_argument('--agent_type', help='agent type filer for waymo', default='all')
    return parser.parse_args()

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
        else:
            print(f"Error: cannot load {pickle_path} from {root_path} with {map}")
            return None
    else:
        assert False, 'either filename or agent_dic should be provided for online process'
    if 'data_dic' not in st.session_state:
        st.session_state.data_dic = {}
    st.session_state.road_dic = nested_numpy_to_list_for_json(road_dic)
    st.session_state.agent_dic = nested_numpy_to_list_for_json(agent_dic)
    # init by default scenario
    st.session_state.route_ids = _sample['route_ids'].tolist()
    st.session_state.data['road_dic'] = st.session_state.road_dic
    st.session_state.data['agent_dic'] = st.session_state.agent_dic
    st.session_state.data['route_ids'] = st.session_state.route_ids
    st.session_state.data['current_file_name'] = file_name
    st.session_state.data['frame_index'] = int(_sample['frame_id'])  # in 20hz
    st.session_state.data['current_scenario_id'] = str(_sample['scenario_id'])
    print('Data Dic Process Done', list(st.session_state.keys()), list(agent_dic['ego'].keys()))

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

args = parse_args()
# Load data from dataset folder, this logic is identical to the one in `runner.py`.
# We use the datasets library to load the data, which is a wrapper around the HuggingFace datasets library.
import logging, os
from datasets import disable_caching
import sys
sys.path.append('../')
from runner import load_dataset
logger = logging.getLogger("__main__")
disable_caching()
# loop all datasets
logger.info("Loading full set of datasets from {}".format(args.saved_dataset_folder))
assert os.path.isdir(args.saved_dataset_folder), "Dataset folder {} does not exist".format(args.saved_dataset_folder)
assert args.task in ["nuplan", "waymo"], "Task must be either nuplan or waymo"
index_root = os.path.join(args.saved_dataset_folder, 'index')
root_folders = os.listdir(index_root)

# if 'test' in root_folders:
#     test_dataset = load_data(index_root, "test", data_args.agent_type, False)

# Fetch new data for given timestamp and update the app state
# Sidebar content
st.sidebar.title("STR Scenario Visualization")
st.sidebar.write('Current Scenario: ', st.session_state.data['current_scenario_id'])

# Initialize data in the app state
if not st.session_state.scenario_ids_by_file:
    if 'val' in root_folders:
        st.session_state.dataset = load_data(root=index_root, split='val', agent_type=args.agent_type, select=False)
        for i, each_sample in enumerate(st.session_state.dataset):
            file_name = each_sample['file_name']
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

if st.session_state.scenario_ids_by_file:
    if not st.session_state.scenario_ids_by_file_readable:
        st.session_state.scenario_ids_by_file_readable = turn_file_name_to_readable_string(st.session_state.scenario_ids_by_file)
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
    load_dictionary_for_file(args.saved_dataset_folder, 'val', st.session_state.dataset[st.session_state.data['selected_index']])

make_prediction = st.sidebar.button('Make Prediction')

# Render the component with given data, and it will return a new timestamp if animating
st.session_state.running_conditions = planning_map(data=st.session_state.data, key="planning_map")
print('debug tics: ', st.session_state.running_conditions, list(st.session_state.data.keys()), st.session_state.data['current_scenario_id'])