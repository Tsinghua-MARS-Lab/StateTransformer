import streamlit as st
from planning_map import planning_map
import argparse
import sys

css = """
<style>
iframe {
    width: 100% !important;
    height: 80vh !important;
}
</style>
"""

# Set basic page config and style
st.set_page_config(layout="wide")
st.markdown(css, unsafe_allow_html=True)



def fetch_new_data(timestamp):
    """Fetch new data from backend for given timestamp."""
    return {"greeting": f"Hello! {timestamp=}", "width": timestamp}

@st.cache_data
def load_data(dataset_folder):
    print('test: ', dataset_folder)
    return dataset_folder

@st.cache_data
def parse_args():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('-p', '--path', help='Dataset Directory', required=True)
    return parser.parse_args()

args = parse_args()
loaded_dictionary = load_data(args.path)

# Initialize data in the app state
if "data" not in st.session_state:
    st.session_state.data = fetch_new_data(0)

# Render the component with given data, and it will return a new timestamp if animating
timestamp = planning_map(data=st.session_state.data, key="planning_map")

# Fetch new data for given timestamp and update the app state
st.session_state.data = fetch_new_data(timestamp)

# Sidebar content
st.sidebar.title("Map Visualization")
