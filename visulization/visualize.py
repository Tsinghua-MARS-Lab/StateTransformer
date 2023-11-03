import streamlit as st
from planning_map import planning_map

css = """
<style>
iframe {
    width: 100% !important;
    height: 80vh !important;
}
</style>
"""


def fetch_new_data(timestamp):
    """Fetch new data from backend for given timestamp."""
    return {"greeting": f"Hello! {timestamp=}", "width": timestamp}


# Initialize data in the app state
if "data" not in st.session_state:
    st.session_state.data = fetch_new_data(0)

# Set basic page config and style
st.set_page_config(layout="wide")
st.markdown(css, unsafe_allow_html=True)

# Render the component with given data, and it will return a new timestamp if animating
timestamp = planning_map(data=st.session_state.data, key="planning_map")

# Fetch new data for given timestamp and update the app state
st.session_state.data = fetch_new_data(timestamp)

# Sidebar content
st.sidebar.title("Map Visualization")
