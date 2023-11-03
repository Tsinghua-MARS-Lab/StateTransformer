import os
import streamlit.components.v1 as components

# Change this if you want to bring this component to production
__RELEASE__ = False

if not __RELEASE__:
    _component_func = components.declare_component(
        "planning_map",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("planning_map", path=build_dir)


def planning_map(key=None, **kwargs):
    return _component_func(key=key, **kwargs)
