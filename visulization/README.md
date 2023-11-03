# Visualization

## Prerequisites

To setup the visualization app with Streamlit in both backend and frontend side, run the following commands:

```bash
pip install streamlit
cd planning_map/frontend
npm install
```

## Development

In development stage, the map frontend needs to be bootstrapped separately. To do so, run the following command first:

```bash
cd planning_map/frontend
npm run start
```

Then, run the following command to start the app in another terminal:

```bash
streamlit run visualize.py
```

## Production

Before using the app in practice, the frontend needs to be built:

```bash
cd planning_map/frontend
npm run build
```

Then, change the `__RELEASE__` variable in the `planning_map/__init__.py`:

```python
# Change this if you want to bring this component to production
__RELEASE__ = True
```

Finally, run the following command to start the app:

```bash
streamlit run visualize.py
```

## Key Files

- `visualize.py`: the main file of the visualization app
- `planning_map/`: the folder containing the map component
  - `__init__.py`: the backend side of the component
  - `frontend/`: the frontend side of the component
    - `src/index.ts`: the render logic in TypeScript (JavaScript)
    - `public/index.html`: the placeholder HTML file
