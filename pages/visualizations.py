import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
from shapely.geometry import Point
import folium
import zipfile
import os
import tempfile


# Utility function to extract CSV from a ZIP file
def extract_zip(filename, data_dir="./data/"):
    with zipfile.ZipFile(f"{data_dir}{filename}.zip", 'r') as zipf:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_csv:
            zipf.extract(f"{filename}.csv", tmp_csv.name)
            data = pd.read_csv(tmp_csv.name)
    return data


# Cached data loaders
@st.cache_data
def load_edges_with_mape():
    return extract_zip("edges_with_MAPE")


@st.cache_data
def load_state_coordinates():
    return extract_zip("state_coordinates")


@st.cache_data
def load_forecast_with_state():
    return extract_zip("EIA930LoadAndForecast_with_states")


# Function to visualize map with nodes and edges on Folium
def visualize_map_with_edges(state_coords, edges_df, load_forecast_df):
    # Initialize Folium map
    us_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

    # Create a lookup dictionary for state coordinates
    coords_dict = state_coords.set_index('state')[['latitude', 'longitude']].to_dict('index')

    # Map nodes to their respective states using load forecast data
    node_to_state = load_forecast_df.set_index('respondent')['state'].to_dict()
    nodes = set(edges_df['node1']).union(set(edges_df['node2']))
    node_states = {node: node_to_state.get(node) for node in nodes if node in node_to_state}

    # Prepare node offsets and group nodes by state
    state_to_nodes, node_offsets = {}, {}
    offset, node_radius = 0.5, 7  # Offset for latitude/longitude and marker radius

    for node, state in node_states.items():
        if state in coords_dict:
            if state not in state_to_nodes:
                state_to_nodes[state] = []

            # Compute position offset based on the node index in its state
            index = len(state_to_nodes[state])
            lat_offset = coords_dict[state]['latitude'] + (index % 3) * offset
            lon_offset = coords_dict[state]['longitude'] + (index // 3) * offset

            # Store offset and add marker
            node_offsets[node] = (lat_offset, lon_offset)
            state_to_nodes[state].append(node)
            folium.CircleMarker(
                location=[lat_offset, lon_offset],
                radius=node_radius,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.8,
                popup=f"{node} ({state})"
            ).add_to(us_map)

    # Draw edges between valid nodes
    for _, row in edges_df.iterrows():
        node1, node2 = row['node1'], row['node2']
        if node1 in node_offsets and node2 in node_offsets:
            folium.PolyLine(
                [node_offsets[node1], node_offsets[node2]],
                color='gray',
                weight=3,
                opacity=0.6
            ).add_to(us_map)
        else:
            st.warning(f"Missing offset mapping for nodes: {node1}, {node2}")

    # Render the map in Streamlit
    st.subheader("US States Map with Nodes and Edges")
    st_folium(us_map, width=800, height=600)


# Main app
st.title("EIA 930 Demand Forecast Visualization")

# Load data
edges_df = load_edges_with_mape()
state_coords = load_state_coordinates()
respondent_with_state = load_forecast_with_state()

# Map visualization
visualize_map_with_edges(state_coords, edges_df, respondent_with_state)
