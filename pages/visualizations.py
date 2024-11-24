import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import folium
import zipfile
import os

def extract_zip(filename, data_dir="./data/"):
    csv_path = f"{data_dir}{filename}.csv"
    zip_path = f"{data_dir}{filename}.zip"

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(data_dir)

    data = pd.read_csv(csv_path)
    os.remove(csv_path)
    return data


@st.cache_data
def load_edges_with_mape():
    return extract_zip("edges_with_MAPE")


@st.cache_data
def load_state_coordinates():
    return extract_zip("state_coordinates")


@st.cache_data
def load_forecast_with_state():
    return extract_zip("EIA930LoadAndForecast_with_states")


@st.cache_data
def load_ba_lat_long():
    return extract_zip("Balancing_Authority_Lat_Long")

def visualize_map_with_edges(edges_df, ba_lat_long):
    us_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

    # Create a lookup dictionary for latitude and longitude by ba_code
    ba_coords = ba_lat_long.set_index('ba_code')[['lat', 'lng']].to_dict('index')

    # Map nodes to coordinates based on ba_code
    nodes = set(edges_df['node1']).union(set(edges_df['node2']))
    node_coords = {node: ba_coords.get(node) for node in nodes if node in ba_coords}

    # Add markers for nodes
    for node, coords in node_coords.items():
        if coords:
            folium.CircleMarker(
                location=[coords['lat'], coords['lng']],
                radius=7,  # Node marker radius
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.8,
                popup=f"{node}"
            ).add_to(us_map)

    # Draw edges between valid nodes
    for _, row in edges_df.iterrows():
        node1, node2 = row['node1'], row['node2']
        if node1 in node_coords and node2 in node_coords:
            lat1, lon1 = node_coords[node1]['lat'], node_coords[node1]['lng']
            lat2, lon2 = node_coords[node2]['lat'], node_coords[node2]['lng']

            tooltip = (
                f"Respondent 1: {row['respondent_x']}<br>"
                f"MAPE Node 1: {row['MAPE_node1']:.3f}<br>"
                f"Respondent 2: {row['respondent_y']}<br>"
                f"MAPE Node 2: {row['MAPE_node2']:.3f}<br>"
                f"Absolute Difference: {row['abs_diff'] * 100:.2f}%"
            )

            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color='gray',
                weight=3,
                opacity=0.6,
                tooltip=tooltip
            ).add_to(us_map)

    # Render the map in Streamlit
    st.subheader("US States Map with Nodes and Edges")
    st_folium(us_map, width=800, height=600)

# Main app
st.title("EIA 930 Demand Forecast Visualization")

# Load data
edges_df = load_edges_with_mape()
ba_lat_long = load_ba_lat_long()

# Map visualization
visualize_map_with_edges(edges_df, ba_lat_long)
