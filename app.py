import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import geopandas as gpd
import joblib
import json
import folium
import zipfile
import os 

def extract_zip(filename):
    csv_path = f"./data/{filename}.csv"
    zip_path = f"./data/{filename}.zip"

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall("./data/")
        print(f"Files extracted to: {csv_path}")

    data = pd.read_csv(f"./data/{filename}.csv")

    os.remove(csv_path)
    return data

def get_state_boundaries(state, state_geojson):
    # Fetch state boundary coordinates from geojson
    state_data = state_geojson[state_geojson['state'] == state]
    if not state_data.empty:
        boundary = state_data['geometry'].values[0]
        return Polygon(boundary['coordinates'][0])
    return None

# Function to check if the node is within the state's boundary
def is_within_boundary(lat, lon, state, state_geojson):
    state_boundary = get_state_boundaries(state, state_geojson)
    if state_boundary:
        point = Point(lon, lat)
        return state_boundary.contains(point)
    return False

# Load edges_with_MAPE.csv
@st.cache_data
def load_edges_with_mape():
    return extract_zip("edges_with_MAPE")

# Load state coordinates from CSV
@st.cache_data
def load_state_coordinates():
    return extract_zip("state_coordinates")

@st.cache_data
def load_forcast_with_state():
    return extract_zip("EIA930LoadAndForecast_with_states")


# Function to visualize map with nodes and edges on Folium
def visualize_map_with_edges(state_coords, edges_df, load_forecast_df):
    # Create a folium map centered on the US
    us_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

    # Create a lookup dictionary for state coordinates
    coords_dict = state_coords.set_index('state')[['latitude', 'longitude']].to_dict('index')

    # Extract unique nodes and map them to states using the load forecast data
    nodes = set(edges_df['node1']).union(set(edges_df['node2']))
    node_to_state = load_forecast_df.set_index('respondent')['state'].to_dict()
    # Map nodes to their respective states
    node_states = {node: node_to_state.get(node) for node in nodes if node in node_to_state}
    # Group nodes by state
    state_to_nodes = {}
    node_offsets = {}
    for node, state in node_states.items():
        if state in coords_dict:
            if state not in state_to_nodes:
                state_to_nodes[state] = []
            state_to_nodes[state].append(node)

# Add markers for each valid node
    offset = .5  # Increased offset value for latitude/longitude
    node_radius = 7  # Increased radius for larger markers
    for node, state in node_states.items():
        if state in coords_dict:
            if state not in state_to_nodes:
                state_to_nodes[state] = []

            # Calculate offset for the node based on its index in the state's nodes
            index = len(state_to_nodes[state])
            lat_offset = coords_dict[state]['latitude'] + (index % 3) * offset
            lon_offset = coords_dict[state]['longitude'] + (index // 3) * offset

            # Add the node's offset to the mapping
            node_offsets[node] = (lat_offset, lon_offset)

            # Add the node to the state's list
            state_to_nodes[state].append(node)

            # Add a marker for the node
            folium.CircleMarker(
                location=[lat_offset, lon_offset],
                radius=node_radius,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.8,
                popup=f"{node} ({state})"
            ).add_to(us_map)

    # Add edges between valid nodes using offset positions
    for _, row in edges_df.iterrows():
        node1, node2 = row['node1'], row['node2']

        if node1 in node_offsets and node2 in node_offsets:
            lat1, lon1 = node_offsets[node1]
            lat2, lon2 = node_offsets[node2]

            # Draw the edge using the offset positions
            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color='gray',
                weight=3,  # Slightly thicker edges
                opacity=0.6
            ).add_to(us_map)
        else:
            st.warning(f"Missing offset mapping for nodes: {node1}, {node2}")

    # Display the map in Streamlit
    st.subheader("US States Map with Nodes and Edges")
    st_folium(us_map, width=800, height=600)  # Adjust map dimensions

# Load evaluation results from the saved JSON file
def load_evaluation_results():
    with open("./data/evaluation_results.json", "r") as file:
        return json.load(file)

# Load different models
@st.cache_resource
def load_models():
    linear_regression = joblib.load("./models/linear_regression_model.pkl")
    random_forest = joblib.load("./models/random_forest_model.pkl")
    gradient_boosting_model = joblib.load("./models/gradient_boosting_model.pkl")
    return linear_regression, random_forest, gradient_boosting_model

# Function to select the best model based on MAPE
def select_best_model(evaluation_results):
    best_model_name = min(evaluation_results, key=lambda k: evaluation_results[k]["MAPE"])
    return best_model_name

# Function to make predictions using the model
def predict_ldwp(model, input_data):
    # Ensure input_data is a DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit App Layout
def main():
    st.title("EIA 930 Demand Forecast Analysis with Prediction")

    # Load the evaluation results and models
    evaluation_results = load_evaluation_results()
    models = load_models()
    edges_df = load_edges_with_mape()
    state_coords = load_state_coordinates()
    respondent_with_state = load_forcast_with_state()

    # Visualization of network
    st.subheader("Network Visualization of Node Relationships")
    st.write("This graph shows the relationships between nodes with MAPE values and their absolute differences.")
    # visualize_network(edges_df, state_coords)

    # Visualization of map with nodes and edges
    visualize_map_with_edges(state_coords, edges_df, respondent_with_state)

    # Select the best model based on MAPE
    best_model_name = select_best_model(evaluation_results)
    best_model = models[
        ["Linear Regression", "Random Forest", "XGBoost"].index(best_model_name)
    ]

    # Display the selected model and its performance
    st.subheader("Selected Best Model")
    st.write(f"**Model:** {best_model_name}")
    st.write(f"**MAPE:** {evaluation_results[best_model_name]['MAPE']:.4f}")
    st.write(f"**Cross-Validation Mean Score:** {evaluation_results[best_model_name]['Cross-Validation Mean']:.4f}")

    # Display other models and their performance
    st.subheader("Other Models and Their MAPE")
    for model_name, metrics in evaluation_results.items():
        if model_name != best_model_name:
            st.write(f"**{model_name}:**")
            st.write(f"MAPE: {metrics['MAPE']:.4f}")
            st.write(f"Cross-Validation Mean Score: {metrics['Cross-Validation Mean']:.4f}")
            st.write("---")

    # Input features for prediction
    st.subheader("Predict LDWP Demand")
    st.write("Enter the feature values below to predict the LDWP demand:")

    # Feature inputs
    LDWP_lag1 = st.number_input("LDWP Lag 1 (previous period)", value=0.0, step=0.1)
    LDWP_lag24 = st.number_input("LDWP Lag 24 (24 hours prior)", value=0.0, step=0.1)
    CISO = st.number_input("CISO", value=0.0, step=0.1)
    BPAT = st.number_input("BPAT", value=0.0, step=0.1)
    PACE = st.number_input("PACE", value=0.0, step=0.1)
    NEVP = st.number_input("NEVP", value=0.0, step=0.1)
    AZPS = st.number_input("AZPS", value=0.0, step=0.1)
    WALC = st.number_input("WALC", value=0.0, step=0.1)

    # Prepare input data
    input_data = {
        "LDWP_lag1": LDWP_lag1,
        "LDWP_lag24": LDWP_lag24,
        "CISO": CISO,
        "BPAT": BPAT,
        "PACE": PACE,
        "NEVP": NEVP,
        "AZPS": AZPS,
        "WALC": WALC
    }

    # Make prediction
    if st.button("Predict"):
        prediction = predict_ldwp(best_model, input_data)
        st.success(f"Predicted LDWP Demand: {prediction:.2f}")

if __name__ == "__main__":
    main()
