import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import geopandas as gpd
import joblib
import json
import folium

# Load edges_with_MAPE.csv
@st.cache_data
def load_edges_with_mape():
    return pd.read_csv("./data/edges_with_MAPE.csv")

# Load state coordinates from CSV
@st.cache_data
def load_state_coordinates():
    return pd.read_csv("./data/state_coordinates.csv")  # Make sure this CSV has 'STUSPS', 'latitude', 'longitude'

# Function to visualize the network
def visualize_network(edges_df, state_coords):
    G = nx.DiGraph()
    
    # Add nodes with MAPE as an attribute
    for _, row in edges_df.iterrows():
        G.add_node(row['node1'], MAPE=row['MAPE_node1'])
        G.add_node(row['node2'], MAPE=row['MAPE_node2'])
        G.add_edge(row['node1'], row['node2'], weight=row['abs_diff'])
    
    # Node colors based on MAPE
    node_colors = [G.nodes[node]['MAPE'] for node in G.nodes]
    
    # Format edge labels to two decimal places
    edge_labels = {
        (u, v): f"{d['weight']:.4f}" for u, v, d in G.edges(data=True)
    }
    
    # Draw the network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.75, iterations=300)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=1000,
        edge_color="gray",
        font_size=10,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8
    )
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label="MAPE")
    st.pyplot(plt)

# Function to visualize map with nodes and edges on Folium
def visualize_map_with_edges(state_coords, edges_df):
    # Create a folium map centered on the US
    us_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

    # Create markers for each state based on latitude and longitude
    for _, row in state_coords.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=row['STUSPS']
        ).add_to(us_map)

    # Draw edges between states (nodes) using lines
    for _, row in edges_df.iterrows():
        node1 = row['node1']
        node2 = row['node2']

        # Get coordinates for node1 and node2
        lat1, lon1 = state_coords[state_coords['STUSPS'] == node1][['latitude', 'longitude']].values[0]
        lat2, lon2 = state_coords[state_coords['STUSPS'] == node2][['latitude', 'longitude']].values[0]

        # Draw a line between the two states
        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color='gray', weight=2, opacity=0.6).add_to(us_map)

    # Display the map in Streamlit
    st.subheader("US States Map with Nodes and Edges")
    folium_static(us_map)

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

    # Visualization of network
    st.subheader("Network Visualization of Node Relationships")
    st.write("This graph shows the relationships between nodes with MAPE values and their absolute differences.")
    visualize_network(edges_df, state_coords)

    # Visualization of map with nodes and edges
    visualize_map_with_edges(state_coords, edges_df)

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
