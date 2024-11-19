import streamlit as st
import pandas as pd
import joblib
import json

@st.cache_data
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

evaluation_results = load_evaluation_results()
models = load_models()

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