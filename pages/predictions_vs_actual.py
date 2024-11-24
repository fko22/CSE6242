import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load the evaluation results including base models and stacking models
with open("./data/evaluation_results.json", "r") as file:
    evaluation_results = json.load(file)

# Convert evaluation results into a DataFrame for easier visualization
evaluation_df = pd.DataFrame(evaluation_results).T

# Display MAPE comparison bar plot (Base Models + Stacking Models)
st.subheader("Model Performance (MAPE Comparison)")

fig, ax = plt.subplots(figsize=(10, 6))
bars = evaluation_df["MAPE"].plot(kind='bar', ax=ax, color='skyblue')
evaluation_df["MAPE"].plot(kind='bar', ax=ax, color='skyblue')
ax.set_title("Comparison of MAPE for Base and Stacking Models")
ax.set_ylabel("MAPE")
ax.set_xticklabels(evaluation_df.index, rotation=45, ha='right')

# Display numbers on top of the bars
for bar in bars.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # X position
        bar.get_height() ,  # Y position (slightly above the bar)
        f"{bar.get_height():.4f}",  # Format to two decimals
        ha='center',  # Center align
        va='bottom',  # Bottom align
        fontsize=10  # Adjust font size as needed
    )

# Display the plot in Streamlit
st.pyplot(fig)


with open("./data/predictions.json", "r") as file:
    predictions = json.load(file)
# Load actual and predicted values for visualization
predictions_df = pd.DataFrame(predictions)
predictions_df.reset_index(drop=True, inplace=True)
print(predictions_df)
st.subheader("Predictions vs Actual Values")
# Downsample factor (plot every n-th data point)
n = 50  # Adjust this value to control the degree of downsampling

# Loop through each model and plot predictions vs actual values
models = predictions_df.columns[1:]  # Exclude 'Actual' column
for model in models:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Downsampled Actual values
    ax.plot(predictions_df['Actual'][::n], label='Actual', color='black', linewidth=1, linestyle='-')

    # Downsampled Model predictions
    ax.plot(predictions_df[model][::n], label=model, color='blue', linewidth=1, linestyle='-')

    # Add plot details
    ax.set_title(f"Predictions vs Actual Values ({model})")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)

    # Display each plot in Streamlit
    st.pyplot(fig)