import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore

#@st.cache_data
def load_correlation_matrix():
    return pd.read_csv("./data/correlation_matrix.csv", index_col=0)

# Function for clustering analysis
def visualize_clustering(correlation_matrix):
    st.subheader("Clustering Analysis of Respondents")

    # Perform K-Means Clustering
    num_clusters = 5  # Define number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(correlation_matrix)
    correlation_matrix['Cluster'] = cluster_labels

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(correlation_matrix.drop(columns='Cluster'))

    # Calculate Z-scores for PCA components to detect outliers
    pca_zscores = zscore(pca_results, axis=0)
    # Define outliers as those with Z-score > 3 or < -3
    outliers = (pca_zscores > 3) | (pca_zscores < -3)

    # Scatter plot with PCA, highlight outliers
    st.markdown("### PCA Scatter Plot of Clusters with Outliers Highlighted")
    plt.figure(figsize=(8, 6))
    
    # Plot clusters
    for cluster in range(num_clusters):
        indices = correlation_matrix['Cluster'] == cluster
        plt.scatter(pca_results[indices, 0], pca_results[indices, 1], label=f"Cluster {cluster}")

    # Highlight outliers
    outlier_indices = outliers.any(axis=1)  # Outliers in either PC1 or PC2
    plt.scatter(pca_results[outlier_indices, 0], pca_results[outlier_indices, 1], color='red', s=100, label="Outliers", marker='x')

    plt.title("PCA Visualization of Clusters with Outliers")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    st.pyplot(plt)

    st.markdown("### Respondents in Each Cluster")
    cluster_groups = correlation_matrix.reset_index().groupby('Cluster')  # Reset index

    # Create a dictionary to store cluster data for table display
    cluster_dict = {"Cluster": [], "Respondents": [], "Outlier": []}

    for cluster, group in cluster_groups:
        respondent_column = group.columns[0]  # Get the name of the first column (assumed to be respondents)
        # Identify if respondents in this cluster are outliers
        outlier_flag = outlier_indices[group.index].tolist()  # Get outlier flags for this group
        outlier_names = [group[respondent_column].iloc[i] for i, flag in enumerate(outlier_flag) if flag]
        
        if not outlier_names:  # If no outliers, add 'None'
            outlier_names = ['None']

        cluster_dict["Cluster"].append(cluster)
        cluster_dict["Respondents"].append(", ".join(group[respondent_column].tolist()))  # Join respondent names as a single string
        cluster_dict["Outlier"].append(", ".join(outlier_names))  # Add outlier names or 'None'

    # Convert the dictionary to a DataFrame for display
    cluster_table = pd.DataFrame(cluster_dict)
    cluster_table.set_index('Cluster', inplace=True)

    # Render the table in Streamlit (hide the index column)
    st.dataframe(cluster_table.style.hide(axis="index"))

    # Heatmap
    st.markdown("### Heatmap of Correlation Matrix with Cluster Assignments")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix.drop(columns='Cluster'),
        cmap='coolwarm',
        annot=False,
        cbar=True
    )
    plt.title("Heatmap of Correlation Matrix")
    st.pyplot(plt)

    st.markdown("### Anomaly and Outlier Detection in the Correlation Matrix")
    
    # Calculate Z-scores for correlation matrix to detect anomalies
    corr_zscores = zscore(correlation_matrix.drop(columns='Cluster'), axis=0)
    # Define outliers in the correlation matrix (Z-score > 3 or < -3)
    corr_outliers = (corr_zscores > 3) | (corr_zscores < -3)

    # Create a DataFrame to display outlier information
    outlier_info = pd.DataFrame(corr_outliers, columns=correlation_matrix.drop(columns='Cluster').columns)
    outlier_info = outlier_info.replace({True: "Outlier", False: "Normal"})

    # Highlight outliers in the table by using pandas Styler
    def highlight_outliers(val):
        color = 'background-color: red' if val == 'Outlier' else ''
        return color

    # Apply the highlighting function to the DataFrame
    styled_outliers_table = outlier_info.style.applymap(highlight_outliers)

    # Display the table of outliers in the correlation matrix
    st.markdown("#### Outlier Detection in Correlation Matrix")
    st.dataframe(styled_outliers_table, width=1000, height=600)

# Main app
st.title("EIA 930 Anomaly and Outlier Detection")

# Load data
correlation_matrix = load_correlation_matrix()

# Clustering visualization
visualize_clustering(correlation_matrix)
