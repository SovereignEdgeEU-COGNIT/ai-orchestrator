import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_clusters(cluster_dir, num_rows=10000, skip_rows=0, sample_size=100):
    """
    Load datasets from the specified directory, skip initial rows, and randomly sample rows.

    Parameters:
    - cluster_dir: Path to the directory containing the cluster CSV files.
    - num_rows: Number of rows to load from each dataset after skipping the specified rows.
    - skip_rows: Number of rows to skip from the beginning of each dataset.
    - sample_size: Number of rows to randomly sample from the combined data.

    Returns:
    - combined_df: A DataFrame containing the randomly sampled rows from all datasets.
    """
    all_data = []

    for file_name in os.listdir(cluster_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(cluster_dir, file_name)
            print(f"Processing file: {file_name} (skip rows: {skip_rows}, load rows: {num_rows})")

            # Read the dataset with skip_rows and num_rows limit
            df = pd.read_csv(file_path, skiprows=range(1, skip_rows + 1), nrows=num_rows)

            # Parse the 'processes' column as JSON
            df["processes"] = df["processes"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
            all_data.append(df)

    # Combine all data from all clusters into one DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Randomly sample rows from the combined DataFrame
    if sample_size and len(combined_df) > sample_size:
        combined_df = combined_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return combined_df

def extract_process_features(data):
    """
    Extract `process_gpu_utilization`, `process_memory_utilization`, and `process_energy` from the dataset.

    Parameters:
    - data: DataFrame containing the combined data.

    Returns:
    - features: A DataFrame containing GPU utilization, memory utilization, and energy for all processes.
    """
    gpu_utilization = []
    memory_utilization = []
    energy = []

    for processes in data["processes"]:
        for process in processes:
            gpu_utilization.append(process["process_gpu_utilization"])
            memory_utilization.append(process["process_memory_utilization"])
            energy.append(process["process_energy"])

    # Create a DataFrame with the extracted features
    features = pd.DataFrame({
        "GPU Utilization": gpu_utilization,
        "Memory Utilization": memory_utilization,
        "Energy": energy
    })
    return features

def perform_clustering(features, n_clusters, n_init):
    """
    Cluster the features using K-Means clustering.

    Parameters:
    - features: DataFrame containing the extracted features.
    - n_clusters: Number of clusters to form.
    - n_init: Number of initializations to perform.

    Returns:
    - clusters: Cluster labels for each process.
    - kmeans: Trained KMeans model.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    return clusters, kmeans

def plot_clusters_3d(features, clusters, n_clusters, n_init, output_dir):
    """
    Plot the clusters in a 3D scatter plot and save the figure.

    Parameters:
    - features: DataFrame containing the extracted features.
    - clusters: Cluster labels for each process.
    - n_clusters: Number of clusters.
    - n_init: Number of initializations used for clustering.
    - output_dir: Directory to save the plot.
    """
    colors = ['blue', 'green', 'red']
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(features["GPU Utilization"], 
               features["Memory Utilization"], 
               features["Energy"], 
               c=[colors[label % len(colors)] for label in clusters], s=50, alpha=0.8)

    ax.set_title(f"K-Means Clustering (K={n_clusters}, n_init={n_init})", fontsize=16)
    ax.set_xlabel("GPU Utilization", fontsize=14)
    ax.set_ylabel("Memory Utilization", fontsize=14)
    ax.set_zlabel("Energy", fontsize=14)

    # Reverse the GPU Utilization axis
    ax.set_xlim(ax.get_xlim()[::-1])

    # Add grid
    ax.grid(True)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"kmeans_3d_k_{n_clusters}_n_init_{n_init}.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

def plot_clusters_2d(features, clusters, output_dir):
    """
    Plot the clusters in a 2D scatter plot (for K=2) and save the figure.

    Parameters:
    - features: DataFrame containing the extracted features.
    - clusters: Cluster labels for each process.
    - output_dir: Directory to save the plot.
    """
    colors = ['blue', 'green']
    plt.figure(figsize=(10, 8))
    plt.scatter(features["GPU Utilization"], 
                features["Memory Utilization"], 
                c=[colors[label % len(colors)] for label in clusters], s=50, alpha=0.8)

    plt.title("K-Means Clustering (K=2)", fontsize=16)
    plt.xlabel("GPU Utilization", fontsize=14)
    plt.ylabel("Memory Utilization", fontsize=14)

    # # Reverse the GPU Utilization axis
    # plt.gca().invert_xaxis()

    # Add grid
    plt.grid(True)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"kmeans_2d_k_2.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Directory containing the datasets
    cluster_dir = "Code-III-Simulator-Data/random"
    output_dir = "Code-III-Simulator-Data/clustering_figs"

    # Load clusters with skip_rows and random sampling
    data = load_clusters(cluster_dir, num_rows=10000, skip_rows=1500, sample_size=200)

    # Extract features for clustering
    features = extract_process_features(data)

    # Perform clustering with K=3 and K=2 for 3D and 2D visualizations respectively
    n_init = 10

    # For 3D clustering
    # n_clusters_3d = 3
    # clusters_labels_3d, kmeans_model_3d = perform_clustering(features, n_clusters=n_clusters_3d, n_init=n_init)
    # plot_clusters_3d(features, clusters_labels_3d, n_clusters=n_clusters_3d, n_init=n_init, output_dir=output_dir)

    # For 2D clustering
    n_clusters_2d = 2
    clusters_labels_2d, kmeans_model_2d = perform_clustering(features, n_clusters=n_clusters_2d, n_init=n_init)
    plot_clusters_2d(features, clusters_labels_2d, output_dir=output_dir)
