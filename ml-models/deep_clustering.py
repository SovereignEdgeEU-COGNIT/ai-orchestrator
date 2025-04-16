import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

# Custom Dataset Class
class ProcessDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Deep k-Means Autoencoder Model
class DeepKMeansAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3, n_clusters=3):
        super(DeepKMeansAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        # Cluster Centers
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def clustering_loss(self, latent_data):
        """
        Compute clustering loss: Distance between latent points and their closest cluster centers.
        """
        # Compute pairwise distances between latent points and cluster centers
        distances = torch.cdist(latent_data, self.cluster_centers)  # [batch_size, n_clusters]
        # Assign each point to its closest cluster
        min_distances = torch.min(distances, dim=1)[0]  # Minimum distance for each point
        return torch.mean(min_distances ** 2)  # Average squared distance

def train_deep_kmeans(data, input_dim, latent_dim=3, n_clusters=3, epochs=50, batch_size=32, lr=0.001, lambda_clust=0.1):
    """
    Train the Deep k-Means autoencoder with a joint reconstruction and clustering loss.
    """
    dataset = ProcessDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepKMeansAutoencoder(input_dim, latent_dim, n_clusters)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reconstruction_criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            latent, reconstructed = model(batch)
            
            # Compute reconstruction loss
            reconstruction_loss = reconstruction_criterion(reconstructed, batch)
            # Compute clustering loss
            clust_loss = model.clustering_loss(latent)
            # Combined loss
            loss = reconstruction_loss + lambda_clust * clust_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss/len(dataloader):.4f}, "
              f"Reconstruction Loss: {reconstruction_loss.item():.4f}, Clustering Loss: {clust_loss.item():.4f}")
    return model

def perform_latent_kmeans(model, data):
    """
    Perform clustering in the latent space using K-Means with initialized cluster centers.

    Parameters:
    - model: Trained DeepKMeansAutoencoder model.
    - data: Scaled input features.

    Returns:
    - clusters: Predicted cluster labels.
    - latent_data: Latent space representation.
    - kmeans: Fitted KMeans model.
    """
    model.eval()
    with torch.no_grad():
        latent_data, _ = model(torch.tensor(data, dtype=torch.float32))
    latent_data = latent_data.numpy()

    # Use the cluster centers from the model as initialization for KMeans
    kmeans = KMeans(n_clusters=model.n_clusters, init=model.cluster_centers.detach().numpy(), n_init=1, random_state=42)
    clusters = kmeans.fit_predict(latent_data)
    return clusters, latent_data, kmeans

def plot_clusters_3d(latent_data, clusters, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'green', 'blue']
    for i in range(len(colors)):
        cluster_points = latent_data[clusters == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   color=colors[i], label=f"Cluster {i+1}", s=50, alpha=0.8)

    ax.set_title("Deep K-Means Clustering (Latent Space)", fontsize=16)
    ax.set_xlabel("Latent Dim 1", fontsize=14)
    ax.set_ylabel("Latent Dim 2", fontsize=14)
    ax.set_zlabel("Latent Dim 3", fontsize=14)
    ax.legend()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "deep_kmeans_3d.png")
    plt.savefig(plot_path)
    print(f"Saved 3D clustering plot: {plot_path}")
    plt.close()

def plot_clusters_2d(latent_data, clusters, output_dir):
    plt.figure(figsize=(10, 8))
    colors = ['green', 'blue']
    for i in range(len(colors)):
        cluster_points = latent_data[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i], label=f"Cluster {i+1}", s=50, alpha=0.8)

    plt.title("Deep K-Means Clustering (Latent Space, 2D)", fontsize=16)
    plt.xlabel("Latent Dim 1", fontsize=14)
    plt.ylabel("Latent Dim 2", fontsize=14)
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "deep_kmeans_2d.png")
    plt.savefig(plot_path)
    print(f"Saved 2D clustering plot: {plot_path}")
    plt.close()

if __name__ == "__main__":
    input_dir = "Code-III-Simulator-Data/random"
    output_dir = "Code-III-Simulator-Data/deep_clustering_figs"

    from clustering import load_clusters, extract_process_features
    data = load_clusters(input_dir, num_rows=10000, skip_rows=1500, sample_size=200)
    features = extract_process_features(data)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    input_dim = scaled_features.shape[1]
    latent_dim = 3
    n_clusters = 3
    lambda_clust = 0.1

    model = train_deep_kmeans(scaled_features, input_dim, latent_dim, n_clusters,
                              epochs=50, batch_size=32, lr=0.001, lambda_clust=lambda_clust)

    clusters, latent_data, kmeans = perform_latent_kmeans(model, scaled_features)
    plot_clusters_3d(latent_data, clusters, output_dir)
    plot_clusters_2d(latent_data[:, :2], clusters, output_dir)
