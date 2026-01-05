#twitch_data_preprocessing.py
"""
Output:
- Graph similarity matrix S
- Training data: (x, y, p) tuples
- Test data: (x, y, p) tuples
"""

import numpy as np
import pandas as pd
import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

def load_twitch_data(edges_path, features_path):
    """
    Load Twitch edges and features data.
    
    Returns:
        edges_df: DataFrame with mutual follower relationships
        features_df: DataFrame with streamer features
    """
    edges_df = pd.read_csv(edges_path)
    features_df = pd.read_csv(features_path)
    
    print(f"Loaded {len(edges_df)} edges and {len(features_df)} streamers")
    return edges_df, features_df

def build_adjacency_matrix(edges_df, features_df):
    """
    Build adjacency matrix from mutual follower edges.
    Creates a symmetric adjacency matrix where edges represent mutual follows.
    
    Returns:
        A: Sparse adjacency matrix 
        node_to_idx: Dictionary mapping numeric_id to matrix index
        idx_to_node: Dictionary mapping matrix index to numeric_id
    """
    print("Building adjacency matrix from mutual followers...")
    
    # Create mapping from numeric_id to index
    unique_nodes = sorted(features_df['numeric_id'].unique())
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Build edge list with indices
    edges_list = []
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Processing edges"):
        node1, node2 = row['numeric_id_1'], row['numeric_id_2']
        if node1 in node_to_idx and node2 in node_to_idx:
            idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
            edges_list.append((idx1, idx2))
            edges_list.append((idx2, idx1))  # Make symmetric
    
    # Create sparse adjacency matrix
    N = len(unique_nodes)
    rows, cols = zip(*edges_list) if edges_list else ([], [])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    
    return A, node_to_idx, idx_to_node

def normalize_adjacency(A):
    """
    Normalize adjacency matrix for better GNN performance.
    Uses symmetric normalization: D^(-1/2) * A * D^(-1/2)
    """

    A = A.tocsr()
    
    # Add self-loops
    A = A + csr_matrix(np.eye(A.shape[0]))
    
    # Compute degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees_inv_sqrt = np.power(degrees, -0.5, where=degrees != 0)
    degrees_inv_sqrt[degrees == 0] = 0
    
    # Normalize
    D_inv_sqrt = csr_matrix(np.diag(degrees_inv_sqrt))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_norm

def create_similarity_graph(A, features_df, node_to_idx, alpha=0.7):
    """
    Create similarity graph combining mutual follower relationships and feature similarity.
    
    Combine:
    - Mutual follower graph (social structure)
    - Feature similarity (streamer characteristics)
    
    Args:
        A: Normalized adjacency matrix from mutual followers
        features_df: Streamer features DataFrame
        node_to_idx: Node to index mapping
        alpha: Weight for mutual followers vs feature similarity (0-1)
    
    Returns:
        S: Combined similarity matrix
    """

    
    # Convert normalized adjacency to dense tensor
    A_norm = normalize_adjacency(A)
    A_dense = torch.tensor(A_norm.toarray(), dtype=torch.float64)
    
    # Compute feature similarity using views and lifetime
    N = len(node_to_idx)
    feature_matrix = np.zeros((N, 2))  # Use views and lifetime
    
    for idx, row in features_df.iterrows():
        node_id = row['numeric_id']
        if node_id in node_to_idx:
            matrix_idx = node_to_idx[node_id]
            # Log-transform views (handle large variance)
            feature_matrix[matrix_idx, 0] = np.log1p(row['views'])
            # Normalize lifetime to years
            feature_matrix[matrix_idx, 1] = row['life_time'] / 365.0
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    # Compute cosine similarity 
    features_tensor = torch.tensor(feature_matrix, dtype=torch.float64)
    features_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
    feature_sim = torch.mm(features_norm, features_norm.t())
    
    # Combine: alpha * mutual_followers + (1-alpha) * feature_similarity
    S = alpha * A_dense + (1 - alpha) * feature_sim
    
    # Ensure symmetry and non-negativity
    S = (S + S.t()) / 2
    S = torch.clamp(S, min=0)
    
    return S

def create_training_data(features_df, node_to_idx, num_users=2000, 
                         watch_prob_base=0.15, test_split=0.2, seed=42):
    """
    Coalesce training and test data
    
   data format: entries (x_n, y_n, p_n)
    - x_n: vector of ratings for a user (sparse, most entries are 0)
    - y_n: scalar rating to predict
    - p_n: index of streamer whose rating is being predicted
    
    Args:
        features_df: Streamer features
        node_to_idx: Node to index mapping
        S: Similarity matrix
        num_users: Number of synthetic users
        watch_prob_base: Base probability of watching a streamer
        test_split: Fraction of data for testing
        seed: Random seed
    
    Returns:
        x_train, y_train, p_train: Training data
        x_test, y_test, p_test: Test data
    """

    np.random.seed(seed)
    
    N = len(node_to_idx)
    
    # Create preference scores based on features and graph structure
    preference_scores = np.zeros((num_users, N))
    
    for user_idx in range(num_users):
        # Each user has preferences influenced by:
        # - Streamer views (popularity)
        # - Streamer lifetime (activity)
        # - Graph structure (similar streamers)
        
        for streamer_idx, row in features_df.iterrows():
            node_id = row['numeric_id']
            if node_id not in node_to_idx:
                continue
            
            matrix_idx = node_to_idx[node_id]
            
            # Base preference from features
            view_score = np.log1p(row['views']) / 20.0
            activity_score = (row['life_time'] / 365.0) * (1 - row['dead_account'])
            personal_bias = np.random.normal(0, 0.5)
            
            preference = view_score + 0.5 * activity_score + personal_bias
            preference_scores[user_idx, matrix_idx] = preference
    
    # Convert to ratings 
    ratings = 1 + 4 * (1 / (1 + np.exp(-preference_scores)))
    
    # Create data entries (x_n, y_n, p_n)
    x_train_list = []
    y_train_list = []
    p_train_list = []
    
    x_test_list = []
    y_test_list = []
    p_test_list = []
    
    for user_idx in range(num_users):
        # Sample which streamers user has watched
        watch_probs = watch_prob_base * (ratings[user_idx] / 5.0)
        watched = np.random.binomial(1, watch_probs, N).astype(bool)
        watched_indices = np.where(watched)[0]
        
        if len(watched_indices) < 2:
            continue  # Skip users with too few watches
        
        # Split into train/test: hold out some watched streamers for prediction
        n_watched = len(watched_indices)
        n_test = max(1, int(test_split * n_watched))
        test_indices = np.random.choice(watched_indices, n_test, replace=False)
        train_indices = np.setdiff1d(watched_indices, test_indices)
        
        # Training data: predict some watched streamers from others
        for test_idx in test_indices:
            # Input: ratings for train_indices, zeros elsewhere
            x_user = np.zeros(N)
            x_user[train_indices] = ratings[user_idx, train_indices]
            
            # Target: rating for test streamer
            y_target = ratings[user_idx, test_idx]
            p_target = test_idx
            
            x_train_list.append(x_user)
            y_train_list.append(y_target)
            p_train_list.append(p_target)
        
        # Test data: similar split but different streamers
        if len(train_indices) > 0 and len(test_indices) > 0:
            # Use a different subset for test
            test_subset = test_indices[:min(len(test_indices), 2)]
            for test_idx in test_subset:
                x_user = np.zeros(N)
                x_user[train_indices] = ratings[user_idx, train_indices]
                
                y_target = ratings[user_idx, test_idx]
                p_target = test_idx
                
                x_test_list.append(x_user)
                y_test_list.append(y_target)
                p_test_list.append(p_target)
    
    # Convert to tensors
    x_train = torch.tensor(np.array(x_train_list), dtype=torch.float64)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float64)
    p_train = torch.tensor(np.array(p_train_list), dtype=torch.long)
    
    x_test = torch.tensor(np.array(x_test_list), dtype=torch.float64)
    y_test = torch.tensor(np.array(y_test_list), dtype=torch.float64)
    p_test = torch.tensor(np.array(p_test_list), dtype=torch.long)
    
    print(f"Created {len(x_train)} training samples and {len(x_test)} test samples")
    return x_train, y_train, p_train, x_test, y_test, p_test

def main():
    """Main preprocessing pipeline."""
    # Paths 
    edges_path = '/Users/shrishtiroy/Downloads/twitch_gamers/large_twitch_edges.csv'
    features_path = '/Users/shrishtiroy/Downloads/twitch_gamers/large_twitch_features.csv'
    output_path = 'twitch_data.p'
    
    # Load data
    edges_df, features_df = load_twitch_data(edges_path, features_path)
    
    # Build graph from mutual followers
    A, node_to_idx, idx_to_node = build_adjacency_matrix(edges_df, features_df)
    
    # Create similarity graph (combining mutual followers + feature similarity)

    S = create_similarity_graph(A, features_df, node_to_idx, alpha=0.7)
    
    # Create training and test data
    x_train, y_train, p_train, x_test, y_test, p_test = create_training_data(
        features_df, node_to_idx, num_users=2000, watch_prob_base=0.15
    )
    
    data = {
        'S': S,  # Similarity matrix (graph)
        'train': {
            'x': x_train,  # Available ratings for making predictions
            'y': y_train,  # Rating to be predicted
            'p': p_train   # Index of streamer whose rating is being predicted
        },
        'test': {
            'x': x_test,
            'y': y_test,
            'p': p_test
        }
    }
    
    print(f"\nSaving processed data to {output_path}...")
    with open(output_path, 'wb') as f:
        pkl.dump({'data': data}, f)
    
    print(f"Graph: {S.shape[0]} streamers")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

if __name__ == '__main__':
    main()

