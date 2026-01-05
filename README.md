# Twitch Streamer Recommendation System with Graph Neural Networks

A Graph Neural Network (GNN) recommendation system that predicts which Twitch streamers users will watch based on social network structure and streamer features. Processes 168,114 streamers and 6.8 million mutual follower relationships.

## Overview

This system implements collaborative filtering using Graph Neural Networks to recommend Twitch streamers. The approach combines:

- Social network structure: Mutual follower relationships between streamers
- Streamer features: View counts, account lifetime, activity status
- Graph signal processing: Message-passing algorithms that propagate information through the network
- Multi-layer neural networks: Learns patterns in user preferences

The model predicts user preferences by learning how streamers are connected in the social graph and how similar streamers appeal to similar audiences.

## Features

### Graph Construction

The system builds a hybrid similarity graph combining:

1. Social network graph: Built from mutual follower relationships. Edges represent streamers who follow each other.

2. Feature similarity graph: Computed using cosine similarity on streamer features (views, lifetime, activity status). Identifies streamers with similar characteristics.

The similarity matrix S is a weighted combination:
```
S = α × (Mutual Follower Graph) + (1-α) × (Feature Similarity Graph)
```

Default α = 0.6 (60% mutual followers, 40% feature similarity).

### Graph Neural Network Architecture

Three-layer message-passing architecture:

```
Input Signal (User Watch History) 
    ↓
Layer 1: Graph Filter (5 taps) → ReLU → 64 features
    ↓
Layer 2: Graph Filter (5 taps) → ReLU → 32 features  
    ↓
Layer 3: Graph Filter (1 tap) → Output (Predicted Ratings)
```

Each layer applies graph convolution:
```
Z = Σ(k=0 to K) S^k × X × H_k
```

Where:
- S^k represents k-hop neighborhoods in the graph
- X is the input signal (user's watch history)
- H_k are learnable filter coefficients

### Data Processing

- Feature normalization: Log-transform views, standardize lifetime, normalize other features
- Sparse matrix operations: Efficient handling of large graphs using SciPy sparse matrices
- Synthetic training data: Generates user preferences based on streamer features (views, lifetime, activity)
- Graph normalization: Symmetric normalization D^(-1/2) × A × D^(-1/2) with self-loops

### Model Capabilities

1. Sparse data handling: Makes predictions with sparse user watch histories by propagating preferences through graph connections

2. Cold start handling: Recommends new streamers based on features and connections to similar streamers

3. Multi-factor analysis: Considers popularity (view counts), activity (account lifetime, dead account status), social connections (mutual followers), and user patterns

## Dataset

Twitch Gamers Dataset:
- 168,114 streamers (nodes)
- 6,797,557 mutual follower relationships (edges)
- Features per streamer: views, account lifetime (days), mature content flag, affiliate status, account activity status

## Installation

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Installs: PyTorch, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, tqdm

2. Prepare the Twitch dataset:
   - Download the Twitch Gamers dataset
   - Place `large_twitch_edges.csv` and `large_twitch_features.csv` in the project directory

## Usage

### Step 1: Preprocess Data

Build the graph and prepare training data:

```bash
python twitch_data_preprocessing.py
```

This script:
- Loads Twitch edges and features from CSV files
- Constructs adjacency matrix from mutual followers
- Normalizes adjacency matrix (symmetric normalization with self-loops)
- Prepares streamer features (log-transform views, standardize lifetime)
- Creates hybrid similarity graph (combines mutual followers and feature similarity)
- Generates synthetic training data based on streamer features
- Saves processed data to `twitch_data.p`

Expected output:
```
Loading Twitch data...
Loaded 6797557 edges and 168114 streamers
Building adjacency matrix from mutual followers...
Built adjacency matrix: 168114 nodes, 3398778 undirected edges
Preparing streamer features...
Creating similarity graph...
Creating synthetic training data for 2000 users...
Created X training samples and Y test samples
Preprocessing complete!
```

### Step 2: Train GNN

Train the Graph Neural Network:

```bash
python twitch_gnn_training.py
```

This will:
- Load preprocessed graph and training data from `twitch_data.p`
- Initialize 3-layer GNN with configurable architecture
- Train for 10 epochs using Adam optimizer (learning rate 0.05, batch size 20)
- Evaluate on test set after each epoch
- Generate training curves plot: `twitch_gnn_training_curves.png`
- Save trained model to `twitch_gnn_model.pth`

Training output:
```
Using device: cpu  # or cuda if GPU available
Number of streamers (nodes in graph): 168114
Number of training samples: X
Number of test samples: Y

Starting training...
Epoch 1/10: 100%|████████| X/X [XX:XX<00:00, X.XXit/s]
Epoch: 1/10     Train Loss: X.XXXX     Test Loss: X.XXXX
...
```

## Model Architecture

### Hyperparameters

- Layers: 3
- Filter taps: [5, 5, 1] (number of graph hops per layer)
- Feature dimensions: [1, 64, 32, 1] (input → hidden → hidden → output)
- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.05
- Batch size: 20
- Epochs: 10

### Customization

Modify architecture in `twitch_gnn_training.py`:

```python
L = 3  # Number of layers
K_list = [5, 5, 1]  # Filter taps per layer
F_list = [1, 64, 32, 1]  # Feature dimensions
sigma = nn.ReLU()  # Activation function
```

## Technical Details

### Graph Signal Processing

User watch histories are treated as signals on the graph. Each user's preferences are a sparse vector:
- Non-zero entries: streamers the user has watched (with ratings)
- Zero entries: streamers not yet watched

The GNN processes these signals through:
1. Diffusion: Spreading preference information through graph connections
2. Aggregation: Combining information from neighboring streamers
3. Transformation: Learning non-linear patterns via neural network layers

### Message Passing

At each layer:
- Each streamer node receives messages from neighbors
- Messages are weighted by similarity matrix S
- Node aggregates messages and applies transformation
- Process repeats across layers, propagating information through the network

### Feature Integration

Streamer features are used in two ways:
1. Graph construction: Features influence similarity matrix S via cosine similarity
2. Training data generation: Features determine synthetic user preferences (users prefer active, popular streamers)

### Collaborative Filtering

The system implements collaborative filtering by:
- Learning patterns from multiple users' watch histories
- Identifying streamers that similar users enjoy
- Using graph structure to find similar streamers without direct user overlap

## Implementation Details

1. Hybrid graph construction: Combines social network structure with feature-based similarity
2. Multi-hop propagation: Uses graph powers (S^k) to capture k-hop relationships
3. Sparse signal processing: Handles sparse user watch histories efficiently
4. Feature-aware recommendations: Considers streamer characteristics alongside social connections
5. Scalable operations: Processes 168K+ nodes using sparse matrix operations

## Project Structure

```
movie_gnn/
├── twitch_data_preprocessing.py    # Data loading and graph construction
├── twitch_gnn_training.py          # GNN model definition and training
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── large_twitch_edges.csv          # Twitch mutual follower data (required)
├── large_twitch_features.csv       # Twitch streamer features (required)
├── twitch_data.p                   # Processed data (generated)
├── twitch_gnn_model.pth            # Trained model (generated)
└── twitch_gnn_training_curves.png  # Training visualization (generated)
```

## Applications

This system demonstrates GNN applications for:
- Social media recommendations: Using follower networks for content discovery
- Collaborative filtering: Improving recommendation accuracy with graph structure
- Network analysis: Understanding information flow through social graphs
- Feature learning: Discovering patterns in high-dimensional data

## Future Enhancements

Potential improvements:
- Real user watch history data (replacing synthetic data)
- Temporal modeling (preference changes over time)
- Multi-task learning (predict views, churn, etc.)
- Attention mechanisms for neighbor weighting
- GPU acceleration for faster training

## References

- Graph Neural Networks: Message-passing architectures for graph-structured data
- Collaborative Filtering: Recommendation systems using user-item interactions
- Twitch Gamers Dataset: Rozemberczki & Sarkar (2021), "Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings"

---

Built with PyTorch and Graph Neural Networks
