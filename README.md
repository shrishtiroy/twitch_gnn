# Twitch Streamer Recommendation System with GNNs

A Graph Neural Network (GNN) recommendation system that predicts which Twitch streamers users will watch based on social network structure and streamer features. Processes 168,114 streamers and 6.8 million mutual follower relationships.

##Why this project
I wanted to create this project because during quarantine, I spent a lot of time on Twitch watching my favorite streamers. I would interact with people in streamers' chats and they were like my online friends. Eventually, I wanted to find new streamers that my friends watched but also related to my watch history. This is what TwitchGNN aims to do.

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


## Usage




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

User watch histories are treated as signals on the graph. Each user's preferences are a vector:
- Non-zero entries: streamers the user has watched with ratings
- Zero entries: streamers not yet watched

The GNN processes these signals through:
1. Scale: Spreading preference information through graph connections
2. Shift: Combining information from neighboring streamers
3. Sum: Learning non-linear patterns via neural network layers

### Message Passing

At each layer:
- Each streamer node receives messages from neighbors
- Messages are weighted by similarity matrix S
- Node aggregates messages and applies transformation
- Process repeats across layers

### Feature Integration

Streamer features are used in two ways:
1. Graph construction: Features influence similarity matrix S via cosine similarity
2. Training data generation: Features determine synthetic user preferences (users prefer active, popular streamers)



Potential improvements:
- Preference changes over time
- Attention mechanisms for neighbor weighting

## References

- Twitch Gamers Dataset: Rozemberczki & Sarkar (2021), "Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings"


