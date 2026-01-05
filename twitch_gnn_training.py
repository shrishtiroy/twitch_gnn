# twitch_gnn_training.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load processed Twitch data
with open("twitch_data.p", "rb") as f:
    contents = pkl.load(f)
data = contents['data']

# Graph similarity matrix (combines mutual followers + feature similarity)
S = data['S'].to(device)
N = S.shape[0]  # Number of streamers

# Training data
x_train = data['train']['x'].to(device)  # User watch history (sparse)
y_train = data['train']['y'].to(device)  # Rating to predict
p_train = data['train']['p'].to(device)  # Streamer index to predict
n_train = x_train.shape[0]

# Test data
x_test = data['test']['x'].to(device)
y_test = data['test']['y'].to(device)
p_test = data['test']['p'].to(device)
n_test = x_test.shape[0]

# Streamer features 
features = data['features'].to(device)

print(f"Number of streamers (nodes in graph): {N}")
print(f"Number of training samples: {n_train}")
print(f"Number of test samples: {n_test}")

def streamer_mse_loss(y_hat, y, idx_streamer):
    """
    MSE loss for streamer recommendation.
    """
    y_hat = y_hat.squeeze()
    y = y.squeeze()
    idx_streamer = idx_streamer.squeeze()
    
    prediction = y_hat[torch.arange(y_hat.shape[0]), idx_streamer.long()]
    mse = torch.mean((prediction - y) ** 2)
    return mse

def filter_function(x, h, S, b):
    """
    Graph filter function
    """
    B, G, N = x.shape
    K, _, F = h.shape
    
    y = torch.zeros((B, N, F), device=device)
    
    for k in range(K):
        if k == 0:
            S_k = torch.eye(N, device=device)
        else:
            S_k = torch.matrix_power(S, k)
        
        x_k = torch.matmul(x, S_k)
        y += torch.matmul(x_k.permute(0, 2, 1), h[k])
    
    y = y + b
    y = y.permute(0, 2, 1)
    return y

class FilterModule(nn.Module):
    """Graph filter module """
    
    def __init__(self, K, f_in, f_out):
        super().__init__()
        self.K = K
        self.f_in = f_in
        self.f_out = f_out
        
        self.h = nn.Parameter(torch.randn(self.K, self.f_in, self.f_out))
        self.b = nn.Parameter(torch.zeros(self.f_out))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.f_in * self.K)
        self.h.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, x, S):
        return filter_function(x, self.h, S, self.b)

class GNNModule(nn.Module):
    """
    GNN for streamer recommendation.
    Architecture considers:
    - Mutual follower relationships (in graph S)
    - Streamer features (views, lifetime, mutual followers) are incorporated in S
    """
    
    def __init__(self, L, k_list, f_list, sigma):
        super().__init__()
        self.sigma = sigma
        
        gml = []
        for layer in range(L - 1):
            gml.append(FilterModule(k_list[layer], f_list[layer], f_list[layer + 1]).to(device))
        self.gml = gml
        
        self.readout = FilterModule(1, f_list[L - 1], f_list[L])
    
    def forward(self, x, S):
        # Reshape input: (batch_size, num_streamers) -> (batch_size, 1, num_streamers)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for layer in self.gml:
            x = layer(x, S)
            x = self.sigma(x)
        
        x = self.readout(x, S)
        return x

# GNN specifications
L = 3  # Number of layers
K_list = [5, 5, 1]  # Filter taps per layer
F_list = [1, 64, 32, 1]  # Feature dimensions
sigma = nn.ReLU()  # Activation

# Instantiate model
model = GNNModule(L, K_list, F_list, sigma).to(device)

# Training parameters
n_epochs = 10
learning_rate = 0.05
batch_size = 20

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
batch_index = np.append(np.arange(0, n_train, batch_size), n_train)
n_batches = int(np.ceil(n_train / batch_size))

loss_train = []
loss_test = []

for epoch in range(n_epochs):
    random_permutation = np.random.permutation(n_train)
    idx_epoch = [int(i) for i in random_permutation]
    epoch_loss = 0
    
    model.train()
    for batch in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}"):
        this_batch_indices = idx_epoch[batch_index[batch]: batch_index[batch + 1]]
        
        x_train_batch = x_train[this_batch_indices, :].unsqueeze(1)  # (batch, 1, N)
        y_train_batch = y_train[this_batch_indices]
        p_train_batch = p_train[this_batch_indices]
        
        model.zero_grad()
        
        y_hat_train_batch = model(x_train_batch, S)
        loss_value_train = streamer_mse_loss(y_hat_train_batch, y_train_batch, p_train_batch)
        
        loss_value_train.backward()
        optimizer.step()
        
        epoch_loss += loss_value_train.item()
        loss_train.append(loss_value_train.item())
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        x_test_batch = x_test.unsqueeze(1)
        y_hat_test = model(x_test_batch, S)
        test_loss = streamer_mse_loss(y_hat_test, y_test, p_test)
        loss_test.append(test_loss.item())
    
    avg_epoch_loss = epoch_loss / n_batches
    print(f"Epoch: {epoch + 1}/{n_epochs} \t Train Loss: {avg_epoch_loss:.4f} \t Test Loss: {test_loss.item():.4f}\n")

# Plot training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_train, label='Training MSE', alpha=0.7)
plt.xlabel('Batches')
plt.ylabel('Mean Squared Error')
plt.title('GNN Training Loss Evolution')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_test, label='Test MSE', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('GNN Test Loss Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('twitch_gnn_training_curves.png', dpi=150)
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    x_test_batch = x_test.unsqueeze(1)
    y_hat_test = model(x_test_batch, S)
    final_test_mse = streamer_mse_loss(y_hat_test, y_test, p_test)

print(f"\nFinal Test Mean Squared Error: {final_test_mse.item():.5f}")


# Save model
torch.save(model.state_dict(), 'twitch_gnn_model.pth')
