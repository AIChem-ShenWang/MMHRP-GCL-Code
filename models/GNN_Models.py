import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GCN(nn.Module):
    def __init__(self, node_feature_num, channels):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(node_feature_num, channels[0])
        self.conv2 = pyg_nn.GCNConv(channels[0], channels[1])
        self.mlp = nn.Sequential(
            nn.Linear(channels[1], 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, data):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()

        # GAT
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = pyg_nn.global_mean_pool(x, batch=batch)

        # MLP
        x = self.mlp(x)

        return x

class GAT(nn.Module):
    def __init__(self, node_feature_num, channels, heads):
        super(GAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(node_feature_num, channels[0], heads=heads)
        self.conv2 = pyg_nn.GATConv(channels[0] * heads, channels[1], heads=heads)
        self.mlp = nn.Sequential(
            nn.Linear(channels[1] * heads, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )


    def forward(self, data):
        x = data["x"].clone().detach().float()
        edge_index = data["edge_index"].clone().detach()
        batch = data["batch"].clone().detach()

        # GAT
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = pyg_nn.global_mean_pool(x, batch=batch)

        # MLP
        x = self.mlp(x)

        return x

# Model Evaluation Function
import numpy as np

def RMSE(pred, true):
    diff_2 = (pred - true)**2
    return np.sqrt(diff_2.mean())

def R2(pred, true):
    u = ((true - pred) ** 2).sum()
    v = ((true - true.mean()) ** 2).sum()
    r2 = 1 - u / v
    return r2