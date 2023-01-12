# Multi-Objective Recommender System
## Import Libraries
```python
import numpy as np
import pandas as pd
import gc

!pip install -q pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

from itertools import product
import tqdm
from collections import defaultdict

import time

from sklearn.metrics import mean_squared_error

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import to_hetero
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, GraphConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import HGTLoader, NeighborLoader, LinkNeighborLoader, DataLoader

!pip install -q sentence-transformers

!pip3 install -q torch torchvision torchaudio
!pip install -U polars

import polars as pl
```

## Using CUDA GPU
```python
print('Pytorch CUDA Version is ', torch.version.cuda)
```
```python
torch.cuda.empty_cache()
```
```python
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
```
```python
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
```
```python
torch.cuda.is_available()
```
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Load Preprocess Edge Index, Edge Label and Node Features
```python
edge_index = torch.load('.../session-aid_edges_train_and_test.pt')
edge_label = torch.load('.../edge_label_session-aid_edges_train_and_test.pt')
session_feat = torch.load('.../session_features.pt')
aid_feat = torch.load('.../aid_features.pt')
```

## Nodes and Edges Atrributes
```python
## Nodes Atrributes
session_num_nodes = len(edge_index[0].unique()) #df['session'].nunique()
aid_num_nodes = len(edge_index[1].unique()) #df['aid'].nunique()

## Edges Atrributes
edge_index = edge_index
edge_label = edge_label #torch.tensor(df['type'].values).type(torch.int64)
```

## Construct Heterogenous Graph
```python
node_types = {
    'session': {
        'num_nodes': session_num_nodes,
        'x': session_feat
    },
    'aid': {
        'x': aid_feat,
    }
}

edge_types = {
    ('session', 'event', 'aid'): {
        'edge_index': edge_index,
        'edge_label': edge_label
    }
}
```
```python
data = HeteroData({**node_types, **edge_types})
```
```python
>>> data
HeteroData(
  session={
    num_nodes=14571582,
    x=[14571582, 32]
  },
  aid={ x=[1855603, 32] },
  (session, event, aid)={
    edge_index=[2, 223644219],
    edge_label=[223644219]
  }
)
```
```python
>>> data.metadata()
(['session', 'aid'], [('session', 'event', 'aid')])
```

## Construct Undirected Graph
```python
# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
```
```python
# Remove "reverse" label.
del data['aid', 'rev_event', 'session'].edge_label 
```
```python
>>> data
HeteroData(
  session={
    num_nodes=14571582,
    x=[14571582, 32]
  },
  aid={ x=[1855603, 32] },
  (session, event, aid)={
    edge_index=[2, 223644219],
    edge_label=[223644219]
  },
  (aid, rev_event, session)={ edge_index=[2, 223644219] }
)
```

## Train/val/test Link-Level Splits
```python
# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('session', 'event', 'aid')],
    rev_edge_types=[('aid', 'rev_event', 'session')],
)(data)
```

## Construct Mini-Batch Loader
```python
# Define seed edges:
edge_label_index = train_data['session', 'event', 'aid'].edge_label_index
edge_label = train_data['session', 'event', 'aid'].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,  # TODO
    num_neighbors=[5, 2],  # TODO
    neg_sampling_ratio=0.0,  # TODO
    edge_label_index=(('session', 'event', 'aid'), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)
```

## Weight Calculation for Imbalanced Data
```python
# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
weight = torch.bincount(train_data['session', 'aid'].edge_label)
weight = weight.max() / weight
```
```python
>>> weight
tensor([ 1.0000, 11.5101, 38.9133])
```

## Define Loss Function
Weighted RMSE
```python
def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()
```

## Construct Model
**GNNEncoder** encoding node embeddings
```python
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```
**EdgeDecoder** decoding edge labels
```python
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['session'][row], z_dict['aid'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)

        return z.view(-1)
```
**Model** combined `GNNEncoder` and `EdgeDecoder`
```python
class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # encoder and decoder
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def get_embedding(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def get_decoding(self, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
```
Define model
```python
model = Model(hidden_channels=32).to(device)
```
```python
# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:

sampled = next(iter(train_loader))

with torch.no_grad():
    model.encoder(sampled_data.to(device).x_dict, sampled_data.to(device).edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

```
```python
del sampled, optimizer
gc.collect()
```

## Train/test Functions
```python

def train(train_data=sampled_data):
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data[('session', 'event', 'aid')].edge_label_index)
    target = train_data['session', 'aid'].edge_label #/2
    loss = weighted_mse_loss(pred, target, weight.to(device))

    # Embedding
    emb_dict = model.get_embedding(train_data.x_dict, train_data.edge_index_dict)

    loss.backward()
    optimizer.step()
    return float(loss), pred, model, emb_dict
```
```python
@torch.no_grad()
def test(data=data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['session', 'aid'].edge_label_index)
    pred = pred.clamp(min=0, max=2)
    target = data['session', 'aid'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), pred, target
```

## Training Model
Including early-stopping
```python
%%time
import torch.nn.functional as F

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

#model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define a variable to keep track of the best validation loss
best_val_loss = float('inf')

# Define a variable to keep track of the number of consecutive epochs without improvement
early_stopping_count = 0

# Define the threshold for early stopping
early_stopping_threshold = 20

# Train the model
for epoch in range(0, 300):
    total_loss = total_train_rmse = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        sampled_data = sampled_data.to(device)

        loss, pred, model, emb_dict = train(train_data=sampled_data)
        train_rmse, _, _ = test(data=sampled_data)

        total_loss += float(loss) * pred.numel()
        total_train_rmse += float(train_rmse) * pred.numel()
        total_examples += pred.numel()

    val_rmse,_ ,_ = test(data=val_data.to(device))
    test_rmse,_ ,_ = test(data=test_data.to(device))
    print(f'Epoch: {epoch+1:03d}, Loss: {total_loss/total_examples:.4f}, Train: {total_train_rmse/total_examples:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    
    # early stopping
    val_loss = val_rmse
    # Check if the validation loss has improved
    if val_loss < best_val_loss:
        # Update the best validation loss
        best_val_loss = val_loss
        # Save the model parameters
        torch.save(model, '/content/drive/MyDrive/OTTO-Kaggle-Competition/models/best_full_model.pt')
        #torch.save(model.state_dict(), '/content/drive/MyDrive/OTTO-Kaggle-Competition/models/best_full_model.pt')
        # Reset the early stopping count
        early_stopping_count = 0
    else:
        # Increment the early stopping count
        early_stopping_count += 1
        
    if early_stopping_count >= early_stopping_threshold:
        # The validation loss has not improved for `early_stopping_threshold` consecutive epochs, so stop training
        print("Early stopping!")
        break
```

## Construct Validation Loader
```python
# Define the validation seed edges:
edge_label_index = val_data['session', 'event', 'aid'].edge_label_index
edge_label = val_data['session', 'event', 'aid'].edge_label

val_loader = LinkNeighborLoader(
    data=val_data,  # TODO
    num_neighbors=[20, 10],  # TODO
    #neg_sampling_ratio=0.0,  # TODO
    edge_label_index=(('session', 'event', 'aid'), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

sampled_data = next(iter(val_loader))

print("Sampled mini-batch:")
print("===================")
print(sampled_data)

assert sampled_data['session', 'event', 'aid'].edge_label_index.size(1) == 3 * 128
assert sampled_data['session', 'event', 'aid'].edge_label.min() >= 0
assert sampled_data['session', 'event', 'aid'].edge_label.max() <= 2
```

## Compute AUC Scores
```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# You need the labels to binarize
labels = [0, 1, 2]

preds = []
ground_truths = []

for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        _, pred, ground_truth = test(data=sampled_data.to(device))
        preds.append(pred)
        ground_truths.append(ground_truth)
        # TODO: Collect predictions and ground-truths and write them into
        # `preds` and `ground_truths`.
        #raise NotImplementedError

pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

# Binarize `pred` with shape (n_samples, n_classes)
pred = label_binarize(pred.astype(int), classes=labels)

# Binarize `ground_truth` with shape (n_samples, n_classes)
ground_truth = label_binarize(ground_truth, classes=labels)

auc = roc_auc_score(ground_truth, pred, multi_class='ovo', average='weighted')
print()
print(f"Validation AUC: {auc:.4f}")
```
