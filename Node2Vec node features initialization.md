# Node2Vec Features Initialization 
## Using Node2Vec model to preprocessing node features for non-feature's node types
Node2vec is a method for learning low-dimensional representations of nodes in a graph.
It is designed to work with a **single type of node**, but it can be applied to different types of nodes by **treating each node type as a separate graph**, and learning embeddings for each one separately. In this way, multiple node types can be incorporated into a single graph representation, by concatenating the embeddings of each node type.
This will allow to the model to learn a multi-typed node structure.
It is worth mentioning that this works well if the multiple node types are connected to each other in a meaningful way. Otherwise, it would be better to treat them as separate graph.

Example graph:
|   |session|aid|
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 1 | 1 |
| 4 | 1 | 0 |
| 5 | 2 | 0 |

## `session` nodes
[later]

## `aid` nodes
|   |session|aid|prev_aid|
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 |
| 2 | 0 | 2 | 1 |
| 3 | 1 | 1 | 1 |
| 4 | 1 | 0 | 1 |
| 5 | 2 | 0 | 0 |

```python
>>> edge_index
tensor([[0, 1, 2, 1, 0, 0],
        [0, 0, 1, 1, 1, 0]])
        
>>> edge_index.shape
torch.Size([2, 6]) #[2, num_edges]
```

```python
>>> data.num_nodes
3 #number of `aid` unique node [0, 1, 2]
```
```python
model = Node2Vec(edge_index=data.edge_index,
                 embedding_dim=..., 
                 walk_length=...,
                 context_size=...,
                 walks_per_node=...,
                 num_negative_samples=..., 
                 p=0.2, q=0.4, sparse=True).to(device)
```
