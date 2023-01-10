# OTTO-Multi-Objective-Recommender-System
A [Kaggle Competition](https://www.kaggle.com/competitions/otto-recommender-system/overview)
# Version Logs
> Follow [this](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py) example provided by PyTorch Geometric Team

## [DEVELOP1.8.1](https://colab.research.google.com/drive/1DWLNG4t_VM2_6QRQuAl8WdFgANs4L3ec)
  ### Changes
  > Minor changes
  ### Issues

  ### Futures

## [DEVELOP1.8](https://colab.research.google.com/drive/1-0RupSIL7Z5gO3VuaJmirDZn2uRPtGnd#scrollTo=ztCYkbwc-z5g&uniqifier=1)
  ### Changes
  - [X] Use preprocessed `edge_index` and `edge_label` (combined train+test nodes/edges)
  - [X] No longer `torch.nn.Embedding` layer for training in every feed forward
  ### Issues
  - [ ] Memory crash while split data with `T.RandomLinkSplit` in standard high GPU mode (fine in premium >;3)
  - [X] `IndexError: Item index larger than the largest item index`
     > Solved - more detail in notion ;ppp
  ### Futures
  - [ ] Can use node in/out degree as `x`
  - [ ] Consider new model structure also new train/test procedure
  - [ ] Need new function for create submission file
  - [ ] [Here](https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/index.html) algorithms that support multiple edge types in heterogenous graph

## [DEVELOP1.7](https://colab.research.google.com/drive/1OYwFL1Nb0QpBH4AHctdeHqjz_napt5xX#scrollTo=aED_TAobTF6d)
  - [X] ~⚠️ Unstable - System RAM and GPU RAM overload!~
    > UPDATE: Fine to use now ;ppp
## [DEVELOP1.6](https://colab.research.google.com/drive/1LMM4KUubrtgFevA8BQiTAU_4AgC4FOlM#scrollTo=hbuDkO-Nh8kA&uniqifier=2)
  - [X] Added feature embeddings layer in Model
  - [X] Added Polars for better time and memory efficiency
  - [X] Bring back our beloved pre-step(in mini-batch) for lazy initialization
## [DEVELOP1.5](https://colab.research.google.com/drive/1QLt8OBWYSWmHiQQ1OCxaHaN96PyhQTEN#scrollTo=WDkdfZnxEB02&uniqifier=1):
  - [X] Mini-Batch for a very large data (example [here](https://colab.research.google.com/drive/1ksnVuQBPZA7W0nbOokz6nqB0EdDKWOUk#scrollTo=Vi25Z7lFPPjc)) with `LinkNeighborLoader`
  - [X] Added evaluate fucntion (AUC)
  - [X] Now can compute on CUDA GPU
## [DEVELOP1.4](https://colab.research.google.com/drive/1DMHdyKAxoJLJlOynj0p8fUqnySYLpucG#scrollTo=yf2nIQqrAy8y&uniqifier=1): 
  - [x] Back to 1 edge type and predicting the edges event - example [here](https://colab.research.google.com/drive/1ksnVuQBPZA7W0nbOokz6nqB0EdDKWOUk)
  - [X] Added code to load and construct graph for test dataset 
  - [ ] ~Memory-inefficient when call recommendation function~
    > Canceled
## [DEVELOP1.3](https://colab.research.google.com/drive/11m9ztUNqBqe4f8dAWDmJe_7IAIT3yJyv#scrollTo=aED_TAobTF6d): 
  - [x] Added recommendation function and multiple edge types
## [DEVELOP1.2](https://colab.research.google.com/drive/19ku8TR77OhTH0nHgLiEr_PepVnjgCiws): 
  - [x] New code for modeling
---
* [Link Prediction on Heterogeneous Graphs with PyG](https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70)
* [Example AutoEncoder.ipynb](https://colab.research.google.com/drive/1nyufporgJp-j4BqZ6jYwgdMLyz_bJ_es#scrollTo=WjdGbaa8LdU9)
* [Recommender Systems with Graph Neural Networks in PyG](https://colab.research.google.com/drive/1qQEcYrzWJyJpAlwCMcJdFNPlYlyzHPdF#scrollTo=ktxdLosxtgZd)
* https://blog.dataiku.com/graph-neural-networks-link-prediction-part-two

<img src="https://img.freepik.com/free-vector/aesthetic-ocean-background-pastel-glitter-design-vector_53876-157553.jpg?w=2000" width="550"/>
