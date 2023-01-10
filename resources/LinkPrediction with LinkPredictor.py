class GNNStack(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
      super(GNNStack, self).__init__()
      # define layers
      self.convs = torch.nn.ModuleList()
      self.convs.append(SAGEConv(input_dim, hidden_dim))
      self.dropout = dropout
      self.num_layers = num_layers
      self.emb = emb

      assert (self.num_layers >= 1), 'Number of layers is not >= 1'

      for layer in range(self.num_layers - 1):
          self.convs.append(SAGEConv(hidden_dim, hidden_dim))
      
      # post-message-passing
      self.post_mp = torch.nn.Sequential(
          torch.nn.Linear(hidden_dim, hidden_dim),
          torch.nn.Dropout(self.dropout),
          torch.nn.Linear(hidden_dim, output_dim)
      )

  def forward(self, x, edge_index):
      for i in range(self.num_layers):
          x = self.convs[i](x, edge_index)
          x = F.relu(x)
          x = F.dropout(x, p=self.dropout, training=self.training)
      x = self.post_mp(x)

      # return node embeddings after post-message passing if specified
      if self.emb:
          return x
      # else return class probabilities for each node
      return F.log_softmax(x, dim=1)
    
####################################### 
    
class LinkPredictor(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
      super(LinkPredictor, self).__init__()
      # define layers
      self.linears = torch.nn.ModuleList()
      self.linears.append(torch.nn.Linear(in_channels, hidden_channels))
      for _ in range(num_layers - 2):
          self.linears.append(torch.nn.Linear(hidden_channels, hidden_channels))
      self.linears.append(torch.nn.Linear(hidden_channels, out_channels))
      self.dropout = dropout

  def reset_parameters(self):
      for linear in self.linears:
        linear.reset_parameters()

  def forward(self, x_i, x_j):
      x = x_i * x_j # element-wise embeddings
      for linear in self.linears[:-1]:
        x = linear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
      x = self.linears[-1](x)
      return torch.sigmoid(x)

####################################### 
    
from torch_geometric.nn.models.signed_gcn import negative_sampling
def train(model, link_predictor, emb, edge_index, pos_train_edge, batch_size, optimizer):
    '''
    Runs offline training for model, link_predictor and node embeddings given the message
    edges and supervision edges.
    1. Updates node embeddings given the edge index (i.e. the message passing edges)
    2. Computes predictions on the positive supervision edges
    3. Computes predictions on the negative supervision edges (which are sampled)
    4. Computes the loss on the positive and negative edges and updates parameters
    '''
    model.train()
    link_predictor.train()

    train_losses = []
    
    for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True):
        optimizer.zero_grad()

        node_emb = model(emb, edge_index)

        pos_edge = pos_train_edge[edge_id].T
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])

        neg_edge = negative_sampling(edge_index,
                                     num_nodes=emb.shape[0],
                                     num_neg_samples=edge_id.shape[0],
                                     method='dense')
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])

        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return sum(train_losses) / len(train_losses)
  
####################################### 
  
def test(model, predictor, emb, edge_index, split_edge, batch_size, evaluator):
    '''
    Evaluates model on positive and negative test edges
    1. Computes the updated node embeddings given the edge index (i.e. the message passing edges)
    2. Computes predictions on the positive and negative edges
    3. Calculates hits @ k given predictions using the ogb evaluator
    '''
    model.eval()
    predictor.eval()

    node_emb = model(emb, edge_index)

    pos_test_edge = split_edge['test']['edge'].to(emb.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(emb.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K # using the evaluator function in the ogb.linkproppred package
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred
        })[f'hits@{K}']

        results[f'hits@{K}']

    return results

#######################################  
  
train_graph = torch.load('train.pt')
val_graph = torch.load('val.pt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optim_wd = 0
epochs = 300
hidden_dim = 1024
dropout = 0.3
num_layers = 2
lr = 1e-5
node_emb_dim = 1
batch_size = 1024


train_graph = train_graph.to(device)
val_graph = val_graph.to(device)


model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device) # the graph neural network that takes all the node embeddings as inputs to message pass and agregate
link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device)


optimizer = torch.optim.Adam(
    list(model.parameters()) + list(link_predictor.parameters()),
    lr=lr, weight_decay=optim_wd
)


train_loss = train(
	model, 
	link_predictor, 
	torch.tensor(train_graph.x).float().to(device), 
	train_graph.edge_index, 
	train_graph.pos_edge_label_index.T, 
	batch_size, 
	optimizer
)
