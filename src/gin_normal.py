import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, AttentionalAggregation

### GIN convolution
class SimpleGINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(SimpleGINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.eps = 0
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GNN wrapper
class GNN_Costume(torch.nn.Module):
    def __init__(self, num_class, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, graph_pooling="mean"):
        super(GNN_Costume, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.emb_dim = emb_dim
        self.num_class = num_class

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(num_layer):
            self.convs.append(SimpleGINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "attention":
            gate = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, 1)
            )
            self.pool = AttentionalAggregation(gate_nn=gate)
        else:
            raise ValueError(f"Unknown pooling type: {graph_pooling}")

        self.graph_pred_linear = torch.nn.Linear(emb_dim, num_class)

        self.dropout = torch.nn.Dropout(drop_ratio)
        self.activation = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer != self.num_layer - 1:
                h = self.activation(h)
                h = self.dropout(h)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.stack(h_list[1:], dim=0).sum(dim=0)

        h_graph = self.pool(node_representation, batch)
        return self.graph_pred_linear(h_graph)
