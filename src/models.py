import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn.aggr import SumAggregation

from src.conv import GNN_node, GNN_node_Virtualnode

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        # Definisci anche altre parti del modello se necessario (ad esempio, head per la classificazione)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
            self.encoder = nn.Sequential(self.gnn_node, self.gnn_node)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
            self.encoder = nn.Sequential(self.gnn_node, nn.ReLU(), self.gnn_node)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            gate = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), 
                                          torch.nn.BatchNorm1d(2*emb_dim), 
                                          torch.nn.ReLU(), 
                                          torch.nn.Linear(2*emb_dim, 1))
            self.pool = AttentionalAggregation(gate_nn=gate )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        # Estrae le feature dei nodi e l'edge_index dal batch
        x, edge_index = batched_data.x, batched_data.edge_index
        
        # Applica l'encoder: supponendo che self.encoder sia ad esempio una lista di due layer,
        # eseguiamo ciascun layer in sequenza
        h = self.encoder[0](x, edge_index)
        h = self.encoder[1](h, edge_index)
        
        # Aggrega a livello di grafo le feature ottenute usando il global add pool;
        # batched_data.batch contiene gli indici per aggregare i nodi nel grafo di appartenenza
        h_graph_encoded = global_add_pool(h, batched_data.batch)
        
        # Alternativamente, se hai un ulteriore ramo di elaborazione:
        h_node = self.gnn_node(batched_data)  # Ad esempio, una parte del modello che elabora i nodi
        h_graph = self.pool(h_node, batched_data.batch)  # Pooling a livello di grafo
        
        # Infine, passa la rappresentazione aggregata attraverso il layer finale per la predizione
        output = self.graph_pred_linear(h_graph)
        
        return output
