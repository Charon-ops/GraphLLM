import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GATConv as PYGGATConv

class GCN(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x

class GAT2(torch.nn.Module):
    def __init__(self, num_feat, hidden_dimension, num_layers, num_class, dropout, attn_drop, num_of_heads = 1, num_of_out_heads = 1, norm = None):
        super().__init__()
        self.layers = []
        self.bns = []
        if num_layers == 1:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = False, dropout=attn_drop)
        else:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = True, dropout=attn_drop)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))
        self.layers.append(self.conv1)
        for _ in range(num_layers - 2):
            self.layers.append(
                PYGGATConv(hidden_dimension * num_of_heads, hidden_dimension, num_of_heads, concat = True, dropout = dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if num_layers > 1:
            self.layers.append(PYGGATConv(hidden_dimension * num_of_heads, num_class, heads=num_of_out_heads,
                             concat=False, dropout=attn_drop).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)
        self.norm = norm 
        self.num_layers = num_layers
        self.with_bn = True if self.norm == 'BatchNorm' else False
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x