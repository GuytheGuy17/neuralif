import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn # --- FIX: Use a clear and correct alias for the nn module
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import aggr

from neuralif.utils import TwoHop

# --- HELPER CLASSES AND TRANSFORMS (NOW SELF-CONTAINED) ---

class ToLowerTriangular(torch_geometric.transforms.BaseTransform):
    """A transform that keeps only the edges of the lower-triangular part of a graph's adjacency matrix."""
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is None: edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        row, col = edge_index
        mask = row >= col
        
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        return data

def augment_features(data):
    """Augments node features with structural graph properties, as described in the paper."""
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.squeeze()
    num_nodes = data.num_nodes
    
    x = torch.arange(num_nodes, device=edge_index.device, dtype=torch.float).unsqueeze(1)
    data.x = x

    data = torch_geometric.transforms.LocalDegreeProfile()(data)

    row, col = edge_index
    diag_mask = row == col
    
    diag_map = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    diag_map.scatter_(0, row[diag_mask], torch.abs(edge_attr[diag_mask]))
    diag_map = diag_map.unsqueeze(1)

    off_diag_attr = torch.abs(edge_attr.clone())
    off_diag_attr[diag_mask] = 0
    
    row_sums = aggr.SumAggregation()(off_diag_attr, row, dim_size=num_nodes).unsqueeze(1)
    row_maxes = aggr.MaxAggregation()(off_diag_attr, row, dim_size=num_nodes).unsqueeze(1)
    
    dominance = diag_map / (row_sums + 1e-8)
    decay = diag_map / (row_maxes + 1e-8)
    
    dominance_feat = torch.nan_to_num(dominance / (dominance + 1), nan=1.0)
    decay_feat = torch.nan_to_num(decay / (decay + 1), nan=1.0)
    
    data.x = torch.cat([data.x, dominance_feat, decay_feat], dim=1)
    return data

class MLP(nn.Module):
    """A simple multi-layer perceptron."""
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        layers = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i+1]))
            if i < len(width) - 2 or activate_final:
                if activation == 'relu': layers.append(nn.ReLU())
                elif activation == 'tanh': layers.append(nn.Tanh())
                else: raise NotImplementedError(f"Activation '{activation}' not supported.")
        if layer_norm: layers.append(nn.LayerNorm(width[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- CORE MODEL COMPONENTS ---

class MP_Block(nn.Module):
    """The core message-passing block."""
    def __init__(self, node_features, edge_features, hidden_size, activation, aggregate):
        super().__init__()
        # --- FIX: Use the correct alias 'gnn' for the layer ---
        self.conv1 = gnn.GCNConv(node_features, hidden_size, aggr=aggregate)
        self.act1 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.conv2 = gnn.GCNConv(hidden_size, node_features, aggr=aggregate)
        self.act2 = nn.ReLU() if activation == 'relu' else nn.Tanh()
    
    def forward(self, x, edge_index, edge_attr):
        x = self.act1(self.conv1(x, edge_index, edge_attr.squeeze()))
        x = self.act2(self.conv2(x, edge_index, edge_attr.squeeze()))
        return x, edge_attr

# --- MAIN MODEL DEFINITION ---

class NeuralIF(nn.Module):
    """The main Neural Incomplete Factorization (NeuralIF) model."""
    def __init__(self, **kwargs):
        super().__init__()
        self.augment_node_features = kwargs.get("augment_nodes", False)
        self.use_two_hop = kwargs.get("two_hop", False)
        self.use_checkpointing = kwargs.get("checkpointing", True)
        
        node_features = 8 if self.augment_node_features else 1
        latent_size = kwargs.get("latent_size", 8)
        mp_steps = kwargs.get("message_passing_steps", 3)
        activation = kwargs.get("activation", "relu")
        aggregate = kwargs.get("aggregate", "mean")

        self.node_encoder = MLP([node_features, latent_size], activation=activation)
        self.edge_encoder = MLP([1, 1], activation=activation)
        
        self.mps = nn.ModuleList()
        for _ in range(mp_steps):
            self.mps.append(MP_Block(node_features=latent_size, edge_features=1, hidden_size=latent_size, activation=activation, aggregate=aggregate))
            
        self.edge_decoder = MLP([2 * latent_size, latent_size, 1], activation=activation)
        
    def forward(self, data):
        if self.augment_node_features: data = augment_features(data)
        if self.use_two_hop: data = TwoHop()(data)
        data = ToLowerTriangular()(data)
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.node_encoder(x)
        original_edge_attr = edge_attr.clone()

        for layer in self.mps:
            if self.training and self.use_checkpointing:
                x, _ = checkpoint(layer, x, edge_index, original_edge_attr, use_reentrant=False)
            else:
                x, _ = layer(x, edge_index, original_edge_attr)

        return self._transform_output(x, edge_index)

    def _transform_output(self, x, edge_index):
        row, col = edge_index
        edge_values = self.edge_decoder(torch.cat([x[row], x[col]], dim=1))
        
        diag_mask = row == col
        edge_values[diag_mask] = torch.nn.functional.softplus(edge_values[diag_mask])
        
        L = torch.sparse_coo_tensor(
            edge_index, edge_values.squeeze(-1),
            size=(x.size(0), x.size(0)), device=x.device
        ).coalesce()
        
        if self.training:
            return L, torch.mean(torch.abs(edge_values)), None
        else:
            U = torch.sparse_coo_tensor(L.indices().flip(0), L.values(), L.shape).coalesce()
            return L, U, None