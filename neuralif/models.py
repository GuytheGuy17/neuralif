# FILE: neuralif/models.py

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import aggr

from neuralif.utils import TwoHop

# ==============================================================================
# --- HELPER CLASSES AND TRANSFORMS (Faithful Replication) ---
# ==============================================================================

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
    if data.edge_attr.dim() == 1:
        data.edge_attr = data.edge_attr.unsqueeze(-1)
        
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    num_nodes = data.num_nodes
    
    data.x = torch.arange(num_nodes, device=x.device, dtype=torch.float).unsqueeze(1)
    data = torch_geometric.transforms.LocalDegreeProfile()(data)

    row, col = edge_index
    diag_mask = row == col
    
    diag_map = torch.zeros((num_nodes, 1), device=x.device, dtype=x.dtype)
    if torch.any(diag_mask):
        diag_map[row[diag_mask], 0] = torch.abs(edge_attr[diag_mask].squeeze())

    off_diag_attr = torch.abs(edge_attr.clone())
    off_diag_attr[diag_mask] = 0
    
    # --- START: CRITICAL FIX ---
    # Added 'dim=0' to the aggregation calls. This resolves the 'invalid dimension'
    # error by explicitly telling the aggregator to operate along dimension 0
    # of the 1D squeezed tensor.
    row_sums = aggr.SumAggregation()(off_diag_attr.squeeze(-1), row, dim_size=num_nodes, dim=0).unsqueeze(1)
    row_maxes = aggr.MaxAggregation()(off_diag_attr.squeeze(-1), row, dim_size=num_nodes, dim=0).unsqueeze(1)
    # --- END: CRITICAL FIX ---
    
    dominance = diag_map / (row_sums + 1e-8)
    decay = diag_map / (row_maxes + 1e-8)
    
    dominance_feat = torch.nan_to_num(dominance / (dominance + 1), nan=1.0)
    decay_feat = torch.nan_to_num(decay / (decay + 1), nan=1.0)
    
    data.x = torch.cat([data.x, dominance_feat, decay_feat], dim=1)
    return data

class MLP(nn.Module):
    """A flexible multi-layer perceptron from the repository."""
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        width = list(filter(lambda x: x > 0, width))
        if not width or len(width) < 2:
            self.net = nn.Identity()
            return
        lls = nn.ModuleList()
        for k in range(len(width)-1):
            lls.append(nn.Linear(width[k], width[k+1], bias=True))
            if k != (len(width)-2) or activate_final:
                if activation == "relu": lls.append(nn.ReLU())
                elif activation == "tanh": lls.append(nn.Tanh())
                else: raise NotImplementedError(f"Activation '{activation}' not supported")
        if layer_norm: lls.append(nn.LayerNorm(width[-1]))
        self.net = nn.Sequential(*lls)

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# --- CORE MODEL COMPONENTS (Faithful Replication) ---
# ==============================================================================

class GraphNet(nn.Module):
    def __init__(self, node_features, edge_features, global_features, hidden_size,
                 aggregate, activation, skip_connection, edge_features_out):
        super().__init__()
        if aggregate == "sum": self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean": self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max": self.aggregate = aggr.MaxAggregation()
        else: raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")
        
        edge_input_dim = edge_features + (1 if skip_connection else 0) + (2 * node_features) + global_features
        node_input_dim = node_features + edge_features_out + global_features
        
        self.edge_block = MLP([edge_input_dim, hidden_size, edge_features_out], activation=activation)
        self.node_block = MLP([node_input_dim, hidden_size, node_features], activation=activation)
        
    def forward(self, x, edge_index, edge_attr, g=None):
        row, col = edge_index
        if g is not None:
            edge_inputs = torch.cat([g.expand(x[row].shape[0], -1), x[row], x[col], edge_attr], dim=1)
        else:
            edge_inputs = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_embedding = self.edge_block(edge_inputs)
        aggregation = self.aggregate(edge_embedding, row, dim_size=x.size(0))
        if g is not None:
            node_inputs = torch.cat([g.expand(x.shape[0], -1), x, aggregation], dim=1)
        else:
            node_inputs = torch.cat([x, aggregation], dim=1)
        node_embeddings = self.node_block(node_inputs)
        return edge_embedding, node_embeddings, None

class MP_Block(nn.Module):
    """The dual GraphNet message-passing block from the repository, mimicking LU factorization."""
    def __init__(self, skip_connections, first, last, edge_features, node_features, global_features, hidden_size, activation, aggregate):
        super().__init__()
        aggr_list = aggregate if isinstance(aggregate, list) and len(aggregate) == 2 else [aggregate, aggregate]
        edge_feat_in = 1 if first else edge_features
        edge_feat_out = 1 if last else edge_features
        
        self.l1 = GraphNet(
            node_features=node_features, edge_features=edge_feat_in, global_features=global_features,
            hidden_size=hidden_size, aggregate=aggr_list[0], activation=activation,
            skip_connection=(not first and skip_connections), edge_features_out=edge_features
        )
        self.l2 = GraphNet(
            node_features=node_features, edge_features=edge_features, global_features=global_features,
            hidden_size=hidden_size, aggregate=aggr_list[1], activation=activation,
            skip_connection=False, edge_features_out=edge_feat_out
        )
    
    def forward(self, node_embedding, edge_index, edge_embedding, global_features):
        edge_embedding, node_embedding, _ = self.l1(node_embedding, edge_index, edge_embedding, g=global_features)
        edge_index_T = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_embedding, node_embedding, _ = self.l2(node_embedding, edge_index_T, edge_embedding, g=global_features)
        return edge_embedding, node_embedding, global_features

# ==============================================================================
# --- MAIN NeuralIF MODEL DEFINITION (Faithful Replication) ---
# ==============================================================================

class NeuralIF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.augment_node_features = kwargs.get("augment_nodes", False)
        self.skip_connections = kwargs.get("skip_connections", True)
        self.use_two_hop = kwargs.get("two_hop", False)
        
        node_features = 8 if self.augment_node_features else 1
        mp_steps = kwargs.get("message_passing_steps", 3)
        
        self.mps = nn.ModuleList()
        for l in range(mp_steps):
            self.mps.append(MP_Block(
                node_features=node_features,
                edge_features=kwargs.get("edge_features", 1),
                global_features=0,
                hidden_size=kwargs.get("latent_size", 8),
                activation=kwargs.get("activation", "relu"),
                aggregate=kwargs.get("aggregate", "mean"),
                first=(l == 0),
                last=(l == mp_steps - 1),
                skip_connections=self.skip_connections
            ))
        
    def forward(self, data):
        if self.augment_node_features: data = augment_features(data)
        if self.use_two_hop: data = TwoHop()(data)
        data = ToLowerTriangular()(data)
        
        node_embedding, edge_index, edge_embedding = data.x, data.edge_index, data.edge_attr
        a_edges = edge_embedding.clone()
        
        for i, layer in enumerate(self.mps):
            if i > 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
            
            edge_embedding, node_embedding, _ = layer(node_embedding, edge_index, edge_embedding, None)
        
        return self._transform_output(node_embedding, edge_index, edge_embedding)

    def _transform_output(self, x, edge_index, edge_values):
        diag_mask = edge_index[0] == edge_index[1]
        edge_values[diag_mask] = torch.nn.functional.softplus(edge_values[diag_mask]) + 1e-3
        edge_values = edge_values.squeeze(-1)
        
        L = torch.sparse_coo_tensor(
            edge_index, edge_values,
            size=(x.size(0), x.size(0)), device=x.device
        ).coalesce()
        
        l1_penalty = torch.mean(torch.abs(edge_values))

        if self.training:
            return L, l1_penalty, None
        else:
            U = torch.sparse_coo_tensor(L.indices().flip(0), L.values(), L.shape).coalesce()
            return L, U, None