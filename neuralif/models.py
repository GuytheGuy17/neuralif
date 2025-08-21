import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import aggr

from neuralif.utils import TwoHop

# This class transforms the edge indices to ensure they represent a lower triangular matrix.
# It modifies the edge indices and attributes to maintain the lower triangular structure.
class ToLowerTriangular(torch_geometric.transforms.BaseTransform):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is None: edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        row, col = edge_index
        mask = row >= col
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        return data

# This function computes the loss based on the model output and the data.
# It handles both the sketched loss and the improved sketch with PCG loss.
def augment_features(data):
    # This function needs to be aware of the 2D edge features
    if data.edge_attr.dim() == 1: data.edge_attr = data.edge_attr.unsqueeze(-1)
    if data.edge_attr.size(1) == 1: # If only value is present, add placeholder fill-in feature
        fill_in_feature = torch.zeros(data.edge_attr.size(0), 1, device=data.edge_attr.device)
        data.edge_attr = torch.cat([data.edge_attr, fill_in_feature], dim=1)
        
    x, edge_index, edge_attr_val = data.x, data.edge_index, data.edge_attr[:, 0]
    num_nodes = data.num_nodes
    data.x = torch.arange(num_nodes, device=x.device, dtype=torch.float).unsqueeze(1)
    data = torch_geometric.transforms.LocalDegreeProfile()(data)
    row, col = edge_index
    diag_mask = row == col
    diag_map = torch.zeros((num_nodes, 1), device=x.device, dtype=x.dtype)
    if torch.any(diag_mask): diag_map[row[diag_mask], 0] = torch.abs(edge_attr_val[diag_mask].squeeze())
    off_diag_attr = torch.abs(edge_attr_val.clone())
    off_diag_attr[diag_mask] = 0
    row_sums = aggr.SumAggregation()(off_diag_attr, row, dim_size=num_nodes, dim=0).unsqueeze(1)
    row_maxes = aggr.MaxAggregation()(off_diag_attr, row, dim_size=num_nodes, dim=0).unsqueeze(1)
    dominance = diag_map / (row_sums + 1e-8)
    decay = diag_map / (row_maxes + 1e-8)
    dominance_feat = torch.nan_to_num(dominance / (dominance + 1), nan=1.0)
    decay_feat = torch.nan_to_num(decay / (decay + 1), nan=1.0)
    data.x = torch.cat([data.x, dominance_feat, decay_feat], dim=1)
    return data

## This class defines a Multi-Layer Perceptron (MLP) 
class MLP(nn.Module):
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        width = list(filter(lambda x: x > 0, width))
        if not width or len(width) < 2: self.net = nn.Identity(); return
        ## Create a sequence of linear layers with the specified activation function.
        ## The last layer can be activated or not based on the 'activate_final' flag.
        lls = nn.ModuleList()
        for k in range(len(width)-1):
            lls.append(nn.Linear(width[k], width[k+1], bias=True))
            if k != (len(width)-2) or activate_final:
                if activation == "relu": lls.append(nn.ReLU())
                elif activation == "tanh": lls.append(nn.Tanh())
                else: raise NotImplementedError(f"Activation '{activation}' not supported")
        if layer_norm: lls.append(nn.LayerNorm(width[-1]))
        self.net = nn.Sequential(*lls)

        ## This method performs the forward pass through the MLP.
    # It applies the sequence of linear layers and activation functions.
    def forward(self, x): return self.net(x)
    
# This class defines a GraphNet layer that processes node and edge features.
# It aggregates edge features and updates node embeddings based on the specified aggregation method.
class GraphNet(nn.Module):
    def __init__(self, node_features, edge_features, global_features, hidden_size,
                 aggregate, activation, edge_features_out):
        super().__init__()
        ## The aggregation method is chosen based on the provided string.
        if aggregate == "sum": self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean": self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max": self.aggregate = aggr.MaxAggregation()
        else: raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")
        edge_input_dim = edge_features + (2 * node_features) + global_features
        node_input_dim = node_features + edge_features_out + global_features
        # The edge block processes the edge features and the node features.
        # The node block processes the node features and the aggregated edge features.
        self.edge_block = MLP([edge_input_dim, hidden_size, edge_features_out], activation=activation)
        self.node_block = MLP([node_input_dim, hidden_size, node_features], activation=activation)
    
    # This method performs the forward pass through the GraphNet layers.
    # It processes the edge and node features, aggregates them, and returns the updated embeddings.
    def forward(self, x, edge_index, edge_attr, g=None):
        row, col = edge_index
        if g is not None: edge_inputs = torch.cat([g.expand(x[row].shape[0], -1), x[row], x[col], edge_attr], dim=1)
        else: edge_inputs = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_embedding = self.edge_block(edge_inputs)
        aggregation = self.aggregate(edge_embedding, row, dim_size=x.size(0))
        if g is not None: node_inputs = torch.cat([g.expand(x.shape[0], -1), x, aggregation], dim=1)
        else: node_inputs = torch.cat([x, aggregation], dim=1)
        node_embeddings = self.node_block(node_inputs)
        return edge_embedding, node_embeddings, None

# This class defines a block of message passing layers for the NeuralIF model.
# It handles skip connections and the transformation of edge features.
class MP_Block(nn.Module):
    def __init__(self, skip_connections, first, last, edge_features, node_features, global_features, hidden_size, activation, aggregate):
        super().__init__()
        aggr_list = aggregate if isinstance(aggregate, list) and len(aggregate) == 2 else [aggregate, aggregate]
   
        # The first layer has no skip connections, so it only uses the node features.
        # The second layer uses the edge features from the first layer.
        # For subsequent layers, it's the hidden dimension + the original 2 features for the skip connection.
        edge_feat_in = 2 if first else edge_features + (2 if skip_connections else 0)
        # The output is just the value, so it's 1-dimensional if it's the last layer.
        edge_feat_out = 1 if last else edge_features
        # Initialize the GraphNet layers with the specified parameters.
        # The first layer uses the node features, the second layer uses the edge features.
        
        self.l1 = GraphNet(
            node_features=node_features, edge_features=edge_feat_in, global_features=global_features,
            hidden_size=hidden_size, aggregate=aggr_list[0], activation=activation,
            edge_features_out=edge_features
        )
        self.l2 = GraphNet(
            node_features=node_features, edge_features=edge_features, global_features=global_features,
            hidden_size=hidden_size, aggregate=aggr_list[1], activation=activation,
            edge_features_out=edge_feat_out
        )
    # This method is called to perform a forward pass through the block.
    # It processes the node and edge features through the two GraphNet layers.
    def forward(self, node_embedding, edge_index, edge_embedding, global_features):
        edge_embedding, node_embedding, _ = self.l1(node_embedding, edge_index, edge_embedding, g=global_features)
        edge_index_T = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_embedding, node_embedding, _ = self.l2(node_embedding, edge_index_T, edge_embedding, g=global_features)
        return edge_embedding, node_embedding, global_features


# This class defines the NeuralIF model, which is a neural network for solving linear systems.
# It uses message passing and can handle edge features.
# The model can be configured with various parameters such as the number of message passing steps,
# whether to use skip connections, and the activation function.
# The model outputs a lower triangular matrix L and its transpose U, which are used to solve
# linear systems efficiently.
# The model can also augment node features and use two-hop neighborhoods.
class NeuralIF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.augment_node_features = kwargs.get("augment_nodes", True)
        self.skip_connections = kwargs.get("skip_connections", True)
        self.use_two_hop = kwargs.get("two_hop", False)
        node_features = 8 if self.augment_node_features else 1
        mp_steps = kwargs.get("message_passing_steps", 3)
        edge_features = kwargs.get("edge_features", 16) # Increased hidden dim for edges
        self.mps = nn.ModuleList()
        for l in range(mp_steps):
            self.mps.append(MP_Block(
                node_features=node_features, edge_features=edge_features, global_features=0,
                hidden_size=kwargs.get("latent_size", 32), activation=kwargs.get("activation", "relu"),
                aggregate=kwargs.get("aggregate", "mean"), first=(l == 0), last=(l == mp_steps - 1),
                skip_connections=self.skip_connections
            ))

        # The final output is a lower triangular matrix L and its transpose U
        # The edge features are transformed to ensure they are positive and non-zero.
    def forward(self, data):
        if self.augment_node_features: data = augment_features(data)
        if self.use_two_hop: data = TwoHop()(data)
        data = ToLowerTriangular()(data)
        
        node_embedding, edge_index, edge_embedding = data.x, data.edge_index, data.edge_attr
        # The skip connection now holds both original features
        a_edges = edge_embedding.clone()
        
        for i, layer in enumerate(self.mps):
            if i > 0 and self.skip_connections:
                # Concatenate the hidden edge embedding with the original 2 features
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
            edge_embedding, node_embedding, _ = layer(node_embedding, edge_index, edge_embedding, None)
            
        return self._transform_output(node_embedding, edge_index, edge_embedding)

    # This method transforms the output of the model to create the lower triangular matrix L.
    # It ensures that the diagonal entries are positive and non-zero by applying a softplus function   
    def _transform_output(self, x, edge_index, edge_values):
        diag_mask = edge_index[0] == edge_index[1]
        softplus_result = torch.nn.functional.softplus(edge_values[diag_mask].squeeze(-1)) + 1e-6
        edge_values[diag_mask] = softplus_result.unsqueeze(-1)
        
        # Ensure edge_values is 1D for the sparse matrix construction
        edge_values_squeezed = edge_values.squeeze(-1)
        L = torch.sparse_coo_tensor(
            edge_index, edge_values_squeezed, 
            size=(x.size(0), x.size(0)), device=x.device
        ).coalesce()
        # Ensure the diagonal is positive and non-zero
        off_diagonal_values = edge_values_squeezed[~diag_mask]
        l1_penalty = torch.mean(torch.abs(off_diagonal_values))
        
        # Create the upper triangular matrix U by flipping the indices of L
        U = torch.sparse_coo_tensor(L.indices().flip(0), L.values(), L.shape).coalesce()
        
        # Package L and U into a tuple. This becomes the primary output.
        LU_factors = (L, U)

        # Return the tuple in both training and evaluation modes for a consistent output signature.
        if self.training:
            return LU_factors, l1_penalty, None
        else:
            # We also return the L1 penalty as None to match the training signature.
            return LU_factors, None, None