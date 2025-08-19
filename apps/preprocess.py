import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import numpy as np

class AddHeuristicFillIn(BaseTransform):
    """
    A PyTorch Geometric transform that adds candidate "fill-in" edges to a graph
    based on the sparsity pattern of the matrix A^2.
    """
    def __init__(self, K: int = 5):
        """
        Args:
            K (int): The number of top candidate edges to add per row.
        """
        self.K = K

    def __call__(self, data):
        """
        Applies the transform to a single data object.
        """
        if self.K == 0:
            # If K is 0, do nothing but ensure the binary feature exists for consistency
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.unsqueeze(-1)
            
            fill_in_feature = torch.zeros(data.edge_attr.size(0), 1, device=data.edge_attr.device)
            data.edge_attr = torch.cat([data.edge_attr, fill_in_feature], dim=1)
            return data

        num_nodes = data.num_nodes
        
        # Get the SciPy version of the matrix A, ensuring edge_attr is 1D
        edge_values = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr.squeeze()
        A = to_scipy_sparse_matrix(data.edge_index, edge_values, num_nodes).tocsr()
        
        # Calculate the pattern of A^2 to find candidate edges
        A_squared_pattern = (A @ A).astype(bool).tocsr()
        
        # Remove original edges to isolate new candidate edges
        candidate_edges_matrix = A_squared_pattern - A.astype(bool)
        candidate_edges_matrix.eliminate_zeros()
        
        # Use Top-K selection to choose the best new edges for each row
        new_edges_rows = []
        new_edges_cols = []
        
        for i in range(num_nodes):
            # Get the column indices of candidate edges for the current row
            row_candidates = candidate_edges_matrix.getrow(i).indices
            
            if len(row_candidates) > 0:
                # We only add edges for the lower triangle (col < row) to avoid duplicates
                selected_candidates = [col for col in row_candidates if col < i][:self.K]
                
                if selected_candidates:
                    new_edges_rows.extend([i] * len(selected_candidates))
                    new_edges_cols.extend(selected_candidates)

        if not new_edges_rows:
            # If no new edges were found, just ensure the binary feature exists and return
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.unsqueeze(-1)
            fill_in_feature = torch.zeros(data.edge_attr.size(0), 1, device=data.edge_attr.device)
            data.edge_attr = torch.cat([data.edge_attr, fill_in_feature], dim=1)
            return data

        # Create the new edge_index for the fill-in
        # We need to add both (i, j) and (j, i) to keep the graph undirected
        lower_triangle_edges = torch.tensor([new_edges_rows, new_edges_cols], dtype=torch.long)
        upper_triangle_edges = torch.tensor([new_edges_cols, new_edges_rows], dtype=torch.long)
        new_edge_index = torch.cat([lower_triangle_edges, upper_triangle_edges], dim=1)
        
        # Combine original and new edges
        combined_edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
        
        # Create the new edge attributes
        if data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)
            
        # Original edges have their value and a '0' for the is_fill_in flag
        original_attrs = torch.cat([data.edge_attr, torch.zeros(data.edge_attr.size(0), 1, device=data.edge_attr.device)], dim=1)
        
        # New edges have a value of 0 and a '1' for the is_fill_in flag
        new_attrs = torch.zeros(new_edge_index.size(1), 2, device=data.edge_attr.device)
        new_attrs[:, 1] = 1.0
        
        combined_edge_attr = torch.cat([original_attrs, new_attrs], dim=0)
        
        # Update the data object in-place
        data.edge_index = combined_edge_index
        data.edge_attr = combined_edge_attr
        
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'