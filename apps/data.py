import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

def matrix_to_graph_sparse(A, b):
    """Internal helper to convert a SciPy COO matrix to a PyG Data object."""
    edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
    edge_attr = torch.tensor(A.data, dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(b, dtype=torch.float).unsqueeze(1)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def matrix_to_graph(A, b):
    """Converts a SciPy sparse matrix to a PyG Data object."""
    return matrix_to_graph_sparse(coo_matrix(A), b)

def graph_to_matrix(data):
    """Converts a PyG Data object back to a sparse torch tensor."""
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
    b = data.x.squeeze()
    return A, b

class FolderDataset(torch.utils.data.Dataset):
    """A generic dataset that loads all '.pt' graph files from a specified folder."""
    def __init__(self, folder_path):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(folder_path, '*.pt')))
        if not self.files:
            raise FileNotFoundError(f"CRITICAL: No '.pt' files were found in the directory: {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)

def get_dataloader(dataset_path, batch_size=1, mode="train", **kwargs):
    """
    Creates a DataLoader by loading graph files from the provided directory path.
    e.g., dataset_path/train or dataset_path/val
    """
    folder_path = os.path.join(dataset_path, mode)
    dataset = FolderDataset(folder_path=folder_path)
    
    # This function now ignores unused parameters like 'n' and 'spd' for simplicity.
    
    print(f"Successfully created a '{mode}' dataloader.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))