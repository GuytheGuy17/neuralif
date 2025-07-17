import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

def matrix_to_graph(scipy_matrix, b_vector):
    """Efficiently converts a SciPy sparse matrix and a vector 'b' into a PyTorch Geometric Data object."""
    edge_index, edge_attr = from_scipy_sparse_matrix(scipy_matrix)
    node_features = torch.tensor(b_vector, dtype=torch.float).view(-1, 1)
    data = Data(x=node_features, edge_index=edge_index.long(), edge_attr=edge_attr.float())
    return data

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
    
    print(f"Successfully created a '{mode}' dataloader.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))