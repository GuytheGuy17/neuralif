import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from scipy.sparse import coo_matrix

# --- START OF DEFINITIVE FIX ---
# This adds the project's root directory ('neuralif/') to the Python path.
# This allows absolute imports (like 'from apps.preprocess...') to work reliably
# no matter how the script is run.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# --- END OF DEFINITIVE FIX ---

from apps.preprocess import AddHeuristicFillIn

def matrix_to_graph(A, b):
    """
    Converts a SciPy sparse matrix and a NumPy vector into a PyTorch Geometric
    Data object.
    """
    A_coo = A.tocoo()
    edge_index = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)
    edge_attr = torch.tensor(A_coo.data, dtype=torch.float32).unsqueeze(1)
    x = torch.tensor(b, dtype=torch.float32).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def graph_to_matrix(data):
    """
    Converts a PyTorch Geometric Data object back to a PyTorch sparse tensor.
    """
    edge_values = data.edge_attr[:, 0] if data.edge_attr.dim() > 1 else data.edge_attr
    A = torch.sparse_coo_tensor(
        data.edge_index,
        edge_values,
        (data.num_nodes, data.num_nodes)
    )
    b = data.x[:, 0]
    return A, b

class FolderDataset(Dataset):
    """
    A PyTorch Geometric dataset that loads graph data from a folder of .pt files.
    """
    def __init__(self, folder_path, transform=None, pre_transform=None):
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])
        super().__init__(folder_path, transform, pre_transform)

    def len(self):
        return len(self.files)

    def get(self, idx):
        # Explicitly set weights_only=False to load PyG Data objects
        data = torch.load(os.path.join(self.folder_path, self.files[idx]), weights_only=False)
        return data

def get_dataloader(dataset_path, batch_size, mode="train", add_fill_in=False, fill_in_k=0):
    """
    Creates a DataLoader for the specified dataset split, with optional support
    for the heuristic fill-in transform.
    """
    folder_path = os.path.join(dataset_path, mode)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"CRITICAL: No '{mode}' directory found in the dataset path: {dataset_path}")

    transform = AddHeuristicFillIn(K=fill_in_k) if add_fill_in else None
    dataset = FolderDataset(folder_path=folder_path, transform=transform)

    if len(dataset) == 0:
        raise FileNotFoundError(f"CRITICAL: No '.pt' files were found in the directory: {folder_path}")
    
    print(f"Successfully created a '{mode}' dataloader.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")
    if transform:
        print(f" -> Applying transform: {transform}")

    # Set num_workers=0 for robust performance in Colab
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=0)
