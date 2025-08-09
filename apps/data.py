import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from scipy.sparse import coo_matrix

# Add the project's root directory to the Python path for reliable imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from apps.preprocess import AddHeuristicFillIn

def matrix_to_graph(A, b):
    """
    Converts a SciPy sparse matrix and a NumPy vector into a PyTorch Geometric
    Data object, using float32 for memory efficiency.
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
    Creates a DataLoader for the specified dataset split.

    If 'add_fill_in' is True, it will prioritize loading from a pre-processed 
    directory to avoid expensive on-the-fly transformations. If the processed
    directory does not exist, it falls back to real-time transformation.
    """
    # --- START OF PRE-PROCESSING MODIFICATION ---
    # Prioritize loading pre-processed data if it exists and fill-in is requested
    processed_folder_path = os.path.join(dataset_path, "processed", mode)
    
    # Check if we should use the pre-processed data
    use_preprocessed = add_fill_in and os.path.isdir(processed_folder_path)

    if use_preprocessed:
        folder_path = processed_folder_path
        transform = None # The transform has already been applied
        print(f"INFO: Loading PRE-PROCESSED data for '{mode}' split.")
    else:
        # Fallback to original behavior: on-the-fly transform or no transform
        folder_path = os.path.join(dataset_path, mode)
        transform = AddHeuristicFillIn(K=fill_in_k) if add_fill_in else None
    # --- END OF PRE-PROCESSING MODIFICATION ---

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"CRITICAL: No '{mode}' directory found in the expected path: {folder_path}")

    # Create the dataset from the selected folder with the appropriate transform
    dataset = FolderDataset(folder_path=folder_path, transform=transform)

    if len(dataset) == 0:
        raise FileNotFoundError(f"CRITICAL: No '.pt' files were found in the directory: {folder_path}")
    
    print(f"Successfully created a '{mode}' dataloader.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")
    if transform:
        print(f" -> Applying ON-THE-FLY transform: {transform}")

    # Set num_workers=0 for robust performance in Colab and to avoid issues with some transforms
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=0)