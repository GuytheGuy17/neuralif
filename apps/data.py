import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from scipy.sparse import coo_matrix

# Import the new preprocessing transform, assuming it's in apps/preprocess.py
try:
    from preprocess import AddHeuristicFillIn
except ImportError:
    # Provide a dummy class if preprocess.py doesn't exist, so old code doesn't break.
    print("Warning: 'apps/preprocess.py' not found. Fill-in functionality will be disabled.")
    class AddHeuristicFillIn:
        def __init__(self, K=0): pass
        def __call__(self, data): return data

# This is the function that was missing, causing the ImportError.
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


# This utility is used by the loss function
def graph_to_matrix(data):
    """
    Converts a PyTorch Geometric Data object back to a PyTorch sparse tensor.
    """
    # If edge_attr has multiple features (like the is_fill_in flag), only use the first one.
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
        data = torch.load(os.path.join(self.folder_path, self.files[idx]))
        return data

def get_dataloader(dataset_path, batch_size, mode="train", add_fill_in=False, fill_in_k=0):
    """
    Creates a DataLoader for the specified dataset split, with optional support
    for the heuristic fill-in transform.
    """
    folder_path = os.path.join(dataset_path, mode)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"CRITICAL: No '{mode}' directory found in the dataset path: {dataset_path}")

    # --- DEFINITIVE MERGED LOGIC ---
    # Apply the AddHeuristicFillIn transform if the user requests it.
    transform = AddHeuristicFillIn(K=fill_in_k) if add_fill_in else None
    dataset = FolderDataset(folder_path=folder_path, transform=transform)
    # --- END OF DEFINITIVE MERGED LOGIC ---

    if len(dataset) == 0:
        raise FileNotFoundError(f"CRITICAL: No '.pt' files were found in the directory: {folder_path}")
    
    print(f"Successfully created a '{mode}' dataloader.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")
    if transform:
        print(f" -> Applying transform: {transform}")

    # Use num_workers > 0 for performance, but it can cause issues in some environments.
    # Set to 0 if you encounter multiprocessing errors.
    num_workers = 2 if torch.cuda.is_available() else 0

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=num_workers)
