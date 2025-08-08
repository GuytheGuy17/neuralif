import os
import torch
from torch_geometric.data import Dataset, DataLoader
from scipy.sparse import coo_matrix

# Import the new preprocessing transform
from preprocess import AddHeuristicFillIn

class FolderDataset(Dataset):
    """
    A PyTorch Geometric dataset that loads graph data from a folder of .pt files.
    """
    def __init__(self, folder_path, transform=None, pre_transform=None):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
        super().__init__(folder_path, transform, pre_transform)

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(os.path.join(self.folder_path, self.files[idx]))
        return data

def get_dataloader(dataset_path, batch_size, mode="train", add_fill_in=False, fill_in_k=0):
    """
    Creates a DataLoader for the specified dataset split.

    Args:
        add_fill_in (bool): If True, applies the heuristic fill-in transform.
        fill_in_k (int): The 'K' parameter for the fill-in transform.
    """
    folder_path = os.path.join(dataset_path, mode)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"CRITICAL: No '{mode}' directory found in the dataset path: {dataset_path}")

    # --- START OF MODIFICATION ---
    # Apply the new transform if requested
    transform = AddHeuristicFillIn(K=fill_in_k) if add_fill_in else None
    dataset = FolderDataset(folder_path=folder_path, transform=transform)
    # --- END OF MODIFICATION ---

    if len(dataset) == 0:
        raise FileNotFoundError(f"CRITICAL: No '.pt' files were found in the directory: {folder_path}")
    
    print(f"Successfully created a '{mode}' dataloader.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")
    if transform:
        print(f" -> Applying transform: {transform}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))

# This utility is assumed to exist from your original codebase
def graph_to_matrix(data):
    A = torch.sparse_coo_tensor(
        data.edge_index,
        data.edge_attr[:, 0], # Use only the first feature (the value)
        (data.num_nodes, data.num_nodes)
    )
    b = data.x[:, 0]
    return A, b