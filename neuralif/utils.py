import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
import scipy
import scipy.sparse
import time
import os
import psutil

# Add this import at the top of your utils.py file
from torch_geometric.utils import coalesce, remove_self_loops, to_torch_coo_tensor, to_edge_index

# It's good practice to import TYPE_CHECKING for type hints that would cause circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neuralif.models import NeuralIF


class TwoHop(torch_geometric.transforms.BaseTransform):
    def forward(self, data):
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        num_nodes = data.num_nodes
        adj = to_torch_coo_tensor(edge_index, size=num_nodes)
        adj = adj @ adj
        edge_index2, _ = to_edge_index(adj)
        edge_index2, _ = remove_self_loops(edge_index2)
        edge_index = torch.cat([edge_index, edge_index2], dim=1)
        if edge_attr is not None:
            edge_attr2 = edge_attr.new_zeros(edge_index2.size(1), *edge_attr.size()[1:])
            edge_attr = torch.cat([edge_attr, edge_attr2], dim=0)
        data.edge_index, data.edge_attr = coalesce(edge_index, edge_attr, num_nodes)
        return data


def gradient_clipping(model, clip=None):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    return total_norm


def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(dictionary))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_non_zeros(P):
    return torch.linalg.norm(P.flatten(), ord=0)


def frob_norm_sparse(data):
    return torch.pow(torch.sum(torch.pow(data, 2)), 0.5)


def filter_small_values(A, threshold=1e-5):
    return torch.where(torch.abs(A) < threshold, torch.zeros_like(A), A)


def plot_graph(data):
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    filtered_edges = list(filter(lambda x: x[0] != x[1], g.edges()))
    nx.draw(g, edgelist=filtered_edges)
    plt.show()


def print_graph_statistics(data):
    print(data.validate())
    print(data.is_directed())
    print(data.num_nodes)


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper


def test_spd(A):
    np.testing.assert_allclose(A, A.T, atol=1e-6)
    assert np.linalg.eigh(A)[0].min() > 0


def kA_bound(cond, k):
    return 2 * ((torch.sqrt(cond) - 1) / (torch.sqrt(cond) + 1)) ** k


def eigenval_distribution(P, A):
    if P is None:
        return torch.linalg.eigh(A)[0]
    else:
        return torch.linalg.eigh(P @ A @ P.T)[0]


def condition_number(P, A, invert=False, split=True):
    if invert:
        if split:
            P = torch.linalg.solve_triangular(P, torch.eye(P.size()[0], device=P.device, requires_grad=False), upper=False)
        else:
            P = torch.linalg.inv(P)
    if split:
        return torch.linalg.cond(P @ A @ P.T)
    else:
        return torch.linalg.cond(P @ A)


def l1_output_norm(P):
    return torch.sum(torch.abs(P)) / P.size()[0]


def rademacher(n, m=1, device=None):
    if device is None:
        return torch.sign(torch.randn(n, m, requires_grad=False))
    else:
        return torch.sign(torch.randn(n, m, device=device, requires_grad=False))


def torch_sparse_to_scipy(A):
    A = A.coalesce()
    d = A.values().squeeze().numpy()
    i, j = A.indices().numpy()
    A_s = scipy.sparse.coo_matrix((d, (i, j)))
    return A_s


def gershgorin_norm(A, graph=False):
    if graph:
        row, col = A.edge_index
        agg = torch_geometric.nn.aggr.SumAggregation()
        row_sum = agg(torch.abs(A.edge_attr), row)
        col_sum = agg(torch.abs(A.edge_attr), col)
    else:
        n = A.size()[0]
        row_sum = torch.sum(torch.abs(A.to_dense()), dim=1)
        col_sum = torch.sum(torch.abs(A.to_dense()), dim=0)
    gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
    return gamma


time_function = lambda: time.perf_counter()


### START OF NEW CODE ###
def load_checkpoint(checkpoint_path: str, model_class: 'NeuralIF', device: str = 'cpu', weights_name: str = "final_model.pt"):
    """
    Loads a model and its training configuration from a checkpoint folder.

    Args:
        checkpoint_path (str): Path to the folder containing the model and config.
        model_class (torch.nn.Module): The model class to instantiate (e.g., NeuralIF).
        device (str): The device to load the model onto ('cpu' or 'cuda:0').
        weights_name (str): The name of the weights file (e.g., 'final_model.pt').

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded model.
            - config (dict): The training configuration dictionary.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    config_path = os.path.join(checkpoint_path, "config.json")
    weights_path = os.path.join(checkpoint_path, weights_name)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Define the keys that are relevant for model architecture
    model_arg_keys = [
        "latent_size", "message_passing_steps", "skip_connections", "augment_nodes",
        "activation", "aggregate", "two_hop", "edge_features"
    ]
    model_args = {k: v for k, v in config.items() if k in model_arg_keys}
    
    # Instantiate the model with arguments from the config file
    model = model_class(**model_args)
    # Load the state dictionary onto the specified device
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device) # Ensure the model is on the correct device
    model.eval() # Set to evaluation mode

    return model, config
### END OF NEW CODE ###