import argparse
import os
import json
import torch
import numpy as np

from apps.data import get_dataloader
from neuralif.models import NeuralIF
from krylov.cg import preconditioned_conjugate_gradient
from krylov.preconditioner import LearnedPreconditioner

@torch.inference_mode()
def validate(baseline_checkpoint_path, top_k_checkpoint_path, dataset_path, K, sample_idx):
    """
    Performs a detailed validation on a single data sample to compare the baseline
    model with the new Top-K fill-in model. This runs entirely in float32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 1. Load Data (will be float32) ---
    print("--- 1. Loading Data ---")
    test_loader_no_fill_in = get_dataloader(dataset_path, batch_size=1, mode="test", add_fill_in=False)
    original_data = test_loader_no_fill_in.dataset[sample_idx]
    
    test_loader_with_fill_in = get_dataloader(dataset_path, batch_size=1, mode="test", add_fill_in=True, fill_in_k=K)
    augmented_data = test_loader_with_fill_in.dataset[sample_idx]

    print(f"Original number of edges: {original_data.num_edges}")
    print(f"Number of edges after adding K={K} fill-in candidates: {augmented_data.num_edges}")
    print("-" * 30)

    # --- 2. Load Models (float32) ---
    print("\n--- 2. Loading Models ---")
    baseline_model = load_model(baseline_checkpoint_path, device)
    top_k_model = load_model(top_k_checkpoint_path, device)
    print("-" * 30)

    # --- 3. Analyze the Baseline Model ---
    print("\n--- 3. Analyzing BASELINE Model ---")
    prec_baseline = LearnedPreconditioner(original_data.to(device), baseline_model)
    nnz_baseline = prec_baseline.nnz
    print(f"NNZ of the final preconditioner from the baseline model: {nnz_baseline}")
    
    # --- 4. Analyze the Top-K Model ---
    print("\n--- 4. Analyzing TOP-K Model ---")
    prec_top_k = LearnedPreconditioner(augmented_data.to(device), top_k_model, drop_tol=1e-4)
    nnz_top_k = prec_top_k.nnz
    print(f"NNZ of the final preconditioner from the Top-K model (after thresholding): {nnz_top_k}")
    print("-" * 30)

    # --- 5. Compare Solver Performance (in float32) ---
    print("\n--- 5. Comparing Solver Performance ---")
    # Prepare data for the solver, ensuring float32
    A_coo = torch.sparse_coo_tensor(original_data.edge_index, original_data.edge_attr[:, 0],
                                   (original_data.num_nodes, original_data.num_nodes),
                                   dtype=torch.float32).to('cpu')
    A = A_coo.to_sparse_csr()
    b = original_data.x[:, 0].squeeze().to(torch.float32).to('cpu')
    solution = original_data.s.to(torch.float32).squeeze().to('cpu') if hasattr(original_data, "s") else None
    solver_settings = {"max_iter": 2 * original_data.num_nodes, "x0": None, "rtol": 1e-6}

    # Run solver with the baseline preconditioner
    res_baseline, _ = preconditioned_conjugate_gradient(A, b, M=prec_baseline, x_true=solution, **solver_settings)
    iters_baseline = len(res_baseline) - 1
    print(f"Iterations required with BASELINE preconditioner: {iters_baseline}")

    # Run solver with the Top-K preconditioner
    res_top_k, _ = preconditioned_conjugate_gradient(A, b, M=prec_top_k, x_true=solution, **solver_settings)
    iters_top_k = len(res_top_k) - 1
    print(f"Iterations required with TOP-K preconditioner: {iters_top_k}")
    print("-" * 30)

    if iters_top_k < iters_baseline:
        print("\n✅ SUCCESS: The Top-K model produced a more effective preconditioner!")
    else:
        print("\nℹ️ INFO: The Top-K model did not improve upon the baseline for this sample.")


def load_model(checkpoint_path, device):
    """Helper function to load a trained model."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    config_path = os.path.join(checkpoint_path, "config.json")
    weights_path = os.path.join(checkpoint_path, "final_model.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    with open(config_path) as f:
        config = json.load(f)

    model_arg_keys = ["latent_size", "message_passing_steps", "skip_connections", "augment_nodes", "activation", "aggregate", "two_hop", "edge_features"]
    model_args = {k: v for k, v in config.items() if k in model_arg_keys}
    
    model = NeuralIF(**model_args)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and compare a Top-K fill-in model against a baseline.")
    parser.add_argument("--baseline_checkpoint", type=str, required=True, help="Path to the results folder of the baseline model.")
    parser.add_argument("--top_k_checkpoint", type=str, required=True, help="Path to the results folder of the Top-K model.")
    parser.add_argument("--dataset", type=str, default="data/synthetic_small", help="Path to the dataset directory.")
    parser.add_argument("--k", type=int, default=5, help="The 'K' value used for the Top-K model.")
    parser.add_argument("--sample_idx", type=int, default=0, help="The index of the test sample to analyze (e.g., 0 for the first sample).")
    args = parser.parse_args()
    
    validate(args.baseline_checkpoint, args.top_k_checkpoint, args.dataset, args.k, args.sample_idx)