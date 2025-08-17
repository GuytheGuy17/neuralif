import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from neuralif.utils import load_checkpoint
from apps.data import FolderDataset

def main(args):
    """
    Loads trained baseline and Top-K models, generates their preconditioner factors
    for a single sample, and creates a comparative plot of their sparsity patterns.
    """
    print("--- Starting Sparsity Visualization Script ---")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- 1. Load the trained models using the project's utility function ---
    print("\n[1/4] Loading models...")
    try:
        model_k0, config_k0 = load_checkpoint(args.baseline_checkpoint, device=DEVICE)
        model_k1, config_k1 = load_checkpoint(args.top_k_checkpoint, device=DEVICE)
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("Please ensure you have trained the models successfully before running this script.")
        return
    
    
    print("\n[2/4] Loading data samples...")
    # Ensure the models are in evaluation mode
    raw_dataset_path = os.path.join(args.dataset, 'test')
    raw_dataset = FolderDataset(folder_path=raw_dataset_path)
    data_k0 = raw_dataset.get(args.sample_idx).to(DEVICE)

    # The K=1 model needs the augmented graph from the 'processed' folder.
    processed_dataset_path = os.path.join(args.dataset, 'processed', 'test')
    if not os.path.exists(processed_dataset_path):
         print(f"❌ ERROR: Pre-processed data not found at {processed_dataset_path}")
         print(f"Please run 'preprocess_data.py --k {args.k}' first.")
         return
    processed_dataset = FolderDataset(folder_path=processed_dataset_path)
    data_k1 = processed_dataset.get(args.sample_idx).to(DEVICE)
    
    MATRIX_SIZE = data_k0.num_nodes
    print(f"Loaded sample matrix #{args.sample_idx} (size: {MATRIX_SIZE}x{MATRIX_SIZE})")

    # --- 3. Generate preconditioner factors for the sample ---
    print("\n[3/4] Generating preconditioner factors...")
    with torch.no_grad():
        L_k0, _, _ = model_k0(data_k0)
        L_k1_raw, _, _ = model_k1(data_k1)

    # Apply the drop tolerance to the K=1 factor to get the final sparsity
    L_k1_coalesced = L_k1_raw.coalesce()
    mask = torch.abs(L_k1_coalesced.values()) >= args.drop_tol
    
    L_k1_final = torch.sparse_coo_tensor(
        L_k1_coalesced.indices()[:, mask],
        L_k1_coalesced.values()[mask],
        L_k1_coalesced.shape
    ).coalesce()

    # --- 4. Identify fill-in and create the plot ---
    print("\n[4/4] Analyzing sparsity and generating plot...")
    # Get coordinates of non-zero elements
    coords_k0 = set(map(tuple, L_k0.coalesce().indices().t().cpu().numpy()))
    coords_k1_final = set(map(tuple, L_k1_final.indices().t().cpu().numpy()))
    
    fill_in_coords = np.array(list(coords_k1_final - coords_k0)).T
    original_coords_in_k1 = np.array(list(coords_k1_final.intersection(coords_k0))).T
    
    print(f"  -> Baseline NNZ: {len(coords_k0)}")
    print(f"  -> NeuralIF-K NNZ: {len(coords_k1_final)}")
    print(f"  -> Learned Fill-in Entries: {fill_in_coords.shape[1] if fill_in_coords.size > 0 else 0}")

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    # Panel A: Baseline (K=0) Preconditioner
    coords_k0_plot = np.array(list(coords_k0)).T
    axes[0].scatter(coords_k0_plot[1], coords_k0_plot[0], s=0.01, c='royalblue')
    axes[0].set_title(f'Baseline `NeuralIF` (K=0) Factor\nNNZ: {len(coords_k0)}', fontsize=12)
    axes[0].set_xlabel('Column Index', fontsize=10)
    axes[0].set_ylabel('Row Index', fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()

    # Panel B: Flexible (K=1) Preconditioner with Highlighted Fill-in
    if original_coords_in_k1.size > 0:
        axes[1].scatter(original_coords_in_k1[1], original_coords_in_k1[0], s=0.01, c='royalblue', label='Original Structure')
    if fill_in_coords.size > 0:
        axes[1].scatter(fill_in_coords[1], fill_in_coords[0], s=0.01, c='orangered', label='Learned Fill-in')
    axes[1].set_title(f'`NeuralIF-K` (K=1) Factor\nNNZ: {len(coords_k1_final)}', fontsize=12)
    axes[1].set_xlabel('Column Index', fontsize=10)
    axes[1].set_aspect('equal')
    
    fig.suptitle(f'Learned Preconditioner Sparsity for Sample Matrix #{args.sample_idx}', fontsize=16, y=1.02)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, markerscale=150)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_filename = 'sparsity_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot successfully saved as '{output_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize and compare the sparsity of trained preconditioners.")
    parser.add_argument("--baseline_checkpoint", type=str, required=True, help="Path to the results folder of the baseline (K=0) model.")
    parser.add_argument("--top_k_checkpoint", type=str, required=True, help="Path to the results folder of the Top-K model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the root dataset directory.")
    parser.add_argument("--k", type=int, required=True, help="The 'K' value used for the Top-K model and for loading processed data.")
    parser.add_argument("--sample_idx", type=int, default=0, help="The index of the test sample to analyze.")
    parser.add_argument("--drop_tol", type=float, default=1e-4, help="Drop tolerance for the learned factor.")
    args = parser.parse_args()
    
    # Before running, ensure the pre-processed data for the given K exists.
    # This check is crucial.
    processed_test_dir = os.path.join(args.dataset, 'processed', 'test')
    if not os.path.isdir(processed_test_dir):
        print(f"--- WARNING: Pre-processed test data not found. Running pre-processing for K={args.k}... ---")
        os.system(f"python preprocess_data.py --dataset_path {args.dataset} --k {args.k}")
        print("--- Pre-processing complete. ---")

    main(args)