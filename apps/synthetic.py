import os
import argparse
import torch
import numpy as np
import scipy
from scipy.sparse import coo_matrix

# This assumes data.py with matrix_to_graph is in the same directory
from data import matrix_to_graph


def generate_sparse_random(n, alpha=1e-4, random_state=0):
    """Generates a random sparse SPD matrix using the paper's method."""
    rng = np.random.RandomState(random_state)
    
    # Using 1% sparsity to match the paper's synthetic dataset description
    sparsity = 0.01
    nnz = int(sparsity * n ** 2)
    
    # Ensure unique coordinates
    rows_cols = set()
    while len(rows_cols) < nnz:
        rows_cols.add((rng.randint(0, n), rng.randint(0, n)))
    
    rows, cols = zip(*rows_cols)
    vals = np.array([rng.normal(0, 1) for _ in cols])
    
    M = coo_matrix((vals, (rows, cols)), shape=(n, n))
    I = scipy.sparse.identity(n)
    A = (M @ M.T) + alpha * I
    
    b = rng.uniform(0, 1, size=n)
    
    # Generate a high-accuracy solution using CG for test/val sets
    x, _ = scipy.sparse.linalg.cg(A, b)
    
    return A, b, x

def main(args):
    """Main function to generate and save the dataset based on command-line arguments."""
    print(f"Preparing to generate {args.num_samples} samples...")
    print(f" -> Matrix size: {args.matrix_size}x{args.matrix_size}")
    print(f" -> Output dir: {os.path.abspath(args.output_dir)}")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        random_state = args.seed + i
        
        A, b, x = generate_sparse_random(
            n=args.matrix_size, 
            alpha=args.alpha, 
            random_state=random_state
        )
        
        graph = matrix_to_graph(A, b)
        if x is not None:
            graph.s = torch.tensor(x, dtype=torch.float)
        
        save_path = os.path.join(args.output_dir, f'graph_{args.matrix_size}_{i}.pt')
        torch.save(graph, save_path)
        
        if (i + 1) % 100 == 0 or (i + 1) == args.num_samples:
            print(f"  ... generated and saved sample {i + 1} / {args.num_samples}")

    print("-" * 50)
    print("Generation complete.")
    print("-" * 50)


if __name__ == '__main__':
    # --- UPDATED: Using argparse to control generation ---
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets.")
    
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated graph files.")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of graph samples to generate.")
    parser.add_argument('--matrix_size', type=int, default=1024, help="Matrix size (N) for the NxN matrices.")
    parser.add_argument('--alpha', type=float, default=1e-3, help="Regularization parameter for SPD generation.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed for reproducibility.")

    args = parser.parse_args()
    main(args)