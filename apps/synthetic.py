import os
import sys
import argparse
import torch
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from apps.data import matrix_to_graph

def generate_sparse_spd_with_target_density(n, target_density, alpha=1e-3, random_state=0, compute_solution=True, tol=0.1, max_iters=20):
    """
    Generates a random sparse SPD matrix 'A' with a final density close to 'target_density'.
    This version uses the M@M.T method, which is known to produce matrices of the
    correct difficulty (~914 iterations) for a valid comparison to the reference paper.
    """
    rng = np.random.RandomState(random_state)
    target_nnz = int(target_density * n * n)
    if target_nnz <= 0:
        raise ValueError("Target density is too low, resulting in zero or negative target non-zeros.")

    m_density_est = np.sqrt(target_density / n) if target_density > 0 else 0

    A_iter = None
    for i in range(max_iters):
        m_nnz = int(m_density_est * n * n)
        m_nnz = max(1, min(m_nnz, n * n - 1))
        
        rows = rng.randint(0, n, size=m_nnz)
        cols = rng.randint(0, n, size=m_nnz)
        
        unique_coords = sorted(list(set(zip(rows, cols))))
        rows_unique, cols_unique = zip(*unique_coords)
        vals = rng.normal(0, 1, size=len(rows_unique))
        
        M_iter = coo_matrix((vals, (rows_unique, cols_unique)), shape=(n, n)).tocsr()
        A_iter = (M_iter @ M_iter.T)
        
        current_nnz = A_iter.nnz
        if target_nnz == 0:
             error_ratio = 1.0 if current_nnz == 0 else float('inf')
        else:
            error_ratio = current_nnz / target_nnz
        
        if abs(1 - error_ratio) < tol:
            break
        
        adjustment_factor = np.sqrt(1 / error_ratio) if error_ratio > 0 else 2.0
        m_density_est *= (1 + (adjustment_factor - 1) * 0.75)
    else:
        tqdm.write(f"\nWarning: Density target not met for seed {random_state}. Using best effort.")
    
    A = A_iter + alpha * scipy.sparse.identity(n, format='csc')
    
    A.eliminate_zeros()
    tqdm.write(f"Generated matrix with {100 * (A.nnz / n**2) :.4f}% density ({A.nnz} non-zeros) for seed {random_state}")

    b = rng.uniform(0, 1, size=n)
    x = None
    if compute_solution:
        try:
            x, info = scipy.sparse.linalg.cg(A, b, rtol=1e-10, maxiter=max(5000, 2 * n))
            if info != 0:
                 tqdm.write(f"  -> WARNING: CG solve for ground-truth did not converge. Info: {info}")
        except Exception as e:
            tqdm.write(f"  -> WARNING: CG solve for ground-truth failed. Error: {e}")
    
    return A, b, x

def main(args):
    """Main function to generate and save the dataset."""
    print("\n" + "-"*60)
    print("Generating Synthetic Dataset (Faithful to Paper's Difficulty)")
    print(f" -> Number of samples: {args.num_samples}")
    print(f" -> Matrix size: {args.matrix_size}x{args.matrix_size}")
    print(f" -> Target FINAL Matrix Density: {args.density*100:.4f}%")
    print(f" -> Output directory: {os.path.abspath(args.output_dir)}")
    print("-"*60 + "\n")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in tqdm(range(args.num_samples), desc="Generating Samples"):
        random_state = args.seed + i
        
        A, b, x = generate_sparse_spd_with_target_density(
            n=args.matrix_size,
            target_density=args.density,
            alpha=args.alpha, 
            random_state=random_state,
            compute_solution=not args.no_solution
        )
        
        graph = matrix_to_graph(A, b)
        if x is not None:
            graph.s = torch.tensor(x, dtype=torch.float32)
        
        save_path = os.path.join(args.output_dir, f'graph_{args.matrix_size}_{i}.pt')
        torch.save(graph, save_path)
            
    print("\n" + "-"*60, "\nDataset generation complete.\n" + "-"*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generator for synthetic SPD matrices.")
    
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated graph files.")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of graph samples to generate.")
    parser.add_argument('--matrix_size', type=int, default=10000, help="Matrix size (N) for the NxN matrices.")
    parser.add_argument('--density', type=float, default=0.01, help="Target density for the FINAL matrix A.")
    parser.add_argument('--alpha', type=float, default=1e-3, help="Diagonal shift for SPD guarantee.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed.")
    parser.add_argument('--no_solution', action='store_true', help="Skip computing the ground-truth solution 'x'.")

    args = parser.parse_args()
    main(args)
