import warnings
import torch
from apps.data import graph_to_matrix

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

def sketched_loss(L, A, normalized=False):
    """Computes the sketched Frobenius norm ||L*U - A||."""
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat, U_mat = L, L.T
    
    z = torch.randn((A.shape[0], 1), device=L_mat.device, dtype=L_mat.dtype)
    est = L_mat @ (U_mat @ z) - A @ z
    norm = torch.linalg.vector_norm(est, ord=2)
    
    if normalized:
        norm = norm / (torch.linalg.vector_norm(A @ z, ord=2) + 1e-8)
    return norm

# ==============================================================================
# --- START: CRITICAL FIX ---
#
# The original 'iterative_solve' function incorrectly used a Conjugate Gradient
# (CG) method to solve triangular systems (Lx=b). CG is only suitable for
# Symmetric Positive-Definite (SPD) matrices, which L and U are not.
#
# The function below, 'sparse_triangular_solve', correctly and differentiably
# implements forward and backward substitution for sparse matrices, which is the
# required operation for applying the inverse of the preconditioner.
# ==============================================================================

def sparse_triangular_solve(matrix, vector, lower: bool):
    """
    Differentiably solves a sparse triangular system.
    
    Args:
        matrix (torch.sparse_coo_tensor): A sparse triangular matrix (L or U).
        vector (torch.Tensor): The right-hand side vector.
        lower (bool): True for forward substitution (L), False for backward (U).
    
    Returns:
        torch.Tensor: The solution vector x in matrix @ x = vector.
    """
    # Ensure inputs are in the correct format
    matrix_csr = matrix.to_sparse_csr()
    b = vector.squeeze()
    n = b.shape[0]
    x = torch.zeros_like(b)

    if lower:
        # Forward substitution: L @ x = b
        # x_i = (b_i - sum_{j<i} L_ij * x_j) / L_ii
        for i in range(n):
            # Sum over the off-diagonal elements of the row
            row_start = matrix_csr.crow_indices()[i]
            row_end = matrix_csr.crow_indices()[i+1]
            row_cols = matrix_csr.col_indices()[row_start:row_end]
            row_vals = matrix_csr.values()[row_start:row_end]
            
            # Find diagonal and off-diagonal elements
            diag_mask = (row_cols == i)
            diag_val = row_vals[diag_mask]
            
            off_diag_cols = row_cols[~diag_mask]
            off_diag_vals = row_vals[~diag_mask]
            
            # This dot product is differentiable
            off_diag_sum = torch.dot(off_diag_vals, x[off_diag_cols]) if len(off_diag_cols) > 0 else 0
            
            x[i] = (b[i] - off_diag_sum) / diag_val

    else:
        # Backward substitution: U @ x = b
        # x_i = (b_i - sum_{j>i} U_ij * x_j) / U_ii
        for i in range(n - 1, -1, -1):
            # Sum over the off-diagonal elements of the row
            row_start = matrix_csr.crow_indices()[i]
            row_end = matrix_csr.crow_indices()[i+1]
            row_cols = matrix_csr.col_indices()[row_start:row_end]
            row_vals = matrix_csr.values()[row_start:row_end]
            
            # Find diagonal and off-diagonal elements
            diag_mask = (row_cols == i)
            diag_val = row_vals[diag_mask]
            
            off_diag_cols = row_cols[~diag_mask]
            off_diag_vals = row_vals[~diag_mask]
            
            # This dot product is differentiable
            off_diag_sum = torch.dot(off_diag_vals, x[off_diag_cols]) if len(off_diag_cols) > 0 else 0
            
            x[i] = (b[i] - off_diag_sum) / diag_val

    return x.view(-1, 1) # Return as a column vector

# ==============================================================================
# --- END: CRITICAL FIX ---
# ==============================================================================

def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3):
    """A differentiable proxy for PCG convergence."""
    n, device, dtype = A.shape[0], A.device, A.dtype
    b = torch.randn((n, 1), device=device, dtype=dtype)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16
    
    # --- FIX: Use the correct sparse triangular solve ---
    # Apply preconditioner M_inv @ r = U_inv @ (L_inv @ r)
    y = sparse_triangular_solve(L_mat, r, lower=True)
    z = sparse_triangular_solve(U_mat, y, lower=False)
    
    p, residuals, rz_old = z.clone(), [], (r * z).sum()

    for _ in range(cg_steps):
        pAp = (p * (A @ p)).sum()
        if torch.abs(pAp) < 1e-12: break
        alpha = rz_old / pAp
        r = r - alpha * (A @ p)
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        
        # --- FIX: Use the correct sparse triangular solve ---
        y = sparse_triangular_solve(L_mat, r, lower=True)
        z = sparse_triangular_solve(U_mat, y, lower=False)
        
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < 1e-12: break
        p, rz_old = z + (rz_new / rz_old) * p, rz_new
        
    return torch.stack(residuals).mean() if residuals else torch.tensor(1.0, device=device)

def improved_sketch_with_pcg(L, A, **kwargs):
    """Combines the sketched loss with the PCG proxy loss."""
    sketch_loss_val = sketched_loss(L, A, normalized=kwargs.get('normalized', False))
    proxy = pcg_proxy(L[0], L[1], A, cg_steps=kwargs.get('pcg_steps', 3))
    return sketch_loss_val + kwargs.get('pcg_weight', 0.1) * proxy

def loss(output, data, config=None, **kwargs):
    """Main loss dispatcher."""
    with torch.no_grad():
        A, _ = graph_to_matrix(data)
    
    factors = (output, output.T) if not isinstance(output, tuple) else output

    if config == 'sketch_pcg':
        return improved_sketch_with_pcg(factors, A, **kwargs)
    elif config == 'sketched' or config is None:
        return sketched_loss(factors, A, normalized=kwargs.get("normalized", False))
    else:
        raise ValueError(f"Loss configuration '{config}' not supported in this script.")