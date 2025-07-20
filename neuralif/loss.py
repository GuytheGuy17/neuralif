# FILE: neuralif/loss.py (FIXED)

import warnings
import torch
import torch.sparse  # <--- IMPORT THIS
from apps.data import graph_to_matrix

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


# --- START: EFFICIENT IMPLEMENTATION ---
# The original manual implementation with a Python for-loop is replaced
# by a single call to PyTorch's optimized and differentiable sparse solver.
# This is the critical fix for the training bottleneck.
def sparse_triangular_solve(matrix, vector, lower: bool):
    """
    Differentiably solves a sparse triangular system using PyTorch's native,
    highly optimized C++/CUDA implementation. This requires PyTorch >= 1.12.
    """
    # The native solver expects the right-hand-side to be a dense 2D matrix (N, K).
    # We ensure our input vector conforms to this shape.
    if vector.dim() == 1:
        vector = vector.unsqueeze(1)

    # Call the native PyTorch function.
    # Note: Our 'lower=True' corresponds to the function's 'upper=False'.
    solution, _ = torch.sparse.triangular_solve(
        matrix, vector, upper=(not lower), unitriangular=False
    )
    return solution
# --- END: EFFICIENT IMPLEMENTATION ---


def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3):
    """A differentiable proxy for PCG convergence."""
    n, device, dtype = A.shape[0], A.device, A.dtype
    b = torch.randn((n, 1), device=device, dtype=dtype)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16
    
    # These calls will now use the fast, native solver
    y = sparse_triangular_solve(L_mat, r, lower=True)
    z = sparse_triangular_solve(U_mat, y, lower=False)
    p, residuals, rz_old = z.clone(), [], (r * z).sum()

    for _ in range(cg_steps):
        pAp = (p * (A @ p)).sum()
        if torch.abs(pAp) < 1e-12: break
        alpha = rz_old / (pAp + 1e-12)
        r = r - alpha * (A @ p)
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        
        # These calls will also use the fast, native solver
        y = sparse_triangular_solve(L_mat, r, lower=True)
        z = sparse_triangular_solve(U_mat, y, lower=False)
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < 1e-12: break
        p, rz_old = z + (rz_new / (rz_old + 1e-12)) * p, rz_new
        
    return torch.stack(residuals).mean() if residuals else torch.tensor(1.0, device=device)


def sketched_loss(L, A, normalized=False):
    """Computes the sketched Frobenius norm ||L*U - A||. This is the baseline loss."""
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


def improved_sketch_with_pcg(L, A, **kwargs):
    """Combines the sketched loss with the PCG proxy loss for your experiment."""
    L_mat, U_mat = (L, L.T) if not isinstance(L, tuple) else L
    sketch_loss_val = sketched_loss((L_mat, U_mat), A, normalized=kwargs.get('normalized', False))
    proxy = pcg_proxy(L_mat, U_mat, A, cg_steps=kwargs.get('pcg_steps', 3))
    return sketch_loss_val + kwargs.get('pcg_weight', 0.1) * proxy


def loss(output, data, config=None, **kwargs):
    """Main loss dispatcher. Selects the loss function based on the 'config' string."""
    with torch.no_grad():
        A, _ = graph_to_matrix(data)
    
    if config == 'sketch_pcg':
        return improved_sketch_with_pcg(output, A, **kwargs)
    elif config == 'sketched' or config is None:
        # Default to the baseline sketched loss
        return sketched_loss(output, A, normalized=kwargs.get("normalized", False))
    else:
        raise ValueError(f"Loss configuration '{config}' not supported.")