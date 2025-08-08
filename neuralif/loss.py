# FILE: neuralif/loss.py
# (Final, correct version with the fast, approximate triangular solve
# for the pcg_proxy, which is compatible with the entire codebase.)

import warnings
import torch
from torch_geometric.utils import degree

from apps.data import graph_to_matrix

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def iterative_triangular_solve(A_sparse, b_vec, iterations=5):
    """
    Approximates the solution to A*x = b using a few steps of a simple iterative solver.
    This is a fast, differentiable replacement for an exact triangular solve.
    """
    x = torch.zeros_like(b_vec)
    
    if not A_sparse.is_coalesced():
        A_sparse = A_sparse.coalesce()
    
    indices = A_sparse.indices()
    values = A_sparse.values()
    
    diag_mask = indices[0] == indices[1]
    diag_indices = indices[0][diag_mask]
    diag_values = values[diag_mask]
    
    n = A_sparse.shape[0]
    temp_diag = torch.ones(n, device=A_sparse.device, dtype=A_sparse.dtype)
    temp_diag.scatter_(0, diag_indices, diag_values)
    D_inv = 1.0 / (temp_diag + 1e-12)

    # Perform a few steps of the Jacobi-like Richardson iteration
    for _ in range(iterations):
        x = x + D_inv * (b_vec - A_sparse @ x)
        
    return x.unsqueeze(-1)


def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3, preconditioner_solve_steps: int = 5):
    """
    A robust and differentiable PCG proxy that uses the fast iterative triangular solve.
    """
    n, device, dtype = A.shape[0], A.device, A.dtype
    torch.manual_seed(0)
    b = torch.randn((n, 1), device=device, dtype=dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16

    # Use the fast, approximate solver for the preconditioner steps
    y = iterative_triangular_solve(L_mat, r.squeeze(), iterations=preconditioner_solve_steps)
    z = iterative_triangular_solve(U_mat, y.squeeze(), iterations=preconditioner_solve_steps)

    p = z.clone()
    residuals = []
    rz_old = (r * z).sum()

    for i in range(cg_steps):
        Ap = A @ p
        pAp = (p * Ap).sum()
        if torch.abs(pAp) < 1e-12: break
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        y = iterative_triangular_solve(L_mat, r.squeeze(), iterations=preconditioner_solve_steps)
        z = iterative_triangular_solve(U_mat, y.squeeze(), iterations=preconditioner_solve_steps)
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < 1e-12: break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if not residuals:
        return torch.tensor(1.0, device=A.device)

    return torch.stack(residuals).mean()


def sketched_loss(L, A, normalized=False):
    """Computes the structural loss based on the Frobenius norm approximation."""
    L_mat, U_mat = (L, L.t()) if not isinstance(L, tuple) else L
    z = torch.randn((A.shape[0], 1), device=L_mat.device, dtype=L_mat.dtype)
    residual_vec = (L_mat @ (U_mat @ z)) - (A @ z)
    norm = torch.linalg.vector_norm(residual_vec, ord=2)
    if normalized:
        norm = norm / (torch.linalg.vector_norm(A@z, ord=2) + 1e-12)
    return norm


def improved_sketch_with_pcg(L, A, **kwargs):
    """Computes a hybrid loss where BOTH components are normalized."""
    L_mat, U_mat = (L, L.t()) if not isinstance(L, tuple) else L
    sketch_loss_val = sketched_loss((L_mat, U_mat), A, normalized=True)
    proxy_loss_val = pcg_proxy(
        L_mat.coalesce(), U_mat.coalesce(), A, 
        cg_steps=kwargs.get('pcg_steps', 3),
        preconditioner_solve_steps=kwargs.get('preconditioner_solve_steps', 5)
    )
    return sketch_loss_val + kwargs.get('pcg_weight', 0.1) * proxy_loss_val


def loss(output, data, config=None, **kwargs):
    """Computes the loss on the same device as the model output."""
    device = output.device if not isinstance(output, tuple) else output[0].device
    L_factor = output if not isinstance(output, tuple) else output[0]
    A, _ = graph_to_matrix(data)
    A = A.to(device)

    if config == 'sketch_pcg':
        final_loss = improved_sketch_with_pcg(L_factor, A, **kwargs)
    elif config is None or config == "sketched":
        final_loss = sketched_loss(L_factor, A, normalized=kwargs.get('normalized', False))
    else:
        raise ValueError(f"Invalid or undefined loss configuration: {config}")
    return final_loss
