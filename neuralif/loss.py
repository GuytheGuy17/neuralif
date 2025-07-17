# FILE: neuralif/loss.py

import warnings
import torch

from apps.data import graph_to_matrix

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# --- UNCHANGED ORIGINAL FUNCTIONS ---

def frobenius_loss(L, A, sparse=True):
    if type(L) is tuple:
        U, L = L[1], L[0]
    else:
        U = L.T
    if sparse:
        r = L@U - A
        return torch.norm(r)
    else:
        A, L, U = A.to_dense().squeeze(), L.to_dense().squeeze(), U.to_dense().squeeze()
        return torch.linalg.norm(L@U - A, ord="fro")

def sketched_loss(L, A, c=None, normalized=False):
    if type(L) is tuple:
        U, L = L[1], L[0]
    else:
        U = L.T
    eps = 1e-8
    z = torch.randn((A.shape[0], 1), device=L.device)
    est = L@(U@z) - A@z
    norm = torch.linalg.vector_norm(est, ord=2)
    if normalized and c is None:
        norm = norm / torch.linalg.vector_norm(A@z, ord=2)
    elif normalized:
        norm = norm / (c + eps)
    return norm

def supervised_loss(L, A, x):
    if type(L) is tuple:
        U, L = L[1], L[0]
    else:
        U = L.T
    if x is None:
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
    res = L@(U@x) - b
    return torch.linalg.vector_norm(res, ord=2)

def combined_loss(L, A, x, w=1):
    loss1 = sketched_loss(L, A)
    loss2 = supervised_loss(L, A, x)
    return w * loss1 + loss2

# --- NEW FUNCTIONS FOR PCG PROXY LOSS ---

def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3):
    """A differentiable proxy for PCG convergence."""
    def iterative_solve(matrix, vector, iterations=5):
        """A few steps of CG to approximate a solve, e.g., Mz=r."""
        x = torch.zeros_like(vector)
        r = vector - matrix @ x
        p = r.clone()
        rs_old = torch.dot(r.squeeze(), r.squeeze())
        for _ in range(iterations):
            Ap = matrix @ p
            alpha = rs_old / (torch.dot(p.squeeze(), Ap.squeeze()) + 1e-10)
            x, r = x + alpha * p, r - alpha * Ap
            rs_new = torch.dot(r.squeeze(), r.squeeze())
            if torch.sqrt(rs_new) < 1e-8: break
            p, rs_old = r + (rs_new / rs_old) * p, rs_new
        return x

    n, device, dtype = A.shape[0], A.device, A.dtype
    b = torch.randn((n, 1), device=device, dtype=dtype)
    x, r = torch.zeros_like(b), b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16
    
    y = iterative_solve(L_mat, r)
    z = iterative_solve(U_mat, y)
    p, residuals, rz_old = z.clone(), [], (r * z).sum()

    for _ in range(cg_steps):
        Ap, pAp = A @ p, (p * (A @ p)).sum()
        if torch.abs(pAp) < 1e-12: break
        alpha = rz_old / pAp
        x, r = x + alpha * p, r - alpha * Ap
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        
        y = iterative_solve(L_mat, r)
        z = iterative_solve(U_mat, y)
        rz_new = (r * z).sum()
        
        if torch.abs(rz_old) < 1e-12: break
        beta, rz_old = rz_new / rz_old, rz_new
        p = z + beta * p
        
    return torch.stack(residuals).mean() if residuals else torch.tensor(1.0, device=device)

def improved_sketch_with_pcg(L, A, **kwargs):
    """Combines the sketched Frobenius norm loss with the PCG proxy loss."""
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat, U_mat = L, L.T

    sketch_loss_val = sketched_loss(L, A, normalized=kwargs.get('normalized', False))
    proxy = pcg_proxy(L_mat, U_mat, A, cg_steps=kwargs.get('pcg_steps', 3))
    return sketch_loss_val + kwargs.get('pcg_weight', 0.1) * proxy

# --- UPDATED LOSS DISPATCHER ---

def loss(output, data, config=None, **kwargs):
    with torch.no_grad():
        A, _ = graph_to_matrix(data)
    
    preconditioner_factors = (output, output.T) if not isinstance(output, tuple) else output

    if config == 'sketch_pcg':
        # NEW: Call the combined loss function
        l = improved_sketch_with_pcg(preconditioner_factors, A, **kwargs)
    elif config == "sketched" or config is None:
        l = sketched_loss(preconditioner_factors, A, normalized=kwargs.get("normalized", False))
    elif config == "supervised":
        l = supervised_loss(preconditioner_factors, A, data.s.squeeze())
    elif config == "combined":
        l = combined_loss(preconditioner_factors, A, data.s.squeeze())
    elif config == "frobenius":
        l = frobenius_loss(preconditioner_factors, A, sparse=False)
    else:
        raise ValueError(f"Invalid loss configuration: {config}")
        
    return l