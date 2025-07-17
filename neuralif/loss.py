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

def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3):
    """A differentiable proxy for PCG convergence."""
    def iterative_solve(matrix, vector, iterations=5):
        x = torch.zeros_like(vector)
        r = vector - matrix @ x
        p, rs_old = r.clone(), torch.dot(r.squeeze(), r.squeeze())
        for _ in range(iterations):
            alpha = rs_old / (torch.dot(p.squeeze(), (matrix @ p).squeeze()) + 1e-10)
            x, r = x + alpha * p, r - alpha * (matrix @ p)
            rs_new = torch.dot(r.squeeze(), r.squeeze())
            if torch.sqrt(rs_new) < 1e-8: break
            p, rs_old = r + (rs_new / rs_old) * p, rs_new
        return x

    n, device, dtype = A.shape[0], A.device, A.dtype
    b = torch.randn((n, 1), device=device, dtype=dtype)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16
    
    y = iterative_solve(L_mat, r)
    z = iterative_solve(U_mat, y)
    p, residuals, rz_old = z.clone(), [], (r * z).sum()

    for _ in range(cg_steps):
        pAp = (p * (A @ p)).sum()
        if torch.abs(pAp) < 1e-12: break
        alpha = rz_old / pAp
        r = r - alpha * (A @ p)
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        
        y = iterative_solve(L_mat, r)
        z = iterative_solve(U_mat, y)
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