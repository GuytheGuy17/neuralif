import warnings
import torch
from torch_geometric.utils import degree


warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# This function computes the degree of each node in the graph.
# It returns a tensor of degrees, which is useful for various graph algorithms.
def iterative_triangular_solve(A_sparse, b_vec, iterations=5):
    """
    Approximates the solution to A*x = b using a few steps of a simple iterative solver.
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
    D_inv = 1.0 / (temp_diag + 1e-9) 

    for _ in range(iterations):
        x = x + D_inv * (b_vec - A_sparse @ x)
        
    return x.unsqueeze(-1)

# This function implements a proxy for the preconditioned conjugate gradient method.
# It uses a fast iterative triangular solve to approximate the solution.
def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3, preconditioner_solve_steps: int = 5):
    """
    A robust and differentiable PCG proxy that uses the fast iterative triangular solve.
    """
    n, device, dtype = A.shape[0], A.device, A.dtype
    
    eps = torch.finfo(dtype).eps * 100
    
    torch.manual_seed(0)
    b = torch.randn((n, 1), device=device, dtype=dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + eps

    # Solve L * y = r
    y = iterative_triangular_solve(L_mat, r.squeeze(), iterations=preconditioner_solve_steps)
    # Solve U * z = y
    z = iterative_triangular_solve(U_mat, y.squeeze(), iterations=preconditioner_solve_steps)

    # Initial search direction
    p = z.clone()
    residuals = []
    rz_old = (r * z).sum()

    # Main loop for the conjugate gradient method
    for i in range(cg_steps):
        Ap = A @ p
        pAp = (p * Ap).sum()
        if torch.abs(pAp) < eps: break 
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        y = iterative_triangular_solve(L_mat, r.squeeze(), iterations=preconditioner_solve_steps)
        z = iterative_triangular_solve(U_mat, y.squeeze(), iterations=preconditioner_solve_steps)
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < eps: break 
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if not residuals:
        return torch.tensor(1.0, device=A.device)

    return torch.stack(residuals).mean()

# This function computes the structural loss based on the Frobenius norm approximation.
# It uses a random vector to approximate the loss, which is differentiable.

def sketched_loss(L, A, normalized=False):
    """Computes the structural loss based on the Frobenius norm approximation."""
    L_mat, U_mat = (L, L.t()) if not isinstance(L, tuple) else L
    z = torch.randn((A.shape[0], 1), device=L_mat.device, dtype=L_mat.dtype)
    residual_vec = (L_mat @ (U_mat @ z)) - (A @ z)
    norm = torch.linalg.vector_norm(residual_vec, ord=2)
    if normalized:
        a_norm = torch.linalg.vector_norm(A@z, ord=2)
        norm = norm / (a_norm + torch.finfo(a_norm.dtype).eps)
    return norm

# This function computes a hybrid loss that combines the sketched loss and the PCG proxy loss.
# It is designed to be differentiable and uses the preconditioned conjugate gradient method.
def improved_sketch_with_pcg(L, A, **kwargs):
    """Computes a hybrid loss where BOTH components are normalized."""
    L_mat, U_mat = (L, L.t()) if not isinstance(L, tuple) else L
    # Compute the sketched loss
    sketch_loss_val = sketched_loss((L_mat, U_mat), A, normalized=True)
    # Compute the PCG proxy loss
    proxy_loss_val = pcg_proxy(
        L_mat.coalesce(), U_mat.coalesce(), A, 
        cg_steps=kwargs.get('pcg_steps', 3),
        preconditioner_solve_steps=kwargs.get('preconditioner_solve_steps', 5)
    )
    # Return the combined loss
    return sketch_loss_val + kwargs.get('pcg_weight', 0.1) * proxy_loss_val

#  This function computes the loss based on the model output and the data.
# It handles both the sketched loss and the improved sketch with PCG loss.
def loss(output, data, config=None, **kwargs):
    """Computes the loss on the same device as the model output."""
    device = output.device if not isinstance(output, tuple) else output[0].device
    L_factor = output if not isinstance(output, tuple) else output[0]

    if data.edge_attr.dim() > 1 and data.edge_attr.shape[1] > 1:
        # This branch handles graphs that have the 2D fill-in features
        is_fill_in_flag = data.edge_attr[:, 1]
        original_edge_mask = (is_fill_in_flag == 0)
        
        edge_index_A = data.edge_index[:, original_edge_mask]
        edge_attr_A = data.edge_attr[original_edge_mask, 0] # Use only the value feature
    else:
        # This is a fallback for original graphs without the fill-in attribute
        edge_index_A = data.edge_index
        edge_attr_A = data.edge_attr.squeeze()

    # Create the COO tensor for A on the target device
    A = torch.sparse_coo_tensor(
        edge_index_A,
        edge_attr_A,
        (data.num_nodes, data.num_nodes)
    ).to(device)
    

    if config == 'sketch_pcg':
        final_loss = improved_sketch_with_pcg(L_factor, A, **kwargs)
    elif config is None or config == "sketched":
        final_loss = sketched_loss(L_factor, A, normalized=kwargs.get('normalized', False))
    else:
        raise ValueError(f"Invalid or undefined loss configuration: {config}")
    return final_loss