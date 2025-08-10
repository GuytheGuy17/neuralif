import torch

def stopping_criterion(A, rk, b):
    """
    Calculates the relative residual squared norm: ||r_k||^2 / ||b||^2.
    """
    b_norm_sq = torch.inner(b, b)
    # Use an epsilon appropriate for the dtype of b_norm_sq
    if b_norm_sq < torch.finfo(b_norm_sq.dtype).eps:
        return torch.inner(rk, rk)
    return torch.inner(rk, rk) / b_norm_sq


def conjugate_gradient(A, b, x0=None, x_true=None, rtol=1e-6, max_iter=None):
    """
    A correct and optimized version of your pure PyTorch Conjugate Gradient solver.
    """
    if max_iter is None:
        max_iter = 2 * len(b)
    
    # Epsilon for float32 stability. 1e-9 is a safe choice.
    eps = torch.finfo(b.dtype).eps * 100

    x_hat = x0 if x0 is not None else torch.zeros_like(b)
    r = b - A@x_hat
    p = r.clone()
    rs_old = torch.dot(r, r)
    
    res = stopping_criterion(A, r, b)
    err_A_norm_sq = torch.dot(x_hat - x_true, A @ (x_hat - x_true)) if x_true is not None else torch.tensor(0.0, dtype=b.dtype)
    errors = [(err_A_norm_sq, res)]

    if res < rtol:
        return errors, x_hat

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (torch.dot(p, Ap) + eps)
        x_hat = x_hat + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        res = stopping_criterion(A, r, b)
        errors.append((torch.tensor(0.0, dtype=b.dtype), res))

        if res < rtol:
            break
            
        p = r + (rs_new / (rs_old + eps)) * p
        rs_old = rs_new
        
    if x_true is not None:
        final_err_A_norm_sq = torch.dot(x_hat - x_true, A @ (x_hat - x_true))
        errors[-1] = (final_err_A_norm_sq, errors[-1][1])
        
    return errors, x_hat


def preconditioned_conjugate_gradient(A, b, M, x0=None, x_true=None, rtol=1e-6, max_iter=None):
    """
    A corrected and optimized version of your pure PyTorch Preconditioned CG solver.
    """
    if max_iter is None:
        max_iter = 2 * len(b)

    # Epsilon for float32 stability.
    eps = torch.finfo(b.dtype).eps * 100

    x_hat = x0 if x0 is not None else torch.zeros_like(b)
    rk = b - A@x_hat
    
    error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    res = stopping_criterion(A, rk, b)
    errors = [(torch.inner(error_i, A@error_i), res)]

    if res < rtol:
        return errors, x_hat

    zk = M(rk).squeeze()
    pk = zk.clone()
    
    for _ in range(max_iter):
        rz_old = torch.inner(rk, zk)
        Ap = A@pk
        
        alpha = rz_old / (torch.inner(pk, Ap) + eps)
        x_hat = x_hat + alpha * pk
        rk = rk - alpha * Ap 
        
        res = stopping_criterion(A, rk, b)
        errors.append((torch.tensor(0.0, dtype=b.dtype), res))

        if res < rtol:
            break

        zk = M(rk).squeeze()
        rz_new = torch.inner(rk, zk)
        beta = rz_new / (rz_old + eps)
        pk = zk + beta * pk
        
    if x_true is not None:
        final_err_A_norm_sq = torch.dot(x_hat - x_true, A @ (x_hat - x_true))
        errors[-1] = (final_err_A_norm_sq, errors[-1][1])
        
    return errors, x_hat
