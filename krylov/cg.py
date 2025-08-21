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

# This is a pure PyTorch implementation of the Conjugate Gradient method.
def conjugate_gradient(A, b, x0=None, x_true=None, rtol=1e-6, max_iter=None):
    """
    A correct and optimized version of your pure PyTorch Conjugate Gradient solver.
    """
    if max_iter is None:
        max_iter = 2 * len(b)
    
    # Epsilon for float32 stability
    eps = torch.finfo(b.dtype).eps * 100
    # Initialize x_hat and residuals
    x_hat = x0 if x0 is not None else torch.zeros_like(b)
    r = b - A@x_hat # Initial residual
    p = r.clone() # Initial search direction
    rs_old = torch.dot(r, r) # Initial squared norm of the residual
    # Calculate the initial error if x_true is provided
    res = stopping_criterion(A, r, b) # Relative residual squared norm
    err_A_norm_sq = torch.dot(x_hat - x_true, A @ (x_hat - x_true)) if x_true is not None else torch.tensor(0.0, dtype=b.dtype)
    errors = [(err_A_norm_sq, res)] # Store the initial error and residual
    # Check if the initial residual is already below the tolerance
    if res < rtol:
        return errors, x_hat
    # Main CG loop
    for _ in range(max_iter):
        Ap = A @ p # Matrix-vector product
        # Calculate the step size alpha
        alpha = rs_old / (torch.dot(p, Ap) + eps)
        x_hat = x_hat + alpha * p # Update the solution
        r = r - alpha * Ap # Update the residual
        # Calculate the new residual squared norm
        rs_new = torch.dot(r, r)
        # Update the error and residual
        res = stopping_criterion(A, r, b)
        errors.append((torch.tensor(0.0, dtype=b.dtype), res))

        if res < rtol:
            break
        # Calculate the new search direction   
        p = r + (rs_new / (rs_old + eps)) * p
        rs_old = rs_new
    # If x_true is provided, calculate the final error
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

    # Epsilon for float32 stability
    eps = torch.finfo(b.dtype).eps * 100
    # Initialize x_hat and residuals
    x_hat = x0 if x0 is not None else torch.zeros_like(b)
    rk = b - A@x_hat # Initial residual
    # Calculate the initial error if x_true is provided
    error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    res = stopping_criterion(A, rk, b) # Relative residual squared norm
    errors = [(torch.inner(error_i, A@error_i), res)] # Store the initial error and residual
    # Check if the initial residual is already below the tolerance

    if res < rtol:
        return errors, x_hat
    # Initial preconditioner solve
    zk = M(rk).squeeze()
    pk = zk.clone()
    # Initial squared norm of the residual
    for _ in range(max_iter):
        rz_old = torch.inner(rk, zk)
        Ap = A@pk # Matrix-vector product
        # Calculate the step size alpha
        alpha = rz_old / (torch.inner(pk, Ap) + eps)
        x_hat = x_hat + alpha * pk
        rk = rk - alpha * Ap # Update the residual
        # Calculate the new residual squared norm
        res = stopping_criterion(A, rk, b)
        errors.append((torch.tensor(0.0, dtype=b.dtype), res))

        if res < rtol:
            break
        # Calculate the new error if x_true is provided
        zk = M(rk).squeeze() # Preconditioner solve
        rz_new = torch.inner(rk, zk) # Update the inner product 
        beta = rz_new / (rz_old + eps) # Update the search direction
        pk = zk + beta * pk # Update the search direction
        
    if x_true is not None: # Calculate the final error
        final_err_A_norm_sq = torch.dot(x_hat - x_true, A @ (x_hat - x_true))
        errors[-1] = (final_err_A_norm_sq, errors[-1][1])
        
    return errors, x_hat
