import torch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from neuralif.utils import torch_sparse_to_scipy, time_function

class Preconditioner:
    # ... (implementation is unchanged)
    def __init__(self):
        self.time = 0.0
        self.breakdown = False
        self.breakdown_reason = ""
    def __call__(self, b: torch.Tensor) -> torch.Tensor:
        return self.solve(b)
    @property
    def nnz(self):
        return 0
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return b

class Jacobi(Preconditioner):
    # ... (implementation is unchanged)
    def __init__(self, A_torch: torch.Tensor):
        super().__init__()
        start = time_function()
        A_cpu = A_torch.to('cpu').coalesce()
        indices = A_cpu.indices()
        values = A_cpu.values()
        diag_mask = indices[0] == indices[1]
        diag_indices = indices[0][diag_mask]
        diag_values = values[diag_mask]
        n = A_cpu.shape[0]
        full_diag = torch.zeros(n, device='cpu', dtype=A_cpu.dtype)
        full_diag.scatter_(0, diag_indices, diag_values)
        self.inv_diag = 1.0 / (full_diag + 1e-12)
        self.time = time_function() - start
        self._nnz = n
    @property
    def nnz(self):
        return self._nnz
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return self.inv_diag.to(b.device, b.dtype) * b

class ScipyILU(Preconditioner):
    # ... (implementation is unchanged)
    def __init__(self, A_torch: torch.Tensor):
        super().__init__()
        start_time = time_function()
        A_scipy_csc = torch_sparse_to_scipy(A_torch.cpu()).tocsc()
        try:
            self.ilu_op = scipy.sparse.linalg.spilu(A_scipy_csc, drop_tol=0.0, fill_factor=1)
        except RuntimeError as e:
            self.breakdown = True
            self.breakdown_reason = str(e)
            self.ilu_op = None
            print(f"\nWARNING: SciPy ILU factorization failed: {e}")
        self.time = time_function() - start_time
    @property
    def nnz(self):
        if self.ilu_op and hasattr(self.ilu_op, 'L') and hasattr(self.ilu_op, 'U'):
            return self.ilu_op.L.nnz + self.ilu_op.U.nnz
        return 0
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        if self.breakdown: return b
        b_np = b.cpu().numpy()
        x_np = self.ilu_op.solve(b_np)
        return torch.from_numpy(x_np).to(b.device, b.dtype)

class LearnedPreconditioner(Preconditioner):
    def __init__(self, data_on_device, model, drop_tol=1e-6):
        super().__init__()
        self.model = model
        self.drop_tol = drop_tol
        start = time_function()
        self._compute_preconditioner(data_on_device)
        self.time = time_function() - start

    def _compute_preconditioner(self, data_on_device):
        self.model.eval()
        with torch.no_grad():
            L_torch, U_torch, _ = self.model(data_on_device)
        
        # --- START OF MODIFICATION ---
        # Thresholding Step: Remove near-zero entries predicted by the GNN
        # before creating the final solver. This is the "pruning" step.
        L_final = L_torch.coalesce()
        mask = L_final.values().abs() > self.drop_tol
        L_final = torch.sparse_coo_tensor(
            L_final.indices()[:, mask],
            L_final.values()[mask],
            L_final.shape
        ).coalesce()
        
        U_final = torch.sparse_coo_tensor(L_final.indices().flip(0), L_final.values(), L_final.shape).coalesce()
        
        self.L_scipy = torch_sparse_to_scipy(L_final.cpu()).tocsc()
        self.U_scipy = torch_sparse_to_scipy(U_final.cpu()).tocsc()
        # --- END OF MODIFICATION ---
    
    @property
    def nnz(self):
        if self.L_scipy is not None:
            return self.L_scipy.nnz
        return 0

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        # The solve method is now simpler as the factors are pre-computed and pruned
        b_np = b.cpu().numpy()
        y = scipy.sparse.linalg.spsolve_triangular(self.L_scipy, b_np, lower=True)
        x_np = scipy.sparse.linalg.spsolve_triangular(self.U_scipy, y, lower=False)
        return torch.from_numpy(x_np).to(b.device, b.dtype)

def get_preconditioner(data, A_torch_cpu, method: str, model=None, device='cpu') -> Preconditioner:
    if method == "learned":
        if model is None: raise ValueError("A model must be provided for the 'learned' method.")
        # Pass the original data object, as the dataloader has already applied the transform
        data_on_device = data.to(device)
        return LearnedPreconditioner(data_on_device, model)
    
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        return Jacobi(A_torch_cpu)
    elif method == "ic":
        return ScipyILU(A_torch_cpu)
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' not implemented!")