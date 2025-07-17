import torch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.models import NeuralIF

class Preconditioner:
    """A base class to ensure all preconditioners have a consistent interface."""
    def __init__(self):
        self.time = 0.0
        self.breakdown = False

    @property
    def nnz(self):
        return 0

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """Applies the preconditioner M_inv to a vector b."""
        return b # Default is identity (no preconditioning)

class Jacobi(Preconditioner):
    """Jacobi (Diagonal) Preconditioner."""
    def __init__(self, A_torch: torch.Tensor):
        super().__init__()
        start = time_function()
        # Use to_dense() to reliably get the diagonal from a sparse tensor
        self.inv_diag = 1.0 / A_torch.to_dense().diagonal()
        self.time = time_function() - start
        self.nnz = A_torch.shape[0]

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return self.inv_diag * b

class ScipyILU(Preconditioner):
    """Wrapper for the SciPy ILU preconditioner."""
    def __init__(self, ilu_obj):
        super().__init__()
        self._prec = ilu_obj

    @property
    def nnz(self):
        # Access the L and U factors from the stored spilu object
        return self._prec.L.nnz + self._prec.U.nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for SciPy, then back to torch
        b_np = b.cpu().numpy()
        x_np = self._prec.solve(b_np)
        return torch.from_numpy(x_np).to(b.device)

class LearnedPreconditioner(Preconditioner):
    """Wrapper for the learned GNN preconditioner."""
    def __init__(self, data, model):
        super().__init__()
        self.model = model
        self.data = data
        self._computed = False
        self.L_scipy, self.U_scipy = None, None
        
        start = time_function()
        self._compute_preconditioner()
        self.time = time_function() - start

    def _compute_preconditioner(self):
        with torch.no_grad():
            L_torch, U_torch, _ = self.model(self.data)
        
        # Convert torch sparse tensors to scipy sparse matrices for solving
        self.L_scipy = torch_sparse_to_scipy(L_torch.cpu()).tocsc()
        self.U_scipy = torch_sparse_to_scipy(U_torch.cpu()).tocsc()
        self._computed = True
    
    @property
    def nnz(self):
        # Return nnz from the computed scipy matrix
        if not self._computed: self._compute_preconditioner()
        return self.L_scipy.nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """Applies the learned preconditioner M_inv = U_inv * L_inv."""
        if not self._computed: self._compute_preconditioner()
        
        # Use SciPy's robust sparse triangular solve
        b_np = b.cpu().numpy()
        y = scipy.sparse.linalg.spsolve_triangular(self.L_scipy, b_np, lower=True)
        x_np = scipy.sparse.linalg.spsolve_triangular(self.U_scipy, y, lower=False)
        return torch.from_numpy(x_np).to(b.device)

def get_preconditioner(data, method: str, model=None) -> Preconditioner:
    """Factory function to create the specified preconditioner."""
    if method == "learned":
        if model is None: raise ValueError("A model must be provided for the 'learned' method.")
        return LearnedPreconditioner(data, model)

    # For baselines, create PyTorch and SciPy versions of the matrix A
    A_torch = torch.sparse_coo_tensor(
        data.edge_index, data.edge_attr.squeeze(),
        (data.num_nodes, data.num_nodes),
        device=data.x.device, dtype=torch.float64
    ).coalesce()
    
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        return Jacobi(A_torch)
    elif method == "ilu":
        # Convert to SciPy format for the ILU factorization
        A_scipy_csc = torch_sparse_to_scipy(A_torch.cpu()).tocsc()
        start_time = time_function()
        try:
            # Use SciPy's ILU, which is more stable in this environment
            ilu_op = scipy.sparse.linalg.spilu(A_scipy_csc, drop_tol=1e-6, fill_factor=20)
            prec = ScipyILU(ilu_op)
        except Exception as e:
            print(f"\nWARNING: SciPy ILU factorization failed: {e}")
            prec = Preconditioner()
            prec.breakdown = True
        prec.time = time_function() - start_time
        return prec
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' not implemented!")