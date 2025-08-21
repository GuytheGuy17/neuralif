import torch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from neuralif.utils import torch_sparse_to_scipy, time_function
# This is a base class for preconditioners used in Krylov methods.
class Preconditioner:
    def __init__(self):
        self.time = 0.0
        self.breakdown = False
        self.breakdown_reason = ""
    def __call__(self, b: torch.Tensor) -> torch.Tensor:
        # 'b' here is the vector passed from the solver
        return self.solve(b)
    @property
    def nnz(self):
        return 0
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return b
# This is a Jacobi preconditioner that uses the inverse of the diagonal of the matrix A.
class Jacobi(Preconditioner):
    def __init__(self, A_torch: torch.Tensor):
        super().__init__()
        start = time_function()
        # All computation is on the CPU
        A_cpu = A_torch.to('cpu').coalesce()
        indices = A_cpu.indices()
        values = A_cpu.values()
        diag_mask = indices[0] == indices[1] # Mask for diagonal elements
        diag_indices = indices[0][diag_mask] # Indices of diagonal elements
        # Extract the diagonal values
        diag_values = values[diag_mask]
        n = A_cpu.shape[0] # Number of nodes
        # Create a full diagonal tensor
        full_diag = torch.zeros(n, device='cpu', dtype=A_cpu.dtype)
        full_diag.scatter_(0, diag_indices, diag_values) # Fill the diagonal
        self.inv_diag = 1.0 / (full_diag + 1e-12) # Small epsilon to avoid division by zero
        self.time = time_function() - start # Store the time taken for initialisation
        self._nnz = n # Number of non-zero elements in the diagonal
    @property
    def nnz(self):
        return self._nnz # Number of non-zero elements in the diagonal
    
   
    # The method must operate on the input vector 'b' 
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        # 'b' is the input tensor from the solver
        # self.inv_diag is on the CPU
        # Move the inverse diagonal to the correct device for the multiplication
        inv_diag_on_device = self.inv_diag.to(b.device, b.dtype)
        return inv_diag_on_device * b
    



class ScipyILU(Preconditioner):
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
        # Correctly moves input 'b' to CPU for SciPy
        b_np = b.cpu().numpy()
        x_np = self.ilu_op.solve(b_np)
        # Correctly returns tensor to original device
        return torch.from_numpy(x_np).to(b.device, b.dtype)

# This is a learned preconditioner that uses a neural network model to compute the preconditioner.
class LearnedPreconditioner(Preconditioner):
    def __init__(self, data_on_device, model, drop_tol=1e-6):
        super().__init__()
        self.model = model # The model is expected to be a PyTorch model
        self.drop_tol = drop_tol # Drop tolerance for the preconditioner
        start = time_function() # Start timing the preconditioner computation
        self._compute_preconditioner(data_on_device)
        self.time = time_function() - start # Store the time taken for preconditioner computation
    # This method computes the preconditioner using the model.
    # It expects the model to output a lower triangular matrix L.
    def _compute_preconditioner(self, data_on_device): 
        self.model.eval()
        with torch.no_grad():
            L_torch, U_torch, _ = self.model(data_on_device)
        # Ensure L_torch is a sparse COO tensor
        L_final = L_torch.coalesce()
        mask = L_final.values().abs() > self.drop_tol
        # Apply the mask to filter out small values
        L_final = torch.sparse_coo_tensor(
            L_final.indices()[:, mask],
            L_final.values()[mask],
            L_final.shape
        ).coalesce()

        # --- Process U ---
        # 2. Apply the same filtering to the U matrix from the model.
        U_final = U_torch.coalesce()
        mask_U = U_final.values().abs() > self.drop_tol
        U_final = torch.sparse_coo_tensor(
            U_final.indices()[:, mask_U],
            U_final.values()[mask_U],
            U_final.shape
        ).coalesce()
        # Convert to CSR format for efficient solving
        self.L_scipy = torch_sparse_to_scipy(L_final.cpu()).tocsc()
        self.U_scipy = torch_sparse_to_scipy(U_final.cpu()).tocsc()
    
    @property
    def nnz(self):
        if self.L_scipy is not None:
            return self.L_scipy.nnz + self.U_scipy.nnz
        return 0
    # This method solves the linear system L * y = b using SciPy's triangular solver.
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        # Correctly moves input 'b' to CPU for SciPy
        b_np = b.cpu().numpy()
        y = scipy.sparse.linalg.spsolve_triangular(self.L_scipy, b_np, lower=True)
        x_np = scipy.sparse.linalg.spsolve_triangular(self.U_scipy, y, lower=False)
        # Correctly returns tensor to original device
        return torch.from_numpy(x_np).to(b.device, b.dtype)
    
# This function creates a preconditioner based on the method specified.
def get_preconditioner(data, A_torch_cpu, method: str, model=None, device='cpu', drop_tol=1e-6) -> Preconditioner:
    if method == "learned":
        if model is None: raise ValueError("A model must be provided for the 'learned' method.")
        data_on_device = data.to(device)
        return LearnedPreconditioner(data_on_device, model, drop_tol=drop_tol)
    
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        # The constructor correctly uses the CPU tensor A_torch_cpu
        return Jacobi(A_torch_cpu)
    elif method == "ic":
        # The constructor correctly uses the CPU tensor A_torch_cpu
        return ScipyILU(A_torch_cpu)
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' not implemented!")