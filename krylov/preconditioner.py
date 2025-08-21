import torch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import pyamg 

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
    

# In krylov/preconditioner.py

class ScipyIC_Final_Corrected(Preconditioner):
    """
    This is the definitive and final robust implementation of Incomplete Cholesky (IC(0))
    using only the SciPy library.

    It addresses silent failures in high-level wrappers by using the more direct
    and robust `splu` interface to the SuperLU library. It then correctly performs
    the symmetric two-step solve with L and L.T, which is essential for PCG.
    """
    def __init__(self, A_torch: torch.Tensor):
        super().__init__()
        start_time = time_function()
        
        # SuperLU requires the CSC format and float64 for stability.
        A_scipy_csc = torch_sparse_to_scipy(A_torch.cpu().to(torch.float64)).tocsc()
        
        try:
            # Use splu for its robust C-backend (SuperLU). This is more reliable than spilu.
            # We explicitly tell it not to reorder the matrix to preserve the IC(0) structure.
            options = dict(NATURAL=True)
            self.lu_solver = scipy.sparse.linalg.splu(A_scipy_csc, options=options)
            
            # The solver object contains the L and U factors. We will use them manually.
            self._nnz = self.lu_solver.L.nnz

            # Sanity check: If factorization produced trivial results, it's a failure.
            if self._nnz <= A_torch.shape[0]:
                raise RuntimeError("Factorization resulted in a trivial (diagonal) factor.")

        except (RuntimeError, np.linalg.LinAlgError) as e:
            self.breakdown = True
            self.breakdown_reason = str(e)
            self.lu_solver = None
            self._nnz = 0
            print(f"\nWARNING: SciPy IC factorization failed: {e}")
        
        self.time = time_function() - start_time

    @property
    def nnz(self):
        return self._nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        if self.breakdown or self.lu_solver is None:
            return b
        
        b_np = b.cpu().numpy().astype(np.float64)
        
        # The lu_solver.solve method can perform the triangular solves for us.
        # This is the most stable way to do it.
        # P z = r  =>  L L^T z = r
        
        # 1. Forward substitution: Solve L y = r for y
        #    We can use the solver object with U=I and not permuting rows.
        y = self.lu_solver.L @ scipy.sparse.linalg.spsolve_triangular(self.lu_solver.U, b_np, lower=False)


        # 2. Backward substitution: Solve L^T z = y for z
        z = self.lu_solver.U.T @ scipy.sparse.linalg.spsolve_triangular(self.lu_solver.L.T, y, lower=True)

        return torch.from_numpy(z).to(b.device, b.dtype)

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
            LU_factors, _, _ = self.model(data_on_device)
            L_torch, U_torch = LU_factors
        # Ensure L_torch is a sparse COO tensor
        L_final = L_torch.coalesce()
        mask = L_final.values().abs() > self.drop_tol
        # Apply the mask to filter out small values
        L_final = torch.sparse_coo_tensor(
            L_final.indices()[:, mask],
            L_final.values()[mask],
            L_final.shape
        ).coalesce()

        
        # Apply the same filtering to the U matrix from the model.
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
            return self.L_scipy.nnz
        return 0
    # This method solves the linear system using SciPy's triangular solver.
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
        return ScipyIC_Final_Corrected(A_torch_cpu)
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' not implemented!")