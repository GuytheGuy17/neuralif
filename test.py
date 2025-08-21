import argparse
import os
import datetime
import json

import numpy as np
import torch
import torch_geometric

from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.preconditioner import get_preconditioner
from neuralif.models import NeuralIF
from neuralif.utils import time_function, load_checkpoint
from neuralif.logger import TestResults
from apps.data import get_dataloader

# This function loads a model from a checkpoint and returns it along with the training configuration.
@torch.inference_mode()
def test(model, test_loader, device, folder, save_results=False, dataset="random", solver="cg", drop_tol=1e-6):
    if save_results:
        os.makedirs(folder, exist_ok=True)

    print(f"\nTest:\t{len(test_loader.dataset)} samples")
    print(f"Solver:\t{solver} solver\n")
    
    methods = ["learned"] if model is not None else ["baseline", "jacobi", "ic"]
    # Warmup the model if it exists
    for method in methods:
        print(f"Testing {method} preconditioner") 
        test_results = TestResults(method, dataset, folder,
                                   model_name=f"\n{model.__class__.__name__}" if method == "learned" else "",
                                   target=1e-6, solver=solver)
        
        for sample, data in enumerate(test_loader):
            plot = save_results and sample == (len(test_loader.dataset) - 1)

            # Reconstruct the original matrix A on the CPU first
            if data.edge_attr.dim() > 1 and data.edge_attr.shape[1] > 1:
                is_fill_in_flag = data.edge_attr[:, 1]
                original_edge_mask = (is_fill_in_flag == 0)
                edge_index_for_A = data.edge_index[:, original_edge_mask]
                edge_attr_for_A = data.edge_attr[original_edge_mask, 0]
            else:
                edge_index_for_A = data.edge_index
                edge_attr_for_A = data.edge_attr.squeeze()
            # Create the COO tensor for A on CPU
            # This is crucial for Jacobi and ScipyILU preconditioners.
            A_coo_cpu = torch.sparse_coo_tensor(
                edge_index_for_A,
                edge_attr_for_A,
                (data.num_nodes, data.num_nodes),
                dtype=torch.float64
            )
            A_csr_cpu = A_coo_cpu.to_sparse_csr()
            # Move data to the target device
           
            # The preconditioner is ALWAYS created on the CPU using CPU data.
            # This is crucial for Jacobi
            prec = get_preconditioner(data, A_coo_cpu, method, model=model, device=device, drop_tol=drop_tol)
            p_time, breakdown, nnzL = prec.time, prec.breakdown, prec.nnz
            
            # move the tensors needed for the solver to the target device.
            A = A_csr_cpu.to(device)
            b = data.x[:, 0].squeeze().to(torch.float64).to(device)
            solution = data.s.squeeze().to(torch.float64).to(device) if hasattr(data, "s") else None
            
            # Start the solver
            start_solver = time_function()
            solver_settings = {"max_iter": 2 * data.num_nodes, "x0": None, "rtol": test_results.target}

            # Use the preconditioner if it's not the baseline method
            res, final_x = [], None
            if breakdown:
                print(f"  -> WARNING: Preconditioner breakdown on sample {sample}.")
            elif method == "baseline":
                res, final_x = conjugate_gradient(A, b, x_true=solution, **solver_settings)
            else:
                res, final_x = preconditioned_conjugate_gradient(A, b, M=prec, x_true=solution, **solver_settings)
            
            solver_time = time_function() - start_solver
            # Log the results
            iterations = len(res) - 1 if res else -1
            print(f"  [Sample {sample+1}/{len(test_loader.dataset)}] Solved in {solver_time:.2f}s with {iterations} iterations.")
            # Log the results to the test results object
            if res:
                A_norm_sq = np.array([r[0].item() for r in res])
                rel_res_sq = np.array([r[1].item() for r in res])
                b_norm_sq = torch.inner(b.cpu(), b.cpu()).item() # Ensure norms are computed on CPU
                res_norms = np.sqrt(rel_res_sq * b_norm_sq)
                err_norms = np.sqrt(A_norm_sq)
                
                test_results.log_solve(A.shape[0], solver_time, len(res) - 1,
                                       res_norms, err_norms, p_time, 0.0)
                test_results.log(A.cpu()._nnz(), nnzL, plot=plot)
                
        if save_results: test_results.save_results()
        test_results.print_summary()

# This function is used to warm up the model by running a dummy forward pass.
# This is useful to ensure that the model is ready for inference.
def warmup(model, device):
    if model is None: return
    model.to(device)
    model.eval()
    data = torch_geometric.data.Data(
        x=torch.randn(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.randn(2, 1)
    ).to(device)
    _ = model(data)
    print("Model warmup done...")
# This function parses command line arguments for the script.
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="A name for the test run folder.")
    parser.add_argument("--device", type=int, required=False, default=0, help="CUDA device index. Use -1 for CPU.")
    parser.add_argument("--model", type=str, required=False, default="none")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the checkpoint *folder*.")
    parser.add_argument("--weights", type=str, required=False, default="final_model")
    parser.add_argument("--solver", type=str, default="cg")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root folder.")
    parser.add_argument("--subset", type=str, required=False, default="test")
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--drop_tol", type=float, default=1e-6, 
                        help="Tolerance for dropping small values from the learned preconditioner.")
    return parser.parse_args()
# This function generates a synthetic sparse SPD matrix with a target density. 
def main():
    args = argparser()
    
    if torch.cuda.is_available() and args.device >= 0:
        test_device = torch.device(f"cuda:{args.device}")
    else:
        test_device = torch.device("cpu")
        if args.device >= 0:
             print("WARNING: CUDA not found or invalid device ID specified. Defaulting to CPU.")
        else:
             print("INFO: CPU explicitly selected for testing.")
        
    folder = "results/" + (args.name if args.name else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    print(f"\nUsing device: {test_device}")
    
    model = None
    training_config = {}
    if args.model.lower() == "neuralif":
        print("Use model: NeuralIF")
        model, training_config = load_checkpoint(
            checkpoint_path=args.checkpoint,
            model_class=NeuralIF,
            device=test_device,
            weights_name=f"{args.weights}.pt"
        )
        
    elif args.model.lower() != "none":
        print(f"WARNING: Model '{args.model}' not recognized. Running non-data-driven baselines only.")

    warmup(model, test_device)

    use_fill_in = training_config.get("add_fill_in", False)
    fill_in_k_val = training_config.get("fill_in_k", 0)

    print("\nConfiguring test dataloader based on training config:")
    print(f"  -> Add Fill-In: {use_fill_in}")
    print(f"  -> Fill-In K: {fill_in_k_val}")

    testdata_loader = get_dataloader(
        dataset_path=args.dataset,
        batch_size=1,
        mode=args.subset,
        add_fill_in=use_fill_in,
        fill_in_k=fill_in_k_val
    )
    
    test(model, testdata_loader, test_device, folder,
         save_results=args.save, dataset=os.path.basename(args.dataset), 
         solver=args.solver, drop_tol=args.drop_tol)


if __name__ == "__main__":
    main()