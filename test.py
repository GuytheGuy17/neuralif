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
from neuralif.utils import time_function
from neuralif.logger import TestResults
from apps.data import get_dataloader


@torch.inference_mode()
def test(model, test_loader, device, folder, save_results=False, dataset="random", solver="cg"):
    if save_results:
        os.makedirs(folder, exist_ok=True)

    print(f"\nTest:\t{len(test_loader.dataset)} samples")
    print(f"Solver:\t{solver} solver\n")
    
    methods = ["learned"] if model is not None else ["baseline", "jacobi", "ic"]

    for method in methods:
        print(f"Testing {method} preconditioner")
        test_results = TestResults(method, dataset, folder,
                                   model_name=f"\n{model.__class__.__name__}" if method == "learned" else "",
                                   target=1e-6, solver=solver)
        
        for sample, data in enumerate(test_loader):
            plot = save_results and sample == (len(test_loader.dataset) - 1)
            
            # The dataloader has already preprocessed the data if needed
            
            A_coo = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0],
                                       (data.num_nodes, data.num_nodes),
                                       dtype=torch.float64).to('cpu')
            A = A_coo.to_sparse_csr()

            prec = get_preconditioner(data, A_coo, method, model=model, device=device)
            p_time, breakdown, nnzL = prec.time, prec.breakdown, prec.nnz
            
            b = data.x[:, 0].squeeze().to(torch.float64).to('cpu')
            solution = data.s.to(torch.float64).squeeze().to('cpu') if hasattr(data, "s") else None
            
            start_solver = time_function()
            solver_settings = {"max_iter": 2 * data.num_nodes, "x0": None, "rtol": test_results.target}
            
            res, final_x = [], None
            if breakdown:
                print(f"  -> WARNING: Preconditioner breakdown on sample {sample}.")
            elif method == "baseline":
                res, final_x = conjugate_gradient(A, b, x_true=solution, **solver_settings)
            else:
                res, final_x = preconditioned_conjugate_gradient(A, b, M=prec, x_true=solution, **solver_settings)
            
            solver_time = time_function() - start_solver
            
            if res:
                A_norm_sq = np.array([r[0].item() for r in res])
                rel_res_sq = np.array([r[1].item() for r in res])
                b_norm_sq = torch.inner(b, b).item()
                res_norms = np.sqrt(rel_res_sq * b_norm_sq)
                err_norms = np.sqrt(A_norm_sq)
                
                test_results.log_solve(A.shape[0], solver_time, len(res) - 1,
                                       res_norms, err_norms, p_time, 0.0)
                test_results.log(A._nnz(), nnzL, plot=plot)
                
        if save_results: test_results.save_results()
        test_results.print_summary()


def load_checkpoint(model_class, args, device):
    # ... (implementation is unchanged)
    checkpoint_path = args.checkpoint
    config_path = os.path.join(checkpoint_path, "config.json")
    weights_path = os.path.join(checkpoint_path, f"{args.weights}.pt")
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found at {config_path}")
    if not os.path.exists(weights_path): raise FileNotFoundError(f"Weights file not found at {weights_path}")
    with open(config_path) as f:
        config = json.load(f)
    model_arg_keys = ["latent_size", "message_passing_steps", "skip_connections", "augment_nodes", "activation", "aggregate", "two_hop", "edge_features"]
    model_args = {k: v for k, v in config.items() if k in model_arg_keys}
    model = model_class(**model_args)
    print(f"Loading checkpoint from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model


def warmup(model, device):
    # ... (implementation is unchanged)
    if model is None: return
    model.to(device)
    model.eval()
    data = torch_geometric.data.Data(x=torch.randn(2, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2, 2)).to(device)
    _ = model(data)
    print("Model warmup done...")


def argparser():
    parser = argparse.ArgumentParser()
    # ... (most arguments are the same) ...
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="none")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--weights", type=str, required=False, default="final_model")
    parser.add_argument("--solver", type=str, default="cg")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, required=False, default="test")
    parser.add_argument("--samples", type=int, required=False, default=None)
    parser.add_argument("--save", action='store_true', default=False)

    # --- START OF NEW ARGUMENTS ---
    parser.add_argument("--add_fill_in", action='store_true', help="Enable the heuristic fill-in preprocessing for evaluation.")
    parser.add_argument("--fill_in_k", type=int, default=5, help="Number of candidate fill-in edges to add per row for evaluation.")
    # --- END OF NEW ARGUMENTS ---
    
    return parser.parse_args()


def main():
    args = argparser()
    
    if torch.cuda.is_available() and args.device is not None:
        test_device = torch.device(f"cuda:{args.device}")
    else:
        test_device = torch.device("cpu")
        
    folder = "results/" + (args.name if args.name else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    print(f"\nUsing device: {test_device}")
    
    model = None
    if args.model.lower() == "neuralif":
        print("Use model: NeuralIF")
        model = load_checkpoint(NeuralIF, args, test_device)
        model.to(torch.float64) # Ensure model is in correct precision for eval
    elif args.model.lower() != "none":
        print(f"WARNING: Model '{args.model}' not recognized. Running non-data-driven baselines only.")

    warmup(model, test_device)
    
    # --- START OF MODIFICATION ---
    # Pass the new fill-in arguments to the dataloader for the test set
    testdata_loader = get_dataloader(
        dataset_path=args.dataset, 
        batch_size=1, 
        mode=args.subset,
        add_fill_in=args.add_fill_in,
        fill_in_k=args.fill_in_k
    )
    # --- END OF MODIFICATION ---
    
    test(model, testdata_loader, test_device, folder,
         save_results=args.save, dataset=os.path.basename(args.dataset), solver=args.solver)


if __name__ == "__main__":
    main()