import os
import argparse
import json
import torch
import torch_geometric
import time

from apps.data import get_dataloader
from neuralif.utils import count_parameters
from neuralif.loss import loss
from neuralif.models import NeuralIF

# This is the main function that runs the training process.
# It initializes the model, optimizer, and data loader, and runs the training loop.
# It saves the model and configuration if specified.
def main(config):
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config.get("device") is not None else "cpu")
    print(f"Using device: {device}")
    
    folder = os.path.join("results", config["name"])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Results will be saved to: {os.path.abspath(folder)}")
    
    torch_geometric.seed_everything(config["seed"]) # Set the random seed for reproducibility
    
    model_arg_keys = [
        "latent_size", "message_passing_steps", "skip_connections", "augment_nodes",
        "activation", "aggregate", "two_hop", "edge_features"
    ]
    model_args = {k: v for k, v in config.items() if k in model_arg_keys and v is not None} 
    
    # Model is created in float32 by default
    model = NeuralIF(**model_args)
    model.to(device)

    print(f"\nNumber of parameters in model: {count_parameters(model)}\n")
    # Initialise the optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    # Load the model if a checkpoint is provided
    train_loader = get_dataloader(
        dataset_path=config["dataset"], 
        batch_size=config["batch_size"], 
        mode="train",
        add_fill_in=config["add_fill_in"],
        fill_in_k=config["fill_in_k"]
    )
    # Load the model if a checkpoint is provided
    print("--- Starting Training ---")
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss, start_epoch = 0.0, time.perf_counter()
        
        for data in train_loader:
            # Data is loaded as float32 and moved to the device
            data = data.to(device)
            # Ensure the model is in training mode
            optimizer.zero_grad()
            #    We capture it in a new variable `LU_factors`.
            LU_factors, reg, _ = model(data)
            
            loss_kwargs = {
                "pcg_steps": config["pcg_steps"],
                "pcg_weight": config["pcg_weight"],
                "normalized": config["normalize_loss"],
                "preconditioner_solve_steps": config["preconditioner_solve_steps"] 
            }
            
            #    The loss function is designed to handle the tuple output from the model.
            #    It computes the loss based on the LU factors and the data.
            #    The loss function is already designed to handle this.
            main_loss = loss(LU_factors, data, config=config["loss"], **loss_kwargs)
            
            total_loss = main_loss
            if reg is not None and config["regularizer"] > 0:
                total_loss += config["regularizer"] * reg
            #    Backpropagation step
            #    The total loss is computed and backpropagated.
            total_loss.backward()
            #   If gradient clipping is specified, it is applied before the optimizer step
            if config["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            #   Optimizer step
            optimizer.step()
            running_loss += total_loss.item()
            #   Print the loss for the current batch
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']} \t Training Loss: {avg_epoch_loss:.6f} \t Time: {time.perf_counter() - start_epoch:.2f}s")
        #   Save the model state after each epoch if specified
        if config["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")
            
    print("\nTraining complete.")
    if config["save"]:
        torch.save(model.state_dict(), f"{folder}/final_model.pt")

# This function parses command line arguments for the training script.
# It sets default values and allows customisation of the training process.
def argparser():
    parser = argparse.ArgumentParser(description="Training script for NeuralIF.")
    parser.add_argument("--name", type=str, default="training_run")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--regularizer", type=float, default=0.0, help="Weight for L1 penalty. Set > 0 to enable.")
    parser.add_argument("--loss", type=str, default="sketched", choices=['sketched', 'sketch_pcg'], help="Loss function to use.")
    parser.add_argument("--normalize_loss", action='store_true', help="Use normalized Frobenius loss instead of absolute for the 'sketched' baseline.")
    parser.add_argument("--pcg_steps", type=int, default=10, help="Num of steps for the PCG proxy loss.")
    parser.add_argument("--pcg_weight", type=float, default=0.1, help="Weight (lambda) for the PCG proxy loss term.")
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--aggregate", type=str, default="mean")
    parser.add_argument("--edge_features", type=int, default=1)
    parser.add_argument("--preconditioner_solve_steps", type=int, default=5, help="Num of inner CG steps for the triangular solve proxy.")
    parser.add_argument("--augment_nodes", action='store_true', default=True)
    parser.add_argument("--skip_connections", action='store_true', default=True)
    parser.add_argument("--two_hop", action='store_true', default=False)
    parser.add_argument("--add_fill_in", action='store_true', help="Enable the heuristic fill-in preprocessing.")
    parser.add_argument("--fill_in_k", type=int, default=0, help="Number of candidate fill-in edges to add per row.")
    return parser.parse_args()

# This is the entry point of the script.
# It parses the command line arguments and calls the main function with the parsed configuration.
if __name__ == "__main__":
    args = argparser()
    main(vars(args))