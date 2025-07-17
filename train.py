# FILE: train.py

import os
import datetime
import argparse
import json
import torch
import torch_geometric
import time

from apps.data import get_dataloader, graph_to_matrix
from neuralif.utils import count_parameters
from neuralif.logger import TrainResults
from neuralif.loss import loss
from neuralif.models import NeuralIF # We only need the one model

def main(config):
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config.get("device") is not None else "cpu")
    print(f"Using device: {device}")
    
    folder = os.path.join("results", config["name"])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Results will be saved to: {os.path.abspath(folder)}")
    
    torch_geometric.seed_everything(config["seed"])
    
    model_arg_keys = [
        "latent_size", "message_passing_steps", "skip_connections", "augment_nodes",
        "activation", "aggregate", "two_hop", "edge_features"
    ]
    model_args = {k: v for k, v in config.items() if k in model_arg_keys and v is not None}
    
    model = NeuralIF(**model_args)
    model.to(device)
    print(f"\nNumber of parameters in model: {count_parameters(model)}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    train_loader = get_dataloader(dataset_path=config["dataset"], batch_size=config["batch_size"], mode="train")
    
    print("--- Starting Training ---")
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss, start_epoch = 0.0, time.perf_counter()
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # The model returns the L factor and a regularization term
            L_factor, reg, _ = model(data)
            
            # Loss kwargs are passed to the selected loss function
            loss_kwargs = {"pcg_steps": config["pcg_steps"], "pcg_weight": config["pcg_weight"]}
            
            # Calculate the main loss (either 'sketched' or 'sketch_pcg')
            main_loss = loss(L_factor, data, config=config["loss"], **loss_kwargs)
            
            # Add the L1 regularization term from the model
            total_loss = main_loss
            if reg is not None and config["regularizer"] > 0:
                total_loss += config["regularizer"] * reg
            
            total_loss.backward()
            
            if config["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            optimizer.step()
            running_loss += total_loss.item()
            
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']} \t Training Loss: {avg_epoch_loss:.4f} \t Time: {time.perf_counter() - start_epoch:.2f}s")
        
        if config["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")
            
    print("\nTraining complete.")
    if config["save"]:
        torch.save(model.state_dict(), f"{folder}/final_model.pt")

def argparser():
    parser = argparse.ArgumentParser(description="Training script for NeuralIF.")
    # Basic args
    parser.add_argument("--name", type=str, default="training_run")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=42)
    # Dataset
    parser.add_argument("--dataset", type=str, required=True)
    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--regularizer", type=float, default=0.1, help="Weight for L1 penalty from model.")
    # Loss
    parser.add_argument("--loss", type=str, default="sketched", choices=['sketched', 'sketch_pcg'])
    parser.add_argument("--pcg_steps", type=int, default=5, help="Num of steps for PCG proxy loss.")
    parser.add_argument("--pcg_weight", type=float, default=0.1, help="Weight for PCG proxy loss.")
    # Model
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--aggregate", type=str, default="mean")
    parser.add_argument("--edge_features", type=int, default=16)
    parser.add_argument("--augment_nodes", action='store_true', default=True)
    parser.add_argument("--skip_connections", action='store_true', default=True)
    parser.add_argument("--two_hop", action='store_true', default=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    main(vars(args))