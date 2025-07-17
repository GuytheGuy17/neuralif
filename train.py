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
from neuralif.models import NeuralIF

def main(config):
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config.get("device") is not None else "cpu")
    print(f"Using device: {device}")
    
    folder = os.path.join("results", config["name"])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Results will be saved to: {os.path.abspath(folder)}")
    
    torch_geometric.seed_everything(config["seed"])
    
    model_args = {k: config[k] for k in ["latent_size", "message_passing_steps", "skip_connections", "augment_nodes", "global_features", "decode_nodes", "normalize_diag", "activation", "aggregate", "graph_norm", "two_hop", "edge_features"] if k in config}
    
    model = NeuralIF(**model_args)
    model.to(device)
    print(f"\nNumber of parameters in model: {count_parameters(model)}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    train_loader = get_dataloader(config["dataset"], batch_size=config["batch_size"], mode="train")
    
    print("--- Starting Training ---")
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss, start_epoch = 0.0, time.perf_counter()
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, reg, _ = model(data)
            
            loss_kwargs = {
                "pcg_steps": config["pcg_steps"],
                "pcg_weight": config["pcg_weight"],
            }
            l = loss(output, data, config=config["loss"], **loss_kwargs)
            if reg is not None and config["regularizer"] > 0:
                l += config["regularizer"] * reg
            l.backward()
            if config["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            optimizer.step()
            running_loss += l.item()
            
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
    parser.add_argument("--regularizer", type=float, default=0)
    # Loss
    parser.add_argument("--loss", type=str, default="sketched", choices=['sketched', 'sketch_pcg'])
    parser.add_argument("--pcg_steps", type=int, default=3)
    parser.add_argument("--pcg_weight", type=float, default=0.1)
    # Model
    parser.add_argument("--model", type=str, default="neuralif")
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--augment_nodes", action='store_true', default=False)
    parser.add_argument("--skip_connections", action='store_true', default=True)
    # Dummy args for model_args dict
    for arg in ["global_features", "edge_features", "decode_nodes", "normalize_diag", "aggregate", "graph_norm", "two_hop"]:
        parser.add_argument(f'--{arg}', default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    main(vars(args))