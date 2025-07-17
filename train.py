import os
import datetime
import argparse
import pprint
import torch
import torch_geometric
import time

from apps.data import get_dataloader, graph_to_matrix
from neuralif.utils import count_parameters, save_dict_to_file
from neuralif.logger import TrainResults
from neuralif.loss import loss
from neuralif.models import NeuralIF, NeuralPCG, PreCondNet, LearnedLU

# Note: The original 'validate' function has been removed to focus on the training loop changes.
# It can be added back from your original file if needed for your workflow.

def main(config):
    # Setup device
    if config["device"] is None:
        device = "cpu"
    else:
        device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results folder and save config
    folder = "results/" + (config["name"] if config["name"] else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Results will be saved to: {os.path.abspath(folder)}")
    
    torch_geometric.seed_everything(config["seed"])
    
    # Get model arguments from config
    model_args = {k: config[k] for k in ["latent_size", "message_passing_steps", "skip_connections", "augment_nodes", "global_features", "decode_nodes", "normalize_diag", "activation", "aggregate", "graph_norm", "two_hop", "edge_features"] if k in config}
    
    model = NeuralIF(**model_args)
    model.to(device)
    print(f"\nNumber params in model: {count_parameters(model)}\n")
    
    optimizer = torch.optim.AdamW(model.parameters())
    
    # Using your original get_dataloader call structure
    train_loader = get_dataloader(config["dataset"], n=config.get("n", 0), batch_size=config["batch_size"], mode="train")
    
    logger = TrainResults(folder)
    
    print("--- Starting Training ---")
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        start_epoch = time.perf_counter()
        
        for it, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            output, reg, _ = model(data)
            
            # --- UPDATED LOSS CALL ---
            # Create a dictionary of kwargs to pass to the loss function
            loss_kwargs = {
                "pcg_steps": config.get("pcg_steps", 3),
                "pcg_weight": config.get("pcg_weight", 0.1),
            }

            l = loss(output, data, config=config["loss"], **loss_kwargs)
            
            if reg is not None and config.get("regularizer", 0) > 0:
                l = l + config["regularizer"] * reg

            l.backward()
            
            if config.get("gradient_clipping"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            optimizer.step()
            running_loss += l.item()
            
        epoch_time = time.perf_counter() - start_epoch
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']} \t Training Loss: {avg_epoch_loss:.4f} \t Time: {epoch_time:.2f}s")
        
        if config["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")
            
    print("\nTraining complete.")
    if config["save"]:
        logger.save_results()
        torch.save(model.state_dict(), f"{folder}/final_model.pt")

def argparser():
    parser = argparse.ArgumentParser()
    # Basic args
    parser.add_argument("--name", type=str, default="training_run")
    parser.add_argument("--device", type=int)
    parser.add_argument("--save", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=42)
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default="random")
    parser.add_argument("--n", type=int, default=0)

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--regularizer", type=float, default=0)
    
    # Loss function parameters
    parser.add_argument("--loss", type=str, default="sketched", help="Loss function ('sketched' or 'sketch_pcg').")
    
    # --- NEW ARGUMENTS FOR PCG PROXY LOSS ---
    parser.add_argument("--pcg_steps", type=int, default=3, help="Number of PCG steps for the proxy loss.")
    parser.add_argument("--pcg_weight", type=float, default=0.1, help="Weight of the PCG proxy term in the loss.")
    
    # Model parameters from your original file
    parser.add_argument("--model", type=str, default="neuralif")
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--decode_nodes", action='store_true', default=False)
    parser.add_argument("--normalize_diag", action='store_true', default=False)
    parser.add_argument("--aggregate", nargs="*", type=str)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--skip_connections", action='store_true', default=True)
    parser.add_argument("--augment_nodes", action='store_true', default=False)
    parser.add_argument("--global_features", type=int, default=0)
    parser.add_argument("--edge_features", type=int, default=1)
    parser.add_argument("--graph_norm", action='store_true', default=False)
    parser.add_argument("--two_hop", action='store_true', default=False)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()
    main(vars(args))