"""
Experiment Runner for FedAvg vs FedProx on PathMNIST.

Runs:
1. FedAvg (mu=0.0) with seed 0
2. FedProx (mu=0.001) with seed 0
3. FedProx (mu=0.01) with seed 0
4. FedProx (mu=0.1) with seed 0

Logs all runs to Weights & Biases (W&B):
- Global test accuracy & loss
- Per-client test accuracy & loss (Clients 0-4)
- Bonus task: Client update norms and cosine similarity alignment
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import wandb

from fl_core import FLClient, FLServer, set_flat_params


def load_partitioned_data(data_dir="./data/PathMNIST", num_clients=5):
    """Load client train and test splits from .npz files."""
    train_splits = {}
    test_splits = {}
    
    config_file = os.path.join(data_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
            noisy_client = config.get("noisy_client", 4)
    else:
        noisy_client = 4
        
    for c in range(num_clients):
        train_file = os.path.join(data_dir, "train", f"{c}.npz")
        test_file = os.path.join(data_dir, "test", f"{c}.npz")
        
        with open(train_file, "rb") as f:
            train_dict = np.load(f, allow_pickle=True)['data'].tolist()
            train_splits[c] = train_dict
            
        with open(test_file, "rb") as f:
            test_dict = np.load(f, allow_pickle=True)['data'].tolist()
            test_splits[c] = test_dict
            
    return train_splits, test_splits, noisy_client


def build_clients(train_splits, test_splits, noisy_client, device, batch_size=32, local_epochs=2, lr=0.01, mu=0.0):
    clients = []
    for c in range(len(train_splits)):
        client = FLClient(
            client_id=c,
            train_x=train_splits[c]['x'],
            train_y=train_splits[c]['y'],
            test_x=test_splits[c]['x'],
            test_y=test_splits[c]['y'],
            device=device,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=lr,
            mu=mu,
            is_noisy=(c == noisy_client)
        )
        clients.append(client)
    return clients


def run_single_experiment(
    algo_name: str,
    mu: float,
    seed: int,
    train_splits: dict,
    test_splits: dict,
    noisy_client: int,
    device: torch.device,
    args: argparse.Namespace
) -> dict:
    """Run one FL experiment and log to W&B."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    run_name = f"{algo_name.lower()}_seed{seed}" if mu == 0 else f"{algo_name.lower()}_mu{mu}_seed{seed}"
    print(f"\n{'#'*80}", flush=True)
    print(f"Launching Experiment: {run_name} | Algo: {algo_name} | mu: {mu} | Seed: {seed}", flush=True)
    print(f"{'#'*80}", flush=True)
    
    # Initialize Weights & Biases
    wandb_mode = args.wandb_mode
    if os.environ.get("WANDB_API_KEY"):
        wandb_mode = "online" if args.wandb_mode != "disabled" else "disabled"
        
    wandb_run = wandb.init(
        project=args.project,
        name=run_name,
        mode=wandb_mode,
        config={
            "algorithm": algo_name,
            "mu": mu,
            "seed": seed,
            "num_clients": len(train_splits),
            "noisy_client": noisy_client,
            "noise_rate": 0.40,
            "dataset": "PathMNIST",
            "alpha": 0.3,
            "num_rounds": args.num_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr
        }
    )
    
    clients = build_clients(
        train_splits, test_splits, noisy_client, device,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        mu=mu
    )
    
    server = FLServer(
        clients=clients,
        device=device,
        algorithm_name=algo_name,
        mu=mu,
        wandb_run=wandb_run
    )
    
    history = server.fit(num_rounds=args.num_rounds, eval_initial=True)
    
    # Save run summary in W&B
    peak_acc = max(history['global_test_acc'])
    final_acc = history['global_test_acc'][-1]
    final_noisy_acc = history['per_client_test_acc'][noisy_client][-1]
    clean_accs = [history['per_client_test_acc'][c][-1] for c in range(len(clients)) if c != noisy_client]
    avg_clean_acc = float(np.mean(clean_accs))
    
    wandb.summary["peak_global_accuracy"] = peak_acc
    wandb.summary["final_global_accuracy"] = final_acc
    wandb.summary["final_avg_clean_client_acc"] = avg_clean_acc
    wandb.summary["final_noisy_client_acc"] = final_noisy_acc
    
    wandb_run.finish()
    
    return {
        "run_name": run_name,
        "algorithm": algo_name,
        "mu": mu,
        "seed": seed,
        "peak_global_acc": peak_acc,
        "final_global_acc": final_acc,
        "final_avg_clean_acc": avg_clean_acc,
        "final_noisy_client_acc": final_noisy_acc,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Run FedAvg and FedProx experiments on PathMNIST.")
    parser.add_argument("--project", type=str, default="fedavg-vs-fedprox-pathmnist", help="W&B project name")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"],
                        help="W&B logging mode (default: offline, can be synced via `wandb sync`)")
    parser.add_argument("--data_dir", type=str, default="./data/PathMNIST", help="Path to preprocessed PathMNIST data")
    parser.add_argument("--num_rounds", type=int, default=20, help="Communication rounds (default: 20)")
    parser.add_argument("--local_epochs", type=int, default=2, help="Local epochs per round (default: 2)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory to save metric history and plots")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing on device: {device}")
    
    # Ensure data exists
    if not os.path.exists(os.path.join(args.data_dir, "train")):
        print(f"[!] Dataset not found at {args.data_dir}. Generating now...")
        os.system(f"python generate_PathMNIST.py --max_samples 15000 --seed {args.seed}")
        
    train_splits, test_splits, noisy_client = load_partitioned_data(args.data_dir)
    print(f"[*] Loaded data for {len(train_splits)} clients. Noisy client: Client {noisy_client} (40% label noise)")
    
    # Configurations to evaluate:
    # 1. FedAvg (mu=0.0)
    # 2. FedProx (mu=0.001)
    # 3. FedProx (mu=0.01)
    # 4. FedProx (mu=0.1)
    configs = [
        {"algo": "FedAvg", "mu": 0.0},
        {"algo": "FedProx", "mu": 0.001},
        {"algo": "FedProx", "mu": 0.01},
        {"algo": "FedProx", "mu": 0.1},
    ]
    
    all_results = {}
    for cfg in configs:
        res = run_single_experiment(
            algo_name=cfg["algo"],
            mu=cfg["mu"],
            seed=args.seed,
            train_splits=train_splits,
            test_splits=test_splits,
            noisy_client=noisy_client,
            device=device,
            args=args
        )
        all_results[res["run_name"]] = res
        
    # Save metric history to disk
    history_file = os.path.join(args.out_dir, "metrics_history.json")
    with open(history_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] All experiment metrics saved to {history_file}")
    
    # Print final summary comparison table
    print("\n" + "="*95)
    print(f"{'Run Name':<28}{'Algorithm':<12}{'mu':<8}{'Peak Acc (%)':<15}{'Final Acc (%)':<15}{'Clean Avg (%)':<15}{'Noisy C4 (%)'}")
    print("="*95)
    for run_name, res in all_results.items():
        print(f"{run_name:<28}{res['algorithm']:<12}{res['mu']:<8}{res['peak_global_acc']:<15.2f}"
              f"{res['final_global_acc']:<15.2f}{res['final_avg_clean_acc']:<15.2f}{res['final_noisy_client_acc']:.2f}")
    print("="*95 + "\n")
    
    # Automatically generate plots
    print("[*] Generating comparison plots...")
    os.system(f"python plot_results.py --history_file {history_file} --out_dir {args.out_dir}")
    print("[+] Complete pipeline finished successfully!")


if __name__ == "__main__":
    main()
