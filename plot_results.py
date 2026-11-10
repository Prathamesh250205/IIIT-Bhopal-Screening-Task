"""
Plotting Script for Federated Learning Experiments on PathMNIST.

Generates a publication-quality 4-panel comparison figure:
1. Panel (a): Global Test Accuracy across Communication Rounds (FedAvg vs FedProx variants).
2. Panel (b): Per-Client Test Accuracy (Clean Clients 0-3 vs Noisy Client 4).
3. Panel (c): Client Update Norms ||Delta w_i^t|| and Global Update Norm over rounds.
4. Panel (d): Cosine Similarity of Client Updates to Global Update direction.

Saves high-resolution PNG and PDF to the results directory.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_plotting_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'lines.linewidth': 2.0,
        'lines.markersize': 5,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })


def plot_metrics(history_file="./results/metrics_history.json", out_dir="./results"):
    os.makedirs(out_dir, exist_ok=True)
    with open(history_file, "r") as f:
        results = json.load(f)
        
    set_plotting_style()
    
    # Palette definition
    algo_colors = {
        'fedavg_seed0': '#DC2626',           # Crimson Red
        'fedprox_mu0.001_seed0': '#F59E0B',  # Amber
        'fedprox_mu0.01_seed0': '#10B981',   # Emerald Green (Best)
        'fedprox_mu0.1_seed0': '#6366F1',    # Indigo
    }
    algo_labels = {
        'fedavg_seed0': 'FedAvg (mu=0)',
        'fedprox_mu0.001_seed0': 'FedProx (mu=0.001)',
        'fedprox_mu0.01_seed0': 'FedProx (mu=0.01)',
        'fedprox_mu0.1_seed0': 'FedProx (mu=0.1)',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Federated Optimization on PathMNIST (5 Clients, Dirichlet alpha=0.3, 40% Label Noise on Client 4)",
                 fontsize=13, weight='bold', y=0.98)
    
    # -------------------------------------------------------------
    # (a) Global Test Accuracy vs Rounds
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    for run_name, data in results.items():
        hist = data['history']
        rounds = hist['rounds']
        acc = hist['global_test_acc']
        color = algo_colors.get(run_name, '#475569')
        label = algo_labels.get(run_name, run_name)
        ax_a.plot(rounds, acc, marker='o', label=label, color=color)
        
    ax_a.set_title("(a) Global Test Accuracy Comparison")
    ax_a.set_xlabel("Communication Round")
    ax_a.set_ylabel("Global Test Accuracy (%)")
    ax_a.grid(True)
    ax_a.legend(loc='lower right', frameon=True)
    ax_a.set_ylim(bottom=max(0, min(acc) - 5), top=min(100, max(acc) + 5))
    
    # -------------------------------------------------------------
    # (b) Per-Client Accuracy (FedAvg vs Best FedProx)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    best_prox_name = 'fedprox_mu0.01_seed0' if 'fedprox_mu0.01_seed0' in results else list(results.keys())[1]
    fedavg_name = 'fedavg_seed0' if 'fedavg_seed0' in results else list(results.keys())[0]
    
    client_colors = ['#2563EB', '#0D9488', '#D97706', '#9333EA', '#EF4444']
    
    # Plot FedAvg per-client
    rounds = results[fedavg_name]['history']['rounds']
    for cid in range(5):
        c_acc_avg = results[fedavg_name]['history']['per_client_test_acc'][str(cid)]
        c_acc_prox = results[best_prox_name]['history']['per_client_test_acc'][str(cid)]
        
        label_base = f"Client {cid}" if cid != 4 else f"Client {cid} (40% Noise)"
        ax_b.plot(rounds, c_acc_avg, linestyle='--', color=client_colors[cid], alpha=0.5,
                  label=f"{label_base} [FedAvg]" if cid in [0, 4] else None)
        ax_b.plot(rounds, c_acc_prox, linestyle='-', marker='s', markersize=4, color=client_colors[cid],
                  label=f"{label_base} [FedProx]" if cid in [0, 4] else None)
        
    ax_b.set_title(f"(b) Per-Client Test Accuracy (FedAvg vs FedProx mu=0.01)")
    ax_b.set_xlabel("Communication Round")
    ax_b.set_ylabel("Client Test Accuracy (%)")
    ax_b.grid(True)
    ax_b.legend(loc='lower right', frameon=True, fontsize=8)
    
    # -------------------------------------------------------------
    # (c) Client Update Norms ||Delta w_i^t||
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    # For FedAvg vs FedProx, compare update norms
    for cid in range(5):
        c_norms = results[best_prox_name]['history']['per_client_update_norm'][str(cid)]
        lbl = f"Client {cid}" if cid != 4 else f"Client 4 (Noisy)"
        lw = 2.5 if cid == 4 else 1.5
        ax_c.plot(rounds, c_norms, color=client_colors[cid], label=lbl, linewidth=lw)
        
    # Global norm
    g_norms = results[best_prox_name]['history']['global_update_norm']
    ax_c.plot(rounds, g_norms, color='#0F172A', linestyle=':', linewidth=2.5, label='Global Update Norm')
    
    ax_c.set_title(f"(c) Client & Global Update Norms ||Delta w|| (FedProx mu=0.01)")
    ax_c.set_xlabel("Communication Round")
    ax_c.set_ylabel(r"$\ell_2$ Update Norm $||\Delta w||_2$")
    ax_c.grid(True)
    ax_c.legend(loc='upper right', frameon=True)
    
    # -------------------------------------------------------------
    # (d) Cosine Alignment: cos(Delta w_i^t, Delta w_global^t)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    for cid in range(5):
        cos_prox = results[best_prox_name]['history']['per_client_cosine_to_global'][str(cid)]
        lbl = f"Client {cid}" if cid != 4 else f"Client 4 (Noisy)"
        lw = 2.5 if cid == 4 else 1.5
        ax_d.plot(rounds, cos_prox, color=client_colors[cid], label=lbl, linewidth=lw)
        
    ax_d.set_title(f"(d) Directional Alignment: Cosine Similarity to Global Update")
    ax_d.set_xlabel("Communication Round")
    ax_d.set_ylabel(r"Cosine Similarity $\cos(\Delta w_i, \Delta w_{\mathrm{global}})$")
    ax_d.grid(True)
    ax_d.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax_d.legend(loc='lower right', frameon=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    png_path = os.path.join(out_dir, "fedavg_vs_fedprox_pathmnist.png")
    pdf_path = os.path.join(out_dir, "fedavg_vs_fedprox_pathmnist.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    
    print(f"[+] Plots successfully generated:")
    print(f"    - PNG: {png_path}")
    print(f"    - PDF: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_file", type=str, default="./results/metrics_history.json")
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()
    
    plot_metrics(args.history_file, args.out_dir)

