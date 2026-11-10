"""
Core Federated Learning Framework for FedAvg and FedProx on PathMNIST.

Implements:
1. FedAvgCNN architecture matching PFLlib specifications.
2. Local client training supporting both FedAvg (mu=0) and FedProx (mu>0).
3. Server-side weighted aggregation.
4. Comprehensive tracking of global metrics, per-client metrics, update norms,
   and cosine similarities (bonus task).
5. Seamless Weights & Biases (W&B) integration with robust offline/online fallbacks.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class FedAvgCNN(nn.Module):
    """
    Standard benchmark Convolutional Neural Network for Federated Learning (McMahan et al., 2017).
    Adapted for 3-channel (RGB) 28x28 images from PathMNIST with 9 tissue classes.
    """
    def __init__(self, in_features=3, num_classes=9, dim=1024):
        super(FedAvgCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # Output: (32, 12, 12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))  # Output: (64, 4, 4) -> 1024 flat
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all trainable parameters of a model into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """Assign a 1D tensor of flat parameters back to the model's parameters."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two flat vectors with numerical epsilon."""
    norm1 = torch.norm(vec1, p=2)
    norm2 = torch.norm(vec2, p=2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = torch.dot(vec1, vec2)
    return (dot / (norm1 * norm2 + eps)).item()


class FLClient:
    """
    Federated Learning Client supporting FedAvg and FedProx local optimization.
    """
    def __init__(
        self,
        client_id: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        device: torch.device,
        batch_size: int = 32,
        local_epochs: int = 2,
        learning_rate: float = 0.01,
        mu: float = 0.0,
        is_noisy: bool = False
    ):
        self.id = client_id
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.mu = mu
        self.is_noisy = is_noisy
        
        # Build PyTorch DataLoaders
        tx = torch.tensor(train_x, dtype=torch.float32)
        ty = torch.tensor(train_y, dtype=torch.int64)
        self.train_dataset = TensorDataset(tx, ty)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.num_train_samples = len(tx)
        
        vx = torch.tensor(test_x, dtype=torch.float32)
        vy = torch.tensor(test_y, dtype=torch.int64)
        self.test_dataset = TensorDataset(vx, vy)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.num_test_samples = len(vx)
        
        self.model = FedAvgCNN().to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_round(self, global_flat_params: torch.Tensor) -> tuple:
        """
        Execute local training for local_epochs starting from global_flat_params.
        
        For FedProx (mu > 0), solves:
            min_w F_i(w) + (mu / 2) * ||w - w^t||^2
        Via proximal gradient step with PyTorch autograd:
            loss = CE(w) + 0.5 * mu * sum_p ||w_p - w_p^t||^2
        """
        set_flat_params(self.model, global_flat_params)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        epoch_losses = []
        
        # Save snapshot of global reference parameters for FedProx
        global_ref_params = [p.clone().detach().to(self.device) for p in self.model.parameters() if p.requires_grad]
        global_ref_flat = global_flat_params.clone().to(self.device)
        
        for epoch in range(self.local_epochs):
            batch_losses = []
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                ce_loss = self.criterion(outputs, batch_y)
                
                # Add FedProx proximal regularization term to computational graph
                if self.mu > 0:
                    prox_loss = 0.0
                    for p, g in zip(self.model.parameters(), global_ref_params):
                        if p.requires_grad:
                            prox_loss = prox_loss + 0.5 * self.mu * torch.sum((p - g) ** 2)
                    loss = ce_loss + prox_loss
                else:
                    loss = ce_loss
                    
                loss.backward()
                optimizer.step()
                batch_losses.append(ce_loss.item())
                
            epoch_losses.append(np.mean(batch_losses))
            
        final_flat = get_flat_params(self.model).to(self.device)
        update_vec = final_flat - global_ref_flat
        update_norm = torch.norm(update_vec, p=2).item()
        avg_train_loss = float(np.mean(epoch_losses))
        
        return final_flat, update_vec, update_norm, avg_train_loss

    def evaluate(self, model: nn.Module = None) -> tuple:
        """Evaluate accuracy and cross-entropy loss on clean local test set."""
        eval_model = self.model if model is None else model
        eval_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = eval_model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_y.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        avg_loss = total_loss / max(total, 1)
        accuracy = (correct / max(total, 1)) * 100.0
        return accuracy, avg_loss, correct, total


class FLServer:
    """
    Federated Learning Coordinator managing communication rounds, model aggregation,
    bonus metric computation (update norms & cosine alignment), and Weights & Biases logging.
    """
    def __init__(
        self,
        clients: list,
        device: torch.device,
        algorithm_name: str = "FedAvg",
        mu: float = 0.0,
        wandb_run=None
    ):
        self.clients = clients
        self.num_clients = len(clients)
        self.device = device
        self.algorithm_name = algorithm_name
        self.mu = mu
        self.wandb_run = wandb_run
        
        # Initialize global model
        self.global_model = FedAvgCNN().to(device)
        self.total_train_samples = sum(c.num_train_samples for c in self.clients)
        self.client_weights = [c.num_train_samples / self.total_train_samples for c in self.clients]
        
        # History log dictionary
        self.history = {
            'rounds': [],
            'global_test_acc': [],
            'global_test_loss': [],
            'global_update_norm': [],
            'per_client_test_acc': {c.id: [] for c in self.clients},
            'per_client_test_loss': {c.id: [] for c in self.clients},
            'per_client_train_loss': {c.id: [] for c in self.clients},
            'per_client_update_norm': {c.id: [] for c in self.clients},
            'per_client_cosine_to_global': {c.id: [] for c in self.clients},
            'noisy_client_pairwise_cosine': {c.id: [] for c in self.clients if c.id != 4}
        }

    def evaluate_global_model(self) -> dict:
        """Evaluate the current global model across all clients' test sets."""
        per_client_acc = {}
        per_client_loss = {}
        total_correct = 0
        total_test_samples = 0
        total_test_loss = 0.0
        
        for client in self.clients:
            acc, loss, correct, total = client.evaluate(self.global_model)
            per_client_acc[client.id] = acc
            per_client_loss[client.id] = loss
            total_correct += correct
            total_test_samples += total
            total_test_loss += loss * total
            
        global_acc = (total_correct / max(total_test_samples, 1)) * 100.0
        global_loss = total_test_loss / max(total_test_samples, 1)
        
        return {
            'global_acc': global_acc,
            'global_loss': global_loss,
            'per_client_acc': per_client_acc,
            'per_client_loss': per_client_loss
        }

    def train_round(self, round_idx: int) -> dict:
        """
        Execute one complete communication round:
        1. Broadcast global parameters w^t.
        2. Perform local training on each client.
        3. Collect update vectors Delta w_i^t.
        4. Compute global update Delta w_global^t = sum p_i Delta w_i^t.
        5. Compute norms and cosine similarities.
        6. Apply aggregation: w^{t+1} = w^t + Delta w_global^t.
        7. Evaluate global and per-client test performance.
        """
        w_t = get_flat_params(self.global_model).to(self.device)
        
        client_updates = []
        client_norms = []
        client_train_losses = []
        
        # Local client updates
        for client in self.clients:
            client.mu = self.mu
            final_flat, update_vec, update_norm, train_loss = client.train_one_round(w_t)
            client_updates.append(update_vec)
            client_norms.append(update_norm)
            client_train_losses.append(train_loss)
            
        # Weighted FedAvg aggregation of updates: Delta w_global = sum (p_i * Delta w_i)
        delta_global = torch.zeros_like(w_t)
        for p_i, delta_i in zip(self.client_weights, client_updates):
            delta_global += p_i * delta_i
            
        global_norm = torch.norm(delta_global, p=2).item()
        
        # Update global model: w^{t+1} = w^t + Delta w_global
        w_next = w_t + delta_global
        set_flat_params(self.global_model, w_next)
        
        # Compute Cosine Similarities (Bonus Task)
        cosine_to_global = {}
        for client, delta_i in zip(self.clients, client_updates):
            cos_val = compute_cosine_similarity(delta_i, delta_global)
            cosine_to_global[client.id] = cos_val
            
        # Pairwise cosine between clean clients and noisy Client 4
        noisy_update = client_updates[4] if len(client_updates) > 4 else None
        noisy_pairwise = {}
        if noisy_update is not None:
            for client, delta_i in zip(self.clients, client_updates):
                if client.id != 4:
                    pair_cos = compute_cosine_similarity(delta_i, noisy_update)
                    noisy_pairwise[client.id] = pair_cos
                    
        # Global evaluation
        eval_metrics = self.evaluate_global_model()
        
        # Record into history
        self.history['rounds'].append(round_idx)
        self.history['global_test_acc'].append(eval_metrics['global_acc'])
        self.history['global_test_loss'].append(eval_metrics['global_loss'])
        self.history['global_update_norm'].append(global_norm)
        
        for client in self.clients:
            cid = client.id
            self.history['per_client_test_acc'][cid].append(eval_metrics['per_client_acc'][cid])
            self.history['per_client_test_loss'][cid].append(eval_metrics['per_client_loss'][cid])
            self.history['per_client_train_loss'][cid].append(client_train_losses[cid])
            self.history['per_client_update_norm'][cid].append(client_norms[cid])
            self.history['per_client_cosine_to_global'][cid].append(cosine_to_global[cid])
            if cid != 4 and cid in noisy_pairwise:
                self.history['noisy_client_pairwise_cosine'][cid].append(noisy_pairwise[cid])
                
        # Format metrics for W&B logging
        wandb_log_dict = {
            'round': round_idx,
            'global/test_accuracy': eval_metrics['global_acc'],
            'global/test_loss': eval_metrics['global_loss'],
            'global/update_norm': global_norm,
        }
        for client in self.clients:
            cid = client.id
            prefix = f"client_{cid}_noisy" if client.is_noisy else f"client_{cid}"
            wandb_log_dict[f"{prefix}/test_accuracy"] = eval_metrics['per_client_acc'][cid]
            wandb_log_dict[f"{prefix}/test_loss"] = eval_metrics['per_client_loss'][cid]
            wandb_log_dict[f"{prefix}/train_loss"] = client_train_losses[cid]
            wandb_log_dict[f"{prefix}/update_norm"] = client_norms[cid]
            wandb_log_dict[f"{prefix}/cosine_to_global"] = cosine_to_global[cid]
            if cid != 4 and cid in noisy_pairwise:
                wandb_log_dict[f"bonus/pairwise_cosine_clean_{cid}_vs_noisy_4"] = noisy_pairwise[cid]
                
        if self.wandb_run is not None:
            self.wandb_run.log(wandb_log_dict, step=round_idx)
            
        return {
            'round': round_idx,
            'global_acc': eval_metrics['global_acc'],
            'global_loss': eval_metrics['global_loss'],
            'global_norm': global_norm,
            'per_client_acc': eval_metrics['per_client_acc'],
            'client_norms': client_norms,
            'cosine_to_global': cosine_to_global
        }

    def fit(self, num_rounds: int, eval_initial: bool = True):
        """Run full federated learning training loop across num_rounds."""
        print(f"\n{'='*75}", flush=True)
        print(f"Starting {self.algorithm_name} Training (mu={self.mu}) | Rounds: {num_rounds}", flush=True)
        print(f"{'='*75}", flush=True)
        
        # Round 0 evaluation
        if eval_initial:
            init_eval = self.evaluate_global_model()
            print(f"[Round 00 Init] Global Test Acc: {init_eval['global_acc']:.2f}% | "
                  f"Global Test Loss: {init_eval['global_loss']:.4f}", flush=True)
            if self.wandb_run is not None:
                self.wandb_run.log({
                    'round': 0,
                    'global/test_accuracy': init_eval['global_acc'],
                    'global/test_loss': init_eval['global_loss']
                }, step=0)
                
        for r in range(1, num_rounds + 1):
            res = self.train_round(r)
            clean_accs = [res['per_client_acc'][c] for c in range(4)]
            noisy_acc = res['per_client_acc'][4]
            
            print(f"[Round {r:02d}/{num_rounds:02d}] "
                  f"Global Acc: {res['global_acc']:.2f}% | Loss: {res['global_loss']:.4f} | "
                  f"Clean Avg Acc: {np.mean(clean_accs):.2f}% | Noisy C4 Acc: {noisy_acc:.2f}% | "
                  f"C4 Cosine: {res['cosine_to_global'][4]:.3f} | Global Norm: {res['global_norm']:.3f}", flush=True)
            
        print(f"\n[+] {self.algorithm_name} training complete! Peak Global Acc: {max(self.history['global_test_acc']):.2f}%\n", flush=True)
        return self.history
