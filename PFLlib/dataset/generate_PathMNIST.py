"""
Dataset generation script for PathMNIST (MedMNIST) under Federated Learning.

Generates:
1. 5-client non-IID partition using Dirichlet distribution (alpha = 0.3)
2. Injects 40% label noise to Client 4's training data (symmetric class flipping)
3. Keeps test sets 100% clean across all clients for true generalization assessment
4. Exports PFLlib-compatible .npz files for each client (train/ and test/ directories)
"""

import os
import sys
import copy
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import medmnist
from medmnist import PathMNIST


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_or_load_pathmnist(root="~/.medmnist", max_samples=None, seed=42):
    """
    Load PathMNIST using MedMNIST API.
    Combines train and validation splits to form total pool (or subsamples if max_samples is given).
    """
    root = os.path.expanduser(root)
    print(f"[*] Loading PathMNIST dataset from {root}...")
    
    train_data = PathMNIST(split='train', download=True, root=root)
    val_data = PathMNIST(split='val', download=True, root=root)
    test_data = PathMNIST(split='test', download=True, root=root)
    
    # PathMNIST raw images are (N, 28, 28, 3), labels are (N, 1)
    imgs = np.concatenate([train_data.imgs, val_data.imgs, test_data.imgs], axis=0)
    labels = np.concatenate([train_data.labels, val_data.labels, test_data.labels], axis=0).squeeze()
    
    num_classes = len(np.unique(labels))
    print(f"[*] Loaded PathMNIST: {len(imgs)} total samples, {num_classes} classes.")
    
    if max_samples is not None and max_samples < len(imgs):
        print(f"[*] Subsampling {max_samples} samples uniformly across classes for efficient experimentation...")
        rng = np.random.RandomState(seed)
        samples_per_class = max_samples // num_classes
        selected_indices = []
        for c in range(num_classes):
            c_indices = np.where(labels == c)[0]
            rng.shuffle(c_indices)
            selected_indices.extend(c_indices[:samples_per_class])
        
        selected_indices = np.array(selected_indices)
        rng.shuffle(selected_indices)
        imgs = imgs[selected_indices]
        labels = labels[selected_indices]
        print(f"[*] Subsampled pool size: {len(imgs)} samples ({samples_per_class} per class).")
    
    # Normalize images to [0, 1] float32 and convert to channel-first (N, 3, 28, 28)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.transpose(imgs, (0, 3, 1, 2))  # (N, C, H, W)
    
    return imgs, labels, num_classes


def dirichlet_partition(imgs, labels, num_clients=5, alpha=0.3, min_samples_per_client=100, seed=42):
    """
    Partition dataset among clients using a Dirichlet distribution over class proportions.
    
    Args:
        imgs: numpy array of shape (N, C, H, W)
        labels: numpy array of shape (N,)
        num_clients: number of federated clients (default: 5)
        alpha: Dirichlet concentration parameter (default: 0.3)
        min_samples_per_client: minimum sample threshold
    Returns:
        client_data: dict of client_id -> {'x': imgs, 'y': labels}
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    N = len(labels)
    
    client_indices = [[] for _ in range(num_clients)]
    
    # Try until each client has at least min_samples_per_client
    max_attempts = 100
    for attempt in range(max_attempts):
        client_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # Sample Dirichlet proportions
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance sample allocation
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, client_indices)])
            if proportions.sum() == 0:
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            
            # Split indices by cumulative proportions
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, split_points)
            
            for j in range(num_clients):
                client_indices[j].extend(splits[j].tolist())
                
        min_size = min(len(idxs) for idxs in client_indices)
        if min_size >= min_samples_per_client:
            break
            
    print(f"[*] Partition complete after {attempt + 1} attempt(s).")
    
    client_data = {}
    for j in range(num_clients):
        idxs = np.array(client_indices[j])
        np.random.shuffle(idxs)
        client_data[j] = {
            'x': imgs[idxs],
            'y': labels[idxs]
        }
        
    return client_data


def train_test_split_clients(client_data, train_ratio=0.8, seed=42):
    """
    Split each client's partition into train and test subsets.
    """
    np.random.seed(seed)
    train_splits = {}
    test_splits = {}
    
    for client_id, data in client_data.items():
        n = len(data['y'])
        n_train = int(n * train_ratio)
        perm = np.random.permutation(n)
        
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        train_splits[client_id] = {
            'x': data['x'][train_idx],
            'y': data['y'][train_idx].copy()
        }
        test_splits[client_id] = {
            'x': data['x'][test_idx],
            'y': data['y'][test_idx].copy()
        }
        
    return train_splits, test_splits


def inject_label_noise(train_splits, noisy_client_id=4, noise_rate=0.40, num_classes=9, seed=42):
    """
    Inject symmetric label noise into the designated client's training split.
    
    For 40% of the training samples in client `noisy_client_id`, randomly reassign the label
    uniformly to any of the OTHER 8 classes.
    """
    np.random.seed(seed)
    noisy_train = copy.deepcopy(train_splits)
    
    target_data = noisy_train[noisy_client_id]
    y_target = target_data['y']
    n_samples = len(y_target)
    n_noisy = int(n_samples * noise_rate)
    
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)
    
    corrupted_count = 0
    for idx in noisy_indices:
        orig_label = y_target[idx]
        other_labels = [c for c in range(num_classes) if c != orig_label]
        new_label = np.random.choice(other_labels)
        y_target[idx] = new_label
        corrupted_count += 1
        
    print(f"[*] Injected {noise_rate*100:.1f}% label noise into Client {noisy_client_id}: "
          f"{corrupted_count}/{n_samples} training labels flipped across {num_classes} classes.")
    
    return noisy_train


def save_dataset_splits(out_dirs, train_splits, test_splits, num_classes=9, alpha=0.3, noisy_client=4, noise_rate=0.4):
    """
    Save train and test splits to .npz files following PFLlib directory structure:
      out_dir/
        config.json
        train/0.npz ... 4.npz
        test/0.npz ... 4.npz
    """
    for out_dir in out_dirs:
        os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)
        
        num_clients = len(train_splits)
        client_stats = {}
        
        for c in range(num_clients):
            # Save train npz
            train_file = os.path.join(out_dir, "train", f"{c}.npz")
            with open(train_file, 'wb') as f:
                np.savez_compressed(f, data=train_splits[c])
                
            # Save test npz
            test_file = os.path.join(out_dir, "test", f"{c}.npz")
            with open(test_file, 'wb') as f:
                np.savez_compressed(f, data=test_splits[c])
                
            # Class distribution for metadata
            y_train = train_splits[c]['y']
            unique, counts = np.unique(y_train, return_counts=True)
            dist = {int(k): int(v) for k, v in zip(unique, counts)}
            client_stats[c] = {
                'train_samples': len(train_splits[c]['y']),
                'test_samples': len(test_splits[c]['y']),
                'train_class_dist': dist,
                'has_label_noise': (c == noisy_client),
                'label_noise_rate': noise_rate if c == noisy_client else 0.0
            }
            
        config = {
            'dataset': 'PathMNIST',
            'num_clients': num_clients,
            'num_classes': num_classes,
            'partition': 'dirichlet',
            'alpha': alpha,
            'noisy_client': noisy_client,
            'noise_rate': noise_rate,
            'client_stats': client_stats
        }
        
        with open(os.path.join(out_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"[*] Successfully saved dataset to {out_dir}")


def print_client_summary(train_splits, test_splits, num_classes=9, noisy_client=4):
    print("\n" + "="*80)
    print(f"{'Client':<8}{'Train Samples':<16}{'Test Samples':<16}{'Label Noise':<16}{'Top Classes & Counts'}")
    print("="*80)
    for c in range(len(train_splits)):
        n_train = len(train_splits[c]['y'])
        n_test = len(test_splits[c]['y'])
        noise_str = "40% (CORRUPTED)" if c == noisy_client else "0.0% (Clean)"
        
        y_train = train_splits[c]['y']
        unique, counts = np.unique(y_train, return_counts=True)
        top_k = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:4]
        top_str = ", ".join([f"C{k}:{v}" for k, v in top_k])
        
        print(f"Client {c:<2} {n_train:<16}{n_test:<16}{noise_str:<16}{top_str}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate PathMNIST Dirichlet non-IID split with label noise.")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of clients (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration parameter (default: 0.3)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test ratio (default: 0.8)")
    parser.add_argument("--noisy_client", type=int, default=4, help="Client ID to inject label noise into (default: 4)")
    parser.add_argument("--noise_rate", type=float, default=0.40, help="Fraction of training labels corrupted (default: 0.40)")
    parser.add_argument("--max_samples", type=int, default=15000, help="Subsample total dataset pool for fast training (default: 15000, pass 0 for all 107k)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_root", type=str, default="~/.medmnist", help="Directory where MedMNIST downloads raw data")
    
    args = parser.parse_args()
    max_samples = None if args.max_samples <= 0 else args.max_samples
    
    set_seed(args.seed)
    imgs, labels, num_classes = download_or_load_pathmnist(root=args.data_root, max_samples=max_samples, seed=args.seed)
    
    client_data = dirichlet_partition(
        imgs, labels,
        num_clients=args.num_clients,
        alpha=args.alpha,
        min_samples_per_client=50,
        seed=args.seed
    )
    
    train_splits, test_splits = train_test_split_clients(client_data, train_ratio=args.train_ratio, seed=args.seed)
    
    noisy_train_splits = inject_label_noise(
        train_splits,
        noisy_client_id=args.noisy_client,
        noise_rate=args.noise_rate,
        num_classes=num_classes,
        seed=args.seed
    )
    
    print_client_summary(noisy_train_splits, test_splits, num_classes=num_classes, noisy_client=args.noisy_client)
    
    # Save to both local dataset folder and PFLlib dataset folder
    out_dirs = [
        os.path.join(".", "data", "PathMNIST"),
        os.path.join(".", "PFLlib", "dataset", "PathMNIST")
    ]
    save_dataset_splits(out_dirs, noisy_train_splits, test_splits, num_classes=num_classes,
                        alpha=args.alpha, noisy_client=args.noisy_client, noise_rate=args.noise_rate)
    print("[+] PathMNIST data partition and noise generation complete!")


if __name__ == "__main__":
    main()
