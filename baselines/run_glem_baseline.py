#!/usr/bin/env python3
"""
GLEM Baseline Training & Evaluation Script
===========================================

Runs GLEM (Zhao et al., NeurIPS 2023) on the same datasets and evaluation
protocol as CG-Net, producing accuracy numbers for the external SOTA
comparison table.

GLEM uses an EM-iterative framework to jointly train a GNN and a Language
Model on text-attributed graphs, with pseudo-label regularization.

Usage:
    python baselines/run_glem_baseline.py --dataset MUTAG --em-iters 3 --inner-epochs 50
    python baselines/run_glem_baseline.py --dataset all --em-iters 3 --inner-epochs 50
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.glem import GLEMFramework, GLEMGNN, GLEMLM
import utils.gengraph as gengraph
from multimodal.text_encoder import TextEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name):
    """Load graph, labels, and splits for a given dataset."""
    name_lower = dataset_name.lower()
    if name_lower == "mutag":
        folder = "MUTAG"
        prefix = "mutag"
    elif name_lower in ("nlm_chem", "nlmchem"):
        folder = "nlm_chem"
        prefix = "nlm_chem"
    elif name_lower in ("pqaref",):
        folder = "pqaref"
        prefix = "pqaref"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    processed_dir = Path("data") / folder / "processed"
    graph_path = processed_dir / f"{prefix}_graph.gpickle"
    labels_path = processed_dir / f"{prefix}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"
    fallback_pkl = Path("data") / f"{folder}.pkl"

    if graph_path.exists() and labels_path.exists():
        G = nx.read_gpickle(graph_path)
        labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    elif fallback_pkl.exists():
        with open(fallback_pkl, "rb") as fh:
            G, labels_list, _ = pickle.load(fh)
            labels_list = list(map(int, labels_list))
        print(f"Warning: falling back to pickle load for {folder}.")
    else:
        raise FileNotFoundError(
            f"Processed {folder} artefacts not found. "
            f"Run scripts/prepare_{prefix}.py first."
        )

    splits = None
    if splits_path.exists():
        npz = np.load(splits_path)
        splits = {key: np.asarray(npz[key]).astype(np.int64) for key in npz.files}

    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)
    num_classes = int(labels_np.max()) + 1 if labels_np.size > 0 else 0

    sample_node = next(iter(G.nodes()))
    feat_value = G.nodes[sample_node]["feat"]
    input_dim = len(feat_value) if isinstance(feat_value, (np.ndarray, list, tuple)) else int(feat_value)

    return G, labels_list, splits, input_dim, num_classes


def load_texts(dataset_name, G):
    """Load all node texts for a dataset."""
    name_lower = dataset_name.lower()
    if name_lower == "mutag":
        folder = "MUTAG"
    elif name_lower in ("nlm_chem", "nlmchem"):
        folder = "nlm_chem"
    elif name_lower in ("pqaref",):
        folder = "pqaref"
    else:
        folder = dataset_name

    text_dir = Path("data") / "texts" / folder
    if not text_dir.exists():
        # try lowercase, uppercase, etc.
        for variant in [dataset_name, dataset_name.upper(), dataset_name.lower()]:
            candidate = Path("data") / "texts" / variant
            if candidate.exists():
                text_dir = candidate
                break

    texts = []
    for node_id in G.nodes():
        text_path = text_dir / f"node_{node_id}.txt"
        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())
        else:
            texts.append("No description available.")

    return texts


# ──────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    dataset_name,
    em_iters=3,
    inner_epochs=50,
    hidden_dim=64,
    embedding_dim=64,
    text_dim=128,
    lr=0.01,
    weight_decay=5e-4,
    seed=42,
):
    """Train GLEM on a single dataset and return test accuracy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  GLEM Baseline on {dataset_name}")
    print(f"{'='*60}")

    G, labels_list, splits, input_dim, num_classes = load_dataset(dataset_name)
    num_nodes = G.number_of_nodes()

    print(f"  Nodes: {num_nodes}, Edges: {G.number_of_edges()}")
    print(f"  Input dim: {input_dim}, Classes: {num_classes}")

    # Prepare dense tensors
    data = gengraph.preprocess_input_graph(G, labels_list)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], dtype=torch.float)
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)

    # Splits
    if splits is not None:
        train_idx = splits["train"].tolist()
        val_idx = splits.get("val", splits.get("validation", np.array([], dtype=int))).tolist()
        test_idx = splits["test"].tolist()
        if not test_idx and val_idx:
            test_idx = val_idx
        if not val_idx:
            val_idx = test_idx
    else:
        num_train = int(num_nodes * 0.7)
        idx = list(range(num_nodes))
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
        val_idx = test_idx

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Load texts
    print(f"  Loading texts...")
    texts = load_texts(dataset_name, G)
    print(f"  Loaded {len(texts)} texts")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Text encoder (shared between GLEM's LM component)
    print(f"  Initializing text encoder...")
    text_encoder = TextEncoder(output_dim=text_dim, force_fallback=True)

    # Build GNN: input dim = original feature dim + num_classes (for LM pseudo-labels)
    gnn_input_dim = input_dim + num_classes
    gnn = GLEMGNN(
        input_dim=gnn_input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=3,
        dropout=0.5,
    )

    # Build LM: takes text embeddings and predicts classes
    lm = GLEMLM(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        text_encoder=text_encoder,
        dropout=0.3,
    )

    gnn_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    lm_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    print(f"  GNN params: {gnn_params:,}, LM params: {lm_params:,}, Total: {gnn_params + lm_params:,}")

    # Build GLEM framework
    framework = GLEMFramework(
        gnn=gnn,
        lm=lm,
        num_em_iters=em_iters,
        inner_epochs=inner_epochs,
        confidence_threshold=0.5,
        device=device,
    )

    # Move tensors to device
    x_dev = x.to(device)
    adj_dev = adj.to(device)

    # Run EM training
    start_time = time.time()
    print(f"  Starting EM training ({em_iters} iters x {inner_epochs} epochs each)...")
    results = framework.train_em(
        x=x_dev,
        adj=adj_dev,
        labels=labels_np,
        texts=texts,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    elapsed = time.time() - start_time

    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best Val Acc:  {results['best_val_acc']:.4f}")
    print(f"  Best Test Acc: {results['best_test_acc']:.4f}")

    return {
        "dataset": dataset_name,
        "accuracy": results["best_test_acc"],
        "val_accuracy": results["best_val_acc"],
        "time": elapsed,
        "params": gnn_params + lm_params,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed evaluation
# ──────────────────────────────────────────────────────────────────────────────

def multiseed_eval(dataset_name, seeds=(42, 123, 456, 789, 1024), **kwargs):
    """Run GLEM with multiple seeds and report mean ± std."""
    results = []
    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f"  Seed: {seed}")
        print(f"{'─'*60}")
        res = train_and_evaluate(dataset_name, seed=seed, **kwargs)
        results.append(res["accuracy"])

    arr = np.array(results)
    print(f"\n{'='*60}")
    print(f"  GLEM on {dataset_name} — Multi-seed Summary")
    print(f"{'='*60}")
    print(f"  Seeds:   {list(seeds)}")
    print(f"  Acc:     {arr.mean():.4f} ± {arr.std():.4f}")
    print(f"  Range:   [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"{'='*60}")

    return {"mean": arr.mean(), "std": arr.std(), "all": arr.tolist()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLEM Baseline for CG-Net Paper")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "MUTAG", "mutag", "nlm_chem", "nlmchem", "pqaref"],
        help="Dataset to evaluate (default: all)",
    )
    parser.add_argument("--em-iters", type=int, default=3, help="Number of EM iterations")
    parser.add_argument("--inner-epochs", type=int, default=50, help="Inner training epochs per EM iter")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--text-dim", type=int, default=128, help="Text encoder output dim")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--multiseed", action="store_true",
        help="Run with 5 seeds for statistical significance",
    )
    args = parser.parse_args()

    datasets = ["MUTAG", "nlm_chem", "pqaref"] if args.dataset == "all" else [args.dataset]

    all_results = []
    for ds in datasets:
        if args.multiseed:
            res = multiseed_eval(
                ds,
                seeds=(42, 123, 456, 789, 1024),
                em_iters=args.em_iters,
                inner_epochs=args.inner_epochs,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                text_dim=args.text_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            all_results.append({"dataset": ds, "acc_mean": res["mean"], "acc_std": res["std"]})
        else:
            res = train_and_evaluate(
                ds,
                em_iters=args.em_iters,
                inner_epochs=args.inner_epochs,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                text_dim=args.text_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
            all_results.append(res)

    # Summary
    print(f"\n{'='*70}")
    print(f"  GLEM Baseline — Summary")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12} {'Accuracy':>12} {'Params':>10} {'Time(s)':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    for r in all_results:
        if "acc_mean" in r:
            print(f"  {r['dataset']:<12} {r['acc_mean']:.4f}±{r['acc_std']:.4f} {'—':>10} {'—':>10}")
        else:
            print(f"  {r['dataset']:<12} {r['accuracy']:>12.4f} {r['params']:>10,} {r['time']:>10.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
