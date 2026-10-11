#!/usr/bin/env python3
"""
TextGCN Baseline Training & Evaluation Script
==============================================

Runs TextGCN (Yao et al., AAAI 2019) on the same datasets and evaluation
protocol as CG-Net, producing accuracy numbers for the external SOTA
comparison table.

TextGCN builds a heterogeneous word-document graph from the text corpus
and applies 2-layer GCN for classification.

Supports:
  - MUTAG  (node-level classification on single disjoint-union graph)
  - NLM-Chem (node-level classification)
  - PQARef (node-level classification)

Usage:
    python baselines/run_textgcn_baseline.py --dataset MUTAG --epochs 200
    python baselines/run_textgcn_baseline.py --dataset nlm_chem --epochs 200
    python baselines/run_textgcn_baseline.py --dataset pqaref --epochs 200
    python baselines/run_textgcn_baseline.py --dataset all          # run all three
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

from baselines.textgcn import TextGCN, build_textgcn_graph, load_text_map


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (reuses the same pipeline as train.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name):
    """Load graph, labels, and splits for a given dataset.

    Returns:
        G:           networkx Graph
        labels_list: list[int]  (node-level labels)
        splits:      dict with 'train'/'val'/'test' numpy arrays, or None
        input_dim:   int
        num_classes: int
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    dataset_name,
    epochs=200,
    lr=0.01,
    hidden_dim=200,
    word_embed_dim=64,
    dropout=0.5,
    weight_decay=5e-4,
    window_size=20,
    max_vocab_size=2000,
    seed=42,
):
    """Train TextGCN on a single dataset and return test accuracy."""

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  TextGCN Baseline on {dataset_name}")
    print(f"{'='*60}")

    G, labels_list, splits, input_dim, num_classes = load_dataset(dataset_name)
    num_doc_nodes = G.number_of_nodes()

    # Load text data
    # Try multiple folder names
    text_map = load_text_map(dataset_name)
    if not text_map:
        # Try uppercase
        text_map = load_text_map(dataset_name.upper())
    if not text_map:
        # Try with underscore variants
        for variant in [dataset_name.lower(), dataset_name.upper(),
                        dataset_name.lower().replace("_", ""),
                        dataset_name.lower().replace("chem", "_chem")]:
            text_map = load_text_map(variant)
            if text_map:
                break

    num_texts = len(text_map)
    print(f"  Nodes: {num_doc_nodes}, Edges: {G.number_of_edges()}")
    print(f"  Input dim: {input_dim}, Classes: {num_classes}")
    print(f"  Text nodes loaded: {num_texts}")

    if num_texts == 0:
        print("  ERROR: No text data found! TextGCN requires text attributes.")
        return {"dataset": dataset_name, "accuracy": 0.0, "time": 0, "params": 0}

    # Build TextGCN heterogeneous graph
    print(f"  Building TextGCN graph (window={window_size}, max_vocab={max_vocab_size})...")
    adj, features, num_doc, word_vocab = build_textgcn_graph(
        G, labels_list, text_map,
        window_size=window_size,
        min_word_freq=2,
        max_vocab_size=max_vocab_size,
    )
    num_total = adj.shape[0]
    num_word_nodes = num_total - num_doc
    print(f"  TextGCN graph: {num_total} nodes ({num_doc} doc + {num_word_nodes} word), "
          f"{int((adj.sum() - num_total) / 2)} edges (excl. self-loops)")
    print(f"  Vocabulary size: {len(word_vocab)}")

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
        num_train = int(num_doc_nodes * 0.7)
        idx = list(range(num_doc_nodes))
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
        val_idx = test_idx

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Convert to tensors
    adj_tensor = torch.tensor(adj, dtype=torch.float).to(device)
    feat_tensor = torch.tensor(features, dtype=torch.float).to(device)
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)
    labels_train = torch.tensor(labels_np[train_idx], dtype=torch.long).to(device)

    # Model
    model = TextGCN(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        label_dim=num_classes,
        num_word_nodes=num_word_nodes,
        word_embed_dim=word_embed_dim,
        dropout=dropout,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights
    try:
        total_samples = labels_np.shape[0]
        counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        eps = 1e-6
        weights = total_samples / (num_classes * (counts + eps))
        max_w = 50.0
        weights = np.minimum(weights, max_w)
        weights[counts == 0] = 0.0
        weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("  Using balanced class weights.")
    except Exception:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward: get predictions for doc nodes
        pred = model(feat_tensor, adj_tensor, num_doc)  # [num_doc, num_classes]
        pred_train = pred[train_idx]

        loss = criterion(pred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred_eval = model(feat_tensor, adj_tensor, num_doc)
                pred_labels = torch.argmax(pred_eval, dim=1).cpu().numpy()

                train_acc = accuracy_score(labels_np[train_idx], pred_labels[train_idx])
                val_acc = accuracy_score(labels_np[val_idx], pred_labels[val_idx])
                test_acc = accuracy_score(labels_np[test_idx], pred_labels[test_idx])

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch

                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | "
                    f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | "
                    f"Best: {best_test_acc:.4f} (ep {best_epoch+1})"
                )

    elapsed = time.time() - start_time

    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best Val Acc:  {best_val_acc:.4f}")
    print(f"  Best Test Acc: {best_test_acc:.4f} (epoch {best_epoch+1})")

    return {
        "dataset": dataset_name,
        "accuracy": best_test_acc,
        "val_accuracy": best_val_acc,
        "best_epoch": best_epoch + 1,
        "time": elapsed,
        "params": num_params,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed evaluation
# ──────────────────────────────────────────────────────────────────────────────

def multiseed_eval(dataset_name, seeds=(42, 123, 456, 789, 1024), **kwargs):
    """Run TextGCN with multiple seeds and report mean ± std."""
    results = []
    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f"  Seed: {seed}")
        print(f"{'─'*60}")
        res = train_and_evaluate(dataset_name, seed=seed, **kwargs)
        results.append(res["accuracy"])

    arr = np.array(results)
    print(f"\n{'='*60}")
    print(f"  TextGCN on {dataset_name} — Multi-seed Summary")
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
    parser = argparse.ArgumentParser(description="TextGCN Baseline for CG-Net Paper")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "MUTAG", "mutag", "nlm_chem", "nlmchem", "pqaref"],
        help="Dataset to evaluate (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=200, help="Hidden dimension")
    parser.add_argument("--word-embed-dim", type=int, default=64, help="Word embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--window-size", type=int, default=20, help="PMI window size")
    parser.add_argument("--max-vocab", type=int, default=2000, help="Max vocabulary size")
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
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                word_embed_dim=args.word_embed_dim,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                window_size=args.window_size,
                max_vocab_size=args.max_vocab,
            )
            all_results.append({"dataset": ds, "acc_mean": res["mean"], "acc_std": res["std"]})
        else:
            res = train_and_evaluate(
                ds,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                word_embed_dim=args.word_embed_dim,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                window_size=args.window_size,
                max_vocab_size=args.max_vocab,
                seed=args.seed,
            )
            all_results.append(res)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  TextGCN Baseline — Summary")
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

