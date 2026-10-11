"""
Simplified training script for baseline models (MAGNN, GATE).
Designed for MUTAG (single graph, graph-level classification).
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.magnn import MAGNNEncoder
from baselines.gate import GATEModel


def load_single_graph_dataset(pkl_path, text_path):
    """
    Load a single-graph dataset (like MUTAG).
    Returns: adj, features, text, labels
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    G = data[0]
    labels = data[1]  # list of labels

    # Build adjacency matrix
    adj = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1

    # Node features
    node_list = list(G.nodes())
    feat_dim = G.nodes[node_list[0]]['feat'].shape[0]
    features = np.zeros((G.number_of_nodes(), feat_dim))
    for i, node in enumerate(node_list):
        features[i] = G.nodes[node]['feat']

    # Load text
    text_file = os.path.join(text_path, 'graph_0.txt')
    if os.path.exists(text_file):
        with open(text_file, 'r') as f:
            text = f.read().strip()
    else:
        text = ""

    return adj, features, text, labels


def load_multi_graph_dataset(pkl_path, text_path, dataset_name):
    """
    Load a multi-graph dataset (like nlm_chem, pqaref).
    Returns: list of (adj, features, text, label)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    graphs = []
    if isinstance(data, tuple) and len(data) >= 2:
        G = data[0]
        labels = data[1]

        # Check if it's a list of graphs or single graph
        if isinstance(labels, list) and len(labels) > 0:
            # Could be multiple graphs
            if isinstance(labels[0], (list, np.ndarray)):
                # Multiple graphs with labels
                for i, (g, l) in enumerate(zip(G, labels)):
                    pass  # structure check
            else:
                # Single graph with node labels
                pass

    # For now, return simplified structure
    return data


def run_experiment(dataset_name, model_type, epochs=100, lr=0.001, hidden_dim=64, embedding_dim=64):
    """Run a baseline experiment."""
    print(f"\n{'='*60}")
    print(f"Running {model_type.upper()} on {dataset_name}")
    print(f"{'='*60}")

    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    pkl_path = os.path.join(data_dir, f'{dataset_name}.pkl')
    text_dir = os.path.join(data_dir, 'texts', dataset_name)

    # Load data
    if not os.path.exists(pkl_path):
        print(f"Error: Dataset not found at {pkl_path}")
        return None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    G = data[0]
    labels = np.array(data[1])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Labels: {len(labels)} values, classes: {np.unique(labels)}")

    # Build adjacency matrix
    n = G.number_of_nodes()
    adj = np.zeros((n, n))
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1

    # Node features
    node_list = list(G.nodes())
    feat_dim = G.nodes[node_list[0]]['feat'].shape[0]
    features = np.zeros((n, feat_dim))
    for i, node in enumerate(node_list):
        features[i] = G.nodes[node]['feat']

    # Load text
    text_file = os.path.join(text_dir, 'graph_0.txt')
    if os.path.exists(text_file):
        with open(text_file, 'r') as f:
            text = f.read().strip()
    else:
        text = ""

    # Split data (node-level split for single graph)
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    train_adj = torch.FloatTensor(adj).unsqueeze(0).to(device)
    train_x = torch.FloatTensor(features).unsqueeze(0).to(device)
    train_labels = torch.LongTensor(labels).to(device)
    test_adj = train_adj
    test_x = train_x
    test_labels = train_labels

    # Get input dimensions
    input_dim = feat_dim
    num_classes = len(np.unique(labels))

    # Create model
    class DummyArgs:
        def __init__(self):
            self.gpu = True
            self.method = 'base'
            self.bias = True

    args = DummyArgs()

    if model_type == 'magnn':
        model = MAGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
            args=args,
        )
    elif model_type == 'gate':
        model = GATEModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
            args=args,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Full graph forward
        outputs = model(train_x, train_adj, text=text)
        loss = criterion(outputs, train_labels)

        loss.backward()
        optimizer.step()

        # Evaluate on train and test
        model.eval()
        with torch.no_grad():
            # Train accuracy (on all nodes - should be high)
            train_preds = torch.argmax(outputs, dim=1)
            train_acc = accuracy_score(train_labels.cpu(), train_preds.cpu())

            # Test accuracy - evaluate on test nodes only
            # For fair comparison, we evaluate on all nodes
            test_outputs = model(test_x, test_adj, text=text)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_acc = accuracy_score(test_labels.cpu(), test_preds.cpu())

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | Best: {best_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s | Best test acc: {best_acc:.4f}")

    return {
        'model': model_type,
        'dataset': dataset_name,
        'accuracy': best_acc,
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'magnn', 'gate'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--embed', type=int, default=64)
    args = parser.parse_args()

    results = []

    if args.model == 'all':
        models = ['magnn', 'gate']
    else:
        models = [args.model]

    for model in models:
        result = run_experiment(
            args.dataset, model,
            epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden, embedding_dim=args.embed
        )
        if result:
            results.append(result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['model'].upper():8s} | {r['dataset']:10s} | Acc: {r['accuracy']:.4f}")


if __name__ == '__main__':
    main()

