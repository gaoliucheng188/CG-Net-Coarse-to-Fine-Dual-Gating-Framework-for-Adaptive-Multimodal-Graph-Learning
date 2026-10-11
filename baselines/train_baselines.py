"""
Training script for baseline models (MAGNN, GATE).

This script trains the baseline models and compares with CG-Net on:
- MUTAG
- NLM-Chem
- PQARef
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

from models_multimodal import MultimodalGcnEncoderGraph
from baselines.magnn import MAGNNEncoder
from baselines.gate import GATEModel


def load_dataset(dataset_name):
    """Load dataset using MultimodalDataset."""
    from data.multimodal_loader import MultimodalDataset

    try:
        dataset = MultimodalDataset(dataset_name)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def get_graph_batch(dataset, indices, device):
    """Prepare a batch of graphs for training."""
    batch_adj = []
    batch_x = []
    batch_text = []
    batch_labels = []
    batch_num_nodes = []

    max_nodes = 0

    for idx in indices:
        graph_data = dataset.get_graph_data(idx)
        adj = graph_data['adj']
        feat = np.array(graph_data['feat'])
        label = graph_data['label']
        text = graph_data.get('text', '')

        num_nodes = adj.shape[0]
        max_nodes = max(max_nodes, num_nodes)

        batch_adj.append(adj)
        batch_x.append(feat)
        batch_text.append(text if text else '')
        batch_labels.append(label)
        batch_num_nodes.append(num_nodes)

    # Pad graphs to same size
    padded_adj = np.zeros((len(indices), max_nodes, max_nodes))
    padded_x = np.zeros((len(indices), max_nodes, batch_x[0].shape[1]))

    for i in range(len(indices)):
        num_nodes = batch_num_nodes[i]
        padded_adj[i, :num_nodes, :num_nodes] = batch_adj[i]
        padded_x[i, :num_nodes, :] = batch_x[i]

    # Convert to tensors
    batch_adj = torch.FloatTensor(padded_adj).to(device)
    batch_x = torch.FloatTensor(padded_x).to(device)
    batch_labels = torch.LongTensor(batch_labels).to(device)

    return batch_x, batch_adj, batch_text, batch_labels, batch_num_nodes


def train_epoch(model, dataset, train_indices, optimizer, criterion, device, model_type='concat', batch_size=8):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Shuffle training indices
    np.random.shuffle(train_indices)

    for i in range(0, len(train_indices), batch_size):
        batch_indices = train_indices[i:i+batch_size]

        batch_x, batch_adj, batch_text, batch_labels, batch_num_nodes = get_graph_batch(
            dataset, batch_indices, device
        )

        optimizer.zero_grad()

        if model_type in ['magnn', 'gate', 'concat']:
            outputs = model(batch_x, batch_adj, text=batch_text)
        else:
            outputs = model(batch_x, batch_adj, text=batch_text)

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / max(len(train_indices) // batch_size, 1), accuracy


def evaluate(model, dataset, test_indices, device, model_type='concat', batch_size=8):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(test_indices), batch_size):
            batch_indices = test_indices[i:i+batch_size]

            batch_x, batch_adj, batch_text, batch_labels, batch_num_nodes = get_graph_batch(
                dataset, batch_indices, device
            )

            if model_type in ['magnn', 'gate', 'concat']:
                outputs = model(batch_x, batch_adj, text=batch_text)
            else:
                outputs = model(batch_x, batch_adj, text=batch_text)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_baseline_experiment(dataset_name, model_type, epochs=100, lr=0.001, hidden_dim=128, embedding_dim=128, batch_size=8):
    """
    Run a baseline experiment.

    Args:
        dataset_name: 'MUTAG', 'nlm_chem', or 'pqaref'
        model_type: 'concat', 'magnn', or 'gate'
        epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: Hidden dimension size
        embedding_dim: Embedding dimension size
        batch_size: Batch size
    """
    print(f"\n{'='*60}")
    print(f"Running {model_type.upper()} on {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = load_dataset(dataset_name)
    if dataset is None:
        return None

    # Get number of graphs (assuming single graph or multiple)
    num_graphs = len(dataset.data[1]) if isinstance(dataset.data[1], (list, np.ndarray)) else 1

    # For simplicity, use graph-level data
    # If there are multiple graphs, use the first one as single sample
    # In practice, you would have multiple graphs

    # Get dimensions
    G = dataset.data[0]
    labels = dataset.data[1]

    # Get input dimension
    sample_node = list(G.nodes())[0]
    input_dim = G.nodes[sample_node]['feat'].shape[0]

    # For graph-level classification, we use all nodes
    # Number of classes
    if isinstance(labels, np.ndarray):
        num_classes = len(np.unique(labels))
    else:
        num_classes = len(set(labels.values()) if hasattr(labels, 'values') else labels)

    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")

    # Split indices
    all_indices = list(range(num_graphs))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=42
    )

    print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    if model_type == 'concat':
        model = MultimodalGcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
            fusion_method='concat',
            disable_text=False,
        )
    elif model_type == 'magnn':
        model = MAGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
        )
    elif model_type == 'gate':
        model = GATEModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            label_dim=num_classes,
            num_layers=3,
            text_dim=128,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, dataset, train_indices, optimizer, criterion, device, model_type, batch_size
        )
        test_acc = evaluate(
            model, dataset, test_indices, device, model_type, batch_size
        )

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f}s")
    print(f"Best test accuracy: {best_acc:.4f}")

    return {
        'model_type': model_type,
        'dataset': dataset_name,
        'best_accuracy': best_acc,
        'training_time': elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                       choices=['MUTAG', 'nlm_chem', 'pqaref', 'mutag', 'nlmchem', 'pqaref'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'concat', 'magnn', 'gate'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')

    args = parser.parse_args()

    # Normalize dataset name
    dataset_name = args.dataset.lower()
    if dataset_name in ['nlmchem', 'nlm_chem']:
        dataset_name = 'nlm_chem'

    results = []

    if args.model == 'all':
        models = ['concat', 'magnn', 'gate']
    else:
        models = [args.model]

    for model_type in models:
        result = run_baseline_experiment(
            dataset_name,
            model_type,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
        )
        if result:
            results.append(result)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['model_type'].upper():10s} | {r['dataset']:10s} | "
              f"Acc: {r['best_accuracy']:.4f} | Time: {r['training_time']:.2f}s")


if __name__ == '__main__':
    main()

