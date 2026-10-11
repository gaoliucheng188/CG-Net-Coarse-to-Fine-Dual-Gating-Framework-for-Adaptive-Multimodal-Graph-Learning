import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import glob
import re

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MUTAG Atom Mapping
ATOM_MAP = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

def get_atom_label(feat):
    if isinstance(feat, (list, tuple, np.ndarray)):
        idx = np.argmax(feat)
        return ATOM_MAP.get(idx, '?')
    return '?'

def extract_neighborhood(G, node_idx, k=3):
    subset = {node_idx}
    for _ in range(k):
        next_subset = set(subset)
        for n in subset:
            next_subset.update(G.neighbors(n))
        subset = next_subset
    subgraph = G.subgraph(list(subset)).copy()
    return subgraph

def main():
    # 1. Load Data
    pkl_path = 'data/tud_mutag.pkl'
    if not os.path.exists(pkl_path):
        print("Data not found.")
        return
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    G = data[0]
    
    # 2. Extract Subgraph for Node 0
    # Explainer typically uses 3 layers -> 3 hops
    subgraph = extract_neighborhood(G, 0, k=3)
    # Relabel nodes to 0..N-1 for matching with mask
    # But we need to keep track of original labels for atoms
    mapping = {n: i for i, n in enumerate(sorted(subgraph.nodes()))}
    subgraph_relabeled = nx.relabel_nodes(subgraph, mapping)
    
    # 3. Loop over Ablations
    ablations = ['B0', 'B1', 'B2', 'B3']
    log_dir = 'log'
    
    # Pre-calculate layout once so all plots look the same structure-wise
    pos = nx.kamada_kawai_layout(subgraph_relabeled)
    
    # Node Labels & Colors (Static across ablations)
    labels = {}
    node_colors = []
    for u in subgraph_relabeled.nodes():
        # Get original node index
        orig_u = [n for n, i in mapping.items() if i == u][0]
        atom = get_atom_label(G.nodes[orig_u]['feat'])
        labels[u] = f"{atom}\n({orig_u})"
        if atom == 'C': node_colors.append('lightgray')
        elif atom == 'N': node_colors.append('lightblue')
        elif atom == 'O': node_colors.append('red')
        else: node_colors.append('yellow')

    for ablation in ablations:
        print(f"Processing {ablation}...")
        # Find mask file
        # Note: regex to match explain__B0 or explainB0
        pattern = os.path.join(log_dir, f'masked_adj_mutag_*_explain__*{ablation}*node_idx_0graph_idx_-1.npy')
        files = glob.glob(pattern)
        
        if not files:
            print(f"No mask found for {ablation}.")
            continue
            
        # Pick the first match (usually only one per ablation/node)
        mask_path = files[0]
        mask = np.load(mask_path)
        
        # Check size
        if mask.shape[0] != subgraph_relabeled.number_of_nodes():
            print(f"Size mismatch for {ablation}: Mask {mask.shape[0]} vs Subgraph {subgraph_relabeled.number_of_nodes()}")
            continue

        # 4. Plot
        plt.figure(figsize=(8, 6))
        
        # Edge Colors from Mask
        edge_colors = []
        edge_widths = []
        for u, v in subgraph_relabeled.edges():
            if u < mask.shape[0] and v < mask.shape[0]:
                w = mask[u, v]
                # Make high weight edges red/thick
                edge_colors.append(w)
                edge_widths.append(1 + w * 5)
            else:
                edge_colors.append(0)
                edge_widths.append(1)

        nx.draw(subgraph_relabeled, pos, labels=labels, with_labels=True, 
                node_color=node_colors, node_size=500, font_size=10,
                edge_color=edge_colors, edge_cmap=plt.cm.Reds, width=edge_widths,
                edge_vmin=0, edge_vmax=1)
                
        plt.title(f"Semantic Visualization (Node 0, {ablation})")
        out_path = f'analysis_results/semantic_viz_{ablation}.png'
        plt.savefig(out_path)
        plt.close()
        print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
