import os
import numpy as np
import networkx as nx
import glob
import re
import sys
import pickle

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.io_utils as io_utils

# MUTAG Atom Mapping (Standard TUDataset)
ATOM_MAP = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}

def get_atom_label(feat):
    # feat is a list or array (one-hot)
    if isinstance(feat, (list, tuple, np.ndarray)):
        idx = np.argmax(feat)
        return ATOM_MAP.get(idx, 'Unknown')
    return 'Unknown'

def main():
    # 1. Load Dataset (Pickle)
    pkl_path = 'data/tud_mutag.pkl'
    print(f"Loading dataset from {pkl_path}...")
    
    if not os.path.exists(pkl_path):
        print(f"File {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # data is (G, labels, name) or similar tuple
    if isinstance(data, tuple) and len(data) >= 1:
        G = data[0]
    else:
        print("Unexpected data format.")
        return

    print(f"Loaded graph with {G.number_of_nodes()} nodes.")

    # 2. Find masked_adj files
    log_dir = 'log'
    # We look for files explaining node 0 (as per logs)
    pattern = os.path.join(log_dir, 'masked_adj_mutag_*_explain__B*node_idx_0graph_idx_-1.npy')
    files = glob.glob(pattern)
    
    if not files:
        print("No masked_adj files found.")
        return

    print(f"Found {len(files)} masked adjacency files.")
    
    # Group by ablation
    ablations = ['B0', 'B1', 'B2', 'B3']
    results = {b: [] for b in ablations}

    for fpath in files:
        # Parse filename
        match = re.search(r'explain__(B\d)', fpath)
        if not match:
            match = re.search(r'explain(B\d)', fpath)
        
        if match:
            ablation = match.group(1)
            results[ablation].append(fpath)

    # 3. Analyze Top Edges for Node 0
    # The masked_adj is usually the size of the computation graph (neighborhood)
    # OR the full graph?
    # explainer_main.py saves:
    # masked_adj = (explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze())
    # It seems to be the size of the sub-adjacency (neighborhood).
    # We need to know the mapping from sub-adjacency indices to original graph indices.
    # But we don't have that mapping saved in the npy file.
    # However, for Node 0, the neighborhood is likely small and centered at 0.
    # If we assume the explainer extracts the neighborhood of Node 0, 
    # and the mapping preserves order or we can re-extract it.
    
    # To be precise, we should re-extract the neighborhood of Node 0 from G
    # to match the nodes.
    # But we don't have the exact extraction logic here easily without importing explainer.
    # Let's assume the masked_adj corresponds to the neighborhood of Node 0.
    # And usually the center node is index 0 or the mapping is available.
    # Wait, `explainer.extract_neighborhood` returns `new_idx, adj, feat, label, neighbors`.
    # The `masked_adj` is on the `adj` (subgraph).
    # If we don't know the `neighbors` list, we can't map back to G.
    
    # BUT, if we just want to know "Atom X - Atom Y", we can look at the features 
    # of the nodes in the subgraph.
    # The `masked_adj` doesn't contain features.
    # We need to re-run the neighborhood extraction or assume a standard BFS order.
    
    # Let's try to infer from G.
    # We can extract the k-hop neighborhood of Node 0 from G.
    # The explainer usually uses 3 layers (num_gc_layers=3).
    # So it's a 3-hop neighborhood.
    
    # Let's extract 3-hop neighborhood of Node 0.
    center_node = 0
    khop = 3
    subset = {center_node}
    for _ in range(khop):
        next_subset = set(subset)
        for n in subset:
            next_subset.update(G.neighbors(n))
        subset = next_subset
    
    subgraph_nodes = sorted(list(subset))
    # The explainer might re-index them.
    # Usually it maps them to 0..len(subset)-1.
    # And the center node might be re-indexed.
    
    # If we can't map exactly, we can't be 100% sure.
    # However, we can print the distribution of edge types in the neighborhood
    # and see if the mask highlights specific types.
    
    # For now, let's just print the top edges and their weights, 
    # and if possible, guess the atoms if the adjacency matches the subgraph structure.
    
    # Actually, `explainer_main.py` saves `masked_adj`.
    # It does NOT save the node mapping.
    # This is a limitation for post-hoc analysis.
    # But we can try to match the adjacency matrix shape.
    
    for ablation in ablations:
        print(f"\n--- Ablation {ablation} Semantic Analysis (Node 0) ---")
        for fpath in results[ablation]:
            adj = np.load(fpath)
            # Check shape
            # print(f"Adj shape: {adj.shape}")
            
            # Get top edges
            edges = []
            rows, cols = adj.shape
            for i in range(rows):
                for j in range(i+1, cols):
                    w = adj[i, j]
                    if w > 0.01: # Threshold
                        edges.append((i, j, w))
            
            edges.sort(key=lambda x: x[2], reverse=True)
            top_edges = edges[:5]
            
            print(f"Top 5 Edges (Subgraph Indices):")
            for u, v, w in top_edges:
                print(f"  Edge ({u} - {v}): Weight {w:.4f}")
            
            # Attempt to map to atoms if shape matches our extracted subgraph
            if len(subgraph_nodes) == rows:
                # Assume sorted order matches
                print("  (Mapping to Atoms based on sorted subgraph nodes)")
                for u, v, w in top_edges:
                    real_u = subgraph_nodes[u]
                    real_v = subgraph_nodes[v]
                    atom_u = get_atom_label(G.nodes[real_u]['feat'])
                    atom_v = get_atom_label(G.nodes[real_v]['feat'])
                    print(f"  -> {atom_u} (Node {real_u}) - {atom_v} (Node {real_v}) : {w:.4f}")
            else:
                print(f"  (Subgraph size mismatch: Mask {rows} vs Extracted {len(subgraph_nodes)})")

if __name__ == '__main__':
    main()
