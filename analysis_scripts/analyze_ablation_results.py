import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# import pandas as pd

# Configuration
DATASETS = ['mutag', 'nlm_chem', 'pqaref']
ABLATIONS = ['B0', 'B1', 'B2', 'B3']
BASE_DIR = 'log'
OUTPUT_DIR = 'analysis_results'

def get_file_path(dataset, ablation):
    # Pattern: masked_adj_{dataset}_base_h128_o128_explain__{ablation}node_idx_0graph_idx_-1.npy
    # Note the double underscore based on 'find' output
    filename = f"masked_adj_{dataset}_base_h128_o128_explain__{ablation}node_idx_0graph_idx_-1.npy"
    return os.path.join(BASE_DIR, filename)

def compute_sparsity(mask, threshold=0.5):
    # Sparsity: fraction of edges with weight > threshold
    # Or simply mean weight if we consider soft mask
    count = np.sum(mask > threshold)
    total = mask.size
    return count / total, np.mean(mask)

def compute_jaccard(mask1, mask2, top_k=20):
    # Flatten and get indices of top-k edges
    idx1 = np.argsort(mask1.flatten())[-top_k:]
    idx2 = np.argsort(mask2.flatten())[-top_k:]
    
    set1 = set(idx1)
    set2 = set(idx2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def plot_distributions(masks_dict, dataset):
    plt.figure(figsize=(10, 6))
    for name, mask in masks_dict.items():
        # Flatten and filter out very small values for better visualization if needed
        # or just plot all
        values = mask.flatten()
        # values = values[values > 0.01] # Optional filtering
        plt.hist(values, bins=50, alpha=0.5, label=name, density=True)
    
    plt.title(f'Edge Weight Distribution - {dataset}')
    plt.xlabel('Mask Weight')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset}_dist.png'))
    plt.close()

def plot_heatmaps(masks_dict, dataset):
    n = len(masks_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    
    for ax, (name, mask) in zip(axes, masks_dict.items()):
        im = ax.imshow(mask, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(name)
        fig.colorbar(im, ax=ax)
    
    plt.suptitle(f'Masked Adjacency Matrices - {dataset}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset}_heatmaps.png'))
    plt.close()

def main():
    results = []
    
    for dataset in DATASETS:
        print(f"Analyzing {dataset}...")
        masks = {}
        
        # Load masks
        for ablation in ABLATIONS:
            path = get_file_path(dataset, ablation)
            if os.path.exists(path):
                try:
                    mask = np.load(path)
                    masks[ablation] = mask
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"File not found: {path}")
        
        if not masks:
            continue
            
        # Compute stats
        for ablation, mask in masks.items():
            sparsity_05, mean_density = compute_sparsity(mask, threshold=0.5)
            sparsity_01, _ = compute_sparsity(mask, threshold=0.1)
            
            row = {
                'Dataset': dataset,
                'Ablation': ablation,
                'Shape': str(mask.shape),
                'Total Edges': mask.size,
                'Edges > 0.5': np.sum(mask > 0.5),
                'Mean Density': mean_density,
                'Sparsity (>0.5)': sparsity_05,
                'Sparsity (>0.1)': sparsity_01
            }
            results.append(row)
            
        # Compute Jaccard (Comparison with B0)
        if 'B0' in masks:
            base_mask = masks['B0']
            for ablation, mask in masks.items():
                if ablation == 'B0': continue
                # Check if masks are identical
                if np.allclose(base_mask, mask):
                    jaccard = "Identical"
                else:
                    jaccard = compute_jaccard(base_mask, mask, top_k=20) 
                
                for r in results:
                    if r['Dataset'] == dataset and r['Ablation'] == ablation:
                        r['Jaccard (vs B0, top-20)'] = jaccard
        
        # Plots
        plot_distributions(masks, dataset)
        plot_heatmaps(masks, dataset)

    # Print results manually
    print("\n=== Quantitative Analysis Results ===")
    headers = ['Dataset', 'Ablation', 'Edges > 0.5', 'Mean Density', 'Sparsity (>0.5)', 'Jaccard (vs B0)']
    # Print header
    print(f"{headers[0]:<10} {headers[1]:<10} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]:<25}")
    print("-" * 100)
    
    for row in results:
        d = row.get('Dataset', '')
        a = row.get('Ablation', '')
        e05 = str(row.get('Edges > 0.5', 0))
        md = f"{row.get('Mean Density', 0):.4f}"
        s05 = f"{row.get('Sparsity (>0.5)', 0):.4f}"
        jac = row.get('Jaccard (vs B0, top-20)', 'N/A')
        if isinstance(jac, float):
            jac = f"{jac:.4f}"
            
        print(f"{d:<10} {a:<10} {e05:<15} {md:<15} {s05:<15} {jac:<25}")

    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
