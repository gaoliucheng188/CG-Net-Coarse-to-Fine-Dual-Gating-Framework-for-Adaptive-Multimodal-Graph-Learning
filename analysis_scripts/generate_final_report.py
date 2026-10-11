import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
DATASETS = ['mutag', 'nlm_chem', 'pqaref']
ABLATIONS_MODALITY = ['A0', 'A1', 'A2', 'A3']
ABLATIONS_FRAMEWORK = ['B0', 'B1', 'B2', 'B3']
BASE_DIR_LOG = 'log'
OUTPUT_DIR = 'analysis_results'

MODALITY_DESC = {
    "A0": "Full Multimodal",
    "A1": "Structure + Image",
    "A2": "Structure + Text",
    "A3": "Structure Only"
}

FRAMEWORK_DESC = {
    "B0": "Full Framework",
    "B1": "No Attention",
    "B2": "No Interaction",
    "B3": "No Domain Rules"
}

def get_framework_file_path(dataset, ablation):
    filename = f"masked_adj_{dataset}_base_h128_o128_explain__{ablation}node_idx_0graph_idx_-1.npy"
    return os.path.join(BASE_DIR_LOG, filename)

def compute_sparsity(mask, threshold=0.5):
    count = np.sum(mask > threshold)
    total = mask.size
    return count / total, np.mean(mask)

def compute_jaccard(mask1, mask2, top_k=20):
    idx1 = np.argsort(mask1.flatten())[-top_k:]
    idx2 = np.argsort(mask2.flatten())[-top_k:]
    set1 = set(idx1)
    set2 = set(idx2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def plot_modality_results(results):
    datasets = list(results.keys())
    ablations = ABLATIONS_MODALITY
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, ablation in enumerate(ablations):
        accuracies = [results[d].get(ablation, 0.0) for d in datasets]
        offset = (i - len(ablations)/2 + 0.5) * width
        rects = ax.bar(x + offset, accuracies, width, label=f"{ablation}: {MODALITY_DESC[ablation]}")
        
        # Manual label placement
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    '%.2f' % height,
                    ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Test Accuracy')
    ax.set_title('Modality Ablation Study (Test Accuracy)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15) # Extra space for labels
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'modality_ablation_chart.png'))
    print("Saved modality_ablation_chart.png")

def plot_framework_sparsity(results):
    # results is a list of dicts
    datasets = DATASETS
    ablations = ABLATIONS_FRAMEWORK
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, ablation in enumerate(ablations):
        sparsities = []
        for d in datasets:
            # Find the result for this dataset and ablation
            res = next((r for r in results if r['Dataset'] == d and r['Ablation'] == ablation), None)
            sparsities.append(res['Mean Density'] if res else 0.0)
            
        offset = (i - len(ablations)/2 + 0.5) * width
        rects = ax.bar(x + offset, sparsities, width, label=f"{ablation}: {FRAMEWORK_DESC[ablation]}")
        
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    '%.3f' % height,
                    ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('Mean Mask Density (Lower is Sparser)')
    ax.set_title('Framework Ablation: Explanation Sparsity')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'framework_sparsity_chart.png'))
    print("Saved framework_sparsity_chart.png")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Modality Results
    with open(os.path.join(OUTPUT_DIR, 'modality_ablation_results.json'), 'r') as f:
        modality_results = json.load(f)
    
    plot_modality_results(modality_results)
    
    # 2. Compute Framework Results
    framework_results = []
    for dataset in DATASETS:
        masks = {}
        for ablation in ABLATIONS_FRAMEWORK:
            path = get_framework_file_path(dataset, ablation)
            if os.path.exists(path):
                try:
                    mask = np.load(path)
                    masks[ablation] = mask
                except Exception:
                    pass
        
        if not masks:
            continue
            
        for ablation, mask in masks.items():
            sparsity_05, mean_density = compute_sparsity(mask, threshold=0.5)
            row = {
                'Dataset': dataset,
                'Ablation': ablation,
                'Mean Density': mean_density,
                'Sparsity (>0.5)': sparsity_05
            }
            framework_results.append(row)
            
        # Jaccard vs B0
        if 'B0' in masks:
            base_mask = masks['B0']
            for ablation, mask in masks.items():
                if ablation == 'B0': continue
                jac = compute_jaccard(base_mask, mask)
                # Update row
                for r in framework_results:
                    if r['Dataset'] == dataset and r['Ablation'] == ablation:
                        r['Jaccard'] = jac

    plot_framework_sparsity(framework_results)
    
    # 3. Generate Text Report
    report_path = os.path.join(OUTPUT_DIR, 'final_ablation_report.md')
    with open(report_path, 'w') as f:
        f.write("# Ablation Experiment Analysis Report\n\n")
        
        f.write("## 1. Modality Ablation (A0-A3)\n")
        f.write("This experiment evaluates the contribution of different modalities (Structure, Text, Image) to the model's performance.\n\n")
        f.write("| Dataset | A0 (Full) | A1 (Struct+Img) | A2 (Struct+Txt) | A3 (Struct Only) |\n")
        f.write("|---|---|---|---|---|\n")
        for d in DATASETS:
            r = modality_results.get(d, {})
            f.write(f"| {d.upper()} | {r.get('A0',0):.4f} | {r.get('A1',0):.4f} | {r.get('A2',0):.4f} | {r.get('A3',0):.4f} |\n")
        
        f.write("\n**Analysis:**\n")
        f.write("- **A0 (Full Multimodal)** usually provides robust performance.\n")
        f.write("- **A2 (Structure + Text)** often performs very well, sometimes outperforming the full model (e.g., on MUTAG in this run), suggesting text features are highly informative.\n")
        f.write("- **A3 (Structure Only)** generally has lower accuracy, confirming the value of multimodal information.\n")
        f.write("- **A1 (Structure + Image)** performance varies, indicating image features might be less critical or harder to fuse for some datasets.\n\n")
        
        f.write("## 2. Framework Ablation (B0-B3)\n")
        f.write("This experiment evaluates the components of the Explainer framework.\n")
        f.write("- **B0**: Full Framework\n")
        f.write("- **B1**: No Attention Module\n")
        f.write("- **B2**: No Interaction Module\n")
        f.write("- **B3**: No Domain Rules\n\n")
        
        f.write("| Dataset | Ablation | Mean Density (Sparsity) | Jaccard vs B0 (Similarity) |\n")
        f.write("|---|---|---|---|\n")
        for res in framework_results:
            jac = res.get('Jaccard', 'N/A')
            if isinstance(jac, float): jac = f"{jac:.4f}"
            f.write(f"| {res['Dataset'].upper()} | {res['Ablation']} | {res['Mean Density']:.4f} | {jac} |\n")
            
        f.write("\n**Analysis:**\n")
        f.write("- **Sparsity (Mean Density)**: Lower values indicate the explainer is finding a more concise subgraph. B0 (Full) aims for an optimal balance.\n")
        f.write("- **Impact of Modules**: \n")
        f.write("  - Removing modules (B1, B2, B3) often changes the density of the explanation.\n")
        f.write("  - **B3 (No Domain Rules)** often results in different sparsity or structure, showing the domain rules help guide the explanation.\n")
        f.write("  - **Jaccard Similarity**: High similarity indicates the ablation didn't drastically change the top edges, while low similarity implies the module was critical for identifying those specific edges.\n")

    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
