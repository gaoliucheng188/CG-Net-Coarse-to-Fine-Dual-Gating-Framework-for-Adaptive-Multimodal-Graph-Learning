#!/usr/bin/env python3
"""
Ablation Experiment Script for GNN Explainer + Multimodal

This script runs ablation experiments on nlm_chem, pqaref, and tud_mutag datasets.
Groups:
- A0: Full multimodal (structure + text + image) with best fusion method (attention)
- A1: Structure + image (no text)
- A2: Structure + text (no image)
- A3: Structure only (unimodal)

Parameters: --epochs 1000 --hidden-dim 128 --output-dim 128 --num-gc-layers 3 --dropout 0.2 --weight-decay 0.0005 --train-ratio 0.7 --gpu --cuda 0
"""

import subprocess
import os
import sys
import re
import json
import matplotlib.pyplot as plt
import numpy as np

# Base command
BASE_CMD = [
    "python", "train.py",
    "--epochs", "200",  # Reduced for demonstration/speed, increase if needed
    "--hidden-dim", "128",
    "--output-dim", "128",
    "--num-gc-layers", "3",
    "--dropout", "0.2",
    "--weight-decay", "0.0005",
    "--train-ratio", "0.7",
    "--gpu",
    "--cuda", "0"
]

# Datasets
DATASETS = ["mutag", "nlm_chem", "pqaref"]

# Ablation configurations
ABLATIONS = {
    "A0": {  # Full multimodal with attention fusion
        "multimodal": True,
        "fusion_method": "attention",
        "use_text": True,
        "use_image": True,
        "desc": "Full Multimodal (Text+Image)"
    },
    "A1": {  # Structure + image (no text)
        "multimodal": True,
        "fusion_method": "attention",
        "use_text": False,
        "use_image": True,
        "desc": "Structure + Image"
    },
    "A2": {  # Structure + text (no image)
        "multimodal": True,
        "fusion_method": "attention",
        "use_text": True,
        "use_image": False,
        "desc": "Structure + Text"
    },
    "A3": {  # Structure only
        "multimodal": False,
        "fusion_method": None,
        "use_text": False,
        "use_image": False,
        "desc": "Structure Only"
    }
}

def parse_accuracy(output):
    # Look for the last occurrence of test_acc
    # Pattern: test_acc: 0.7570
    matches = re.findall(r"test_acc:\s+(\d+\.\d+)", output)
    if matches:
        return float(matches[-1])
    return 0.0

def run_experiment(dataset, ablation_name, config):
    """Run a single experiment"""
    cmd = BASE_CMD.copy()
    cmd.extend(["--dataset", dataset])
    
    # Add suffix to avoid overwriting checkpoints and logs
    cmd.extend(["--name-suffix", f"_{ablation_name}"])

    if config["multimodal"]:
        cmd.append("--multimodal")
        cmd.extend(["--fusion-method", config["fusion_method"]])
        if not config["use_text"]:
            cmd.append("--disable-text")
        if not config["use_image"]:
            cmd.append("--disable-image")

    print(f"\n{'='*50}")
    print(f"Running {ablation_name} ({config['desc']}) on {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            print("STDERR:")
            print(result.stderr)
            return False, 0.0
        
        acc = parse_accuracy(result.stdout)
        print(f"Result: Test Accuracy = {acc}")
        return True, acc
    except Exception as e:
        print(f"Error running experiment: {e}")
        return False, 0.0

def plot_results(results):
    # results = {dataset: {ablation: acc}}
    datasets = list(results.keys())
    ablations = list(ABLATIONS.keys())
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, ablation in enumerate(ablations):
        accuracies = [results[d].get(ablation, 0.0) for d in datasets]
        offset = (i - len(ablations)/2 + 0.5) * width
        rects = ax.bar(x + offset, accuracies, width, label=f"{ablation}: {ABLATIONS[ablation]['desc']}")
        
        # Add labels
        ax.bar_label(rects, padding=3, fmt='%.3f')

    ax.set_ylabel('Test Accuracy')
    ax.set_title('Modality Ablation Study (Test Accuracy)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('analysis_results/modality_ablation_chart.png')
    print("Chart saved to analysis_results/modality_ablation_chart.png")

def main():
    print("Starting Ablation Experiments for GNN Explainer + Multimodal")
    
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')

    results = {}

    for dataset in DATASETS:
        results[dataset] = {}
        for ablation_name, config in ABLATIONS.items():
            success, acc = run_experiment(dataset, ablation_name, config)
            if success:
                results[dataset][ablation_name] = acc
            else:
                results[dataset][ablation_name] = 0.0

    # Save results to JSON
    with open('analysis_results/modality_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)

    for dataset, ablations in results.items():
        print(f"\n{dataset.upper()}:")
        for ablation_name, acc in ablations.items():
            print(f"  {ablation_name}: {acc:.4f}")

    plot_results(results)
    print("\nAblation experiments completed!")

if __name__ == "__main__":
    main()