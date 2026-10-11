#!/usr/bin/env python3
"""
Script to generate and run Modal Ablation experiment commands (A0-A3) for GNN Explainer + Multimodal
"""

import subprocess
import os
import sys
import time

# Base command template for training
BASE_CMD_TEMPLATE = [
    "python", "train.py",
    "--epochs", "500", 
    "--hidden-dim", "128",
    "--output-dim", "128",
    "--num-gc-layers", "3",
    "--dropout", "0.2",
    "--weight-decay", "0.0005",
    "--train-ratio", "0.7",
    "--gpu",
    "--cuda", "1"
]

# Base command template for explainer
EXPLAINER_CMD_TEMPLATE = [
    "python", "explainer_main.py",
    "--epochs", "200",
    "--hidden-dim", "128",
    "--output-dim", "128",
    "--num-gc-layers", "3",
    "--dropout", "0.2",
    "--mask-act", "sigmoid",
    "--explain-node", "0",
    "--gpu",
    "--cuda", "1"
]

# Datasets to run on
DATASETS = ["mutag", "nlm_chem", "pqaref"]

# Modal Ablation configurations (A0-A3)
# Modified: All use fusion=concat (B1 style) as base
ABLATIONS = {
    # --- Modal Ablations ---
    "A0": {
        "description": "Full Multimodal (Concat)",
        "train": True,
        "model_suffix": "A0",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": []
    },
    "A1": {
        "description": "No Text (Concat)",
        "train": True,
        "model_suffix": "A1",
        "common_args": ["--multimodal", "--fusion-method", "concat", "--disable-text"],
        "explainer_args": []
    },
    "A2": {
        "description": "No Image (Concat)",
        "train": True,
        "model_suffix": "A2",
        "common_args": ["--multimodal", "--fusion-method", "concat", "--disable-image"],
        "explainer_args": []
    },
    "A3": {
        "description": "Structure Only (No Multimodal)",
        "train": True,
        "model_suffix": "A3",
        "common_args": [], # No --multimodal flag
        "explainer_args": []
    }
}

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Assumes this script is in <root>/scripts/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def run_command(cmd, log_file, cwd):
    """Run a command and log its output."""
    cmd_str = ' '.join(cmd)
    print(f"Running: {cmd_str}")
    with open(log_file, 'a') as f:
        f.write(f"\nCMD: {cmd_str}\n")
        f.write("-" * 50 + "\n")
        
    try:
        # Run and stream output to stdout while capturing for log
        process = subprocess.Popen(
            cmd, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for line in process.stdout:
            print(line, end='') # Stream to console
            output_lines.append(line)
            
        process.wait()
        
        with open(log_file, 'a') as f:
            f.write(''.join(output_lines))
            f.write(f"\nExit Code: {process.returncode}\n")
            
        return process.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        with open(log_file, 'a') as f:
            f.write(f"EXECUTION ERROR: {e}\n")
        return False

def run_modal_experiments():
    root_dir = get_project_root()
    log_file = os.path.join(root_dir, "ablation_experiments_modal.log")
    
    print(f"Project Root: {root_dir}")
    print(f"Logging to: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write("Modal Ablation Experiments Log (A0-A3)\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    for dataset in DATASETS:
        results[dataset] = {}
        print(f"\n{'='*40}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*40}")
        
        for ablation_name, config in ABLATIONS.items():
            print(f"\n--- Experiment: {ablation_name} ({config['description']}) ---")
            
            # 1. Training Phase
            train_success = True
            if config["train"]:
                train_cmd = BASE_CMD_TEMPLATE.copy()
                train_cmd.extend(["--dataset", dataset])
                train_cmd.extend(["--name-suffix", config["model_suffix"]])
                train_cmd.extend(config["common_args"])
                
                print(f"Training model for {ablation_name}...")
                train_success = run_command(train_cmd, log_file, root_dir)
            else:
                print(f"Skipping training for {ablation_name} (Reusing {config['model_suffix']})")

            # 2. Explainer Phase
            explain_success = False
            if train_success:
                expl_cmd = EXPLAINER_CMD_TEMPLATE.copy()
                expl_cmd.extend(["--dataset", dataset])
                expl_cmd.extend(["--name-suffix", config["model_suffix"]]) # Load the correct model
                expl_cmd.extend(["--explainer-suffix", ablation_name]) # Save logs with ablation name
                expl_cmd.extend(config["common_args"])
                expl_cmd.extend(config["explainer_args"])
                
                print(f"Running explainer for {ablation_name}...")
                explain_success = run_command(expl_cmd, log_file, root_dir)
            else:
                print(f"Skipping explainer for {ablation_name} due to training failure.")

            results[dataset][ablation_name] = {
                "train": "SUCCESS" if train_success else ("SKIPPED" if not config["train"] else "FAILED"),
                "explain": "SUCCESS" if explain_success else "FAILED"
            }

    # Summary
    print("\n" + "="*60)
    print("MODAL ABLATION SUMMARY")
    print("="*60)
    for dataset, ablations in results.items():
        print(f"\n{dataset.upper()}:")
        for name, status in ablations.items():
            print(f"  {name}: Train={status['train']}, Explain={status['explain']}")

if __name__ == "__main__":
    run_modal_experiments()
