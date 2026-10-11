#!/usr/bin/env python3
"""
Script to generate and run Module Ablation experiment commands (B1-B4) for GNN Explainer + Multimodal
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

# Module Ablation configurations (B1-B4)
# Modified: Base training uses fusion=concat (B1). Explainer uses UMA (Attention) internally.
ABLATIONS = {
    # --- Base Model (B1) ---
    "B1_Base": {
        "description": "Base Model (Concat Fusion)",
        "train": True,
        "model_suffix": "B1",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": [] # This entry is mainly for training the base model
    },
    # --- Module Ablations (on B1 Base) ---
    "B0": {
        "description": "Full Framework (Concat Base + Explainer UMA)",
        "train": False, # Reuse B1
        "model_suffix": "B1",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": ["--explainer-uma"] # Enable UMA in explainer
    },
    "B1": {
        "description": "No UMA (Concat Base + Explainer Concat)",
        "train": False, # Reuse B1
        "model_suffix": "B1",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": [] # No UMA flag
    },
    "B2": {
        "description": "No Interaction Loss (Lambda ID = 0)",
        "train": False, # Reuse B1
        "model_suffix": "B1",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": ["--explainer-uma", "--lambda-ID", "0.0"]
    },
    "B3": {
        "description": "No Domain Loss (Lambda Domain = 0)",
        "train": False, # Reuse B1
        "model_suffix": "B1",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": ["--explainer-uma", "--lambda-domain", "0.0"]
    },
    "B4": {
        "description": "No Consistency Loss (Lambda Consist = 0)",
        "train": False, # Reuse B1
        "model_suffix": "B1",
        "common_args": ["--multimodal", "--fusion-method", "concat"],
        "explainer_args": ["--explainer-uma", "--lambda-consist", "0.0"]
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

def run_module_experiments():
    root_dir = get_project_root()
    log_file = os.path.join(root_dir, "ablation_experiments_module.log")
    
    print(f"Project Root: {root_dir}")
    print(f"Logging to: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write("Module Ablation Experiments Log (B1-B4)\n")
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
    print("MODULE ABLATION SUMMARY")
    print("="*60)
    for dataset, ablations in results.items():
        print(f"\n{dataset.upper()}:")
        for name, status in ablations.items():
            print(f"  {name}: Train={status['train']}, Explain={status['explain']}")

if __name__ == "__main__":
    run_module_experiments()
