#!/usr/bin/env python3
"""
Script to run Framework Module Ablation Experiments (B0-B3) for MM-GNNExplainer.
"""

import subprocess
import os
import sys
import time

# Base command template for Explainer
# Note: We assume the model is already trained or we are just running the explainer.
# However, the user prompt implies "Experiments", which usually includes training if the ablation affects training.
# BUT, the ablations B1, B2, B3 are about "Unified Attention Layer", "Interaction Detection Head", "Domain Rule Base".
# These seem to be components of the *Explainer*, not the base GNN model.
# So we should run `explainer_main.py`.
# We need a trained model checkpoint first.
# I will assume we need to train a base model first (B0 equivalent training) and then run explainer with different flags.
# OR, if these modules are part of the *training* of the explainer (if the explainer is parametric), we need to train the explainer.
# GNNExplainer is usually an optimization process per instance, not a trained model.
# MM-GNNExplainer seems to follow GNNExplainer (optimization based).
# So we just run `explainer_main.py` with different flags on a pre-trained GNN model.

# We need a base GNN model trained on the dataset.
# I will add a step to train the base model if it doesn't exist, or just assume it exists.
# To be safe, I'll include a training step for the base model (B0 style) for each dataset.

PYTHON_EXEC = "/home/liuchenggao/miniconda3/envs/pytorch/bin/python"

BASE_CMD_TEMPLATE = [
    PYTHON_EXEC, "train.py",
    "--epochs", "1000",
    "--hidden-dim", "128",
    "--output-dim", "128",
    "--num-gc-layers", "3",
    "--dropout", "0.2",
    "--weight-decay", "0.0005",
    "--train-ratio", "0.7",
    "--gpu",
    "--cuda", "0",
]

# default to multimodal + bilinear fusion (matches run_all behavior where multimodal is enabled)
# BASE_CMD_TEMPLATE.extend(["--multimodal", "--fusion-method", "bilinear"])

# Explainer command template (match run_all_ablation_commands.py)
EXPLAINER_CMD_TEMPLATE = [
    PYTHON_EXEC, "explainer_main.py",
    "--epochs", "200",
    "--hidden-dim", "128",
    "--output-dim", "128",
    "--num-gc-layers", "3",
    "--dropout", "0.2",
    "--mask-act", "sigmoid",
    "--explain-node", "0",
    "--gpu",
    "--cuda", "0",
]

# use multimodal + bilinear by default for explainer runs
# EXPLAINER_CMD_TEMPLATE.extend(["--multimodal", "--fusion-method", "bilinear"]) 

# For compatibility with older names in this script, keep TRAIN_CMD_TEMPLATE as alias
TRAIN_CMD_TEMPLATE = BASE_CMD_TEMPLATE

DATASETS = ["mutag", "nlm_chem", "pqaref"]

ABLATIONS = {
    "B0": {
        "desc": "Baseline (Full Framework)",
        "flags": []
    },
    "B1": {
        "desc": "No Modal Attention Module",
        "flags": ["--ablation-no-attention"]
    },
    "B2": {
        "desc": "No Modal Interaction Detection Module",
        "flags": ["--ablation-no-interaction"]
    },
    "B3": {
        "desc": "No Domain Rule Base",
        "flags": ["--ablation-no-domain-rules"]
    }
}

def run_command(cmd, log_file):
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    with open(log_file, 'a') as f:
        f.write(f"\nRunning: {cmd_str}\n")
        f.write("-" * 20 + "\n")
        
    try:
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        with open(log_file, 'a') as f:
            f.write(result.stdout)
            f.write("\nSUCCESS\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command. Exit code: {e.returncode}")
        with open(log_file, 'a') as f:
            f.write(e.stdout)
            f.write(f"\nFAILED with code {e.returncode}\n")
        return False

def main():
    log_file = "framework_ablation_results.log"
    with open(log_file, 'w') as f:
        f.write("Framework Ablation Experiments (B0-B3)\n")
        f.write("========================================\n")

    for dataset in DATASETS:
        print(f"\nProcessing Dataset: {dataset}")
        
        # Determine fusion method based on dataset
        # pqaref checkpoint seems to be trained with attention
        # mutag and nlm_chem seem to be trained with bilinear (default)
        fusion_method = "bilinear"
        if dataset == "pqaref":
            fusion_method = "attention"
            
        # 1. Train Base Model (if needed)
        # Check if checkpoint exists to skip training
        ckpt_name = f"{dataset}_base_h128_o128.pth.tar"
        ckpt_path = os.path.join("ckpt", ckpt_name)
        
        if os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} exists. Skipping training.")
        else:
            print(f"Training base model for {dataset}...")
            train_cmd = TRAIN_CMD_TEMPLATE.copy()
            train_cmd.extend(["--dataset", dataset])
            train_cmd.extend(["--multimodal", "--fusion-method", fusion_method])
            
            if not run_command(train_cmd, log_file):
                print(f"Skipping {dataset} due to training failure.")
                continue

        # 2. Run Ablations
        for name, config in ABLATIONS.items():
            print(f"  Running {name}: {config['desc']}")
            expl_cmd = EXPLAINER_CMD_TEMPLATE.copy()
            expl_cmd.extend(["--dataset", dataset])
            expl_cmd.extend(["--multimodal", "--fusion-method", fusion_method])
            expl_cmd.extend(config["flags"])
            
            # Add suffix to distinguish logs
            # Do NOT use --name-suffix as it changes the checkpoint path to load
            expl_cmd.extend(["--explainer-suffix", f"_{name}"])

            run_command(expl_cmd, log_file)

    print(f"\nAll experiments completed. Check {log_file} for details.")

if __name__ == "__main__":
    main()
