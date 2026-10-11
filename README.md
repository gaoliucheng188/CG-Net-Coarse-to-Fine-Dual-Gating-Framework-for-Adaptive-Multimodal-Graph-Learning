# CG-Net: Coarse-to-Fine Dual Gating for Adaptive Multimodal Graph Learning

This repository contains the source code for **CG-Net**, a multimodal graph learning framework that combines graph structure with node-level text and image information. The project includes data preprocessing, multimodal feature fusion, model training, explanation generation, ablation experiments, and evaluation utilities.

> **Project status:** This repository is an active research codebase. Exact commands and available datasets can depend on the local data files and model checkpoints. See the scripts and configuration files for the current implementation details.

## Overview

CG-Net builds graph representations and fuses them with text and image features. The codebase provides:

- Graph, text, and image feature loading and preprocessing.
- Configurable multimodal fusion (including `concat`, `bilinear`, and `attention` paths).
- GNN model training and multimodal explanation workflows.
- Framework/module ablation scripts and analysis utilities.
- Dataset-specific preparation scripts, including TUD-MUTAG preprocessing.

Core implementation is in [`models_multimodal.py`](models_multimodal.py), [`multimodal/`](multimodal/), [`data/multimodal_loader.py`](data/multimodal_loader.py), and [`explainer/`](explainer/).

## Repository layout

| Path | Purpose |
|------|---------|
| `train.py` | Train graph or multimodal graph models. |
| `explainer_main.py` | Run the explanation workflow. |
| `models.py`, `models_multimodal.py`, `models_pyg.py` | Graph and multimodal model implementations. |
| `multimodal/` | Text/image encoders and multimodal fusion modules. |
| `data/` | Dataset loader, raw/processed data, and generated modalities. |
| `scripts/` | Dataset preparation and experiment runners. |
| `analysis_scripts/` | Result analysis, reporting, and visualization utilities. |

## Setup

Use a Python environment compatible with the installed PyTorch and graph/data-science packages. The dependency versions in [`requirements.txt`](requirements.txt) are inherited from the original codebase and may need adjustment for your Python, operating system, and CUDA version.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your system and CUDA setup by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Some text-encoder configurations may also require the optional `transformers` package and locally available pretrained model files; consult the encoder code and use the fallback option when applicable.

## Data preparation

The multimodal loader expects a preprocessed graph pickle plus optional node-level text and image directories under `data/`. Dataset preparation scripts are in `scripts/`.

For the included TUD-MUTAG raw files, prepare the graph and derived multimodal artifacts with:

```bash
python scripts/prepare_mutag.py --raw-dir data/tud_mutag/raw --out-dir data
```

The script creates a preprocessed `MUTAG.pkl` along with supporting processed files and node/graph text and image artifacts. Check the script's `--help` output for split, seed, and overwrite options. Other datasets may have their own input format and preparation requirements; see the corresponding `scripts/prepare_*.py` files.

## Usage

### Train a model

A basic training invocation is:

```bash
python train.py --dataset MUTAG --epochs 1000 --hidden-dim 128 --output-dim 128 --num-gc-layers 3
```

For multimodal training, enable multimodal inputs and choose a fusion method supported by the current implementation:

```bash
python train.py --dataset MUTAG --multimodal --fusion-method bilinear \
  --epochs 1000 --hidden-dim 128 --output-dim 128 --num-gc-layers 3
```

To see all options available in the current checkout:

```bash
python train.py --help
```

Use `--gpu --cuda 0` to request GPU execution where CUDA is available. Dataset names, required files, and defaults are determined by the dataset loader and training code; verify that the matching preprocessed files exist before starting a run.

### Generate explanations

Run the explainer against a supported dataset and its trained checkpoint:

```bash
python explainer_main.py --dataset MUTAG --multimodal --fusion-method bilinear \
  --explain-node 0 --epochs 200
```

See all explainer options with:

```bash
python explainer_main.py --help
```

The explanation run depends on a compatible trained model checkpoint and the associated dataset artifacts. Output locations and checkpoint naming are controlled by the command-line options and project utilities.

### Run ablations

Experiment and ablation helpers are available in `scripts/`, including:

```bash
python scripts/run_framework_ablation.py
```

Review the script before running it: experiment runners can start multiple training/explanation jobs, may assume particular datasets/checkpoints, and some scripts contain machine-specific paths or settings that should be adapted to your environment.

## Evaluation and analysis

The repository includes analysis utilities under `analysis_scripts/` and top-level evaluation scripts for metrics, sensitivity, fidelity, and case studies. Their inputs and output paths vary by experiment. Use each script's help text or inspect its argument/configuration section before running it.

## Notes and limitations

- Large pretrained weights and some experiment artifacts may not be included in the repository. Obtain or generate the required assets separately.
- Text and image modalities are optional at data-loading time, but experiments that rely on them need the corresponding features or source files.
- The checked-in `requirements.txt` pins older package versions; compatibility with modern Python/PyTorch may require environment-specific updates.
- Results can vary with random seeds, dataset preprocessing, checkpoint selection, and hardware/software versions.

## Citation

If you use this code in academic work, please cite the CG-Net paper associated with this project. Add the final bibliographic details here once the publication information is available.

```bibtex
@article{cgnet,
  title   = {CG-Net: Coarse-to-Fine Dual Gating Framework for Adaptive Multimodal Graph Learning},
  author  = {TODO},
  journal = {Expert Systems with Applications},
  year    = {TODO}
}
```

## License

No license is specified in this repository at the time of writing. Please check with the project maintainers before redistributing or reusing the code.
