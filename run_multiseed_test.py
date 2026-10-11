#!/usr/bin/env python
"""
Multi-Seed Statistical Significance Testing (Task 4)
Runs experiments with multiple random seeds and computes mean ± std.

Usage:
    # Run MUTAG with 5 seeds
    python run_multiseed_test.py --dataset mutag --ckpt-template "ckpt/{dataset}_base_h128_o128_full_modal.pth.tar" --seeds 0 1 2 3 4 --gpu

    # Run all datasets with 5 seeds
    python run_multiseed_test.py --dataset mutag nlm_chem pqaref --seeds 0 1 2 3 4 --gpu
"""
import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Seed Statistical Testing')
    parser.add_argument('--dataset', nargs='+', default=['mutag'],
                        help='Dataset(s) to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024],
                        help='Random seeds to use')
    parser.add_argument('--ckpt-template', type=str,
                        default="ckpt/{dataset}_base_h128_o128_full_modal.pth.tar",
                        help='Checkpoint path template with {dataset} placeholder')
    parser.add_argument('--text-tau', type=float, default=0.7,
                        help='Text tau threshold')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU')
    parser.add_argument('--cuda', type=str, default='0',
                        help='CUDA device')
    parser.add_argument('--explainer-epochs', type=int, default=20,
                        help='Explainer epochs per sample')
    parser.add_argument('--output-dir', type=str, default='analysis_results/multiseed',
                        help='Output directory for results')
    return parser.parse_args()

def run_single_evaluation(dataset, seed, ckpt_path, args):
    """Run single evaluation and parse accuracy results."""
    cmd = [
        'python', 'evaluate_explainer_gating.py',
        '--dataset', dataset,
        '--ckpt-path', ckpt_path,
        '--text-tau', str(args.text_tau),
        '--explainer-epochs', str(args.explainer_epochs),
    ]
    if args.gpu:
        cmd.append('--gpu')
        cmd.extend(['--cuda', args.cuda])

    # Set random seed for this run
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)

    print(f"\n{'='*60}")
    print(f"Running {dataset} with seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600  # 1 hour timeout
        )

        output = result.stdout + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)  # Print last 2000 chars

        # Parse results from output
        results = {
            'seed': seed,
            'dataset': dataset,
            'baseline_full': None,
            'baseline_notext': None,
            'hard_only': None,
            'soft_only': None,
            'ours': None,
            'hard_bypass_count': None,
            'avg_text_score': None,
        }

        for line in output.split('\n'):
            line = line.strip()
            if 'Baseline (Always Full)' in line and ':' in line:
                try:
                    results['baseline_full'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Ablation (Always Drop Text)' in line and ':' in line:
                try:
                    results['baseline_notext'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Variant A (Hard Router Only)' in line and ':' in line:
                try:
                    results['hard_only'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Variant B (Soft Fusion Only)' in line and ':' in line:
                try:
                    results['soft_only'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'OURS (Dual Gating Framework)' in line and ':' in line:
                try:
                    results['ours'] = float(line.split(':')[1].split()[0].strip())
                except:
                    pass
            elif 'Hard Bypassed Text (O(1))' in line:
                try:
                    parts = line.split(':')[1].strip().split()
                    results['hard_bypass_count'] = int(parts[0].split('/')[0])
                except:
                    pass
            elif 'Avg Soft Text Attention' in line:
                try:
                    results['avg_text_score'] = float(line.split(':')[1].strip())
                except:
                    pass

        return results, output

    except subprocess.TimeoutExpired:
        print(f"Timeout for {dataset} seed={seed}")
        return None, ""
    except Exception as e:
        print(f"Error for {dataset} seed={seed}: {e}")
        return None, ""

def compute_statistics(all_results, method='ours'):
    """Compute mean and std for a specific method across seeds."""
    values = [r[method] for r in all_results if r and r[method] is not None]
    if values:
        return np.mean(values), np.std(values), values
    return None, None, []

def format_result(mean, std):
    """Format as mean ± std string."""
    if mean is not None and std is not None:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    return "N/A"

def main():
    args = parse_args()

    # Create output directory
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}  # dataset -> list of run results
    summary_data = []

    print(f"\n{'#'*70}")
    print(f"# Multi-Seed Statistical Significance Testing")
    print(f"# Datasets: {args.dataset}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Text Tau: {args.text_tau}")
    print(f"{'#'*70}")

    for dataset in args.dataset:
        ckpt_path = args.ckpt_template.format(dataset=dataset)

        if not os.path.exists(ckpt_path):
            print(f"\n⚠️  Checkpoint not found: {ckpt_path}")
            print(f"   Skipping {dataset}")
            continue

        dataset_results = []

        for seed in args.seeds:
            result, _ = run_single_evaluation(dataset, seed, ckpt_path, args)
            if result:
                dataset_results.append(result)
                all_results[dataset] = dataset_results

        # Compute statistics for this dataset
        if dataset_results:
            print(f"\n{'='*60}")
            print(f"📊 Results for {dataset.upper()}:")
            print(f"{'='*60}")

            methods = ['baseline_full', 'baseline_notext', 'hard_only', 'soft_only', 'ours']
            method_names = ['Baseline (Full)', 'Baseline (No Text)', 'Hard Only', 'Soft Only', 'OURS (Dual)']

            dataset_summary = {'dataset': dataset, 'seeds': len(dataset_results)}

            for method, name in zip(methods, method_names):
                mean, std, values = compute_statistics(dataset_results, method)
                dataset_summary[f'{method}_mean'] = mean
                dataset_summary[f'{method}_std'] = std
                dataset_summary[f'{method}_values'] = values

                formatted = format_result(mean, std)
                print(f"  {name:25s}: {formatted}")

            # Also compute bypass rate
            bypass_rates = [r['hard_bypass_count'] for r in dataset_results if r.get('hard_bypass_count') is not None]
            if bypass_rates:
                bypass_mean = np.mean(bypass_rates)
                bypass_std = np.std(bypass_rates)
                dataset_summary['bypass_rate_mean'] = bypass_mean
                dataset_summary['bypass_rate_std'] = bypass_std
                print(f"  {'Hard Bypass Rate':25s}: {bypass_mean:.1f} ± {bypass_std:.1f}")

            summary_data.append(dataset_summary)

    # Save detailed results
    detailed_path = outdir / f'multiseed_detailed_{timestamp}.json'
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {detailed_path}")

    # Save summary table
    if summary_data:
        summary_path = outdir / f'multiseed_summary_{timestamp}.csv'

        rows = []
        for s in summary_data:
            row = {
                'Dataset': s['dataset'].upper(),
                'N_Seeds': s['seeds'],
                'Baseline_Full': format_result(s.get('baseline_full_mean'), s.get('baseline_full_std')),
                'Baseline_NoText': format_result(s.get('baseline_notext_mean'), s.get('baseline_notext_std')),
                'Hard_Only': format_result(s.get('hard_only_mean'), s.get('hard_only_std')),
                'Soft_Only': format_result(s.get('soft_only_mean'), s.get('soft_only_std')),
                'OURS_Dual': format_result(s.get('ours_mean'), s.get('ours_std')),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        print(f"💾 Summary table saved to: {summary_path}")

        # Print LaTeX-ready table
        print(f"\n{'='*70}")
        print("📋 LaTeX-Ready Table:")
        print(f"{'='*70}")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Multi-Seed Statistical Results (Mean ± Std)}")
        print("\\begin{tabular}{l|c|c|c|c|c}")
        print("\\hline")
        print("Dataset & Baseline & No Text & Hard Only & Soft Only & OURS \\\\")
        print("\\hline")

        for s in summary_data:
            dataset = s['dataset'].upper()
            cols = [
                format_result(s.get('baseline_full_mean'), s.get('baseline_full_std')),
                format_result(s.get('baseline_notext_mean'), s.get('baseline_notext_std')),
                format_result(s.get('hard_only_mean'), s.get('hard_only_std')),
                format_result(s.get('soft_only_mean'), s.get('soft_only_std')),
                format_result(s.get('ours_mean'), s.get('ours_std')),
            ]
            print(f"{dataset} & {cols[0]} & {cols[1]} & {cols[2]} & {cols[3]} & {cols[4]} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    print(f"\n{'#'*70}")
    print(f"# Multi-Seed Testing Complete!")
    print(f"{'#'*70}")

if __name__ == '__main__':
    main()

