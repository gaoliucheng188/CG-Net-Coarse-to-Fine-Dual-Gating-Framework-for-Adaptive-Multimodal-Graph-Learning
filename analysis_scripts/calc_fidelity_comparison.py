#!/usr/bin/env python
"""
Compare Fidelity Results: Gated vs Non-Gated Explainer Masks

This script analyzes fidelity results from fidelity_results.csv
and compares Baseline vs Gated variants.

Usage:
    python analysis_scripts/calc_fidelity_comparison.py
"""

import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict

def parse_variant_from_filename(filename):
    """
    Parse variant information from mask filename.

    Examples:
    - masked_adj_mutag_base_h128_o128_A0_explain_A0node_idx_0... -> Gated_A0
    - masked_adj_mutag_base_h128_o128_explain__B0node_idx_0...   -> Baseline_B0
    - masked_adj_mutag_base_h128_o128_explainnode_idx_0...        -> Baseline
    - masked_adj_mutag_base_h20_o20_explainnode_idx_0...          -> Legacy_h20o20
    """
    import re
    basename = os.path.basename(filename).replace('.npy', '')
    parts = basename.split('_')

    # Extract hidden/output dims to identify legacy experiments
    hidden_match = re.search(r'h(\d+)_o(\d+)', basename)
    if hidden_match:
        h_dim, o_dim = hidden_match.groups()
        if h_dim != '128' or o_dim != '128':
            return f'Legacy_h{h_dim}o{o_dim}'

    # Find 'explain' in parts - but note it might be 'explainnode' (no underscore)
    explain_idx = -1
    for i, part in enumerate(parts):
        if part.startswith('explain'):
            explain_idx = i
            break

    if explain_idx == -1:
        return 'Baseline'

    # Look for variant pattern in parts near 'explain'
    # Check part after 'explain': e.g., "A0node" -> extract "A0"
    if explain_idx + 1 < len(parts):
        next_part = parts[explain_idx + 1]
        match = re.match(r'^([AB]\d+)', next_part)
        if match:
            return f'Gated_{match.group(1)}'

    # Check part before 'explain': e.g., "A0_explain"
    if explain_idx >= 1:
        prev_part = parts[explain_idx - 1]
        if re.match(r'^[AB]\d+$', prev_part, re.IGNORECASE):
            return f'Gated_{prev_part.upper()}'

    # Check for double underscore pattern: "explain__B0"
    if explain_idx + 2 < len(parts):
        next_next_part = parts[explain_idx + 2]
        match = re.match(r'^([AB]\d+)', next_next_part)
        if match:
            return f'Baseline_{match.group(1)}'

    # No variant found - it's a baseline without specific variant
    return 'Baseline'


def parse_dataset_from_filename(filename):
    """Extract dataset name from filename."""
    basename = os.path.basename(filename)
    datasets = ['mutag', 'nlm_chem', 'pqaref', 'syn1']
    for ds in datasets:
        if ds in basename:
            return ds
    return 'unknown'


def load_fidelity_results(csv_path):
    """Load fidelity results from CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print(f"Please run run_fidelity.sh first to generate this file.")
        return None
    df = pd.read_csv(csv_path)
    return df


def analyze_results(df):
    """Analyze fidelity results by dataset and variant.

    Supports both old format (only file column) and new format (gated/variant columns).
    Filters out Legacy variants (non-128x128 experiments).
    """
    results = defaultdict(lambda: defaultdict(list))

    # Check if new format with gated/variant columns
    if 'gated' in df.columns and 'variant' in df.columns:
        # New format: use gated/variant columns directly
        for _, row in df.iterrows():
            dataset = parse_dataset_from_filename(row['file'])
            variant = row['variant']
            gated = row['gated']

            # Skip Legacy variants (non-128x128 experiments)
            if variant.startswith('Legacy'):
                continue

            results[dataset][variant].append({
                'fidelity_plus': row['fidelity_plus'],
                'fidelity_minus': row['fidelity_minus'],
                'gated': gated
            })
    else:
        # Old format: parse from filename only
        for _, row in df.iterrows():
            filename = row['file']
            dataset = parse_dataset_from_filename(filename)
            variant = parse_variant_from_filename(filename)

            # Skip Legacy variants
            if variant.startswith('Legacy'):
                continue

            results[dataset][variant].append({
                'fidelity_plus': row['fidelity_plus'],
                'fidelity_minus': row['fidelity_minus']
            })

    return results


def compute_summary(results):
    """Compute summary statistics."""
    summary = {}

    for dataset, variants in results.items():
        summary[dataset] = {}
        for variant, records in variants.items():
            if records:
                f_plus_values = [r['fidelity_plus'] for r in records]
                f_minus_values = [r['fidelity_minus'] for r in records]
                # Get gated status (all records for same variant should have same gated value)
                gated_values = [r.get('gated', False) for r in records]
                is_gated = any(gated_values)  # True if any record is gated

                summary[dataset][variant] = {
                    'count': len(records),
                    'is_gated': is_gated,
                    'mean_f_plus': np.mean(f_plus_values),
                    'std_f_plus': np.std(f_plus_values),
                    'mean_f_minus': np.mean(f_minus_values),
                    'std_f_minus': np.std(f_minus_values)
                }
    return summary


def create_comparison_report(summary):
    """Create a formatted comparison report."""
    print("\n" + "="*80)
    print("FIDELITY COMPARISON REPORT: GATED VS BASELINE")
    print("="*80)

    all_comparisons = []

    for dataset in sorted(summary.keys()):
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        variants = summary[dataset]

        print(f"\n{'Variant':<20} {'Gated':<8} {'Count':<8} {'F+ Mean':<12} {'F+ Std':<12} {'F- Mean':<12} {'F- Std':<12}")
        print("-"*80)

        for variant in sorted(variants.keys()):
            stats = variants[variant]
            gated_str = "Yes" if stats.get('is_gated', False) else "No"
            print(f"{variant:<20} {gated_str:<8} {stats['count']:<8} {stats['mean_f_plus']:<12.6f} "
                  f"{stats['std_f_plus']:<12.6f} {stats['mean_f_minus']:<12.6f} {stats['std_f_minus']:<12.6f}")

        # Separate Gated vs Baseline comparisons
        gated_variants = {k: v for k, v in variants.items() if v.get('is_gated', False)}
        baseline_variants = {k: v for k, v in variants.items() if not v.get('is_gated', False)}

        if baseline_variants and gated_variants:
            print(f"\n{'Comparison Analysis':<30}")
            print("-"*70)

            # Use first baseline as reference
            baseline_key = list(baseline_variants.keys())[0]
            baseline = baseline_variants[baseline_key]

            for variant, gated in sorted(gated_variants.items()):
                f_plus_diff = gated['mean_f_plus'] - baseline['mean_f_plus']
                f_minus_diff = gated['mean_f_minus'] - baseline['mean_f_minus']
                better_fplus = "✓" if f_plus_diff > 0 else ""
                better_fminus = "✓" if f_minus_diff > 0 else "↓ (lower is better for F-)"

                print(f"\n{variant} vs {baseline_key}:")
                print(f"  F+ Change: {f_plus_diff:+.6f} {better_fplus}")
                print(f"  F- Change: {f_minus_diff:+.6f} {better_fminus}")

                all_comparisons.append({
                    'Dataset': dataset,
                    'Variant': variant.replace('Gated_', ''),
                    'Baseline_Variant': baseline_key,
                    'Baseline_F+': baseline['mean_f_plus'],
                    'Gated_F+': gated['mean_f_plus'],
                    'F+_Improvement': f_plus_diff,
                    'Baseline_F-': baseline['mean_f_minus'],
                    'Gated_F-': gated['mean_f_minus'],
                    'F-_Improvement': f_minus_diff
                })

        # If no clear Baseline, still report Gated variants
        if baseline_variants and not gated_variants:
            print(f"\n(Note: Only Baseline variants found for this dataset)")

        if not baseline_variants and gated_variants:
            print(f"\n(Note: Only Gated variants found for this dataset - no comparison available)")

    return all_comparisons


def print_conclusion(all_comparisons):
    """Print conclusion from comparison results."""
    if not all_comparisons:
        print("\nNo comparison data available.")
        return
    
    print("\n" + "="*80)
    print("OVERALL CONCLUSION")
    print("="*80)
    
    df = pd.DataFrame(all_comparisons)
    
    print("\nAggregated Results by Dataset:")
    print("-"*70)
    
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        avg_improvement = subset['F+_Improvement'].mean()
        better_count = (subset['F+_Improvement'] > 0).sum()
        total_count = len(subset)
        
        status = "✓" if avg_improvement > 0 else "✗"
        print(f"{dataset.upper()}: Avg F+ Improvement = {avg_improvement:+.6f} {status} ({better_count}/{total_count} variants better)")
    
    overall_avg = df['F+_Improvement'].mean()
    if overall_avg > 0:
        print(f"\n✓ Gated variants show overall F+ improvement of {overall_avg:.6f}")
    else:
        print(f"\n✗ Gated variants show overall F+ decrease of {abs(overall_avg):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Compare Fidelity Results')
    parser.add_argument('--csv', default='analysis_results/semantics/fidelity_results.csv',
                        help='Path to fidelity results CSV')
    parser.add_argument('--output', default='analysis_results/gated_vs_baseline_comparison.csv',
                        help='Output comparison CSV path')
    args = parser.parse_args()
    
    df = load_fidelity_results(args.csv)
    if df is None:
        return
    
    print(f"Loaded {len(df)} records from {args.csv}")
    
    results = analyze_results(df)
    summary = compute_summary(results)
    all_comparisons = create_comparison_report(summary)
    print_conclusion(all_comparisons)
    
    if all_comparisons:
        out_df = pd.DataFrame(all_comparisons)
        out_df.to_csv(args.output, index=False)
        print(f"\nComparison results saved to: {args.output}")


if __name__ == '__main__':
    main()
