import numpy as np
import os
from pathlib import Path

OUT_DIR = Path('analysis_results')
OUT_DIR.mkdir(exist_ok=True)

MASK_DIR = Path('log')
masks = {}
for k in ['B0','B1','B2','B3']:
    p = MASK_DIR / f'masked_adj_mutag_base_h128_o128_explain__{k}node_idx_0graph_idx_-1.npy'
    if p.exists():
        masks[k] = np.load(p)
    else:
        print('Missing', p)

if 'B0' not in masks:
    raise SystemExit('Base mask B0 missing')

base = masks['B0'].flatten()
results = []

# Try scipy
use_scipy = False
try:
    from scipy import stats
    use_scipy = True
except Exception:
    use_scipy = False

def permutation_pvalue(x, y, n_perms=5000):
    # paired permutation test on differences
    d = x - y
    obs = np.mean(d)
    n = len(d)
    count = 0
    rng = np.random.default_rng(0)
    for _ in range(n_perms):
        signs = rng.choice([1,-1], size=n)
        perm = d * signs
        if abs(np.mean(perm)) >= abs(obs):
            count += 1
    return (count + 1) / (n_perms + 1)

for k in ['B1','B2','B3']:
    if k not in masks:
        print('Skipping', k)
        continue
    arr = masks[k].flatten()
    if arr.shape != base.shape:
        print('Shape mismatch', k)
        continue
    diff = arr - base
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)
    # paired t-test / wilcoxon if available
    t_p = 'NA'
    w_p = 'NA'
    if use_scipy:
        try:
            t_stat, t_p = stats.ttest_rel(arr, base)
        except Exception as e:
            t_p = f'err:{e}'
        try:
            w_stat, w_p = stats.wilcoxon(arr, base)
        except Exception as e:
            w_p = f'err:{e}'
    else:
        # fallback permutation
        t_p = permutation_pvalue(arr, base, n_perms=2000)
        w_p = 'perm_fallback'

    res = {
        'comparison': f'{k}-B0',
        'mean_B0': float(np.mean(base)),
        'mean_B': float(np.mean(arr)),
        'mean_diff': float(mean_diff),
        'median_diff': float(median_diff),
        't_p': t_p,
        'w_p': w_p
    }
    results.append(res)

# Save CSV
import csv
out_csv = OUT_DIR / 'mutag_paired_tests.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('Wrote', out_csv)
print('\nSummary:')
for r in results:
    print(r)
