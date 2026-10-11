import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

BASE_DIR = Path('log')
OUT_DIR = Path('analysis_results')
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ['nlm_chem','pqaref']
ABLATIONS = ['B0','B1','B2','B3']

for dataset in DATASETS:
    # build mask paths
    masks = {}
    for a in ABLATIONS:
        p = BASE_DIR / f'masked_adj_{dataset}_base_h128_o128_explain__{a}node_idx_0graph_idx_-1.npy'
        if p.exists():
            masks[a] = np.load(p)
        else:
            print('Missing', p)
    if 'B0' not in masks:
        print('Skipping', dataset, 'no B0')
        continue
    for a in ['B1','B2','B3']:
        if a not in masks:
            print('Skipping pair', a, 'for', dataset)
            continue
        ma = masks[a]
        mb = masks['B0']
        if ma.shape != mb.shape:
            print('Shape mismatch', dataset, a)
            continue
        diff = ma - mb
        np.save(OUT_DIR / f'{dataset}_diff_{a}_minus_B0.npy', diff)
        # heatmap
        vmax = np.max(np.abs(diff))
        if vmax == 0: vmax = 1.0
        plt.figure(figsize=(5,5))
        plt.imshow(diff, cmap='bwr', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(f'{dataset} {a}-B0')
        plt.savefig(OUT_DIR / f'{dataset}_{a}_minus_B0_heatmap.png')
        plt.close()
        # side-by-side
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(mb, cmap='viridis', vmin=0, vmax=1)
        plt.title('B0')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(ma, cmap='viridis', vmin=0, vmax=1)
        plt.title(a)
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(diff, cmap='bwr', vmin=-vmax, vmax=vmax)
        plt.title(f'{a}-B0')
        plt.colorbar()
        plt.suptitle(f'Masked Adjacency Comparison - {dataset}')
        plt.savefig(OUT_DIR / f'{dataset}_{a}_vs_B0_comparison.png')
        plt.close()
        # top-k
        flat = diff.flatten()
        k = min(20, flat.size)
        idx = np.argsort(np.abs(flat))[-k:][::-1]
        rows = []
        for ind in idx:
            i,j = np.unravel_index(ind, diff.shape)
            rows.append((i,j, mb[i,j], ma[i,j], diff[i,j]))
        import csv
        csv_path = OUT_DIR / f'{dataset}_{a}_vs_B0_top{k}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row','col','B0_value',f'{a}_value','diff'])
            for r in rows:
                writer.writerow(r)
        print('Wrote', csv_path)

print('Done')
