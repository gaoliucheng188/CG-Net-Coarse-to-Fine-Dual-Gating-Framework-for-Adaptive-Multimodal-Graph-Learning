import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = 'log'
OUT_DIR = 'analysis_results'
os.makedirs(OUT_DIR, exist_ok=True)

pairs = [('B1','B0'), ('B2','B0'), ('B3','B0')]
mask_paths = {k: os.path.join(BASE_DIR, f'masked_adj_mutag_base_h128_o128_explain__{k}node_idx_0graph_idx_-1.npy') for k in ['B0','B1','B2','B3']}

for a,b in pairs:
    pa = mask_paths[a]
    pb = mask_paths[b]
    if not os.path.exists(pa):
        print(f'Missing {pa}, skipping {a} vs {b}')
        continue
    if not os.path.exists(pb):
        print(f'Missing {pb}, skipping {a} vs {b}')
        continue
    ma = np.load(pa)
    mb = np.load(pb)
    # Ensure shapes
    if ma.shape != mb.shape:
        print(f'Shape mismatch {a}:{ma.shape} vs {b}:{mb.shape}, skipping')
        continue
    diff = ma - mb
    # Save diff numpy
    np.save(os.path.join(OUT_DIR, f'mutag_diff_{a}_minus_{b}.npy'), diff)

    # Plot diff heatmap
    vmax = np.max(np.abs(diff))
    if vmax == 0:
        vmax = 1.0
    plt.figure(figsize=(5,5))
    plt.imshow(diff, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title(f'MUTAG {a} - {b}')
    plt.savefig(os.path.join(OUT_DIR, f'mutag_{a}_minus_{b}_heatmap.png'))
    plt.close()

    # Plot side-by-side for context
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.imshow(mb, cmap='viridis', vmin=0, vmax=1)
    plt.title(b)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(ma, cmap='viridis', vmin=0, vmax=1)
    plt.title(a)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(diff, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.title(f'{a}-{b}')
    plt.colorbar()
    plt.suptitle('Masked Adjacency Comparison - MUTAG')
    plt.savefig(os.path.join(OUT_DIR, f'mutag_{a}_vs_{b}_comparison.png'))
    plt.close()

    # Top-k changes by absolute diff
    flat = diff.flatten()
    k = min(20, flat.size)
    idx = np.argsort(np.abs(flat))[-k:][::-1]
    rows = []
    for ind in idx:
        i,j = np.unravel_index(ind, diff.shape)
        val_a = ma[i,j]
        val_b = mb[i,j]
        dval = diff[i,j]
        rows.append((i,j,val_b,val_a,dval))
    # Save CSV
    import csv
    csv_path = os.path.join(OUT_DIR, f'mutag_{a}_vs_{b}_top{str(k)}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row','col',f'{b}_value',f'{a}_value','diff'])
        for r in rows:
            writer.writerow(r)
    print(f'Wrote {csv_path} and heatmaps for {a} vs {b}')

print('Done')
