#!/usr/bin/env python3
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AR = os.path.join(ROOT, 'analysis_results')

imgs = [
    os.path.join(AR, 'mutag_B1_vs_B0_comparison.png'),
    os.path.join(AR, 'mutag_B2_vs_B0_comparison.png'),
    os.path.join(AR, 'mutag_B3_vs_B0_comparison.png'),
]

semantic_img = os.path.join(AR, 'semantic_viz_B0.png')

def main():
    # A4 portrait
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(7, 4, hspace=0.6, wspace=0.4)

    # Title
    ax_title = fig.add_subplot(gs[0:1, :])
    ax_title.axis('off')
    title = 'Week Report: MM-GNNExplainer Analysis (MUTAG)'
    summary = (
        "Objective: Analyze the impact of removing Attention (B1), Interaction (B2), and Domain Rules (B3).\n"
        "1. Qualitative: B0 (Baseline) focuses on C-C bonds in the ring structure (see Semantic Viz).\n"
        "   B2 (No Interaction) produces a denser mask, highlighting almost all edges (less selective).\n"
        "2. Quantitative: B2 has significantly higher mask density (p<0.05). B0 is more sparse/faithful.\n"
        "3. Semantic: Top edges for B0 are C-C (Weight > 0.98). B1 drops some C-C edges."
    )
    ax_title.text(0, 0.8, title, fontsize=16, weight='bold')
    ax_title.text(0, 0.2, summary, fontsize=10, wrap=True)

    # Semantic Viz
    ax_sem = fig.add_subplot(gs[1:3, 0:2])
    ax_sem.axis('off')
    if os.path.exists(semantic_img):
        im = Image.open(semantic_img)
        ax_sem.imshow(im)
        ax_sem.set_title("Semantic Interpretation (B0)", fontsize=10)
    else:
        ax_sem.text(0.5, 0.5, "Semantic Viz Missing", ha='center')

    # Comparison Plots
    for i, img_path in enumerate(imgs):
        # Position: Row 1-2 (Col 2-3) for first, then Row 3-4...
        # Let's just put them in rows 3, 4, 5
        row = 3 + i
        ax = fig.add_subplot(gs[row, :])
        ax.axis('off')
        if os.path.exists(img_path):
            im = Image.open(img_path)
            ax.imshow(im)
            ax.set_title(os.path.basename(img_path).replace('_', ' '), fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Missing: ' + os.path.basename(img_path), ha='center')

    out_pdf = os.path.join(AR, 'week_report_final.pdf')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', out_pdf)

if __name__ == '__main__':
    main()
