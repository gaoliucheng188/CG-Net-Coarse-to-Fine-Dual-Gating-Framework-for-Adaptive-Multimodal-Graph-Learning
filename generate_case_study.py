#!/usr/bin/env python
"""
Generate Case Study Visualizations for the paper.
Extracts representative samples for Hard Router and Soft Fusion cases.

Usage:
    python generate_case_study.py --dataset mutag --ckpt-path ckpt/mutag_base_h128_o128_full_modal.pth.tar --gpu
"""
import os
import sys
import argparse
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import utils.gengraph as gengraph
from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
from explainer.explain_multimodal import MultimodalExplainer


def ensure_args(args):
    """Ensure args has all necessary attributes."""
    args.bmname = getattr(args, 'bmname', None)
    args.method = getattr(args, 'method', 'base')
    args.bias = getattr(args, 'bias', True)
    args.mask_act = getattr(args, 'mask_act', 'sigmoid')
    args.mask_bias = getattr(args, 'mask_bias', False)
    args.explainer_uma = getattr(args, 'explainer_uma', False)
    args.num_gc_layers = getattr(args, 'num_gc_layers', 3)
    args.lr = getattr(args, 'lr', 0.5)
    args.clip = getattr(args, 'clip', 2.0)
    args.num_epochs = getattr(args, 'num_epochs', 20)
    args.opt = getattr(args, 'opt', 'adam')
    args.weight_decay = getattr(args, 'weight_decay', 0.0)
    args.dropout = getattr(args, 'dropout', 0.0)
    args.opt_scheduler = getattr(args, 'opt_scheduler', 'none')
    args.opt_restart = getattr(args, 'opt_restart', 0)
    args.opt_decay_step = getattr(args, 'opt_decay_step', 100)
    args.opt_decay_rate = getattr(args, 'opt_decay_rate', 1.0)
    args.logdir = getattr(args, 'logdir', 'log')
    args.name_suffix = getattr(args, 'name_suffix', '')
    args.explainer_suffix = getattr(args, 'explainer_suffix', '')
    return args


def extract_neighborhood(G, node_idx, k=3):
    """Extract k-hop neighborhood."""
    subset = {node_idx}
    for _ in range(k):
        next_subset = set(subset)
        for n in subset:
            next_subset.update(G.neighbors(n))
        subset = next_subset
    return G.subgraph(list(subset)).copy()


def plot_case(G, mask, center_node, out_path, title, info_dict, hops=3):
    """Plot a case study visualization."""
    sub = extract_neighborhood(G, center_node, k=hops)
    nodes_sorted = sorted(sub.nodes())
    mapping = {n: i for i, n in enumerate(nodes_sorted)}
    # Keep mapping from rel_id to orig_id
    rev_mapping = {v: k for k, v in mapping.items()}
    sub_rel = nx.relabel_nodes(sub, mapping)

    # Node labels and colors - use relabeled IDs as keys
    labels = {}
    node_colors_list = []
    for rel_n in sub_rel.nodes():
        orig_n = rev_mapping[rel_n]
        feat = G.nodes[orig_n].get('feat')
        atom = '?'
        if isinstance(feat, (list, tuple, np.ndarray)):
            atom = str(int(np.argmax(feat)))
        if orig_n == center_node:
            node_colors_list.append('gold')
            labels[rel_n] = f"{atom}\n({orig_n})*"
        else:
            node_colors_list.append('lightblue')
            labels[rel_n] = f"{atom}\n({orig_n})"

    # Mask submatrix - use original node IDs for indexing
    size = len(nodes_sorted)
    sub_mask = np.zeros((size, size))
    for i, ni in enumerate(nodes_sorted):
        for j, nj in enumerate(nodes_sorted):
            if ni < mask.shape[0] and nj < mask.shape[1]:
                sub_mask[i, j] = mask[ni, nj]

    # Edge styling - use relabeled IDs
    edge_colors = []
    edge_widths = []
    for u, v in sub_rel.edges():
        # sub_mask uses original IDs (sorted order), but relabeled graph uses 0-based indices
        # nodes_sorted is already sorted, so node at index i has original ID nodes_sorted[i]
        ui = mapping.get(rev_mapping[u], u)
        vi = mapping.get(rev_mapping[v], v)
        w = sub_mask[ui, vi] if ui < sub_mask.shape[0] and vi < sub_mask.shape[1] else 0
        edge_colors.append(w)
        edge_widths.append(1 + float(w) * 4)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(sub_rel, seed=42)

    # Draw
    nx.draw_networkx_nodes(sub_rel, pos, node_color=node_colors_list, node_size=600, ax=ax)
    nx.draw_networkx_labels(sub_rel, pos, labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(sub_rel, pos, edge_color=edge_colors,
                           width=edge_widths, edge_cmap=plt.cm.Reds,
                           edge_vmin=0, edge_vmax=1, ax=ax)

    # Title
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Info text
    info_text = '\n'.join([f"{k}: {v}" for k, v in info_dict.items()])
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Edge Importance')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='mutag')
    parser.add_argument('--ckpt-path', default='')
    parser.add_argument('--text-tau', type=float, default=0.7)
    parser.add_argument('--explainer-epochs', type=int, default=20)
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outdir', default='analysis_results/case_study')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--output-dim', type=int, default=128)
    parser.add_argument('--text-dim', type=int, default=300)
    parser.add_argument('--image-dim', type=int, default=2048)
    parser.add_argument('--cuda', default='0')
    known, unknown = parser.parse_known_args()
    args = ensure_args(known)

    dataset_name = args.dataset
    dataset_folder = dataset_name
    file_prefix = dataset_name
    if dataset_name.lower() == 'mutag':
        dataset_folder = 'MUTAG'
        file_prefix = 'mutag'

    processed_dir = Path('data') / dataset_folder / 'processed'
    graph_path = processed_dir / f"{file_prefix}_graph.gpickle"
    labels_path = processed_dir / f"{file_prefix}_labels.csv"
    splits_path = processed_dir / 'split_masks.npz'

    print(f"Loading {dataset_name} graph...")
    G = nx.read_gpickle(graph_path)
    labels_df = pd.read_csv(labels_path, header=None)
    labels_list = labels_df[0].astype(int).tolist()
    num_nodes = G.number_of_nodes()

    # preprocess
    data = gengraph.preprocess_input_graph(G, labels_list)
    x = torch.tensor(data['feat'], dtype=torch.float)
    adj = torch.tensor(data['adj'], dtype=torch.float)

    device = torch.device(f"cuda:{args.cuda}" if args.gpu and torch.cuda.is_available() else "cpu")
    x = x.to(device)
    adj = adj.to(device)

    print("Loading multimodal features...")
    multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=args.gpu)
    full_text_list, full_image_list = [], []
    for node_id in range(num_nodes):
        node_data = multimodal_loader.get_node_data(node_id)
        full_text_list.append(node_data.get('text', ''))
        full_image_list.append(node_data.get('image_path', ''))

    # load model
    if not args.ckpt_path:
        raise ValueError("Please provide --ckpt-path")
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))

    sample_node = next(iter(G.nodes()))
    input_dim = len(G.nodes[sample_node]['feat']) if isinstance(G.nodes[sample_node]['feat'], (list, tuple, np.ndarray)) else int(G.nodes[sample_node]['feat'])

    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    text_dim = args.text_dim
    image_dim = args.image_dim

    if 'conv_first.weight' in sd:
        hidden_dim = sd['conv_first.weight'].shape[1]
    if 'conv_last.weight' in sd:
        output_dim = sd['conv_last.weight'].shape[1]
    if 'text_encoder.fc.weight' in sd:
        text_dim = sd['text_encoder.fc.weight'].shape[0]
    if 'image_encoder.fc.weight' in sd:
        image_dim = sd['image_encoder.fc.weight'].shape[0]

    num_gc_layers = 1
    conv_block_keys = [k for k in sd.keys() if k.startswith('conv_block.') and k.endswith('.weight')]
    if 'conv_first.weight' in sd and 'conv_last.weight' in sd:
        num_gc_layers = len(conv_block_keys) + 2

    fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')

    # Detect project_concat and concat_graph
    project_concat = True
    if 'pred_model.weight' in sd:
        pred_w_shape = sd['pred_model.weight'].shape[1]
        if fusion_method == 'concat':
            expected_concat_dim = output_dim + text_dim + image_dim
            if pred_w_shape == expected_concat_dim or pred_w_shape == 384:
                project_concat = False

    disable_text = 'text_encoder.fc.weight' not in sd
    disable_image = 'image_encoder.fc.weight' not in sd

    # Deduce disable flags from fusion concat_dim
    graph_dim = hidden_dim * (num_gc_layers - 1) + output_dim
    if 'fusion.fc.weight' in sd:
        concat_dim = sd['fusion.fc.weight'].shape[1]
        if concat_dim == graph_dim + text_dim:
            disable_image = True
        elif concat_dim == graph_dim + image_dim:
            disable_text = True
        elif concat_dim != graph_dim + text_dim + image_dim:
            graph_dim = output_dim
            if concat_dim == graph_dim + text_dim:
                disable_image = True
            elif concat_dim == graph_dim + image_dim:
                disable_text = True

    print(f"Model: hidden={hidden_dim}, output={output_dim}, text={text_dim}, image={image_dim}, fusion={fusion_method}")
    print(f"Disable: text={disable_text}, image={disable_image}, project_concat={project_concat}")

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=output_dim,
        label_dim=int(max(labels_list)) + 1, num_layers=num_gc_layers, bn=False, args=args,
        text_dim=text_dim, image_dim=image_dim, fusion_method=fusion_method,
        disable_text=disable_text, disable_image=disable_image,
        project_concat=project_concat
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    # predictions
    print("Computing predictions...")
    with torch.no_grad():
        model.fusion.disable_text = False
        model.fusion.disable_image = False
        ypred_full, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
        if ypred_full.dim() == 3 and ypred_full.shape[0] == 1:
            ypred_full = ypred_full.squeeze(0)
        prob_full_all = F.softmax(ypred_full, dim=-1).cpu()

        model.fusion.disable_text = True
        model.fusion.disable_image = False
        ypred_notext, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
        if ypred_notext.dim() == 3 and ypred_notext.shape[0] == 1:
            ypred_notext = ypred_notext.squeeze(0)
        prob_notext_all = F.softmax(ypred_notext, dim=-1).cpu()

    # test idx
    test_idx = []
    if splits_path.exists():
        npz = np.load(splits_path)
        test_idx = np.asarray(npz.get('test', [])).astype(np.int64).tolist()
    if not test_idx:
        test_idx = list(range(int(num_nodes * 0.8), num_nodes))

    # Create explainer
    test_label_np = np.array(labels_list, dtype=int)
    pred_fake_np = prob_full_all.unsqueeze(0).numpy()

    explainer = MultimodalExplainer(
        model=model,
        adj=data['adj'],
        feat=data['feat'],
        label=np.expand_dims(test_label_np, axis=0),
        pred=pred_fake_np,
        train_idx=[],
        args=args,
        writer=None,
        print_training=False,
        graph_mode=False,
        graph_idx=0,
        use_uma=False,
    )
    explainer.args.multimodal_loader = multimodal_loader

    # Find cases
    found = {'hard': None, 'soft': None}
    TEXT_TAU = args.text_tau

    print(f"Searching for cases with tau={TEXT_TAU}...")

    for i, idx in enumerate(test_idx):
        probs_graph = prob_notext_all[idx].clone()
        probs_full = prob_full_all[idx].clone()
        vals, _ = torch.topk(probs_graph, 2)
        margin = (vals[0] - vals[1]).item()

        explainer.args.num_epochs = args.explainer_epochs
        masked_adj, text_score, image_score = explainer.explain(
            node_idx=idx, graph_idx=0, graph_mode=False, unconstrained=False
        )
        text_score = float(text_score)
        soft_fused_probs = (1.0 - text_score) * probs_graph + text_score * probs_full

        pred_notext = int(torch.argmax(probs_graph).item())
        pred_soft = int(torch.argmax(soft_fused_probs).item())
        gt = int(test_label_np[idx])

        # Hard bypass candidate
        if found['hard'] is None and margin > TEXT_TAU:
            found['hard'] = {
                'node': idx, 'margin': margin, 'text_score': text_score,
                'pred_graph': pred_notext, 'pred_full': int(torch.argmax(probs_full).item()),
                'pred_soft': pred_soft, 'gt': gt
            }
            found['hard']['mask'] = masked_adj.astype(float)

        # Soft correction candidate
        if found['soft'] is None and pred_notext != gt and pred_soft == gt:
            found['soft'] = {
                'node': idx, 'margin': margin, 'text_score': text_score,
                'pred_graph': pred_notext, 'pred_full': int(torch.argmax(probs_full).item()),
                'pred_soft': pred_soft, 'gt': gt
            }
            found['soft']['mask'] = masked_adj.astype(float)

        if found['hard'] is not None and found['soft'] is not None:
            break

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_idx)}...")

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    summary = {'dataset': dataset_name, 'text_tau': TEXT_TAU, 'hard_case': None, 'soft_case': None}

    if found['hard']:
        n = found['hard']['node']
        info = {
            'Node': n,
            'Margin': f"{found['hard']['margin']:.3f}",
            'Text Score': f"{found['hard']['text_score']:.3f}",
            'GT Label': found['hard']['gt'],
            'Pred (Graph)': found['hard']['pred_graph'],
            'Pred (Soft)': found['hard']['pred_soft'],
        }
        out_img = outdir / f'case_hard_bypass_node_{n}.png'
        title = f'Case 1: Hard Router Bypass (tau={TEXT_TAU})'
        plot_case(G, found['hard']['mask'], n, out_img, title, info, hops=args.hops)
        summary['hard_case'] = {k: v for k, v in found['hard'].items() if k != 'mask'}

    if found['soft']:
        n = found['soft']['node']
        info = {
            'Node': n,
            'Margin': f"{found['soft']['margin']:.3f}",
            'Text Score': f"{found['soft']['text_score']:.3f}",
            'GT Label': found['soft']['gt'],
            'Pred (Graph)': found['soft']['pred_graph'],
            'Pred (Soft)': found['soft']['pred_soft'],
        }
        out_img = outdir / f'case_soft_correct_node_{n}.png'
        title = f'Case 2: Soft Fusion Correction (tau={TEXT_TAU})'
        plot_case(G, found['soft']['mask'], n, out_img, title, info, hops=args.hops)
        summary['soft_case'] = {k: v for k, v in found['soft'].items() if k != 'mask'}

    # Save summary
    with open(outdir / 'case_study_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {outdir / 'case_study_summary.json'}")


if __name__ == '__main__':
    main()

