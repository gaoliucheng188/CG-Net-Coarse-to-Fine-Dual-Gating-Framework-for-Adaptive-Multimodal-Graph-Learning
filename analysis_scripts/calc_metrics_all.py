import os
import sys
import torch
import numpy as np
import pickle
import networkx as nx
import argparse
import glob
import re
import pandas as pd
from types import SimpleNamespace

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.io_utils as io_utils
import utils.graph_utils as graph_utils
import models
from models_multimodal import MultimodalGcnEncoderNode, MultimodalGcnEncoderGraph
from data.multimodal_loader import MultimodalDataset

def load_model_and_data(ckpt_path, args):
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda' if args.gpu else 'cpu', weights_only=False)
    
    cg = checkpoint['cg']
    input_dim = cg['feat'].shape[-1]
    num_classes = cg['pred'].shape[-1]
    
    is_graph_classification = len(cg['adj'].shape) == 3
    
    pred_weight = checkpoint['model_state'].get('pred_model.weight')
    project_concat = True
    concat_graph = True
    
    # Heuristic for dimension mismatch
    if pred_weight is not None:
        ckpt_pred_dim = pred_weight.shape[1]
        if ckpt_pred_dim != args.output_dim:
            if ckpt_pred_dim == 384 and args.output_dim == 128:
                args.text_dim = 128
                args.image_dim = 128
                args.fusion_method = 'concat'
                project_concat = False
                concat_graph = False
    
    if args.multimodal:
        if is_graph_classification:
             model = MultimodalGcnEncoderGraph(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.output_dim,
                label_dim=num_classes,
                num_layers=args.num_gc_layers,
                bn=args.bn,
                args=args,
                text_dim=args.text_dim,
                image_dim=args.image_dim,
                fusion_method=args.fusion_method,
                force_text_fallback=args.force_text_fallback,
                disable_text=args.disable_text,
                disable_image=args.disable_image,
                project_concat=project_concat,
                concat=concat_graph,
            )
        else:
            model = MultimodalGcnEncoderNode(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.output_dim,
                label_dim=num_classes,
                num_layers=args.num_gc_layers,
                bn=args.bn,
                args=args,
                text_dim=args.text_dim,
                image_dim=args.image_dim,
                fusion_method=args.fusion_method,
                force_text_fallback=args.force_text_fallback,
                disable_text=args.disable_text,
                disable_image=args.disable_image,
                project_concat=project_concat,
                concat=concat_graph,
            )
    else:
        if is_graph_classification:
            model = models.GcnEncoderGraph(
                input_dim,
                args.hidden_dim,
                args.output_dim,
                num_classes,
                args.num_gc_layers,
                bn=args.bn,
                args=args,
            )
        else:
            model = models.GcnEncoderNode(
                input_dim,
                args.hidden_dim,
                args.output_dim,
                num_classes,
                args.num_gc_layers,
                bn=args.bn,
                args=args,
            )
    
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
        
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()
    
    return model, cg, is_graph_classification

def extract_neighborhood(adj, feat, labels, node_idx, graph_idx=0, n_hops=3):
    rows, cols = np.where(adj > 0)
    graph = nx.Graph()
    graph.add_edges_from(zip(rows, cols))
    
    neighbors = {node_idx}
    current_shell = {node_idx}
    
    for _ in range(n_hops):
        next_shell = set()
        for node in current_shell:
            if node in graph:
                next_shell.update(graph.neighbors(node))
        next_shell = next_shell - neighbors
        neighbors.update(next_shell)
        current_shell = next_shell
        
    neighbors = sorted(list(neighbors))
    try:
        node_idx_new = neighbors.index(node_idx)
    except ValueError:
        node_idx_new = 0
    
    sub_adj = adj[neighbors][:, neighbors]
    sub_feat = feat[neighbors]
    sub_label = labels[neighbors] if labels is not None else None
    
    return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

def calculate_metrics(model, adj, x, node_idx, mask, args, text_feats=None, image_feats=None):
    device = next(model.parameters()).device
    adj_t = torch.tensor(adj, dtype=torch.float).to(device)
    x_t = torch.tensor(x, dtype=torch.float).to(device)
    
    if len(adj_t.shape) == 2:
        adj_t = adj_t.unsqueeze(0)
        x_t = x_t.unsqueeze(0)
    
    t_feats = None
    i_feats = None
    if args.multimodal:
        t_feats = text_feats
        i_feats = image_feats

    # 1. Original Prediction (Confidence)
    with torch.no_grad():
        out = model(x_t, adj_t, text_feats=t_feats, image_feats=i_feats)
        if isinstance(out, tuple): ypred = out[0]
        else: ypred = out
        
        if ypred.dim() == 2:
            if ypred.shape[0] == 1: pred_orig = torch.softmax(ypred[0], dim=0)
            else: pred_orig = torch.softmax(ypred[node_idx], dim=0)
        elif ypred.dim() == 3:
             pred_orig = torch.softmax(ypred[0, node_idx], dim=0)
        else:
             pred_orig = torch.softmax(ypred[0], dim=0)

        label_idx = torch.argmax(pred_orig).item()

    # 2. Sparsity
    mask_t = torch.tensor(mask, dtype=torch.float).to(device)
    mask_bin = (mask_t > 0.5).float()
    sparsity = 1.0 - mask_bin.mean().item()

    # 3. Fidelity+ (Keep): 1 if prediction unchanged when keeping important elements
    adj_keep = adj_t * mask_bin
    with torch.no_grad():
        out = model(x_t, adj_keep, text_feats=t_feats, image_feats=i_feats)
        if isinstance(out, tuple): ypred_keep = out[0]
        else: ypred_keep = out

        if ypred_keep.dim() == 2:
            if ypred_keep.shape[0] == 1: pred_keep = torch.softmax(ypred_keep[0], dim=0)
            else: pred_keep = torch.softmax(ypred_keep[node_idx], dim=0)
        elif ypred_keep.dim() == 3:
             pred_keep = torch.softmax(ypred_keep[0, node_idx], dim=0)
        else:
             pred_keep = torch.softmax(ypred_keep[0], dim=0)

        label_keep = torch.argmax(pred_keep).item()
        fid_plus = 1.0 if label_keep == label_idx else 0.0

    # 4. Fidelity- (Remove): 1 if prediction changed when removing important elements
    adj_remove = adj_t * (1 - mask_bin)
    with torch.no_grad():
        out = model(x_t, adj_remove, text_feats=t_feats, image_feats=i_feats)
        if isinstance(out, tuple): ypred_remove = out[0]
        else: ypred_remove = out

        if ypred_remove.dim() == 2:
            if ypred_remove.shape[0] == 1: pred_remove = torch.softmax(ypred_remove[0], dim=0)
            else: pred_remove = torch.softmax(ypred_remove[node_idx], dim=0)
        elif ypred_remove.dim() == 3:
             pred_remove = torch.softmax(ypred_remove[0, node_idx], dim=0)
        else:
             pred_remove = torch.softmax(ypred_remove[0], dim=0)

        label_remove = torch.argmax(pred_remove).item()
        fid_minus = 1.0 if label_remove != label_idx else 0.0

    return fid_plus, fid_minus, sparsity, None

def get_args_for_model(model_name):
    # Default args
    args = SimpleNamespace(
        dataset='mutag',
        hidden_dim=128,
        output_dim=128,
        num_gc_layers=3,
        bn=False,
        gpu=True,
        multimodal=True,
        text_dim=300,
        image_dim=2048,
        fusion_method='concat',
        force_text_fallback=False,
        disable_text=False,
        disable_image=False,
        dropout=0.2,
        method='base',
        bias=True
    )
    
    if 'nlm_chem' in model_name:
        args.dataset = 'nlm_chem'
    elif 'pqaref' in model_name:
        args.dataset = 'pqaref'
        args.text_dim = 1024 # PQAREF text dim
    
    if '_A3' in model_name:
        args.multimodal = False
    elif '_A1' in model_name:
        args.disable_text = True
    elif '_A2' in model_name:
        args.disable_image = True
        
    return args

def main():
    mask_files = glob.glob(os.path.join('log', 'masked_adj_*.npy'))
    print(f"Found {len(mask_files)} mask files.")
    
    # Group by model
    model_groups = {}
    for f in mask_files:
        # Extract model name
        # Pattern: masked_adj_{MODEL_NAME}_explain_{EXPLAINER_SUFFIX}node_idx...
        # We need to be careful.
        # Example: masked_adj_mutag_base_h128_o128_A0_explain_A0node_idx_0graph_idx_-1.npy
        # Model: mutag_base_h128_o128_A0
        
        match = re.search(r'masked_adj_(.+)_explain', os.path.basename(f))
        if match:
            model_name = match.group(1)
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(f)
    
    results = []
    
    for model_name, files in model_groups.items():
        print(f"Processing model: {model_name} ({len(files)} masks)")
        
        ckpt_path = f'ckpt/{model_name}.pth.tar'
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found. Skipping.")
            continue
            
        args = get_args_for_model(model_name)
        
        try:
            model, cg, is_graph_classification = load_model_and_data(ckpt_path, args)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
            
        mm_loader = None
        if args.multimodal:
            mm_loader = MultimodalDataset(args.dataset, gpu=args.gpu)
            
        for mask_file in files:
            try:
                match = re.search(r'explain_(.+)node_idx_(\d+)graph_idx_(-?\d+)', mask_file)
                explainer_suffix = "unknown"
                if match:
                    explainer_suffix = match.group(1)
                    node_idx = int(match.group(2))
                    graph_idx = int(match.group(3))
                else:
                    # Fallback regex
                    match = re.search(r'node_idx_(\d+)graph_idx_(-?\d+)', mask_file)
                    if match:
                        node_idx = int(match.group(1))
                        graph_idx = int(match.group(2))
                    else:
                        continue

                if graph_idx == -1: graph_idx = 0
                
                mask = np.load(mask_file)
                
                if is_graph_classification:
                    adj = cg['adj'][graph_idx]
                    feat = cg['feat'][graph_idx]
                else:
                    adj = cg['adj'][0]
                    feat = cg['feat'][0]
                
                if mask.shape[0] == adj.shape[0]:
                    sub_adj = adj
                    sub_feat = feat
                    sub_node_idx = node_idx
                else:
                    sub_node_idx, sub_adj, sub_feat, _, neighbors = extract_neighborhood(adj, feat, None, node_idx, graph_idx)
                    if mask.shape[0] != sub_adj.shape[0]:
                        print(f"Shape mismatch for {mask_file}. Skipping.")
                        continue
                
                t_feat = None
                i_feat = None
                if mm_loader:
                    if is_graph_classification:
                        graph_data = mm_loader.get_graph_data(graph_idx)
                        t_feat = graph_data.get('text')
                        i_feat = graph_data.get('image_path')
                    else:
                        # Node classification logic (simplified)
                        pass

                fid_p, fid_m, sparsity, conf = calculate_metrics(model, sub_adj, sub_feat, sub_node_idx, mask, args, text_feats=t_feat, image_feats=i_feat)
                
                results.append({
                    'model': model_name,
                    'explainer': explainer_suffix,
                    'file': os.path.basename(mask_file),
                    'fidelity_plus': fid_p,
                    'fidelity_minus': fid_m,
                    'sparsity': sparsity,
                    'confidence': conf
                })
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
                continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv('analysis_results/ablation_metrics_full.csv', index=False)
        print("Saved results to analysis_results/ablation_metrics_full.csv")
        print(df.groupby(['model', 'explainer'])[['fidelity_plus', 'fidelity_minus', 'sparsity', 'confidence']].mean())

if __name__ == '__main__':
    main()

