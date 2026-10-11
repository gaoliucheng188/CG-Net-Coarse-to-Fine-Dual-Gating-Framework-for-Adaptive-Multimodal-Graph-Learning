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

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.io_utils as io_utils
import utils.graph_utils as graph_utils
import models
from models_multimodal import MultimodalGcnEncoderNode, MultimodalGcnEncoderGraph
from data.multimodal_loader import MultimodalDataset

def load_model_and_data(ckpt_path, args):
    print(f"Loading model and data from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda' if args.gpu else 'cpu', weights_only=False)
    
    cg = checkpoint['cg']
    input_dim = cg['feat'].shape[-1]
    num_classes = cg['pred'].shape[-1]
    
    # Determine if graph classification or node classification
    is_graph_classification = len(cg['adj'].shape) == 3
    
    sd = checkpoint['model_state']
    
    # Extract dimensions from checkpoint
    # GraphConv weight shape: [input_dim, output_dim]
    # conv_first.weight: [input_dim, hidden_dim]  -> shape[0]=input_dim, shape[1]=hidden_dim
    # conv_last.weight: [hidden_dim, output_dim] -> shape[0]=hidden_dim, shape[1]=output_dim
    # conv_block.X.weight: [hidden_dim, hidden_dim] -> shape[0]=hidden_dim, shape[1]=hidden_dim
    
    # Extract input_dim from conv_first.weight
    input_dim = cg['feat'].shape[-1]  # Use data dimension as fallback
    if 'conv_first.weight' in sd:
        input_dim = sd['conv_first.weight'].shape[0]  # conv_first: [input_dim, hidden_dim]
    
    # Extract hidden_dim from conv_first.weight
    hidden_dim = args.hidden_dim
    if 'conv_first.weight' in sd:
        hidden_dim = sd['conv_first.weight'].shape[1]  # conv_first: [input_dim, hidden_dim]
    
    # Extract output_dim from conv_last.weight
    args.output_dim = args.output_dim  # Default from args
    if 'conv_last.weight' in sd:
        args.output_dim = sd['conv_last.weight'].shape[1]  # conv_last: [hidden_dim, output_dim]
        
    print(f"Extracted dims from checkpoint: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={args.output_dim}")
    
    # Extract num_gc_layers by counting conv layers in checkpoint
    # Pattern: conv_first, conv_block.0, conv_block.1, ..., conv_last
    num_gc_layers = 1  # conv_first
    i = 0
    while f'conv_block.{i}.weight' in sd:
        num_gc_layers += 1
        i += 1
    num_gc_layers += 1  # conv_last
    
    # Debug: Also check for other possible layer naming patterns
    if num_gc_layers == 2:  # Only conv_first and conv_last, no conv_block
        # Check for alternative naming
        for key in sd.keys():
            if 'conv' in key.lower() and 'weight' in key.lower():
                print(f"DEBUG: Found layer key: {key} -> shape {sd[key].shape}")
    
    args.num_gc_layers = num_gc_layers
    print(f"Detected num_gc_layers from checkpoint: {num_gc_layers}")
    
    # Debug: Calculate expected fusion dimensions
    graph_dim_concat = hidden_dim * (num_gc_layers - 1) + args.output_dim
    graph_dim_no_concat = args.output_dim
    print(f"DEBUG: graph_dim_concat={graph_dim_concat}, graph_dim_no_concat={graph_dim_no_concat}")
        
    # Extract modality dimensions from checkpoint
    # text_encoder.fc.weight shape: [output_dim, input_dim] = [text_dim, text_embedding_dim]
    # image_encoder.fc.weight shape: [output_dim, input_dim] = [image_dim, image_embedding_dim]
    if 'text_encoder.fc.weight' in sd:
        args.text_dim = sd['text_encoder.fc.weight'].shape[0]  # Output dimension
        print(f"Text encoder output dim from checkpoint: {args.text_dim}")
    elif args.multimodal:
        args.text_dim = 300
        
    if 'image_encoder.fc.weight' in sd:
        args.image_dim = sd['image_encoder.fc.weight'].shape[0]  # Output dimension
        print(f"Image encoder output dim from checkpoint: {args.image_dim}")
    elif args.multimodal:
        args.image_dim = 0  # Assume disabled if not in checkpoint
        
    args.fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')
    
    # Determine if modalities are enabled (check if encoder weights exist in checkpoint)
    disable_text = 'text_encoder.fc.weight' not in sd
    disable_image = 'image_encoder.fc.weight' not in sd
    
    print(f"Modality detection: disable_text={disable_text}, disable_image={disable_image}")
    
    # Determine concat_graph by comparing fusion input dim with expected dims
    project_concat = True
    concat_graph = False  # Default assumption
    
    if 'fusion.fc.weight' in sd:
        fusion_in_dim = sd['fusion.fc.weight'].shape[1]
        
        # Calculate graph_feature_dim based on num_layers and concat
        # From models.py: pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim (if concat)
        graph_dim_concat = hidden_dim * (num_gc_layers - 1) + args.output_dim
        graph_dim_no_concat = args.output_dim
        
        # Effective modality dims (considering disable flags)
        effective_text_dim = 0 if disable_text else args.text_dim
        effective_image_dim = 0 if disable_image else args.image_dim
        
        # Expected fusion input dims
        expected_with_concat = graph_dim_concat + effective_text_dim + effective_image_dim
        expected_without_concat = graph_dim_no_concat + effective_text_dim + effective_image_dim
        
        print(f"Fusion analysis: fusion_in_dim={fusion_in_dim}")
        print(f"  graph_dim_concat={graph_dim_concat}, graph_dim_no_concat={graph_dim_no_concat}")
        print(f"  effective_text_dim={effective_text_dim}, effective_image_dim={effective_image_dim}")
        print(f"  expected_concat={expected_with_concat}, expected_no_concat={expected_without_concat}")
        
        # Match to determine concat_graph
        if fusion_in_dim == expected_with_concat:
            concat_graph = True
            print("Detected: concat_graph=True")
        elif fusion_in_dim == expected_without_concat:
            concat_graph = False
            print("Detected: concat_graph=False")
        else:
            print(f"Warning: No exact match. Trying to infer configuration...")
            # Try different combinations to find the correct one
            found_config = False
            for test_concat in [True, False]:
                for test_disable_text in [True, False]:
                    for test_disable_image in [True, False]:
                        test_graph_dim = graph_dim_concat if test_concat else graph_dim_no_concat
                        test_text = 0 if test_disable_text else args.text_dim
                        test_image = 0 if test_disable_image else args.image_dim
                        test_expected = test_graph_dim + test_text + test_image
                        if fusion_in_dim == test_expected:
                            concat_graph = test_concat
                            disable_text = test_disable_text
                            disable_image = test_disable_image
                            print(f"Detected: concat={test_concat}, disable_text={test_disable_text}, disable_image={test_disable_image}")
                            found_config = True
                            break
                    if found_config:
                        break
                if found_config:
                    break
            
            if not found_config:
                print(f"Warning: Could not determine concat_graph. Using concat_graph=True as fallback.")
                concat_graph = True
    
    # Update args.hidden_dim with the extracted value from checkpoint
    args.hidden_dim = hidden_dim
    print(f"Using hidden_dim={args.hidden_dim} from checkpoint")
    print(f"Detected model config: fusion_method={args.fusion_method}, disable_text={disable_text}, disable_image={disable_image}, project_concat={project_concat}, concat={concat_graph}")
    
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
                disable_text=disable_text,
                disable_image=disable_image,
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
                disable_text=disable_text,
                disable_image=disable_image,
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
    """Returns the neighborhood of a given node."""
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
    
    # Map old indices to new indices
    try:
        node_idx_new = neighbors.index(node_idx)
    except ValueError:
        # Should not happen if node_idx is in graph
        node_idx_new = 0
    
    sub_adj = adj[neighbors][:, neighbors]
    sub_feat = feat[neighbors]
    sub_label = labels[neighbors] if labels is not None else None
    
    return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

def calculate_fidelity(model, adj, x, node_idx, mask, args, text_feats=None, image_feats=None):
    device = next(model.parameters()).device
    
    adj_t = torch.tensor(adj, dtype=torch.float).to(device)
    x_t = torch.tensor(x, dtype=torch.float).to(device)
    
    # Ensure dimensions are correct for the model
    # If model expects batch dim, add it
    if len(adj_t.shape) == 2:
        adj_t = adj_t.unsqueeze(0)
        x_t = x_t.unsqueeze(0)
    
    t_feats = None
    i_feats = None
    if args.multimodal:
        t_feats = text_feats
        i_feats = image_feats

    # 1. Original Prediction
    with torch.no_grad():
        out = model(x_t, adj_t, text_feats=t_feats, image_feats=i_feats)
        if isinstance(out, tuple):
            ypred = out[0]
        else:
            ypred = out
        
        # Handle output shape
        if ypred.dim() == 2: # (batch, classes) or (nodes, classes)
            if ypred.shape[0] == 1: # Graph classification or single node batch
                pred_orig = torch.softmax(ypred[0], dim=0)
            else: # Node classification full graph
                pred_orig = torch.softmax(ypred[node_idx], dim=0)
        elif ypred.dim() == 3: # (batch, nodes, classes)
             pred_orig = torch.softmax(ypred[0, node_idx], dim=0)
        else:
             pred_orig = torch.softmax(ypred[0], dim=0) # Fallback

        label_idx = torch.argmax(pred_orig).item()

    # Threshold mask
    mask_t = torch.tensor(mask, dtype=torch.float).to(device)
    mask_bin = (mask_t > 0.5).float()

    # 2. Fidelity+ (Keep): measure if prediction stays consistent when keeping important edges
    # Fidelity+ = 1 if argmax prediction remains the same after keeping important elements
    adj_keep = adj_t * mask_bin
    with torch.no_grad():
        out = model(x_t, adj_keep, text_feats=t_feats, image_feats=i_feats)
        if isinstance(out, tuple):
            ypred_keep = out[0]
        else:
            ypred_keep = out

        if ypred_keep.dim() == 2:
            if ypred_keep.shape[0] == 1:
                pred_keep = torch.softmax(ypred_keep[0], dim=0)
            else:
                pred_keep = torch.softmax(ypred_keep[node_idx], dim=0)
        elif ypred_keep.dim() == 3:
             pred_keep = torch.softmax(ypred_keep[0, node_idx], dim=0)
        else:
             pred_keep = torch.softmax(ypred_keep[0], dim=0)

        label_keep = torch.argmax(pred_keep).item()
        # Fidelity+ = 1 if prediction unchanged (consistent), 0 otherwise
        fid_plus = 1.0 if label_keep == label_idx else 0.0

    # 3. Fidelity- (Remove): measure if prediction changes when removing important edges
    # Fidelity- = 1 if argmax prediction changes after removing important elements
    adj_remove = adj_t * (1 - mask_bin)
    with torch.no_grad():
        out = model(x_t, adj_remove, text_feats=t_feats, image_feats=i_feats)
        if isinstance(out, tuple):
            ypred_remove = out[0]
        else:
            ypred_remove = out

        if ypred_remove.dim() == 2:
            if ypred_remove.shape[0] == 1:
                pred_remove = torch.softmax(ypred_remove[0], dim=0)
            else:
                pred_remove = torch.softmax(ypred_remove[node_idx], dim=0)
        elif ypred_remove.dim() == 3:
             pred_remove = torch.softmax(ypred_remove[0, node_idx], dim=0)
        else:
             pred_remove = torch.softmax(ypred_remove[0], dim=0)

        label_remove = torch.argmax(pred_remove).item()
        # Fidelity- = 1 if prediction changed (removed critical elements), 0 otherwise
        fid_minus = 1.0 if label_remove != label_idx else 0.0

    return fid_plus, fid_minus

def parse_variant_from_filename(filename):
    """
    Parse variant information from mask filename.

    Examples:
    - masked_adj_mutag_h128_o128_gated_node_idx_0...    -> Gated (new format)
    - masked_adj_mutag_h128_o128_baseline_node_idx_0... -> Baseline (new format)
    - masked_adj_mutag_base_h128_o128_explainnode_idx_0... -> Baseline (old format)
    - masked_adj_mutag_base_h20_o20_explainnode_idx_0... -> Legacy_h20o20
    """
    import os
    import re
    basename = os.path.basename(filename).replace('.npy', '')
    parts = basename.split('_')

    # Check for new format: *_gated_* or *_baseline_*
    if '_gated_' in basename:
        return 'Gated'
    if '_baseline_' in basename:
        return 'Baseline'

    # Extract hidden/output dims to identify legacy experiments
    hidden_match = re.search(r'h(\d+)_o(\d+)', basename)
    if hidden_match:
        h_dim, o_dim = hidden_match.groups()
        if h_dim != '128' or o_dim != '128':
            return f'Legacy_h{h_dim}o{o_dim}'

    # Find 'explain' in parts - for old format
    explain_idx = -1
    for i, part in enumerate(parts):
        if part.startswith('explain'):
            explain_idx = i
            break

    if explain_idx == -1:
        return 'Baseline'

    # Look for variant pattern in parts near 'explain'
    if explain_idx + 1 < len(parts):
        next_part = parts[explain_idx + 1]
        match = re.match(r'^([AB]\d+)', next_part)
        if match:
            return f'Gated_{match.group(1)}'

    if explain_idx >= 1:
        prev_part = parts[explain_idx - 1]
        if re.match(r'^[AB]\d+$', prev_part, re.IGNORECASE):
            return f'Gated_{prev_part.upper()}'

    if explain_idx + 2 < len(parts):
        next_next_part = parts[explain_idx + 2]
        match = re.match(r'^([AB]\d+)', next_next_part)
        if match:
            return f'Baseline_{match.group(1)}'

    return 'Baseline'


def is_gated_from_filename(filename):
    """Determine if the mask is from a gated (with routing) or baseline explainer."""
    import os
    import re
    basename = os.path.basename(filename).lower()

    # Check for new format
    if '_gated_' in basename.lower():
        return True
    if '_baseline_' in basename.lower():
        return False

    parts = os.path.basename(filename).replace('.npy', '').split('_')

    # Check if it's legacy h20_o20 or other non-standard
    hidden_match = re.search(r'h(\d+)_o(\d+)', filename)
    if hidden_match:
        h_dim, o_dim = hidden_match.groups()
        if h_dim != '128' or o_dim != '128':
            return False  # Legacy experiments - not relevant for current analysis

    # Find 'explain' in parts
    explain_idx = -1
    for i, part in enumerate(parts):
        if part.startswith('explain'):
            explain_idx = i
            break

    if explain_idx == -1:
        return False

    # If there's a variant marker BEFORE 'explain', it's gated
    if explain_idx >= 1:
        prev_part = parts[explain_idx - 1]
        if re.match(r'^[AB]\d+$', prev_part, re.IGNORECASE):
            return True

    # If 'explain' is followed by a variant pattern like "A0node", it's gated
    if explain_idx + 1 < len(parts):
        next_part = parts[explain_idx + 1]
        if re.match(r'^[ab]\d+', next_part):
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mutag')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--output-dim', type=int, default=128)
    parser.add_argument('--num-gc-layers', type=int, default=3)
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multimodal', action='store_true', default=True)
    parser.add_argument('--text-dim', type=int, default=300)
    parser.add_argument('--image-dim', type=int, default=2048)
    parser.add_argument('--fusion-method', default='bilinear')
    parser.add_argument('--force-text-fallback', action='store_true')
    parser.add_argument('--disable-text', action='store_true')
    parser.add_argument('--disable-image', action='store_true')
    parser.add_argument('--mask-dir', default='analysis_results/stats_data')
    parser.add_argument('--method', default='base')
    parser.add_argument('--name-suffix', default='')
    parser.add_argument('--bias', dest='bias', action='store_const', const=False, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--assign-ratio', type=float, default=0.1)
    parser.add_argument('--num-pool', type=int, default=1)
    parser.add_argument('--linkpred', action='store_true')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--group-by-variant', action='store_true')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    args = parser.parse_args()
    
    # Load Checkpoint
    ckpt_path = args.model_path if args.model_path else f'ckpt/{args.dataset}_base_h128_o128.pth.tar'
    if not os.path.exists(ckpt_path):
        ckpts = glob.glob(f'ckpt/{args.dataset}*.pth.tar')
        if ckpts:
            ckpt_path = ckpts[0]
        else:
            print("Checkpoint not found.")
            return
    
    model, cg, is_graph_classification = load_model_and_data(ckpt_path, args)
    
    # Load Multimodal Data Loader if needed
    mm_loader = None
    if args.multimodal:
        mm_loader = MultimodalDataset(args.dataset, gpu=args.gpu)

    # Find masks
    mask_files = glob.glob(os.path.join(args.mask_dir, f'masked_adj_{args.dataset}*.npy'))
    if not mask_files:
        mask_files = glob.glob(os.path.join('log', f'masked_adj_{args.dataset}*.npy'))
        
    print(f"Found {len(mask_files)} mask files.")
    
    results = []
    
    for mask_file in mask_files:
        match = re.search(r'node_idx_(\d+)[_]?graph_idx_(-?\d+)', mask_file)
        if not match:
            continue
            
        node_idx = int(match.group(1))
        graph_idx = int(match.group(2))
        
        mask = np.load(mask_file)
        
        if graph_idx == -1:
            graph_idx = 0
            
        # Get data from cg
        if is_graph_classification:
            adj = cg['adj'][graph_idx]
            feat = cg['feat'][graph_idx]
            # label = cg['label'][graph_idx]
        else:
            adj = cg['adj'][0] # Assuming single graph for node classification
            feat = cg['feat'][0]
            # label = cg['label'][0]
            
        # Handle subgraph extraction if mask is smaller than full graph
        # For baseline masks (1x1 placeholder), we use full graph
        if mask.shape == (1, 1):
            # Baseline mask is a placeholder, use full graph
            sub_adj = adj
            sub_feat = feat
            sub_node_idx = node_idx
            neighbors = list(range(adj.shape[0]))
        elif mask.shape[0] == adj.shape[0]:
            sub_adj = adj
            sub_feat = feat
            sub_node_idx = node_idx
        else:
            # Re-extract neighborhood to match mask
            # Note: This assumes the mask was generated using the same extraction logic
            sub_node_idx, sub_adj, sub_feat, _, neighbors = extract_neighborhood(adj, feat, None, node_idx, graph_idx)
            if mask.shape[0] != sub_adj.shape[0]:
                print(f"Shape mismatch for {mask_file}: Mask {mask.shape}, Subgraph {sub_adj.shape}. Skipping.")
                continue
        
        # Get multimodal features
        t_feat = None
        i_feat = None
        
        if mm_loader:
            if is_graph_classification:
                graph_data = mm_loader.get_graph_data(graph_idx)
                t_feat = graph_data.get('text')
                i_feat = graph_data.get('image_path')
            else:
                # Node classification
                # We need features for the nodes in the subgraph
                # For MultimodalGcnEncoderNode, it expects a list of strings
                # We need to get text for each node in the subgraph
                # sub_node_idx is the index in the subgraph
                # neighbors contains original node indices
                
                t_feat = []
                i_feat = []
                for original_node_idx in neighbors:
                    node_data = mm_loader.get_node_data(original_node_idx)
                    t_feat.append(node_data.get('text', ''))
                    i_feat.append(node_data.get('image_path'))

        fid_p, fid_m = calculate_fidelity(model, sub_adj, sub_feat, sub_node_idx, mask, args, text_feats=t_feat, image_feats=i_feat)

        # 解析变体信息
        is_gated = is_gated_from_filename(mask_file)
        variant = parse_variant_from_filename(mask_file)

        # Skip Legacy variants (non-128x128 experiments) - 跳过非标准实验
        if variant.startswith('Legacy'):
            print(f"Skipped (Legacy): {os.path.basename(mask_file)}")
            continue

        results.append({
            'file': os.path.basename(mask_file),
            'gated': is_gated,
            'variant': variant,
            'fidelity_plus': fid_p,
            'fidelity_minus': fid_m
        })

        print(f"Processed {os.path.basename(mask_file)}: Gated={is_gated}, Variant={variant}, F+={fid_p:.4f}, F-={fid_m:.4f}")

    if results:
        df = pd.DataFrame(results)
        if args.output:
            out_path = args.output
        else:
            out_path = os.path.join(args.mask_dir, 'fidelity_results.csv')
        df.to_csv(out_path, index=False)
        print(f"Saved fidelity results to {out_path}")
        print("Average Fidelity+:", df['fidelity_plus'].mean())
        print("Average Fidelity-:", df['fidelity_minus'].mean())
    else:
        print("No results computed.")

if __name__ == '__main__':
    main()

