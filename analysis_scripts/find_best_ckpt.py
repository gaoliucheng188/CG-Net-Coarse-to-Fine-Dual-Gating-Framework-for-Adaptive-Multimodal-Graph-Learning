import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import accuracy_score

from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
import utils.gengraph as gengraph

class Args:
    pass

def map_args_from_sd(args, sd):
    args.hidden_dim = sd['conv_block.0.weight'].shape[0] if 'conv_block.0.weight' in sd else 128
    args.output_dim = sd['conv_last.weight'].shape[1] if 'conv_last.weight' in sd else 128
    args.text_dim = sd['text_encoder.fc.weight'].shape[0] if 'text_encoder.fc.weight' in sd else 300
    args.image_dim = sd['image_encoder.fc.weight'].shape[0] if 'image_encoder.fc.weight' in sd else 2048
    args.fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')
    
    conv_block_keys = [k for k in sd.keys() if k.startswith('conv_block.') and k.endswith('.weight')]
    args.num_gc_layers = 1
    if 'conv_first.weight' in sd and 'conv_last.weight' in sd:
        args.num_gc_layers = len(conv_block_keys) + 2
        
    project_concat = True
    if 'pred_model.weight' in sd:
        pred_w_shape = sd['pred_model.weight'].shape[1]
        if args.fusion_method == 'concat':
            expected_concat_dim = args.output_dim + args.text_dim + args.image_dim
            if pred_w_shape == expected_concat_dim or pred_w_shape == 384:
                project_concat = False
                
    args.disable_text = False
    args.disable_image = False
    graph_dim = args.hidden_dim * (args.num_gc_layers - 1) + args.output_dim
    
    if 'fusion.fc.weight' in sd:
        concat_dim = sd['fusion.fc.weight'].shape[1]
        if concat_dim == graph_dim + args.text_dim: args.disable_image = True
        elif concat_dim == graph_dim + args.image_dim: args.disable_text = True
    elif 'pred_model.weight' in sd and not project_concat:
        pred_w_shape = sd['pred_model.weight'].shape[1]
        if pred_w_shape == graph_dim + args.text_dim: args.disable_image = True
        elif pred_w_shape == graph_dim + args.image_dim: args.disable_text = True
                
    return args, project_concat

def find_best_mutag_ckpts():
    print("Loading MUTAG Dataset and Features...")
    processed_dir = Path("data/MUTAG/processed")
    G = nx.read_gpickle(processed_dir / "mutag_graph.gpickle")
    labels_list = pd.read_csv(processed_dir / "mutag_labels.csv", header=None)[0].astype(int).tolist()
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)
    num_classes = int(labels_np.max()) + 1
    
    splits_path = processed_dir / "split_masks.npz"
    test_idx = np.asarray(np.load(splits_path).get("test", [])).astype(np.int64).tolist()
    
    sample_node = next(iter(G.nodes()))
    input_dim = len(G.nodes[sample_node]["feat"]) if isinstance(G.nodes[sample_node]["feat"], (np.ndarray, list, tuple)) else int(G.nodes[sample_node]["feat"])
    
    data = gengraph.preprocess_input_graph(G, labels_list)
    x = torch.tensor(data["feat"], dtype=torch.float)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    adj = adj.to(device)
    
    multimodal_loader = MultimodalDataset(dataset_name="mutag", gpu=torch.cuda.is_available())
    full_text_list, full_image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        full_text_list.append(node_data.get("text", ""))
        full_image_list.append(node_data.get("image_path", ""))
        
    gt_labels = labels_np[test_idx].tolist()
    
    print("\nScanning 'ckpt/' for .pth.tar files...")
    ckpt_files = glob.glob('ckpt/**/*.pth.tar', recursive=True)
    
    # 按文件修改时间排序，最新的在前面
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(ckpt_files)} checkpoint files. Starting fast evaluation (newest first)...\n")
    
    print(f"{'Checkpoint Name':<40} | {'Full Acc':<10} | {'No Text Acc':<11} | {'Status'}")
    print("-" * 80)
    
    results = []
    
    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            
            args = Args()
            args.bn = False
            args.num_epochs = 1
            args.gpu = torch.cuda.is_available()
            args.cuda = 0
            args.multimodal = True
            args.text_tau = 0.5
            args, project_concat = map_args_from_sd(args, sd)
            
            model = MultimodalGcnEncoderNode(
                input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
                label_dim=num_classes, num_layers=args.num_gc_layers, bn=args.bn, args=args,
                text_dim=args.text_dim, image_dim=args.image_dim, fusion_method=args.fusion_method,
                project_concat=project_concat, disable_text=args.disable_text,
                disable_image=args.disable_image
            )
            model.load_state_dict(sd, strict=False)
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                # Full 
                model.fusion.disable_text = False
                ypred_full, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
                if ypred_full.dim() == 3: ypred_full = ypred_full.squeeze(0)
                pred_full = torch.argmax(ypred_full, dim=-1).cpu().numpy()[test_idx].tolist()
                
                # No Text
                model.fusion.disable_text = True
                ypred_notext, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
                if ypred_notext.dim() == 3: ypred_notext = ypred_notext.squeeze(0)
                pred_notext = torch.argmax(ypred_notext, dim=-1).cpu().numpy()[test_idx].tolist()
                
            acc_full = accuracy_score(gt_labels, pred_full)
            acc_notext = accuracy_score(gt_labels, pred_notext)
            
            name = Path(ckpt_path).parts[-2] + '/' + Path(ckpt_path).name 
            status = "GOOD" if acc_full > 0.90 else "WEAK"
            if status == "GOOD" and acc_full >= 0.92:
                status = "⭐ PERFECT"
                
            results.append({'ckpt': name, 'full': acc_full, 'notext': acc_notext})
            print(f"{name:<40} | {acc_full:.4f}{' '*4} | {acc_notext:.4f}{' '*4} | {status}")
            
        except Exception as e:
            import traceback
            print(f"\n=======================")
            print(f"Failed on: {ckpt_path}")
            traceback.print_exc()
            print(f"=======================\n")
            print(f"{Path(ckpt_path).name:<40} | {'ERROR':<10} | {'ERROR':<11} | Skipped")
            break

    print("\nSearch complete.")

if __name__ == "__main__":
    find_best_mutag_ckpts()
