import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import accuracy_score

import configs
from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
import utils.io_utils as io_utils
import utils.gengraph as gengraph

def run_adaptive_gating_evaluation(args):
    dataset_name = args.dataset
    
    # Handle case-sensitivity for Linux environments
    dataset_folder = dataset_name
    file_prefix = dataset_name
    if dataset_name.lower() == "mutag":
        dataset_folder = "MUTAG"
        file_prefix = "mutag"
        
    processed_dir = Path("data") / dataset_folder / "processed"
    graph_path = processed_dir / f"{file_prefix}_graph.gpickle"
    labels_path = processed_dir / f"{file_prefix}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"

    print(f"Loading {dataset_name} dataset for Adaptive Gating Evaluation...")
    G = nx.read_gpickle(graph_path)
    labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)
    num_classes = int(labels_np.max()) + 1
    
    test_idx = []
    if splits_path.exists():
        npz = np.load(splits_path)
        test_idx = np.asarray(npz.get("test", [])).astype(np.int64).tolist()
    
    if not test_idx:
        num_nodes = G.number_of_nodes()
        test_idx = list(range(int(num_nodes * 0.8), num_nodes))

    sample_node = next(iter(G.nodes()))
    input_dim = len(G.nodes[sample_node]["feat"]) if isinstance(G.nodes[sample_node]["feat"], (np.ndarray, list, tuple)) else int(G.nodes[sample_node]["feat"])

    data = gengraph.preprocess_input_graph(G, labels_list)
    x = torch.tensor(data["feat"], dtype=torch.float)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    
    device = torch.device(f"cuda:{args.cuda}" if args.gpu and torch.cuda.is_available() else "cpu")
    x = x.to(device)
    adj = adj.to(device)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)

    print("Loading multimodal features...")
    multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=args.gpu)
    text_list, image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        text_list.append(node_data["text"])
        image_list.append(node_data["image_path"])
    
    print("Initializing Multi-modal GNN Model...")
    # Load metadata from checkpoint to align dimensions
    ckpt_path = args.ckpt_path if hasattr(args, 'ckpt_path') and args.ckpt_path else None
    if not ckpt_path and hasattr(args, 'ckptdir') and args.ckptdir and os.path.isfile(args.ckptdir):
        ckpt_path = args.ckptdir
        
    project_concat = True
    if ckpt_path:
        tmp_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = tmp_ckpt.get("model_state", tmp_ckpt.get("state_dict", tmp_ckpt))
        args.hidden_dim = sd['conv_block.0.weight'].shape[0] if 'conv_block.0.weight' in sd else 128
        args.output_dim = sd['conv_last.weight'].shape[1] if 'conv_last.weight' in sd else 128
        args.text_dim = sd['text_encoder.fc.weight'].shape[0] if 'text_encoder.fc.weight' in sd else 128
        args.image_dim = sd['image_encoder.fc.weight'].shape[0] if 'image_encoder.fc.weight' in sd else 128
        args.fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')
        
        # Check num_layers based on conv_block
        conv_block_keys = [k for k in sd.keys() if k.startswith('conv_block.') and k.endswith('.weight')]
        args.num_gc_layers = 1
        if 'conv_first.weight' in sd and 'conv_last.weight' in sd:
            args.num_gc_layers = len(conv_block_keys) + 2
        
        # Check if project_concat was false during training
        project_concat = True
        if 'pred_model.weight' in sd:
            pred_w_shape = sd['pred_model.weight'].shape[1]
            out_dim = args.output_dim
            # If pred_model inputs sum_of_components, project_concat was False.
            if args.fusion_method == 'concat':
                expected_concat_dim = args.output_dim + args.text_dim + args.image_dim
                if pred_w_shape == expected_concat_dim or pred_w_shape == 384:
                    project_concat = False

        args.disable_text = False
        args.disable_image = False
        
        # Deduce disable flags from fusion concat_dim
        graph_dim = args.hidden_dim * (args.num_gc_layers - 1) + args.output_dim
        if 'fusion.fc.weight' in sd:
            concat_dim = sd['fusion.fc.weight'].shape[1]
            if concat_dim == graph_dim + args.text_dim: args.disable_image = True
            elif concat_dim == graph_dim + args.image_dim: args.disable_text = True
            elif concat_dim != graph_dim + args.text_dim + args.image_dim:
                # For GcnEncoderNode, pred_input_dim uses output_dim, not the progressive concatenation
                graph_dim = args.output_dim
                if concat_dim == graph_dim + args.text_dim: args.disable_image = True
                elif concat_dim == graph_dim + args.image_dim: args.disable_text = True
        elif 'pred_model.weight' in sd and not project_concat:
            pred_w_shape = sd['pred_model.weight'].shape[1]
            if pred_w_shape == graph_dim + args.text_dim: args.disable_image = True
            elif pred_w_shape == graph_dim + args.image_dim: args.disable_text = True
            elif pred_w_shape != graph_dim + args.text_dim + args.image_dim:
                # Try with output_dim only
                graph_dim = args.output_dim
                if pred_w_shape == graph_dim + args.text_dim: args.disable_image = True
                elif pred_w_shape == graph_dim + args.image_dim: args.disable_text = True

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
        label_dim=num_classes, num_layers=args.num_gc_layers, bn=args.bn, args=args,
        text_dim=args.text_dim, image_dim=args.image_dim, fusion_method=args.fusion_method,
        project_concat=project_concat, disable_text=getattr(args, 'disable_text', False),
        disable_image=getattr(args, 'disable_image', False)
    )
    model = model.to(device)

    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    
    model.eval()

    # Define conditions to test
    conditions = {
        "full": {"disable_text": False, "disable_image": False},
        "no_text": {"disable_text": True, "disable_image": False},
        "no_image": {"disable_text": False, "disable_image": True},
        "no_both": {"disable_text": True, "disable_image": True}
    }

    results_dict = {}
    with torch.no_grad():
        for cond_name, flags in conditions.items():
            model.fusion.disable_text = flags["disable_text"]
            model.fusion.disable_image = flags["disable_image"]
            ypred, _ = model(x, adj, text_feats=text_list, image_feats=image_list)
            if ypred.dim() == 3 and ypred.shape[0] == 1:
                ypred = ypred.squeeze(0)
            probs = F.softmax(ypred, dim=-1).cpu()
            results_dict[cond_name] = probs

    # Evaluation Baselines
    gt_labels = labels_tensor[test_idx].numpy()
    
    # 1. Base (Full Multimodal) Accuracy
    preds_full = torch.argmax(results_dict["full"][test_idx], dim=-1).numpy()
    acc_full = accuracy_score(gt_labels, preds_full)
    
    # 2. Always Drop Text Accuracy
    preds_no_text = torch.argmax(results_dict["no_text"][test_idx], dim=-1).numpy()
    acc_no_text = accuracy_score(gt_labels, preds_no_text)

    # 3. Always Drop Image Accuracy
    preds_no_image = torch.argmax(results_dict["no_image"][test_idx], dim=-1).numpy()
    acc_no_image = accuracy_score(gt_labels, preds_no_image)
    
    # 4. Always Drop Both Accuracy
    preds_no_both = torch.argmax(results_dict["no_both"][test_idx], dim=-1).numpy()
    acc_no_both = accuracy_score(gt_labels, preds_no_both)

    # 5. OURS: Heuristic Confidence Gating (Adaptive Gating)
    # IDEA: We don't know the ground truth. We only know the *max probability* (confidence) of each prediction.
    # If adding a modality significantly *decreases* the model's confidence compared to not having it, 
    # it indicates that the modality is adding noise/uncertainty.
    
    final_adaptive_preds = []
    
    # 阈值：去模态后的置信度比全模态的置信度高出多少，我们就认为全模态被“拖后腿”了
    # 尝试基于置信度反转 (Entropy/Margin) 
    
    for i, idx in enumerate(test_idx):
        prob_full = results_dict["full"][idx]
        prob_no_text = results_dict["no_text"][idx]
        prob_no_image = results_dict["no_image"][idx]
        
        # 计算 Top-1 置信度 vs Top-2 的 Margin (分类置信裕度)
        def get_margin(p):
            sorted_p, _ = torch.sort(p, descending=True)
            if len(sorted_p) > 1:
                return (sorted_p[0] - sorted_p[1]).item()
            return sorted_p[0].item()

        margin_full = get_margin(prob_full)
        margin_no_text = get_margin(prob_no_text)
        margin_no_image = get_margin(prob_no_image)
        
        # 门控决策：谁的分类 Margin 最大就在推理时采用谁的预测
        # 加入一个保守阈值 TAU_CONF：只有当单模态的置信度显著高于全模态时，才执行模态抛弃
        TAU_CONF = getattr(args, 'tau_conf', 0.1)  # 改为可配置
        best_mode = "full"
        max_margin = margin_full
        
        if margin_no_text > max_margin + TAU_CONF:
            max_margin = margin_no_text
            best_mode = "no_text"
            
        if margin_no_image > max_margin + TAU_CONF:
            max_margin = margin_no_image
            best_mode = "no_image"
            
        if best_mode == "no_text":
            final_adaptive_preds.append(torch.argmax(prob_no_text).item())
        elif best_mode == "no_image":
            final_adaptive_preds.append(torch.argmax(prob_no_image).item())
        else:
            final_adaptive_preds.append(torch.argmax(prob_full).item())
            
    acc_adaptive = accuracy_score(gt_labels, final_adaptive_preds)
    
    # 统计采用各分支的次数
    counts = {"full": 0, "no_text": 0, "no_image": 0}
    for p_full, p_notext, p_noimage, p_final in zip(preds_full, preds_no_text, preds_no_image, final_adaptive_preds):
        if p_final == p_notext and p_final != p_full: counts["no_text"] += 1
        elif p_final == p_noimage and p_final != p_full: counts["no_image"] += 1
        else: counts["full"] += 1
    
    print("\n" + "="*50)
    print("🎯 Inference-Time Adaptive Gating Results (No Ground Truth Required) 🎯")
    print("="*50)
    print(f"1. Baseline (Always Full)         : {acc_full:.4f}")
    print(f"2. Ablation (Always Drop Text)    : {acc_no_text:.4f}")
    print(f"3. Ablation (Always Drop Image)   : {acc_no_image:.4f}")
    print(f"4. Ablation (Graph Only)          : {acc_no_both:.4f}")
    print("-" * 50)
    print(f"🚀 OURS (Heuristic Confidence Gating) : {acc_adaptive:.4f} (TAU={TAU_CONF})")
    print(f"   -> Choices made: Full: {counts['full']}, Drop Text: {counts['no_text']}, Drop Image: {counts['no_image']}")
    print("="*50)
    
    if acc_adaptive > acc_full:
        print(f"\n✅ SUCCESS! Adaptive gating improved accuracy by {(acc_adaptive - acc_full)*100:.2f} percentage points!")
    else:
        print("\n⚠️ Adaptive gating didn't beat the baseline yet. We may need to tune TAU_CONFIDENCE or use Entropy.")

if __name__ == "__main__":
    import sys
    # Intercept --ckpt-path hack before configs.arg_parse() complains
    ckpt_path_val = None
    if "--ckpt-path" in sys.argv:
        idx = sys.argv.index("--ckpt-path")
        ckpt_path_val = sys.argv[idx + 1]
        # Remove them so configs.arg_parse doesn't crash on unrecognized arg
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    tau_conf_val = 0.1
    if "--tau-conf" in sys.argv:
        idx = sys.argv.index("--tau-conf")
        tau_conf_val = float(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    args = configs.arg_parse()
    args.multimodal = True
    
    if ckpt_path_val is not None:
        args.ckpt_path = ckpt_path_val
    elif not hasattr(args, 'ckpt_path'):
        args.ckpt_path = ""
        
    args.tau_conf = tau_conf_val

    # Fallbacks if not provided by CLI
    if not hasattr(args, 'dataset') or not args.dataset: args.dataset = "mutag"
    if not hasattr(args, 'hidden_dim') or not args.hidden_dim: args.hidden_dim = 128
    if not hasattr(args, 'output_dim') or not args.output_dim: args.output_dim = 128
    if not hasattr(args, 'text_dim') or not args.text_dim: args.text_dim = 300
    if not hasattr(args, 'image_dim') or not args.image_dim: args.image_dim = 2048
    if not hasattr(args, 'num_gc_layers') or not args.num_gc_layers: args.num_gc_layers = 3
    if not hasattr(args, 'fusion_method') or not args.fusion_method: args.fusion_method = "concat"

    run_adaptive_gating_evaluation(args)
