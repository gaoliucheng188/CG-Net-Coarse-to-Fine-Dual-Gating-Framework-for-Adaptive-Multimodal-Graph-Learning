import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

import configs
from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
import utils.io_utils as io_utils

def extract_logits(args):
    # 1. 准备和加载数据 (类似 train.py 中的 mutag_task)
    dataset_name = "MUTAG"
    processed_dir = Path("data") / dataset_name / "processed"
    graph_path = processed_dir / "mutag_graph.gpickle"
    labels_path = processed_dir / "mutag_labels.csv"
    splits_path = processed_dir / "split_masks.npz"

    print(f"Loading {dataset_name} dataset...")
    G = nx.read_gpickle(graph_path)
    labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)
    num_classes = int(labels_np.max()) + 1
    
    # 提取测试集索引 (如果没有 split_masks, 默认选取最后 20% 作为测试)
    test_idx = []
    if splits_path.exists():
        npz = np.load(splits_path)
        test_idx = np.asarray(npz.get("test", [])).astype(np.int64).tolist()
    
    if not test_idx:
        print("Warning: split_masks.npz not found or missing test split. Using final 20% as test.")
        num_nodes = G.number_of_nodes()
        test_idx = list(range(int(num_nodes * 0.8), num_nodes))

    sample_node = next(iter(G.nodes()))
    feat_value = G.nodes[sample_node]["feat"]
    input_dim = len(feat_value) if isinstance(feat_value, (np.ndarray, list, tuple)) else int(feat_value)

    # 处理图特征 (全图输入，但只提取 test_idx 的结果)
    # GCN 往往需要全图 adj 做传播
    import utils.gengraph as gengraph
    data = gengraph.preprocess_input_graph(G, labels_list)
    x = torch.tensor(data["feat"], dtype=torch.float)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    
    device = torch.device(f"cuda:{args.cuda}" if args.gpu and torch.cuda.is_available() else "cpu")
    x = x.to(device)
    adj = adj.to(device)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)

    # 2. 加载多模态特征
    print("Loading multimodal features...")
    multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=args.gpu)
    text_list, image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        text_list.append(node_data["text"])
        image_list.append(node_data["image_path"])
    
    # 3. 初始化并加载模型
    print("Initializing Multi-modal GNN Model...")
    
    # 优先解析 ckpt 里的元数据
    ckpt_path = args.ckptdir if args.ckptdir and os.path.isfile(args.ckptdir) else (args.ckptdir + ".pth.tar" if args.ckptdir and os.path.isfile(args.ckptdir + ".pth.tar") else None)
    if ckpt_path:
        # Pre-load to inspect the shape and align our instantiation
        tmp_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = tmp_ckpt.get("model_state", tmp_ckpt.get("state_dict", tmp_ckpt))
        
        args.hidden_dim = sd['conv_block.0.weight'].shape[0] if 'conv_block.0.weight' in sd else 128
        args.output_dim = sd['pred_model.weight'].shape[1] if 'pred_model.weight' in sd else 128
        
        if 'text_encoder.fc.weight' in sd:
            t_out, t_in = sd['text_encoder.fc.weight'].shape
            args.text_dim = t_out
        else:
            args.text_dim = 128
            
        if 'image_encoder.fc.weight' in sd:
            i_out, i_in = sd['image_encoder.fc.weight'].shape
            args.image_dim = i_out
        else:
            args.image_dim = 128
            
        if 'fusion.query_proj.weight' in sd:
            args.fusion_method = 'attention'
        elif 'fusion.bilinear.weight' in sd:
            args.fusion_method = 'bilinear'
        elif 'fusion.fc.weight' in sd and sd['fusion.fc.weight'].shape[1] == args.hidden_dim:
            # Often if project_concat=False and just graph_dim is mapped, but since fusion_fc weight is [128,128]
            # it might be attention without query proj or just concat with different dimensions
            args.fusion_method = 'concat'
        else:
            args.fusion_method = 'concat'
            
        print(f"[Auto-Align] Resolved arch -> hidden: {args.hidden_dim}, out: {args.output_dim}, text: {args.text_dim}, image: {args.image_dim}, fusion: {args.fusion_method}")
    else:
        if not hasattr(args, "text_dim"): args.text_dim = 128
        if not hasattr(args, "image_dim"): args.image_dim = 128
        if not hasattr(args, "fusion_method"): args.fusion_method = 'concat'

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
        fusion_method=args.fusion_method
    )
    model = model.to(device)

    # 从指定路径加载模型参数 (你需要指定 --ckptdir)
    if args.ckptdir and args.ckptdir != 'ckpt':
        # Check standard checkpoints
        ckpt_path = None
        # 如果 args.ckptdir 是一个文件，直接使用
        if os.path.isfile(args.ckptdir):
            ckpt_path = args.ckptdir
        else:
            for ext in ["", ".pth", ".pt", ".pth.tar"]:
                if os.path.isfile(args.ckptdir + ext):
                    ckpt_path = args.ckptdir + ext
                    break
        
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}...")
            # PyTorch >= 2.6 security restriction fix
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # 兼容带有 module. 或者其他嵌套前缀的权重字典
            state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {args.ckptdir}. Using RANDOM initialized weights for debugging!")
    else:
        print("Warning: No --ckptdir provided. Using RANDOM initialized weights for debugging!")

    model.eval()

    # 4. 定义四种干预组: full, drop-text, drop-image, drop-both
    conditions = {
        "full": {"disable_text": False, "disable_image": False},
        "no_text": {"disable_text": True, "disable_image": False},
        "no_image": {"disable_text": False, "disable_image": True},
        "no_both": {"disable_text": True, "disable_image": True}
    }

    results_dict = {}

    with torch.no_grad():
        for cond_name, flags in conditions.items():
            print(f"Running inference for condition: {cond_name}...")
            # 动态修改 fusion 的门控状态
            model.fusion.disable_text = flags["disable_text"]
            model.fusion.disable_image = flags["disable_image"]

            ypred, _ = model(x, adj, text_feats=text_list, image_feats=image_list)
            
            # ypred shape: [num_nodes, num_classes] (假设是一维结构，基于 train.py 推测)
            if ypred.dim() == 3 and ypred.shape[0] == 1:
                ypred = ypred.squeeze(0)
            
            # 取Softmax获得概率
            probs = F.softmax(ypred, dim=-1).cpu()
            
            results_dict[cond_name] = probs

    # 5. 汇总记录为 CSV
    print("Aggregating results for Test Set...")
    records = []
    
    for idx in test_idx:
        gt_label = labels_tensor[idx].item()
        
        record = {
            "node_id": idx,
            "ground_truth": gt_label
        }
        
        # 提取各个 condition 下对 *正确类别(GT)* 的预测概率
        for cond_name in conditions.keys():
            probs = results_dict[cond_name][idx]
            gt_prob = probs[gt_label].item()
            pred_class = torch.argmax(probs).item()
            
            record[f"prob_{cond_name}"] = gt_prob
            record[f"pred_{cond_name}"] = pred_class
            
        records.append(record)

    df = pd.DataFrame(records)
    
    # 简单计算一个分型基准词缀: Delta Text 和 Delta Image
    df['delta_text'] = df['prob_full'] - df['prob_no_text']  # 文本的独立贡献
    df['delta_image'] = df['prob_full'] - df['prob_no_image'] # 图像的独立贡献

    out_dir = Path("analysis_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mutag_audit_results.csv"
    df.to_csv(out_path, index=False)
    
    print(f"Extraction complete! Results saved to: {out_path}")
    print("\nPreview of extracted probabilities:")
    print(df[['node_id', 'ground_truth', 'prob_full', 'prob_no_text', 'prob_no_image', 'prob_no_both']].head())

if __name__ == "__main__":
    args = configs.arg_parse()
    # 强制开启多模态开关
    args.multimodal = True
    args.dataset = "mutag"  # 默认使用 mutag
    
    # 修正维度以匹配训练好的 checkpoint (mutag_base_h128_o128)
    args.hidden_dim = 128
    args.output_dim = 128
    args.text_dim = 300
    args.image_dim = 2048
    args.num_gc_layers = 3
    args.fusion_method = "concat" # 假设是concat，或根据具体需要调整
    
    extract_logits(args)

