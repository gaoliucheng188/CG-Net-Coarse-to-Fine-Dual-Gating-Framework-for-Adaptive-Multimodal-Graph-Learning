import os
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
import utils.gengraph as gengraph

def find_cases(dataset_name, tau_conf, target_type):
    # 路径解析
    dataset_folder = dataset_name
    file_prefix = dataset_name
    if dataset_name.lower() == "mutag":
        dataset_folder = "MUTAG"
        file_prefix = "mutag"
        
    ckpt_path = f"ckpt/{dataset_name}_base_h20_o20.pth.tar"
    processed_dir = Path("data") / dataset_folder / "processed"
    graph_path = processed_dir / f"{file_prefix}_graph.gpickle"
    labels_path = processed_dir / f"{file_prefix}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"

    # 加载图和标签
    G = nx.read_gpickle(graph_path)
    labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)
    
    test_idx = []
    if splits_path.exists():
        npz = np.load(splits_path)
        test_idx = np.asarray(npz.get("test", [])).astype(np.int64).tolist()
    if not test_idx:
        test_idx = list(range(int(G.number_of_nodes() * 0.8), G.number_of_nodes()))

    # 构建并加载模型
    print(f"🔄 正在加载 {dataset_name} 模型及多模态特征...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 简易参数组装
    class DummyArgs: pass
    args = DummyArgs()
    args.gpu = torch.cuda.is_available()
    args.method = "base"
    args.mask_act = "sigmoid"
    args.mask_bias = False
    args.name_suffix = ""
    args.explainer_suffix = ""
    args.bias = True
    
    tmp_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = tmp_ckpt.get("model_state", tmp_ckpt.get("state_dict", tmp_ckpt))
    
    args.hidden_dim = sd['conv_block.0.weight'].shape[0] if 'conv_block.0.weight' in sd else 128
    args.output_dim = sd['conv_last.weight'].shape[1] if 'conv_last.weight' in sd else 128
    args.text_dim = sd['text_encoder.fc.weight'].shape[0] if 'text_encoder.fc.weight' in sd else 128
    args.image_dim = sd['image_encoder.fc.weight'].shape[0] if 'image_encoder.fc.weight' in sd else 128
    args.fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')
    
    conv_block_keys = [k for k in sd.keys() if k.startswith('conv_block.') and k.endswith('.weight')]
    args.num_gc_layers = len(conv_block_keys) + 2 if 'conv_first.weight' in sd else 1
    args.bn = False # 可忽略
    
    input_dim = len(G.nodes[0]["feat"]) if isinstance(G.nodes[0]["feat"], (np.ndarray, list, tuple)) else 10
    num_classes = int(labels_np.max()) + 1

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
        label_dim=num_classes, num_layers=args.num_gc_layers, bn=False, args=args,
        text_dim=args.text_dim, image_dim=args.image_dim, fusion_method=args.fusion_method,
        project_concat=True, disable_text=False, disable_image=False
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    # 准备输入
    data = gengraph.preprocess_input_graph(G, labels_list)
    x = torch.tensor(data["feat"], dtype=torch.float).to(device)
    adj = torch.tensor(data["adj"], dtype=torch.float).to(device)
    
    multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=args.gpu)
    text_list, image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        text_list.append(node_data["text"])
        image_list.append(node_data["image_path"])

    results_dict = {}
    with torch.no_grad():
        for cond_name, d_text in [("full", False), ("no_text", True)]:
            model.fusion.disable_text = d_text
            ypred, _ = model(x, adj, text_feats=text_list, image_feats=image_list)
            if ypred.dim() == 3 and ypred.shape[0] == 1: ypred = ypred.squeeze(0)
            results_dict[cond_name] = F.softmax(ypred, dim=-1).cpu()

    # 寻找 Target Cases
    found_cases = []
    print(f"\n🎯 开始搜寻目标样本 (Target Type: {target_type}) ")
    print("-" * 60)
    
    for i, idx in enumerate(test_idx):
        prob_full = results_dict["full"][idx]
        prob_no_text = results_dict["no_text"][idx]
        gt = labels_np[idx]
        
        pred_full = torch.argmax(prob_full).item()
        pred_no_text = torch.argmax(prob_no_text).item()
        
        def get_margin(p):
            sorted_p, _ = torch.sort(p, descending=True)
            return (sorted_p[0] - sorted_p[1]).item() if len(sorted_p) > 1 else sorted_p[0].item()

        margin_full = get_margin(prob_full)
        margin_no_text = get_margin(prob_no_text)
        
        # 门控决策逻辑
        best_mode = "full"
        if margin_no_text > margin_full + tau_conf:
            best_mode = "no_text"
            
        final_pred = pred_no_text if best_mode == "no_text" else pred_full
        
        # 判断是否符合条件
        if target_type == "rescue":
            # 纠错局：Baseline预测错，剔除文本后门控预测对
            if pred_full != gt and final_pred == gt and best_mode == "no_text":
                found_cases.append({"node_id": idx, "label": gt, "pred_full": pred_full, "pred_ours": final_pred})
        
        elif target_type == "audit":
            # 节流局：Baseline预测对，门控成功剔除了文本，并且预测依然对
            if pred_full == gt and final_pred == gt and best_mode == "no_text":
                found_cases.append({"node_id": idx, "label": gt, "pred_full": pred_full, "pred_ours": final_pred})
                
    if not found_cases:
        print("😭 没有找到符合条件的样本。")
    else:
        print(f"🎉 成功找到 {len(found_cases)} 个符合条件的样本！前 5 个 ID 为：")
        for case in found_cases[:5]:
            print(f"   -> Graph Node ID: {case['node_id']} | True Label: {case['label']} | Base Pred: {case['pred_full']} | Ours Pred: {case['pred_ours']} (Dropped Text)")

if __name__ == "__main__":
    # MUTAG 纠错局找寻 (Tau建议用我们测出来的最佳阈值 0.3)
    find_cases("mutag", tau_conf=0.3, target_type="rescue")
    
    # NLM_CHEM 节流局找寻 (Tau 可以在平稳区间任选，比如 0.1)
    find_cases("nlm_chem", tau_conf=0.1, target_type="audit")

