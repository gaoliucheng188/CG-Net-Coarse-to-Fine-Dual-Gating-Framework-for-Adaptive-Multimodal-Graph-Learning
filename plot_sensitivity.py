import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import Namespace

import configs
from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
import utils.gengraph as gengraph
from explainer.explain_multimodal import MultimodalExplainer

def get_ckpt_path(dataset):
    """根据已知最佳模型的命名规则自动寻找对应的checkpoint"""
    # 根据截图中最新的 12月4日 权重文件命名（它们直接是 .pth.tar 文件，而不是文件夹）
    paths_to_try = [
        f"ckpt/{dataset}_base_h128_o128_A0.pth.tar",   # 最新优先 (12月4日)
        f"ckpt/{dataset}_base_h128_o128__A0.pth.tar",  # 次新 (11月28日)
        f"ckpt/{dataset}_base_h128_o128_A1.pth.tar",   # 备选
        f"ckpt/{dataset}_base_h128_o128.pth.tar",      # 兜底
        f"ckpt/{dataset}_base_h20_o20.pth.tar"
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return p
            
    # 终极兜底机制：递归搜索 ckpt/ 目录下包含该 dataset 且为 pth/pth.tar 的文件
    if os.path.exists("ckpt"):
        for root, dirs, files in os.walk("ckpt"):
            for f in files:
                if dataset in f and (f.endswith(".pth") or f.endswith(".pth.tar")):
                    candidate = os.path.join(root, f)
                    print(f"\n⚠️ Auto-fallback found: {candidate}")
                    return candidate
                    
    raise FileNotFoundError(f"Cannot find best checkpoint for {dataset}. Please ensure 'ckpt/' exists and has a trained model.")

def analyze_dataset_sensitivity(dataset_name, gpu, base_args):
    """收集一个数据集在所有测试节点上的 explainer text_scores"""
    # 路径与兼容性处理
    dataset_folder = dataset_name
    file_prefix = dataset_name
    if dataset_name.lower() == "mutag":
        dataset_folder = "MUTAG"
        file_prefix = "mutag"

    processed_dir = Path("data") / dataset_folder / "processed"
    graph_path = processed_dir / f"{file_prefix}_graph.gpickle"
    labels_path = processed_dir / f"{file_prefix}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"

    print(f"\n[{dataset_name.upper()}] Loading Dataset...")
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
    
    device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
    x = x.to(device)
    adj = adj.to(device)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)

    multimodal_loader = MultimodalDataset(dataset_name=dataset_name, gpu=(gpu is not None))
    full_text_list, full_image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        full_text_list.append(node_data.get("text", ""))
        full_image_list.append(node_data.get("image_path", ""))
    
    ckpt_path = get_ckpt_path(dataset_name)
    print(f"[{dataset_name.upper()}] Best ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    # Dynamically build args namespace for model initialization
    args = Namespace(**vars(base_args))
    args.dataset = dataset_name
    args.multimodal_loader = multimodal_loader
    args.hidden_dim = sd['conv_block.0.weight'].shape[0] if 'conv_block.0.weight' in sd else 128
    args.output_dim = sd['conv_last.weight'].shape[1] if 'conv_last.weight' in sd else 128
    args.text_dim = sd['text_encoder.fc.weight'].shape[0] if 'text_encoder.fc.weight' in sd else 128
    args.image_dim = sd['image_encoder.fc.weight'].shape[0] if 'image_encoder.fc.weight' in sd else 128
    args.fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')
    
    conv_block_keys = [k for k in sd.keys() if k.startswith('conv_block.') and k.endswith('.weight')]
    args.num_gc_layers = 1
    if 'conv_first.weight' in sd and 'conv_last.weight' in sd:
        args.num_gc_layers = len(conv_block_keys) + 2

    # Robust dimension and architecture inference
    project_concat = True
    args.concat = True
    args.disable_text = False
    args.disable_image = False

    graph_dim_cT = args.hidden_dim * (args.num_gc_layers - 1) + args.output_dim
    graph_dim_cF = args.output_dim

    if 'pred_model.weight' in sd:
        pred_w_shape = sd['pred_model.weight'].shape[1]
        
        if args.fusion_method == 'concat':
            # Full multimodal
            if pred_w_shape == graph_dim_cT + args.text_dim + args.image_dim:
                project_concat, args.concat = False, True
            elif pred_w_shape == graph_dim_cF + args.text_dim + args.image_dim:
                project_concat, args.concat = False, False
            # Disable image
            elif pred_w_shape == graph_dim_cT + args.text_dim:
                project_concat, args.concat, args.disable_image = False, True, True
            elif pred_w_shape == graph_dim_cF + args.text_dim:
                project_concat, args.concat, args.disable_image = False, False, True
            # Disable text
            elif pred_w_shape == graph_dim_cT + args.image_dim:
                project_concat, args.concat, args.disable_text = False, True, True
            elif pred_w_shape == graph_dim_cF + args.image_dim:
                project_concat, args.concat, args.disable_text = False, False, True

    elif 'fusion.fc.weight' in sd:
        # projected concat
        concat_dim = sd['fusion.fc.weight'].shape[1]
        
        # Calculate exactly how much text/image dimension could be by subtracting graph features from concat_dim
        # 684 might be for example graph=640 (128x5) + text/image=44. Or graph=128 + text/image=556.
        # We need this to NOT fail on `model.load_state_dict(sd)`
        # If we can't match it exactly, we reconstruct the fusion dimension logic by forcefully syncing model.fusion.fc
        
        if concat_dim == graph_dim_cT + args.text_dim + args.image_dim:
            args.concat = True
        elif concat_dim == graph_dim_cF + args.text_dim + args.image_dim:
            args.concat = False
        elif concat_dim == graph_dim_cT + args.text_dim:
            args.concat, args.disable_image = True, True
        elif concat_dim == graph_dim_cF + args.text_dim:
            args.concat, args.disable_image = False, True
        elif concat_dim == graph_dim_cT + args.image_dim:
            args.concat, args.disable_text = True, True
        elif concat_dim == graph_dim_cF + args.image_dim:
            args.concat, args.disable_text = False, True
        else:
            # Fallback if text_dim and image_dim saved inside the weights are fundamentally different from 128
            # E.g., text_dim=44, graph_dim=640
            args.concat = (concat_dim > graph_dim_cT)

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
        label_dim=num_classes, num_layers=args.num_gc_layers, bn=args.bn, args=args,
        text_dim=args.text_dim, image_dim=args.image_dim, fusion_method=args.fusion_method,
        project_concat=project_concat, disable_text=args.disable_text,
        disable_image=args.disable_image, concat=args.concat
    )
    
    # CRITICAL FIX for structural differences mapped inside state_dict:
    # E.g., The text tokenizer output might be 44-dim instead of 128 like MUTAG.
    # The actual stored dimensions of fusion FC must perfectly match!
    # If the constructed multimodal layer doesn't match the checkpt, resync it!
    if 'fusion.fc.weight' in sd and model.fusion.fc is not None:
        ckpt_fc_in = sd['fusion.fc.weight'].shape[1]
        model.fusion.fc = torch.nn.Linear(ckpt_fc_in, args.output_dim)
        # also update model.pred_input_dim
        model.pred_input_dim = args.output_dim
        model.pred_model = model.build_pred_layers(model.pred_input_dim, [], num_classes)
        
    if 'pred_model.weight' in sd:
        ckpt_pred_in = sd['pred_model.weight'].shape[1]
        if hasattr(model, 'pred_model') and model.pred_input_dim != ckpt_pred_in:
            model.pred_input_dim = ckpt_pred_in
            model.pred_model = model.build_pred_layers(model.pred_input_dim, [], num_classes)

    
    # CRITICAL FIX: The GcnEncoderGraph gcn_forward unconditionally concatenates all layers
    # but forgets to truncate when concat=False in Node classification. We must monkey patch it
    # here to avoid the 640 vs 384 dimension crash during forward().
    if not getattr(args, 'concat', True):
        original_gcn_forward = model.gcn_forward
        def patched_gcn_forward(*c_args, **c_kwargs):
            x_tensor, adj_att_tensor = original_gcn_forward(*c_args, **c_kwargs)
            # Truncate to just the output of the last conv layer
            last_dim = model.conv_last.output_dim if hasattr(model.conv_last, 'output_dim') else args.output_dim
            x_tensor = x_tensor[:, :, -last_dim:]
            return x_tensor, adj_att_tensor
        model.gcn_forward = patched_gcn_forward

    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[{dataset_name.upper()}] Pre-computing baseline predictions...")
    with torch.no_grad():
        # Full Modal predictions
        model.fusion.disable_text = False
        model.fusion.disable_image = False
        
        original_text_encoder = model.text_encoder
        original_image_encoder = model.image_encoder
        
        class DummyEncoder(torch.nn.Module):
            def __init__(self, output_dim, dev):
                super().__init__()
                self.output_dim = output_dim
                self.device = dev
                class DummyFC:
                    def __init__(self, out_dim):
                        self.out_features = out_dim
                self.fc = DummyFC(output_dim)
            def _get_batch_size(self, x):
                if x is None:
                    return 0
                if isinstance(x, str):
                    return 1
                if isinstance(x, torch.Tensor):
                    return x.size(0) if x.dim() > 0 else 1
                if isinstance(x, (list, tuple)):
                    return len(x)
                return 1
            def forward(self, t_list, *args, **kwargs):
                batch_size = self._get_batch_size(t_list)
                dummy = torch.zeros(batch_size, self.output_dim, device=self.device)
                if dummy.numel() > 0:
                    dummy[0, 0] = 1e-9 # Prevent fusion.py from stripping it via torch.all() == 0
                return dummy
            def encode_text(self, t_list, *args, **kwargs):
                return self.forward(t_list, *args, **kwargs)
            def encode_image(self, image_input, *args, **kwargs):
                return self.forward(image_input, *args, **kwargs)
                
        dummy_text = DummyEncoder(getattr(original_text_encoder, 'output_dim', args.text_dim), device)
        dummy_image = DummyEncoder(getattr(original_image_encoder, 'output_dim', args.image_dim), device)
        
        # Build completely disconnected graph (only self-loops) for "No Graph" prediction
        adj_eye = torch.eye(adj.size(-1), device=adj.device)
        if adj.dim() == 3:
            adj_eye = adj_eye.unsqueeze(0).expand(adj.size(0), -1, -1)
            
        prob_all_dict = {}
        
        # Eval all 8 combinations of (Has Text, Has Image, Has Graph)
        for has_t in [True, False]:
            for has_i in [True, False]:
                for has_g in [True, False]:
                    model.text_encoder = original_text_encoder if has_t else dummy_text
                    model.image_encoder = original_image_encoder if has_i else dummy_image
                    
                    if hasattr(model, '_text_cache'): model._text_cache.clear()
                    if hasattr(model, '_image_cache'): model._image_cache.clear()
                    
                    curr_adj = adj if has_g else adj_eye
                    ypred, _ = model(x, curr_adj, text_feats=full_text_list, image_feats=full_image_list)
                    if ypred.dim() == 3 and ypred.shape[0] == 1: ypred = ypred.squeeze(0)
                    
                    prob_all_dict[(has_t, has_i, has_g)] = F.softmax(ypred, dim=-1).cpu()

        # Restore state and clear cache again
        model.text_encoder = original_text_encoder
        model.image_encoder = original_image_encoder
        if hasattr(model, '_text_cache'): model._text_cache.clear()
        if hasattr(model, '_image_cache'): model._image_cache.clear()
        
    # We MUST restore the state_dict to ensure the clean state before Explainer runs.
    if getattr(model.fusion, 'fc', None) is not None:
        expected_in = sd['fusion.fc.weight'].shape[1] if 'fusion.fc.weight' in sd else model.fusion.fc.in_features
        if model.fusion.fc.in_features != expected_in:
            model.fusion.fc = torch.nn.Linear(expected_in, model.fusion.fc.out_features).to(device)
            
    model.load_state_dict(sd, strict=False)

    test_label_np = labels_tensor.numpy()
    pred_fake_np = prob_all_dict[(True, True, True)].unsqueeze(0).numpy()
    
    explainer = MultimodalExplainer(
        model=model, adj=data["adj"], feat=data["feat"], label=np.expand_dims(test_label_np, axis=0),
        pred=pred_fake_np, train_idx=[], args=args, writer=None, print_training=False,
        graph_mode=False, graph_idx=0, use_uma=False
    )
    
    # CRITICAL FIX: The explainer creates its own randomly initialized text_encoder/image_encoder!
    # We must patch them to share the actual trained weights, otherwise text_score optimization gets garbage.
    if getattr(explainer, 'text_encoder', None) is not None:
        explainer.text_encoder.load_state_dict(model.text_encoder.state_dict())
    if getattr(explainer, 'image_encoder', None) is not None:
        explainer.image_encoder.load_state_dict(model.image_encoder.state_dict())

    # CRITICAL FIX 2: PyTorch __call__ ignores instance monkey-patched .forward()
    # We must wrap the model in a custom module so `explainer` uses the properly 
    # broadcasted 3D text_feats array when evaluating `self.model(...)` !
    class ModelWrapperForExplainer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            # Expose necessary submodules to satisfying Explainer's hooks
            self.fusion = inner.fusion
            self.text_encoder = inner.text_encoder
            self.image_encoder = inner.image_encoder
            
        def forward(self, x, adj, text=None, image_path=None, batch_num_nodes=None, **kwargs):
            text_arg = kwargs.get('text_feats', text)
            if isinstance(text_arg, torch.Tensor) and text_arg.dim() == 2:
                num_nodes = x.size(1)
                kwargs['text_feats'] = text_arg.unsqueeze(1).expand(-1, num_nodes, -1)
                text = None # clear to force using kwargs['text_feats']
                
            image_arg = kwargs.get('image_feats', image_path)
            if isinstance(image_arg, torch.Tensor) and image_arg.dim() == 2:
                num_nodes = x.size(1)
                kwargs['image_feats'] = image_arg.unsqueeze(1).expand(-1, num_nodes, -1)
                image_path = None
                
            return self.inner(x, adj, text=text, image_path=image_path, batch_num_nodes=batch_num_nodes, **kwargs)

    # Inject the wrapper into the explainer context
    model_wrapper = ModelWrapperForExplainer(model)
    explainer.model = model_wrapper
    
    print(f"[{dataset_name.upper()}] Extracting text scores via Explainer ({len(test_idx)} nodes)...")
    
    # Storage for fast simulation
    node_results = []
    
    args.num_epochs = 50  # Fix to 50 for all stability
    
    for idx in tqdm(test_idx):
        gt = test_label_np[idx]
        
        # 预先存储8种情况的预测结果
        preds = {}
        for has_t in [True, False]:
            for has_i in [True, False]:
                for has_g in [True, False]:
                    preds[f"{has_t}_{has_i}_{has_g}"] = torch.argmax(prob_all_dict[(has_t, has_i, has_g)][idx]).item()
        
        # 核心优化点：每个节点解释提取仅执行一次！
        masked_adj, text_score, image_score = explainer.explain(
            node_idx=idx, graph_idx=0, graph_mode=False, unconstrained=False
        )

        # Graph score from explainer output adjacency mask.
        if masked_adj is None or np.size(masked_adj) == 0:
            graph_score = 0.5
        else:
            graph_score = float(np.clip(np.mean(masked_adj), 0.0, 1.0))
            
        node_results.append({
            'gt': gt,
            'preds': preds,
            'text_score': text_score,
            'image_score': image_score,
            'graph_score': graph_score
        })
        
    def normalize_scores(score_key):
        scores = [r[score_key] for r in node_results]
        s_min, s_max = min(scores), max(scores)
        print(f"[DEBUG] {dataset_name} {score_key}: min={s_min:.6f}, max={s_max:.6f}, mean={np.mean(scores):.4f}")
        if s_max > s_min:
            for r in node_results:
                r[score_key] = (r[score_key] - s_min) / (s_max - s_min)
        else:
            for r in node_results:
                r[score_key] = 0.5

    normalize_scores('text_score')
    normalize_scores('image_score')
    normalize_scores('graph_score')
        
    return node_results

def simulate_threshold(node_results, threshold, mode='joint'):
    """在给定阈值下离线推演：返回准确率与预测变化率。"""
    final_preds = []
    gts = []
    changed = 0
    for r in node_results:
        gts.append(r['gt'])
        base_pred = r['preds']["True_True_True"]

        if mode == 'text':
            t_keep = r['text_score'] >= threshold
            i_keep = True
            g_keep = True
        elif mode == 'image':
            t_keep = True
            i_keep = r['image_score'] >= threshold
            g_keep = True
        elif mode == 'graph':
            t_keep = True
            i_keep = True
            g_keep = r['graph_score'] >= threshold
        else:  # joint
            t_keep = r['text_score'] >= threshold
            i_keep = r['image_score'] >= threshold
            g_keep = r['graph_score'] >= threshold

        pred = r['preds'][f"{t_keep}_{i_keep}_{g_keep}"]
        final_preds.append(pred)
        if pred != base_pred:
            changed += 1

    acc = accuracy_score(gts, final_preds)
    change_rate = changed / len(node_results) if node_results else 0.0
    return acc, change_rate

def main():
    datasets = ["mutag", "nlm_chem", "pqaref"]
    taus = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ... 0.9]
    
    # 模拟构建 base args
    base_args = configs.arg_parse()
    base_args.multimodal = True
    base_args.gpu = True
    base_args.cuda = "0"
    base_args.lr = 0.5
    base_args.clip = 2.0
    base_args.mask_act = "sigmoid"
    base_args.mask_bias = False
    base_args.explainer_suffix = ""  # 修复在这里，增加这个必需的属性
    base_args.method = "base"
    base_args.name_suffix = ""

    mode_order = ['joint', 'text', 'image', 'graph']
    mode_labels = {
        'joint': 'Joint Gating (T+I+G)',
        'text': 'Text-only Gating',
        'image': 'Image-only Gating',
        'graph': 'Graph-only Gating',
    }
    mode_colors = {
        'joint': '#1f77b4',
        'text': '#ff7f0e',
        'image': '#2ca02c',
        'graph': '#d62728',
    }
    mode_markers = {
        'joint': 'o',
        'text': 's',
        'image': '^',
        'graph': 'D',
    }

    all_acc = {}
    all_change = {}
    
    for ds in datasets:
        # 收集该数据集 Explainer 给出的置信度分数
        node_results = analyze_dataset_sensitivity(ds, gpu=0, base_args=base_args)
        
        all_acc[ds] = {m: [] for m in mode_order}
        all_change[ds] = {m: [] for m in mode_order}

        for tau in taus:
            for mode in mode_order:
                acc, chg = simulate_threshold(node_results, tau, mode=mode)
                all_acc[ds][mode].append(acc)
                all_change[ds][mode].append(chg)

        print(f"[{ds.upper()}] Joint Acc: {[f'{a:.4f}' for a in all_acc[ds]['joint']]}")
        print(f"[{ds.upper()}] Joint Pred-Change: {[f'{c:.4f}' for c in all_change[ds]['joint']]}")

    os.makedirs('analysis_results', exist_ok=True)

    # ==== 图1：每个数据集的多模态准确率曲线 ====
    fig_acc, axes_acc = plt.subplots(1, len(datasets), figsize=(18, 5), sharey=True)
    if len(datasets) == 1:
        axes_acc = [axes_acc]
    for i, ds in enumerate(datasets):
        ax = axes_acc[i]
        for mode in mode_order:
            ax.plot(
                taus,
                all_acc[ds][mode],
                marker=mode_markers[mode],
                color=mode_colors[mode],
                linewidth=2,
                markersize=6,
                label=mode_labels[mode],
            )
        ax.set_title(ds.upper())
        ax.set_xlabel('Threshold ($\\tau$)')
        ax.set_xticks(taus)
        ax.grid(True, linestyle='--', alpha=0.5)
    axes_acc[0].set_ylabel('Test Accuracy')
    handles, labels = axes_acc[-1].get_legend_handles_labels()
    fig_acc.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, frameon=False)
    fig_acc.suptitle('Sensitivity Curves by Modality (Accuracy)', fontsize=14, y=1.02)
    fig_acc.tight_layout(rect=[0, 0.08, 1, 1])

    # ==== 图2：每个数据集的预测变化率曲线 ====
    fig_chg, axes_chg = plt.subplots(1, len(datasets), figsize=(18, 5), sharey=True)
    if len(datasets) == 1:
        axes_chg = [axes_chg]
    for i, ds in enumerate(datasets):
        ax = axes_chg[i]
        for mode in mode_order:
            ax.plot(
                taus,
                all_change[ds][mode],
                marker=mode_markers[mode],
                color=mode_colors[mode],
                linewidth=2,
                markersize=6,
                label=mode_labels[mode],
            )
        ax.set_title(ds.upper())
        ax.set_xlabel('Threshold ($\\tau$)')
        ax.set_xticks(taus)
        ax.grid(True, linestyle='--', alpha=0.5)
    axes_chg[0].set_ylabel('Prediction Change Rate')
    handles2, labels2 = axes_chg[-1].get_legend_handles_labels()
    fig_chg.legend(handles2, labels2, loc='lower center', ncol=4, fontsize=10, frameon=False)
    fig_chg.suptitle('Sensitivity Curves by Modality (Prediction Change)', fontsize=14, y=1.02)
    fig_chg.tight_layout(rect=[0, 0.08, 1, 1])

    # ==== 旧图：跨数据集 Joint 曲线，保留方便对比 ====
    plt.figure(figsize=(8, 6))
    ds_markers = ['o', 's', '^']
    ds_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ds_labels = ['MUTAG', 'NLM-CHEM', 'PQAREF']
    for i, ds in enumerate(datasets):
        plt.plot(
            taus,
            all_acc[ds]['joint'],
            marker=ds_markers[i],
            color=ds_colors[i],
            linewidth=2,
            markersize=8,
            label=ds_labels[i],
        )
    plt.title('Joint-Gating Threshold Sensitivity (Cross-Dataset)', fontsize=14, pad=15)
    plt.xlabel('Threshold ($\\tau$)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xticks(taus)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    pdf_path = 'analysis_results/sensitivity_analysis.pdf'
    png_path = 'analysis_results/sensitivity_analysis.png'
    csv_path = 'analysis_results/sensitivity_analysis.csv'
    pdf_acc_path = 'analysis_results/sensitivity_by_modality_accuracy.pdf'
    png_acc_path = 'analysis_results/sensitivity_by_modality_accuracy.png'
    pdf_chg_path = 'analysis_results/sensitivity_by_modality_change.pdf'
    png_chg_path = 'analysis_results/sensitivity_by_modality_change.png'
    csv_acc_path = 'analysis_results/sensitivity_by_modality_accuracy.csv'
    csv_chg_path = 'analysis_results/sensitivity_by_modality_change.csv'
    
    plt.savefig(pdf_path, format='pdf', dpi=300)
    plt.savefig(png_path, format='png', dpi=300)
    fig_acc.savefig(pdf_acc_path, format='pdf', dpi=300)
    fig_acc.savefig(png_acc_path, format='png', dpi=300)
    fig_chg.savefig(pdf_chg_path, format='pdf', dpi=300)
    fig_chg.savefig(png_chg_path, format='png', dpi=300)
    
    # 保存CSV（旧版 joint + 新版 by-modality）
    df_joint = pd.DataFrame({ds: all_acc[ds]['joint'] for ds in datasets}, index=[f"Tau_{t:.1f}" for t in taus])
    df_joint.to_csv(csv_path)

    rows_acc = []
    rows_chg = []
    for ds in datasets:
        for i, tau in enumerate(taus):
            for mode in mode_order:
                rows_acc.append({'dataset': ds, 'tau': float(tau), 'mode': mode, 'accuracy': all_acc[ds][mode][i]})
                rows_chg.append({'dataset': ds, 'tau': float(tau), 'mode': mode, 'change_rate': all_change[ds][mode][i]})
    pd.DataFrame(rows_acc).to_csv(csv_acc_path, index=False)
    pd.DataFrame(rows_chg).to_csv(csv_chg_path, index=False)
    
    print("\n" + "="*50)
    print("✅ Sensitivity Analysis Complete!")
    print(f"Saved Figure (PDF): {pdf_path}")
    print(f"Saved Figure (PNG): {png_path}")
    print(f"Saved Figure (PDF): {pdf_acc_path}")
    print(f"Saved Figure (PNG): {png_acc_path}")
    print(f"Saved Figure (PDF): {pdf_chg_path}")
    print(f"Saved Figure (PNG): {png_chg_path}")
    print(f"Saved Data        : {csv_path}")
    print(f"Saved Data        : {csv_acc_path}")
    print(f"Saved Data        : {csv_chg_path}")
    print("="*50)

if __name__ == "__main__":
    main()

