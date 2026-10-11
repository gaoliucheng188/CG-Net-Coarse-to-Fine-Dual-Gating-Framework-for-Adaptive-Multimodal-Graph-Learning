#!/usr/bin/env python
"""
Generate masks for ALL test nodes using the gated explainer.
This script generates masks for fidelity comparison.

Usage:
    python generate_all_masks.py --dataset mutag --model-path ckpt/mutag_base_h128_o128_full_modal.pth.tar --gpu
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn.functional as F
from pathlib import Path
from models_multimodal import MultimodalGcnEncoderNode, MultimodalGcnEncoderGraph
from data.multimodal_loader import MultimodalDataset
import utils.gengraph as gengraph
from explainer.explain_multimodal import MultimodalExplainer

def run_generate_masks(args):
    # 确保 args 有必要的属性（参考 utils/io_utils.py gen_prefix）
    args.bmname = getattr(args, 'bmname', None)
    args.method = getattr(args, 'method', 'base')
    args.hidden_dim = getattr(args, 'hidden_dim', 128)
    args.output_dim = getattr(args, 'output_dim', 128)
    args.bias = getattr(args, 'bias', True)
    args.name_suffix = getattr(args, 'name_suffix', '')
    args.explainer_suffix = getattr(args, 'explainer_suffix', '')
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
    args.logdir = getattr(args, 'logdir', 'log')  # 新增

    dataset_name = args.dataset
    dataset_folder = dataset_name
    file_prefix = dataset_name
    if dataset_name.lower() == "mutag":
        dataset_folder = "MUTAG"
        file_prefix = "mutag"

    processed_dir = Path("data") / dataset_folder / "processed"
    graph_path = processed_dir / f"{file_prefix}_graph.gpickle"
    labels_path = processed_dir / f"{file_prefix}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"

    print(f"Loading {dataset_name} dataset...")
    G = nx.read_gpickle(graph_path)
    labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)

    # Get test indices
    test_idx = []
    if splits_path.exists():
        npz = np.load(splits_path)
        test_idx = np.asarray(npz.get("test", [])).astype(np.int64).tolist()
    if not test_idx:
        num_nodes = G.number_of_nodes()
        test_idx = list(range(int(num_nodes * 0.8), num_nodes))

    print(f"Test set size: {len(test_idx)} nodes")

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
    full_text_list, full_image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        full_text_list.append(node_data.get("text", ""))
        full_image_list.append(node_data.get("image_path", ""))

    # Load model
    ckpt_path = args.model_path
    if not ckpt_path:
        raise ValueError("Please provide --model-path")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    args.hidden_dim = sd.get('conv_block.0.weight', torch.zeros(1)).shape[0] if 'conv_block.0.weight' in sd else args.hidden_dim
    args.output_dim = sd.get('conv_last.weight', torch.zeros(1)).shape[1] if 'conv_last.weight' in sd else args.output_dim
    args.text_dim = sd.get('text_encoder.fc.weight', torch.zeros(1)).shape[0] if 'text_encoder.fc.weight' in sd else args.text_dim
    args.image_dim = sd.get('image_encoder.fc.weight', torch.zeros(1)).shape[0] if 'image_encoder.fc.weight' in sd else args.image_dim
    args.fusion_method = 'attention' if 'fusion.query_proj.weight' in sd else ('bilinear' if 'fusion.bilinear.weight' in sd else 'concat')

    # Check num_layers based on conv_block
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
    # Deduce disable flags from fusion concat_dim
    graph_dim = args.hidden_dim * (args.num_gc_layers - 1) + args.output_dim
    if 'fusion.fc.weight' in sd:
        concat_dim = sd['fusion.fc.weight'].shape[1]
        if concat_dim == graph_dim + args.text_dim:
            args.disable_image = True
        elif concat_dim == graph_dim + args.image_dim:
            args.disable_text = True
        elif concat_dim != graph_dim + args.text_dim + args.image_dim:
            graph_dim = args.output_dim
            if concat_dim == graph_dim + args.text_dim:
                args.disable_image = True
            elif concat_dim == graph_dim + args.image_dim:
                args.disable_text = True

    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    text_dim = args.text_dim
    image_dim = args.image_dim
    num_gc_layers = args.num_gc_layers
    fusion_method = args.fusion_method
    disable_text = args.disable_text
    disable_image = args.disable_image

    print(f"Model config: hidden={hidden_dim}, output={output_dim}, text={text_dim}, image={image_dim}, fusion={fusion_method}")
    print(f"Disable: text={disable_text}, image={disable_image}, project_concat={project_concat}")

    # 计算 input_dim
    if 'conv_first.weight' in sd:
        input_dim = sd['conv_first.weight'].shape[0]
    else:
        # 从 graph 数据获取
        sample_node = next(iter(G.nodes()))
        feat = G.nodes[sample_node]["feat"]
        if isinstance(feat, (np.ndarray, list, tuple)):
            input_dim = len(feat)
        else:
            input_dim = 128

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=output_dim,
        label_dim=int(labels_np.max()) + 1, num_layers=num_gc_layers, bn=False, args=args,
        text_dim=text_dim, image_dim=image_dim, fusion_method=fusion_method,
        disable_text=disable_text, disable_image=disable_image,
        project_concat=project_concat
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    # Baseline predictions
    with torch.no_grad():
        ypred_full, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
        if ypred_full.dim() == 3 and ypred_full.shape[0] == 1:
            ypred_full = ypred_full.squeeze(0)
        prob_full_all = F.softmax(ypred_full, dim=-1).cpu()

    test_label_np = labels_tensor.numpy()
    pred_fake_np = prob_full_all.unsqueeze(0).numpy()

    # Create explainer
    explainer = MultimodalExplainer(
        model=model, adj=data["adj"], feat=data["feat"],
        label=np.expand_dims(test_label_np, axis=0), pred=pred_fake_np,
        train_idx=[], args=args, writer=None, print_training=False,
        graph_mode=False, graph_idx=0, use_uma=False,
    )
    # 确保 explainer.args 有 multimodal_loader
    explainer.args.multimodal_loader = multimodal_loader

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating masks for {len(test_idx)} test nodes...")
    print("This may take a while...")

    # 统计
    n_baseline = 0
    n_gated = 0

    for i, idx in enumerate(test_idx):
        # Use the gating logic: if margin > tau, use baseline; else use soft fusion
        probs_graph = prob_full_all[idx].clone()
        vals, _ = torch.topk(probs_graph, 2)
        margin = (vals[0] - vals[1]).item()

        if margin > args.text_tau:
            # Hard bypass: no explainer needed, save identity mask
            gated_str = "baseline"
            mask = np.ones((1, 1))  # Placeholder
            np.save(output_dir / f"masked_adj_{dataset_name}_h128_o128_{gated_str}_node_idx_{idx}_graph_idx_0.npy", mask)
            n_baseline += 1
        else:
            # Soft fusion: run explainer
            gated_str = "gated"
            explainer.args.num_epochs = args.explainer_epochs
            masked_adj, text_score, image_score = explainer.explain(
                node_idx=idx, graph_idx=0, graph_mode=False, unconstrained=False
            )
            np.save(output_dir / f"masked_adj_{dataset_name}_h128_o128_{gated_str}_node_idx_{idx}_graph_idx_0.npy", masked_adj)
            n_gated += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(test_idx):
            print(f"[{i+1}/{len(test_idx)}] baseline={n_baseline}, gated={n_gated}")

    print(f"\nDone! Generated masks in {output_dir}")
    print(f"Summary: baseline={n_baseline}, gated={n_gated}")


def main():
    # 先解析我们的自定义参数
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--dataset', default='mutag')
    custom_parser.add_argument('--model-path', default='')
    custom_parser.add_argument('--output-dir', default='log/gated_masks')
    custom_parser.add_argument('--text-tau', type=float, default=0.7)
    custom_parser.add_argument('--explainer-epochs', type=int, default=20)
    custom_parser.add_argument('--hidden-dim', type=int, default=128)
    custom_parser.add_argument('--output-dim', type=int, default=128)
    custom_parser.add_argument('--text-dim', type=int, default=300)
    custom_parser.add_argument('--image-dim', type=int, default=2048)
    custom_parser.add_argument('--cuda', default='0')
    custom_parser.add_argument('--gpu', dest='gpu', action='store_const', const=True, default=False)

    # 分离自定义参数和剩余参数
    known, unknown = custom_parser.parse_known_args()
    args = known

    if not args.model_path:
        # Try to find default model
        default_path = f"ckpt/{args.dataset}_base_h128_o128_full_modal.pth.tar"
        if os.path.exists(default_path):
            args.model_path = default_path
            print(f"Using default model: {args.model_path}")
        else:
            raise ValueError("Please provide --model-path")

    run_generate_masks(args)


if __name__ == '__main__':
    main()

