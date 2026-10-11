import os
import argparse
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import configs
from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
import utils.io_utils as io_utils
import utils.gengraph as gengraph
from explainer.explain_multimodal import MultimodalExplainer

def run_explainer_gating_evaluation(args):
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

    print(f"Loading {dataset_name} dataset for Explainer-Guided Gating...")
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
    args.multimodal_loader = multimodal_loader

    # Optional surface-noise corruption: applied to the loader's text_map BEFORE
    # full_text_list is built, so both the full-fusion baseline and the explainer
    # see degraded text (simulates deployed TAG data where text is unreliable).
    if getattr(args, 'text_noise', None):
        corrupt_text_map(multimodal_loader, args.text_noise,
                         getattr(args, 'noise_rate', 0.15),
                         getattr(args, 'seed', 42))

    full_text_list, full_image_list = [], []
    for node_id in range(G.number_of_nodes()):
        node_data = multimodal_loader.get_node_data(node_id)
        full_text_list.append(node_data.get("text", ""))
        full_image_list.append(node_data.get("image_path", ""))
    
    print("Initializing Multi-modal GNN Model...")
    ckpt_path = args.ckpt_path if hasattr(args, 'ckpt_path') and args.ckpt_path else None
    if not ckpt_path and hasattr(args, 'ckptdir') and args.ckptdir and os.path.isfile(args.ckptdir):
        ckpt_path = args.ckptdir
        
    if ckpt_path and os.path.isdir(ckpt_path):
        import glob
        # 很多时候模型权重会被保存在同名目录的同级位置，并且带有 .pth.tar 后缀
        if os.path.isfile(ckpt_path + '.pth.tar'):
            ckpt_path = ckpt_path + '.pth.tar'
        elif os.path.isfile(ckpt_path.rstrip('/') + '.pth.tar'):
            ckpt_path = ckpt_path.rstrip('/') + '.pth.tar'
        else:
            # 使用递归搜索 (recursive=True) 应对潜在的子文件夹嵌套结构
            possible_ckpts = glob.glob(os.path.join(ckpt_path, '**', '*.pth.tar'), recursive=True) + glob.glob(os.path.join(ckpt_path, '**', '*.pth'), recursive=True)
            if possible_ckpts:
                best_ckpts = [p for p in possible_ckpts if 'best' in os.path.basename(p)]
                ckpt_path = best_ckpts[0] if best_ckpts else possible_ckpts[-1]
            else:
                raise ValueError(f"No .pth.tar or .pth files found in or alongside directory: {ckpt_path}")

    if not ckpt_path: raise ValueError("Checkpoint path missing.")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
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

    # Infer architecture from checkpoint tensor shapes to avoid size mismatch
    # (e.g. pred_model [C,384] vs wrongly-built [C,640]).
    graph_dim_full = args.hidden_dim * max(args.num_gc_layers - 1, 0) + args.output_dim
    graph_dim_last = args.output_dim
    tdim = int(args.text_dim)
    idim = int(args.image_dim)

    project_concat = True
    args.disable_text = False
    args.disable_image = False
    pred_w_shape = None
    if 'pred_model.weight' in sd:
        pred_w_shape = int(sd['pred_model.weight'].shape[1])

    # Prefer fusion.fc input width when present (projected fusion)
    if 'fusion.fc.weight' in sd:
        concat_dim = int(sd['fusion.fc.weight'].shape[1])
        project_concat = True
        # Match fusion input = graph + optional text/image
        matched = False
        for gdim in (graph_dim_full, graph_dim_last):
            if concat_dim == gdim + tdim + idim:
                args.disable_text, args.disable_image = False, False
                matched = True
                break
            if concat_dim == gdim + tdim:
                args.disable_text, args.disable_image = False, True
                matched = True
                break
            if concat_dim == gdim + idim:
                args.disable_text, args.disable_image = True, False
                matched = True
                break
            if concat_dim == gdim:
                args.disable_text, args.disable_image = True, True
                matched = True
                break
        if not matched:
            # Fallback: assume graph+text only (common TAG setting)
            args.disable_image = True
    elif pred_w_shape is not None and args.fusion_method == 'concat':
        # No fusion.fc: MultimodalFusion with project_concat=False feeds concat into pred_model
        project_concat = False
        matched = False
        # Enumerate (graph_dim, disable_text, disable_image)
        for gdim in (graph_dim_full, graph_dim_last):
            options = [
                (gdim + tdim + idim, False, False),
                (gdim + tdim, False, True),
                (gdim + idim, True, False),
                (gdim, True, True),
            ]
            for width, dt, di in options:
                if width == pred_w_shape:
                    args.disable_text, args.disable_image = dt, di
                    matched = True
                    break
            if matched:
                break
        if not matched:
            # Last resort: common paper config graph+text only
            project_concat = False
            args.disable_image = True
            # If still impossible, try project_concat path
            if pred_w_shape in (args.output_dim, graph_dim_full, graph_dim_last):
                project_concat = True

    # Final sanity: rebuild flags so pred_model width matches ckpt when possible.
    # MultimodalGcnEncoderNode uses parent pred_input_dim as graph_dim (= graph_dim_full).
    if pred_w_shape is not None and args.fusion_method == 'concat' and not project_concat:
        gdim = graph_dim_full
        need = pred_w_shape
        if need == gdim + tdim + idim:
            args.disable_text, args.disable_image = False, False
        elif need == gdim + tdim:
            args.disable_text, args.disable_image = False, True
        elif need == gdim + idim:
            args.disable_text, args.disable_image = True, False
        elif need == gdim:
            # Classifier is graph-only (multi-layer GCN concat).
            args.disable_text, args.disable_image = True, True
        else:
            print(f"[warn] Could not exactly match pred_model in_dim={need} "
                  f"with graph_dim={gdim}, text={tdim}, image={idim}.", flush=True)

    # Freeze the architecture-level modality mask used at build time.
    # Runtime Full/DropText MUST respect these; force-enabling extra mods changes
    # concat width under project_concat=False and crashes (e.g. 640 vs 384).
    built_disable_text = bool(args.disable_text)
    built_disable_image = bool(args.disable_image)

    print(
        f"[arch] fusion={args.fusion_method} project_concat={project_concat} "
        f"text_dim={tdim} image_dim={idim} disable_text={built_disable_text} "
        f"disable_image={built_disable_image} pred_in={pred_w_shape} "
        f"graph_dim_full={graph_dim_full} num_gc_layers={args.num_gc_layers}",
        flush=True,
    )
    if built_disable_text and built_disable_image:
        print(
            "[arch] NOTE: this checkpoint's pred_model matches graph features only "
            f"(in_dim={pred_w_shape}). Full and DropText will be identical. "
            "For true multimodal NLM/PQA, retrain with --multimodal (Path A style).",
            flush=True,
        )

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
        label_dim=num_classes, num_layers=args.num_gc_layers, bn=args.bn, args=args,
        text_dim=args.text_dim, image_dim=args.image_dim, fusion_method=args.fusion_method,
        project_concat=project_concat,
        disable_text=built_disable_text,
        disable_image=built_disable_image,
    )

    # If still mismatched, rebuild once using measured pred width heuristics
    built_in = model.pred_model.weight.shape[1]
    if pred_w_shape is not None and built_in != pred_w_shape:
        print(f"[arch] rebuild: model pred_in={built_in} != ckpt pred_in={pred_w_shape}", flush=True)
        if pred_w_shape == graph_dim_full:
            built_disable_text, built_disable_image = True, True
            project_concat = False
        elif pred_w_shape == graph_dim_full + tdim:
            built_disable_text, built_disable_image = False, True
            project_concat = False
        elif pred_w_shape == graph_dim_full + idim:
            built_disable_text, built_disable_image = True, False
            project_concat = False
        elif pred_w_shape == graph_dim_full + tdim + idim:
            built_disable_text, built_disable_image = False, False
            project_concat = False
        elif pred_w_shape == args.output_dim:
            project_concat = True
        model = MultimodalGcnEncoderNode(
            input_dim=input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim,
            label_dim=num_classes, num_layers=args.num_gc_layers, bn=args.bn, args=args,
            text_dim=args.text_dim, image_dim=args.image_dim, fusion_method=args.fusion_method,
            project_concat=project_concat,
            disable_text=built_disable_text,
            disable_image=built_disable_image,
        )
        print(
            f"[arch] rebuilt pred_in={model.pred_model.weight.shape[1]} "
            f"disable_text={built_disable_text} disable_image={built_disable_image} "
            f"project_concat={project_concat}",
            flush=True,
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # BERT keys often missing when ckpt used hashing/fallback text encoder — OK.
        miss_show = [k for k in missing if 'bert.' not in k][:8]
        if miss_show:
            print(f"[load] missing non-bert keys (first 8): {miss_show}", flush=True)
        else:
            print(f"[load] missing only bert/* keys (fallback text encoder ckpt) count={len(missing)}", flush=True)
    if unexpected:
        print(f"[load] unexpected keys (first 8): {list(unexpected)[:8]}", flush=True)
    ckpt_pred_shape = tuple(sd['pred_model.weight'].shape) if 'pred_model.weight' in sd else None
    if ckpt_pred_shape is not None and tuple(model.pred_model.weight.shape) != ckpt_pred_shape:
        raise RuntimeError(
            f"pred_model shape still mismatched: model={tuple(model.pred_model.weight.shape)} "
            f"ckpt={ckpt_pred_shape}."
        )
    model = model.to(device)
    model.eval()

    print("Pre-computing baseline predictions...")
    with torch.no_grad():
        # Full Modal = modalities the checkpoint was built with (NOT force-all-on).
        model.fusion.disable_text = built_disable_text
        model.fusion.disable_image = built_disable_image
        ypred_full, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
        if ypred_full.dim() == 3 and ypred_full.shape[0] == 1:
            ypred_full = ypred_full.squeeze(0)
        prob_full_all = F.softmax(ypred_full, dim=-1).cpu()

        # Drop Text: turn text off; keep image as in the built architecture.
        # Under project_concat=False, input width is fixed at build time, so we only
        # drop text if the model was built WITH text; otherwise Full==DropText.
        model.fusion.disable_text = True
        model.fusion.disable_image = built_disable_image
        ypred_notext, _ = model(x, adj, text_feats=full_text_list, image_feats=full_image_list)
        if ypred_notext.dim() == 3 and ypred_notext.shape[0] == 1:
            ypred_notext = ypred_notext.squeeze(0)
        # If built without text, concat width would break if fusion still expects text.
        # Rebuild-safe path: when text was never part of pred, DropText == Full.
        if built_disable_text:
            prob_notext_all = prob_full_all.clone()
        else:
            prob_notext_all = F.softmax(ypred_notext, dim=-1).cpu()

        # Restore built modality mask for explainer / soft path
        model.fusion.disable_text = built_disable_text
        model.fusion.disable_image = built_disable_image

    test_label_np = labels_tensor.numpy()
    pred_fake_np = prob_full_all.unsqueeze(0).numpy()
    # Graph-only (text-disabled) probabilities drive the modality-safe mask target.
    pred_graph_np = prob_notext_all.unsqueeze(0).numpy()

    explainer = MultimodalExplainer(
        model=model,
        adj=data["adj"],
        feat=data["feat"],
        label=np.expand_dims(test_label_np, axis=0),
        pred=pred_fake_np,
        pred_graph=pred_graph_np,
        train_idx=[],
        args=args,
        writer=None,
        print_training=False,
        graph_mode=False,
        graph_idx=0,
        use_uma=args.explainer_uma if hasattr(args, 'explainer_uma') else False,
    )

    preds_ours = []
    preds_hard_only = []
    preds_soft_only = []
    baseline_full_preds = []
    baseline_notext_preds = []
    gt_labels = []
    margins = []
    is_easy_flags = []
    is_disagree_flags = []

    print(f"\nStarting Architectural Ablation & Gating Evaluation (tau={args.text_tau})...")
    print("⚠️  Note: This run includes 'Soft Fusion Only' variant, which forces Explainer on ALL samples. It may take longer.")
    TEXT_TAU = args.text_tau  # Use arg

    num_dropped_ours = 0
    sum_text_score_ours = 0.0
    t_loop0 = time.perf_counter()
    t_soft_total = 0.0

    for idx in tqdm(test_idx):
        gt = test_label_np[idx]
        gt_labels.append(gt)

        pred_full = torch.argmax(prob_full_all[idx]).item()
        pred_graph = torch.argmax(prob_notext_all[idx]).item()
        baseline_full_preds.append(pred_full)
        baseline_notext_preds.append(pred_graph)

        # ---- Coarse-to-Fine Dual Gating (Scheme 2) ----
        probs_graph = prob_notext_all[idx].clone()
        probs_full = prob_full_all[idx].clone()

        vals, _ = torch.topk(probs_graph, 2)
        margin = (vals[0] - vals[1]).item()
        margins.append(margin)
        easy = margin > TEXT_TAU
        is_easy_flags.append(easy)
        is_disagree_flags.append(pred_full != pred_graph)

        # --- Variant A: Hard Router Only ---
        if easy:
            preds_hard_only.append(pred_graph)
        else:
            # Fallback to Full Modal prediction directly, without explainer soft score
            preds_hard_only.append(pred_full)

        # --- Variant B: Soft Fusion Only ---
        # Generate Explainer text_score (Needed for Soft-Only AND potentially for Ours)
        args.num_epochs = args.explainer_epochs
        t_s0 = time.perf_counter()
        masked_adj, text_score, image_score = explainer.explain(
            node_idx=idx, graph_idx=0, graph_mode=False, unconstrained=False
        )
        t_soft_total += time.perf_counter() - t_s0
        text_score = round(float(text_score), 6)

        soft_fused_probs = (1.0 - text_score) * probs_graph + text_score * probs_full
        pred_soft = torch.argmax(soft_fused_probs).item()
        preds_soft_only.append(pred_soft)

        # --- OURS: Coarse-to-Fine Dual Gating ---
        if easy:
            preds_ours.append(pred_graph)
            num_dropped_ours += 1
        else:
            preds_ours.append(pred_soft)
            sum_text_score_ours += text_score

    t_loop = time.perf_counter() - t_loop0

    acc_full = accuracy_score(gt_labels, baseline_full_preds)
    acc_no_text = accuracy_score(gt_labels, baseline_notext_preds)

    acc_hard_only = accuracy_score(gt_labels, preds_hard_only)
    acc_soft_only = accuracy_score(gt_labels, preds_soft_only)
    acc_ours = accuracy_score(gt_labels, preds_ours)

    num_soft_ours = len(test_idx) - num_dropped_ours
    avg_text_score_ours = sum_text_score_ours / num_soft_ours if num_soft_ours > 0 else 0

    def _subset_acc(y_true, y_pred, mask):
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            return float("nan"), 0
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        return accuracy_score(yt, yp), len(idx)

    easy_mask = is_easy_flags
    hard_mask = [not e for e in is_easy_flags]
    disagree_mask = is_disagree_flags

    methods = {
        "Always Full": baseline_full_preds,
        "DropText": baseline_notext_preds,
        "Variant A": preds_hard_only,
        "Variant B": preds_soft_only,
        "CG-Net": preds_ours,
    }

    print("\n" + "=" * 75)
    print("Architectural Ablation Results (Dual Gating Framework)")
    print("=" * 75)
    print(f"1. Baseline (Always Full)            : {acc_full:.4f}")
    print(f"2. Ablation (Always Drop Text)       : {acc_no_text:.4f}")
    print("-" * 75)
    print(f"3. Variant A (Hard Router Only)      : {acc_hard_only:.4f}")
    print(f"   -> Details: margin>{TEXT_TAU} drops text, else uses raw Full model.")
    print(f"4. Variant B (Soft Fusion Only)      : {acc_soft_only:.4f}")
    print(f"   -> Details: Forces Explainer on 100% samples (Heavy computation!).")
    print("-" * 75)
    print(f"OURS (Dual Gating Framework)         : {acc_ours:.4f}")
    print(f"   -> Hard Bypassed Text (O(1))      : {num_dropped_ours} / {len(test_idx)} samples")
    print(f"   -> Bypass ratio                   : {100.0 * num_dropped_ours / max(len(test_idx), 1):.1f}%")
    print(f"   -> Avg Soft Text Attention (Rest) : {avg_text_score_ours:.6f}")
    print("=" * 75)

    # ---- Split analysis for claims A/B ----
    print("\n" + "=" * 75)
    print(f"SPLIT ANALYSIS (tau={TEXT_TAU})  [for motivation/framework claims]")
    print("=" * 75)
    n_easy = sum(1 for e in easy_mask if e)
    n_hard = sum(1 for h in hard_mask if h)
    n_disc = sum(1 for d in disagree_mask if d)
    print(f"Easy (margin>{TEXT_TAU}): {n_easy}/{len(test_idx)}")
    print(f"Hard (margin<={TEXT_TAU}): {n_hard}/{len(test_idx)}")
    print(f"Disagree (graph pred != full pred): {n_disc}/{len(test_idx)}")
    print("-" * 75)
    print(f"{'Method':<14} {'All':>8} {'Easy':>8} {'Hard':>8} {'Disagree':>10}")
    for name, preds in methods.items():
        a_all = accuracy_score(gt_labels, preds)
        a_easy, _ = _subset_acc(gt_labels, preds, easy_mask)
        a_hard, _ = _subset_acc(gt_labels, preds, hard_mask)
        a_disc, _ = _subset_acc(gt_labels, preds, disagree_mask)
        def fmt(v):
            return f"{v:.4f}" if v == v else "  n/a "
        print(f"{name:<14} {fmt(a_all):>8} {fmt(a_easy):>8} {fmt(a_hard):>8} {fmt(a_disc):>10}")
    print("-" * 75)
    # Claim helpers
    soft_hard, _ = _subset_acc(gt_labels, preds_soft_only, hard_mask)
    full_hard, _ = _subset_acc(gt_labels, baseline_full_preds, hard_mask)
    ours_hard, _ = _subset_acc(gt_labels, preds_ours, hard_mask)
    a_hard_only_hard, _ = _subset_acc(gt_labels, preds_hard_only, hard_mask)
    print("B-claim check (need Soft/CG-Net > Full on HARD to support Soft quality):")
    if n_hard == 0:
        print("  HARD set empty at this tau — cannot test Soft quality.")
    else:
        print(f"  Full  on HARD: {full_hard:.4f}")
        print(f"  Soft  on HARD: {soft_hard:.4f}  (delta vs Full: {soft_hard - full_hard:+.4f})")
        print(f"  VarA  on HARD: {a_hard_only_hard:.4f}")
        print(f"  CGNet on HARD: {ours_hard:.4f}  (delta vs Full: {ours_hard - full_hard:+.4f})")
        if soft_hard > full_hard + 1e-12:
            print("  => Soft > Full on HARD: weak support for claim B (re-run multi-seed to confirm).")
        elif soft_hard + 1e-12 >= full_hard:
            print("  => Soft ≈ Full on HARD: Soft does not hurt; still no Acc boost for B.")
        else:
            print("  => Soft < Full on HARD: does NOT support claim B at this setting.")
    print("A-claim check (routing efficiency):")
    print(f"  Bypass={num_dropped_ours}/{len(test_idx)} ({100.0 * num_dropped_ours / max(len(test_idx), 1):.1f}%), "
          f"CG-Net All Acc={acc_ours:.4f}, Full All Acc={acc_full:.4f}, delta={acc_ours - acc_full:+.4f}")
    print("=" * 75)

    # ---- Timing (supports motivation efficiency story) ----
    print("\n" + "=" * 75)
    print("TIMING (this process; Soft runs on ALL nodes for Variant B)")
    print("=" * 75)
    n = max(len(test_idx), 1)
    print(f"Test nodes                 : {n}")
    print(f"Loop wall time (Soft+all)  : {t_loop:.2f} s  ({t_loop / n:.4f} s/node)")
    print(f"Soft explain total         : {t_soft_total:.2f} s  ({t_soft_total / n:.4f} s/node)")
    print(f"Approx CG-Net Soft calls   : {num_soft_ours} hard nodes only")
    if num_soft_ours > 0:
        # Estimate CG-Net soft cost if soft were only on hard nodes (same per-call cost)
        est_cgnet_soft = t_soft_total * (num_soft_ours / n)
        print(f"Est. Soft time if hard-only : {est_cgnet_soft:.2f} s  "
              f"(saves ~{t_soft_total - est_cgnet_soft:.2f} s vs Soft-on-all)")
    print("Note: Always Full has 0 Soft calls; its cost is precompute + one fused forward.")
    print("      Use bypass ratio + Est. Soft time for claim A efficiency narrative.")
    print("=" * 75)

    # ---- Efficiency benchmark (deployable per-branch latency) ----
    if getattr(args, "efficiency_bench", False):
        benchmark_efficiency(
            model=model, x=x, adj=adj, device=device, test_idx=test_idx,
            text_list=full_text_list, image_list=full_image_list,
            n_test=len(test_idx), n_hard=n_hard, t_soft_total=t_soft_total,
            args=args,
        )


def corrupt_text(text, mode, rate, rng):
    """Apply a surface-noise corruption to node text, simulating deployed TAG data.

    mode 'typo': replace a `rate` fraction of alphabetic characters with a random
                 alphabet character (simulates typos).
    mode 'ocr': replace confusion-prone character pairs (i/l, o/0, c/e, etc.)
                with probability `rate`, simulating OCR errors.
    Deterministic under a caller-supplied rng so runs are reproducible.
    """
    if not isinstance(text, str) or not text:
        return text
    chars = list(text)
    replace = False
    if mode == 'typo':
        for j, ch in enumerate(chars):
            if ch.isalpha() and rng.random() < rate:
                chars[j] = rng.choice('abcdefghijklmnopqrstuvwxyz')
                replace = True
    elif mode == 'ocr':
        confusions = {'i': 'l', 'l': 'i', 'o': '0', '0': 'o', 'c': 'e', 'e': 'c',
                      '1': 'l', 'l': '1', 'n': 'm', 'm': 'n'}
        for j, ch in enumerate(chars):
            low = ch.lower()
            if low in confusions and rng.random() < rate:
                chars[j] = confusions[low]
                replace = True
    return "".join(chars) if replace else text


def corrupt_text_map(loader, mode, rate, seed):
    """Corrupt the loader's text_map in place before evaluation.

    Called once before full_text_list is built, so BOTH the full-fusion baseline
    pass and the per-sample explainer reads observe the corrupted text — matching
    a deployment where node text is degraded. 'none' is a no-op.
    """
    import random as _random
    rng = _random.Random(seed)
    if mode is None or mode == 'none' or rate <= 0:
        return
    tmap = getattr(loader, 'text_map', {})
    for k in list(tmap.keys()):
        tmap[k] = corrupt_text(tmap.get(k, ''), mode, rate, rng)
    print(f"[noise] corrupted text_map in place (mode={mode}, rate={rate}, seed={seed})", flush=True)


def benchmark_efficiency(model, x, adj, device, test_idx, text_list, image_list,
                         n_test, n_hard, t_soft_total, args):
    """Measure deployable per-branch latency with CUDA events (honest, no FLOP counter).

    Model costs (whole-graph forwards, per test split):
      Always Full = one fused full-graph forward that encodes all text.
      Drop Text   = one graph-only forward (router / topology-only floor).
      CG-Net      = graph-only forward (router) + per-hard-node text encoding
                    + per-hard-node Soft Masker optimization.
    The text cache is cleared before every timed full forward so the measured cost
    reflects real encoding, not cache hits.
    """
    if n_test == 0:
        print("Efficiency: empty test split, skip."); return
    cuda = (device.type == "cuda")

    def _sync():
        if cuda:
            torch.cuda.synchronize(device)

    def _time(fn, repeats=20):
        for _ in range(3):
            fn()
        _sync()
        if cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(repeats):
                fn()
            e.record()
            _sync()
            return s.elapsed_time(e) / 1000.0 / repeats
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats

    full_text, full_img = bool(args.disable_text), bool(args.disable_image)
    had_text_cache = hasattr(model.text_encoder, "_text_cache")
    had_img_cache = hasattr(model.image_encoder, "_image_cache")

    def _clear_caches():
        if had_text_cache:
            model.text_encoder._text_cache = {}
        if had_img_cache:
            model.image_encoder._image_cache = {}

    # Always Full: built modality config, text encoded fresh each call.
    def _fn_full():
        _clear_caches()
        model.fusion.disable_text = full_text
        model.fusion.disable_image = full_img
        model(x, adj, text_feats=text_list, image_feats=image_list)

    # Drop Text / router: text (and image, for a topology-only floor) disabled.
    def _fn_graph():
        model.fusion.disable_text = True
        model.fusion.disable_image = True
        model(x, adj, text_feats=text_list, image_feats=image_list)

    # Restore built config afterwards.
    try:
        t_full = _time(_fn_full)
        t_graph = _time(_fn_graph)
    finally:
        model.fusion.disable_text = full_text
        model.fusion.disable_image = full_img

    n_hard = max(n_hard, 0)
    n_easy = n_test - n_hard
    t_text_all = max(t_full - t_graph, 0.0)
    t_text_per = t_text_all / max(n_test, 1)
    t_soft_per = t_soft_total / max(n_test, 1)   # per-call soft cost (uniform)
    t_cgnet = t_graph + n_hard * (t_text_per + t_soft_per)
    t_always = t_full
    t_drop = t_graph
    text_time_avoided = n_easy * t_text_per

    print("\n" + "=" * 78)
    print("EFFICIENCY BENCHMARK (deployable per-branch latency, GPU-timed)")
    print("=" * 78)
    print(f"Test split                : {n_test} nodes  (easy={n_easy}, hard={n_hard})")
    print(f"Full-graph forward (Full) : {t_full * 1e3:8.3f} ms")
    print(f"Graph-only forward (Drop) : {t_graph * 1e3:8.3f} ms")
    print(f"Text branch all (est)     : {t_text_all * 1e3:8.3f} ms  ({t_text_per * 1e3:.4f} ms/node)")
    print(f"Soft Masker per call      : {t_soft_per * 1e3:8.3f} ms")
    print("-" * 78)
    print(f"{'Method':<16}{'latency(ms)':>14}{'vs AlwaysFull':>14}")
    print(f"{'Always Full':<16}{t_always * 1e3:>14.3f}{1.0:>14.3f}")
    print(f"{'Drop Text':<16}{t_drop * 1e3:>14.3f}{(t_always / t_drop if t_drop > 0 else float('nan')):>14.3f}")
    if t_cgnet > 0:
        print(f"{'CG-Net':<16}{t_cgnet * 1e3:>14.3f}{(t_always / t_cgnet):>14.3f}")
    print("-" * 78)
    print(f"Text time avoided by bypass: {text_time_avoided * 1e3:.2f} ms "
          f"({100.0 * text_time_avoided / max(t_always, 1e-12):.1f}% of Always-Full text-side time)")
    print(f"CG-Net speedup vs AlwaysFull: {(t_always / t_cgnet) if t_cgnet > 0 else float('nan'):.3f}x "
          f"at {100.0 * n_easy / max(n_test, 1):.1f}% bypass")
    print("NOTE: latency is machine-specific (this GPU); the bypass-driven relative saving is "
          "architecture-level and hardware-independent.", flush=True)

    import json
    from datetime import datetime
    out = Path("analysis_results/canonical")
    out.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": getattr(args, "dataset", "?"), "tau": getattr(args, "text_tau", 0.7),
        "n_test": n_test, "n_easy": n_easy, "n_hard": n_hard,
        "t_full_ms": t_always * 1e3, "t_graph_ms": t_drop * 1e3,
        "t_text_all_ms": t_text_all * 1e3, "t_text_per_ms": t_text_per * 1e3,
        "t_soft_per_ms": t_soft_per * 1e3, "t_cgnet_ms": t_cgnet * 1e3,
        "speedup_vs_full": (t_always / t_cgnet) if t_cgnet > 0 else None,
        "text_time_avoided_ms": text_time_avoided * 1e3,
        "bypass_frac": n_easy / max(n_test, 1),
    }
    p = out / f"efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w") as f:
        json.dump(row, f, indent=2)
    print("EFFICIENCY ->", p, flush=True)


if __name__ == "__main__":
    import sys
    # Intercept --ckpt-path, --explainer-epochs, etc. hack before configs.arg_parse() complains
    ckpt_path_val = None
    if "--ckpt-path" in sys.argv:
        idx = sys.argv.index("--ckpt-path")
        ckpt_path_val = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    text_tau_val = 0.5
    if "--text-tau" in sys.argv:
        idx = sys.argv.index("--text-tau")
        text_tau_val = float(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    image_tau_val = 0.5
    if "--image-tau" in sys.argv:
        idx = sys.argv.index("--image-tau")
        image_tau_val = float(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    explainer_epochs_val = 20
    if "--explainer-epochs" in sys.argv:
        idx = sys.argv.index("--explainer-epochs")
        explainer_epochs_val = int(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    seed_val = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed_val = int(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    eff_bench_val = False
    if "--efficiency-bench" in sys.argv:
        sys.argv.remove("--efficiency-bench")
        eff_bench_val = True

    text_noise_val = None
    if "--text-noise" in sys.argv:
        idx = sys.argv.index("--text-noise")
        text_noise_val = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    noise_rate_val = 0.15
    if "--noise-rate" in sys.argv:
        idx = sys.argv.index("--noise-rate")
        noise_rate_val = float(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    args = configs.arg_parse()
    args.multimodal = True
    
    if ckpt_path_val is not None:
        args.ckpt_path = ckpt_path_val
    else:
        args.ckpt_path = ""
        
    args.text_tau = text_tau_val
    args.image_tau = image_tau_val
    args.explainer_epochs = explainer_epochs_val
    args.text_noise = text_noise_val
    args.noise_rate = noise_rate_val

    # Reproducibility: seed torch/numpy/cudnn from --seed (falls back to PYTHONHASHSEED).
    # Soft Masker is a test-time optimization; without seeding, dict-order/cudnn
    # nondeterminism makes its learned text_score unstable (observed 0.05-0.94 on MUTAG).
    seed_val = seed_val if seed_val is not None else int(os.environ.get("PYTHONHASHSEED", "42"))
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] torch/numpy/cudnn seeded with {seed_val}", flush=True)

    # Main overrides only if missing
    if not hasattr(args, 'dataset') or not args.dataset: args.dataset = "mutag"
    if not hasattr(args, 'hidden_dim') or not args.hidden_dim: args.hidden_dim = 128
    if not hasattr(args, 'output_dim') or not args.output_dim: args.output_dim = 128
    if not hasattr(args, 'text_dim') or not args.text_dim: args.text_dim = 300
    if not hasattr(args, 'image_dim') or not args.image_dim: args.image_dim = 2048
    if not hasattr(args, 'num_gc_layers') or not args.num_gc_layers: args.num_gc_layers = 3
    if not hasattr(args, 'fusion_method') or not args.fusion_method: args.fusion_method = "concat"
    if not hasattr(args, 'logdir') or not args.logdir: args.logdir = "log"
    if not hasattr(args, 'opt') or not args.opt: args.opt = "adam"

    # Explainer Params
    args.lr = 0.5
    args.clip = 2.0
    args.num_epochs = 20
    args.explainer_uma = False
    args.mask_act = "sigmoid"
    args.mask_bias = False
    args.explainer_suffix = ""
    # Modality-safe fusion prior: text must earn its weight, else s_T falls back to 0.
    args.lambda_mod = getattr(args, "lambda_mod", 0.5)
    args.efficiency_bench = eff_bench_val
    args.seed = seed_val

    if not os.path.exists("log"):
        os.makedirs("log")

    run_explainer_gating_evaluation(args)
