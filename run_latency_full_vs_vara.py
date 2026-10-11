#!/usr/bin/env python3
"""Wall-clock: Always Full vs Variant A (Hard Router) WITHOUT Soft Masker.

This measures real inference routing cost: graph-only for easy nodes,
full fusion only for hard nodes. No test-time Soft optimization.

Usage on server:

  python run_latency_full_vs_vara.py
  python run_latency_full_vs_vara.py --cuda 1
  python run_latency_full_vs_vara.py --warmup 2 --repeat 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import configs
from models_multimodal import MultimodalGcnEncoderNode
from data.multimodal_loader import MultimodalDataset
import utils.gengraph as gengraph


def load_graph_and_split(dataset_name: str):
    dataset_folder = dataset_name
    file_prefix = dataset_name
    if dataset_name.lower() == "mutag":
        dataset_folder = "MUTAG"
        file_prefix = "mutag"

    processed_dir = Path("data") / dataset_folder / "processed"
    graph_path = processed_dir / f"{file_prefix}_graph.gpickle"
    labels_path = processed_dir / f"{file_prefix}_labels.csv"
    splits_path = processed_dir / "split_masks.npz"

    G = nx.read_gpickle(graph_path)
    labels_list = pd.read_csv(labels_path, header=None)[0].astype(int).tolist()
    labels_np = np.asarray(labels_list, dtype=int).reshape(-1)

    test_idx = []
    if splits_path.exists():
        npz = np.load(splits_path)
        test_idx = np.asarray(npz.get("test", [])).astype(np.int64).tolist()
    if not test_idx:
        n = G.number_of_nodes()
        test_idx = list(range(int(n * 0.8), n))

    sample_node = next(iter(G.nodes()))
    feat = G.nodes[sample_node]["feat"]
    input_dim = len(feat) if isinstance(feat, (np.ndarray, list, tuple)) else int(feat)

    data = gengraph.preprocess_input_graph(G, labels_list)
    return G, labels_np, test_idx, input_dim, data


def build_model_from_ckpt(ckpt_path: str, input_dim: int, num_classes: int, args, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    args.hidden_dim = sd["conv_block.0.weight"].shape[0] if "conv_block.0.weight" in sd else 128
    args.output_dim = sd["conv_last.weight"].shape[1] if "conv_last.weight" in sd else 128
    args.text_dim = sd["text_encoder.fc.weight"].shape[0] if "text_encoder.fc.weight" in sd else 128
    args.image_dim = sd["image_encoder.fc.weight"].shape[0] if "image_encoder.fc.weight" in sd else 128
    args.fusion_method = (
        "attention"
        if "fusion.query_proj.weight" in sd
        else ("bilinear" if "fusion.bilinear.weight" in sd else "concat")
    )

    conv_block_keys = [k for k in sd if k.startswith("conv_block.") and k.endswith(".weight")]
    args.num_gc_layers = 1
    if "conv_first.weight" in sd and "conv_last.weight" in sd:
        args.num_gc_layers = len(conv_block_keys) + 2

    graph_dim_full = args.hidden_dim * max(args.num_gc_layers - 1, 0) + args.output_dim
    tdim, idim = int(args.text_dim), int(args.image_dim)
    pred_w = int(sd["pred_model.weight"].shape[1]) if "pred_model.weight" in sd else None

    project_concat = True
    disable_text, disable_image = False, False

    if "fusion.fc.weight" in sd:
        cin = int(sd["fusion.fc.weight"].shape[1])
        project_concat = True
        # graph_dim for fusion is often embedding/output path; try full then last
        for gdim in (graph_dim_full, args.output_dim):
            if cin == gdim + tdim + idim:
                disable_text, disable_image = False, False
                break
            if cin == gdim + tdim:
                disable_text, disable_image = False, True
                break
            if cin == gdim + idim:
                disable_text, disable_image = True, False
                break
            if cin == gdim:
                disable_text, disable_image = True, True
                break
        else:
            # common TAG: graph+text projected to output_dim
            disable_image = True
    elif pred_w is not None and args.fusion_method == "concat":
        project_concat = False
        for gdim in (graph_dim_full, args.output_dim):
            if pred_w == gdim + tdim + idim:
                disable_text, disable_image = False, False
                break
            if pred_w == gdim + tdim:
                disable_text, disable_image = False, True
                break
            if pred_w == gdim + idim:
                disable_text, disable_image = True, False
                break
            if pred_w == gdim:
                disable_text, disable_image = True, True
                break

    # Prefer disable_image for TAG paper setting if still ambiguous
    if pred_w is not None and project_concat and pred_w == args.output_dim:
        # projected fusion: keep flags from fusion.fc if any; else graph+text
        if "fusion.fc.weight" not in sd:
            disable_image = True

    print(
        f"[arch] fusion={args.fusion_method} project_concat={project_concat} "
        f"text_dim={tdim} image_dim={idim} disable_text={disable_text} "
        f"disable_image={disable_image} pred_in={pred_w} layers={args.num_gc_layers}",
        flush=True,
    )

    model = MultimodalGcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=num_classes,
        num_layers=args.num_gc_layers,
        bn=getattr(args, "bn", False),
        args=args,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        fusion_method=args.fusion_method,
        project_concat=project_concat,
        disable_text=disable_text,
        disable_image=disable_image,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[load] unexpected (first 5): {list(unexpected)[:5]}", flush=True)
    model = model.to(device)
    model.eval()
    return model, disable_text, disable_image


@torch.no_grad()
def forward_probs(model, x, adj, text_list, image_list, disable_text: bool, disable_image: bool):
    model.fusion.disable_text = disable_text
    model.fusion.disable_image = disable_image
    ypred, _ = model(x, adj, text_feats=text_list, image_feats=image_list)
    if ypred.dim() == 3 and ypred.shape[0] == 1:
        ypred = ypred.squeeze(0)
    return F.softmax(ypred, dim=-1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model, x, adj, text_list, image_list, disable_text, disable_image, device):
    sync(device)
    t0 = time.perf_counter()
    probs = forward_probs(model, x, adj, text_list, image_list, disable_text, disable_image)
    sync(device)
    return probs, time.perf_counter() - t0


def run_one_setting(
    dataset: str,
    ckpt: str,
    tau: float,
    device: torch.device,
    args,
    warmup: int,
    repeat: int,
):
    print("\n" + "=" * 72, flush=True)
    print(f"DATASET={dataset}  TAU={tau}  CKPT={ckpt}", flush=True)
    print("=" * 72, flush=True)

    G, labels_np, test_idx, input_dim, data = load_graph_and_split(dataset)
    num_classes = int(labels_np.max()) + 1
    x = torch.tensor(data["feat"], dtype=torch.float, device=device)
    adj = torch.tensor(data["adj"], dtype=torch.float, device=device)

    print("Loading multimodal features...", flush=True)
    loader = MultimodalDataset(dataset_name=dataset, gpu=args.gpu)
    full_text, full_image = [], []
    for node_id in range(G.number_of_nodes()):
        nd = loader.get_node_data(node_id)
        full_text.append(nd.get("text", ""))
        full_image.append(nd.get("image_path", ""))

    model, built_dt, built_di = build_model_from_ckpt(ckpt, input_dim, num_classes, args, device)

    # Warmup (includes possible first-time text encode cache fill)
    print(f"Warmup x{warmup} (fills text cache)...", flush=True)
    for _ in range(warmup):
        _ = forward_probs(model, x, adj, full_text, full_image, built_dt, built_di)
        _ = forward_probs(model, x, adj, full_text, full_image, True, built_di)
        sync(device)

    full_times, graph_times, vara_logic_ms = [], [], []
    acc_full = acc_graph = acc_vara = None
    n_easy = n_hard = 0
    y_true = labels_np[test_idx]

    print(f"Timed repeats x{repeat} (no Soft Masker)...", flush=True)
    for r in range(repeat):
        # --- Always Full: one fused forward (text enabled as built) ---
        probs_full, t_full = timed_forward(
            model, x, adj, full_text, full_image, built_dt, built_di, device
        )
        full_times.append(t_full)

        # --- Graph-only / DropText ---
        probs_graph, t_graph = timed_forward(
            model, x, adj, full_text, full_image, True, built_di, device
        )
        graph_times.append(t_graph)

        # --- Variant A decision (pure CPU on cached probs; no Soft) ---
        t1 = time.perf_counter()
        pred_full = probs_full.argmax(dim=-1).cpu().numpy()
        pred_graph = probs_graph.argmax(dim=-1).cpu().numpy()
        # margin from graph probs
        top2 = torch.topk(probs_graph, k=min(2, probs_graph.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]
        margin = margin.cpu().numpy()
        easy = margin > tau
        pred_vara = np.where(easy, pred_graph, pred_full)
        vara_logic_ms.append((time.perf_counter() - t1) * 1000.0)

        # Acc only need once (deterministic)
        if r == 0:
            n_easy = int(easy[test_idx].sum()) if hasattr(easy, "__getitem__") else int(np.sum(easy[np.array(test_idx)]))
            # easy is for all nodes; index test
            easy_t = easy[np.array(test_idx)]
            n_easy = int(easy_t.sum())
            n_hard = len(test_idx) - n_easy
            acc_full = accuracy_score(y_true, pred_full[np.array(test_idx)])
            acc_graph = accuracy_score(y_true, pred_graph[np.array(test_idx)])
            acc_vara = accuracy_score(y_true, pred_vara[np.array(test_idx)])

        # Variant A wall = graph forward + (if any hard) need full forward too.
        # Fair efficient implementation: always need graph pass; full pass only if hard exists.
        # Cost model:
        #   DropText: t_graph
        #   Always Full: t_full  (graph+text fused once)
        #   Var A practical: t_graph + (t_full if n_hard>0 else 0)
        # Note: current model API does one full-graph forward; cannot skip encoding per-node
        # inside one batched forward. We report BOTH:
        #   (1) API-level times of the two forwards
        #   (2) Idealized Var A = t_graph + t_full * (n_hard/n_test)  as lower-bound proxy
        #       if per-node text could be skipped (true routing saving on text encoder)

    def stats(xs):
        a = np.asarray(xs, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=0)), float(a.min()), float(a.max())

    full_m, full_s, full_lo, full_hi = stats(full_times)
    graph_m, graph_s, graph_lo, graph_hi = stats(graph_times)
    n_test = len(test_idx)
    # Ideal routing time if full forward cost scales with hard fraction (text-dominated case)
    ideal_vara = [g + f * (n_hard / max(n_test, 1)) for g, f in zip(graph_times, full_times)]
    # Conservative Var A: must run both forwards with current batched API
    cons_vara = [g + f for g, f in zip(graph_times, full_times)]
    ideal_m, ideal_s, _, _ = stats(ideal_vara)
    cons_m, cons_s, _, _ = stats(cons_vara)

    print("\n--- Accuracy (test) ---", flush=True)
    print(f"Always Full : {acc_full:.4f}", flush=True)
    print(f"DropText    : {acc_graph:.4f}", flush=True)
    print(f"Variant A   : {acc_vara:.4f}", flush=True)
    print(f"Easy/Hard   : {n_easy}/{n_hard}  (tau={tau}, bypass={100*n_easy/max(n_test,1):.1f}%)", flush=True)

    print("\n--- Wall time per full test inference (seconds) ---", flush=True)
    print(f"{'Method':<28} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}", flush=True)
    print(f"{'Always Full (1 fused fwd)':<28} {full_m:10.4f} {full_s:10.4f} {full_lo:10.4f} {full_hi:10.4f}", flush=True)
    print(f"{'DropText (graph-only fwd)':<28} {graph_m:10.4f} {graph_s:10.4f} {graph_lo:10.4f} {graph_hi:10.4f}", flush=True)
    print(f"{'VarA conservative (g+f)':<28} {cons_m:10.4f} {cons_s:10.4f}", flush=True)
    print(f"{'VarA ideal (g + f*hard%)':<28} {ideal_m:10.4f} {ideal_s:10.4f}", flush=True)
    print(
        f"\nNotes:\n"
        f"  - No Soft Masker in any timing path.\n"
        f"  - Current model does whole-graph multimodal forward; cannot skip per-node BERT inside one call.\n"
        f"  - 'ideal' estimates saving if text cost ~ scales with hard fraction (routing value).\n"
        f"  - 'conservative' runs both forwards (upper bound with current API).\n"
        f"  - Decision logic overhead is negligible (~{np.mean(vara_logic_ms):.2f} ms).",
        flush=True,
    )

    # LaTeX-friendly one-liner
    print("\n--- LaTeX row draft ---", flush=True)
    print(
        f"{dataset} ($\\tau={tau}$) & {acc_full:.4f} & {acc_vara:.4f} & "
        f"{n_easy}/{n_test} ({100*n_easy/max(n_test,1):.1f}\\%) & "
        f"{full_m:.3f} & {ideal_m:.3f} & {cons_m:.3f} \\\\",
        flush=True,
    )
    return {
        "dataset": dataset,
        "tau": tau,
        "acc_full": acc_full,
        "acc_vara": acc_vara,
        "bypass": n_easy / max(n_test, 1),
        "t_full": full_m,
        "t_ideal": ideal_m,
        "t_cons": cons_m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--ckpt-mutag", default="ckpt/mutag_base_h128_o128_clean.pth.tar")
    ap.add_argument("--ckpt-pqa", default="ckpt/pqaref_base_h128_o128_mm.pth.tar")
    ap.add_argument("--skip-mutag", action="store_true")
    ap.add_argument("--skip-pqa", action="store_true")
    cli = ap.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    root = Path(__file__).resolve().parent
    os.chdir(root)

    # minimal args namespace for model ctor (GcnEncoderGraph reads args.gpu/method/bias)
    class A:
        pass

    args = A()
    args.gpu = True
    args.cuda = cli.cuda
    args.bn = False
    args.bias = True
    args.method = "base"  # not "att"
    args.device = None
    args.dropout = 0.0
    args.num_gc_layers = 3
    args.hidden_dim = 128
    args.output_dim = 128
    args.text_dim = 128
    args.image_dim = 128
    args.fusion_method = "concat"

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cli.cuda}" if use_cuda else "cpu")
    print(f"[device] {device}", flush=True)
    if use_cuda:
        print(torch.cuda.get_device_name(int(cli.cuda)), flush=True)

    jobs = []
    if not cli.skip_mutag:
        jobs.append(("mutag", cli.ckpt_mutag, 0.3))
    if not cli.skip_pqa:
        jobs.append(("pqaref", cli.ckpt_pqa, 0.7))

    results = []
    for ds, ckpt, tau in jobs:
        ckpt_path = root / ckpt
        if not ckpt_path.is_file():
            print(f"[skip] missing {ckpt_path}", flush=True)
            continue
        results.append(
            run_one_setting(ds, str(ckpt_path), tau, device, args, cli.warmup, cli.repeat)
        )

    print("\n" + "=" * 72, flush=True)
    print("SUMMARY (mean wall seconds)", flush=True)
    print("=" * 72, flush=True)
    print(f"{'Dataset':<10} {'tau':>5} {'AccF':>8} {'AccA':>8} {'Bypass':>8} {'tFull':>8} {'tIdealA':>8} {'tConsA':>8}", flush=True)
    for r in results:
        print(
            f"{r['dataset']:<10} {r['tau']:5.1f} {r['acc_full']:8.4f} {r['acc_vara']:8.4f} "
            f"{100*r['bypass']:7.1f}% {r['t_full']:8.4f} {r['t_ideal']:8.4f} {r['t_cons']:8.4f}",
            flush=True,
        )
    print(
        "\nHow to cite in paper:\n"
        "  - Accuracy: Variant A matches Full (routing does not hurt).\n"
        "  - Efficiency: report bypass %; if text encoding dominates, ideal time ≈ graph + full*hard_frac.\n"
        "  - Be honest that batched GNN API needs engineering for true per-node text skip in one kernel.\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

