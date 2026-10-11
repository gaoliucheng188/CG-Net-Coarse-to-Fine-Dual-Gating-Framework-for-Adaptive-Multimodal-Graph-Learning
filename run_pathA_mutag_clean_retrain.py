#!/usr/bin/env python3
"""Path A: clean-text MUTAG retrain + Soft/CG-Net eval.

Avoids bash CRLF issues. Run on server:

  python run_pathA_mutag_clean_retrain.py
  python run_pathA_mutag_clean_retrain.py --epochs 300 --cuda 0
  python run_pathA_mutag_clean_retrain.py --skip-train --ckpt ckpt/mutag_base_h128_o128_clean.pth.tar
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> int:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=os.environ.get("CUDA_ID", "0"))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "300")))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--output", type=int, default=128)
    ap.add_argument("--text-dim", type=int, default=300)
    ap.add_argument("--image-dim", type=int, default=2048)
    ap.add_argument("--fusion", default="concat", choices=["concat", "attention", "bilinear", "add"])
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--soft-epochs", type=int, default=20)
    ap.add_argument("--suffix", default="clean")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--ckpt", default="", help="If set, use this ckpt for eval")
    ap.add_argument("--keep-image", action="store_true", help="Do NOT pass --disable-image")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)
    py = sys.executable

    print("=" * 60, flush=True)
    print("Path A: clean-text MUTAG retrain + pred-class Soft eval", flush=True)
    print(f"cuda={args.cuda} epochs={args.epochs} fusion={args.fusion}", flush=True)
    print("=" * 60, flush=True)

    # GPU check
    code = run([py, "-c", f"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name({int(args.cuda)}))"])
    if code != 0:
        print("[error] CUDA check failed", flush=True)
        return code

    # 1) clean texts
    text_dir = root / "data" / "texts" / "MUTAG"
    leaky = False
    if text_dir.is_dir():
        for p in text_dir.glob("node_*.txt"):
            s = p.read_text(encoding="utf-8", errors="ignore")
            if "classified as mutagenic" in s or "classified as non-mutagenic" in s:
                leaky = True
                break
    if leaky or not text_dir.is_dir():
        print("[1/4] Regenerating label-free MUTAG texts ...", flush=True)
        if text_dir.is_dir():
            backup = root / "data" / "texts" / "MUTAG_leaky_backup"
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(text_dir, backup)
        code = run([py, "scripts/prepare_mutag.py", "--overwrite", "--label-free-text"])
        if code != 0:
            return code
    else:
        print("[1/4] Texts already label-free", flush=True)
    sample = text_dir / "node_0.txt"
    if sample.exists():
        print("[sample]", sample.read_text(encoding="utf-8", errors="ignore")[:180], flush=True)

    # expected ckpt path from io_utils.gen_prefix:
    # mutag_base_h128_o128_clean.pth.tar  (dataset_method_h_o_suffix)
    expected = root / "ckpt" / f"mutag_base_h{args.hidden}_o{args.output}_{args.suffix}.pth.tar"

    # 2) train
    if not args.skip_train:
        print("[2/4] Training multimodal GCN on clean MUTAG ...", flush=True)
        cmd = [
            py, "train.py",
            "--dataset", "mutag",
            "--multimodal",
            "--method", "base",
            "--gpu", "--cuda", str(args.cuda),
            "--hidden-dim", str(args.hidden),
            "--output-dim", str(args.output),
            "--num-gc-layers", "3",
            "--text-dim", str(args.text_dim),
            "--image-dim", str(args.image_dim),
            "--fusion-method", args.fusion,
            "--lr", str(args.lr),
            "--epochs", str(args.epochs),
            "--batch-size", "1",
            "--name-suffix", args.suffix,
            "--ckptdir", "ckpt",
            "--logdir", "log/mutag_clean_retrain",
        ]
        if not args.keep_image:
            cmd.append("--disable-image")
        code = run(cmd)
        if code != 0:
            print("[error] train.py failed", flush=True)
            return code
    else:
        print("[2/4] skip train", flush=True)

    # 3) resolve ckpt
    ckpt = Path(args.ckpt) if args.ckpt else expected
    if not ckpt.is_file():
        # search fallbacks
        cands = sorted((root / "ckpt").glob(f"mutag*{args.suffix}*.pth.tar"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            cands = sorted((root / "ckpt").glob("mutag*.pth.tar"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            ckpt = cands[0]
            print(f"[warn] expected missing; using newest: {ckpt}", flush=True)
        else:
            print("[error] no mutag ckpt found under ckpt/", flush=True)
            return 1
    print(f"[3/4] ckpt = {ckpt}", flush=True)

    # 4) eval
    print("[4/4] Soft/CG-Net eval (pred-class) ...", flush=True)
    code = run([
        py, "run_predclass_softmasker.py", "--smoke",
        "--cuda", str(args.cuda),
        "--tau", str(args.tau),
        "--epochs", str(args.soft_epochs),
        "--ckpt-mutag", str(ckpt),
    ])
    print("[done]", flush=True)
    print(f"ckpt: {ckpt}", flush=True)
    print("Inspect latest logs_predclass_*/eval_mutag_tau*.txt", flush=True)
    print("Success criterion: Always Full should NOT stay ~0.38 if train/eval texts match.", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

