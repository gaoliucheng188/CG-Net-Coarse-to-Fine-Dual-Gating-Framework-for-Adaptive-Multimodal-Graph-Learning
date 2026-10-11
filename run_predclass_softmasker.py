#!/usr/bin/env python3
"""Pred-class Soft Masker eval runner (no bash CRLF issues).

Usage on server:
  python run_predclass_softmasker.py --smoke
  python run_predclass_softmasker.py
  python run_predclass_softmasker.py --with-tau
  python run_predclass_softmasker.py --smoke --cuda 0 --epochs 20
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, log_path: Path | None = None) -> int:
    print(">>>", " ".join(cmd), flush=True)
    if log_path is None:
        return subprocess.call(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Only MUTAG @ tau")
    parser.add_argument("--with-tau", action="store_true", help="Also sweep tau")
    parser.add_argument("--tau", type=float, default=float(os.environ.get("TAU", "0.7")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "20")))
    parser.add_argument("--cuda", type=str, default=os.environ.get("CUDA_ID", "0"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--ckpt-mutag",
        default=os.environ.get("CKPT_MUTAG", "ckpt/mutag_base_h128_o128.pth.tar"),
    )
    parser.add_argument(
        "--ckpt-nlm",
        default=os.environ.get("CKPT_NLM", "ckpt/nlm_chem_base_h128_o128.pth.tar"),
    )
    parser.add_argument(
        "--ckpt-pqa",
        default=os.environ.get("CKPT_PQA", "ckpt/pqaref_base_h128_o128.pth.tar"),
    )
    parser.add_argument("--skip-clean-text", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    os.chdir(root)

    logdir = root / f"logs_predclass_{time.strftime('%Y%m%d_%H%M%S')}"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] logs -> {logdir}", flush=True)

    # GPU check
    if not args.cpu:
        try:
            import torch
            ok = torch.cuda.is_available()
            print(f"[gpu] torch.cuda.is_available()={ok}", flush=True)
            if ok:
                cid = int(args.cuda)
                print(f"[gpu] device_count={torch.cuda.device_count()}", flush=True)
                name = (
                    torch.cuda.get_device_name(cid)
                    if cid < torch.cuda.device_count()
                    else "INVALID"
                )
                print(f"[gpu] using cuda:{cid} name={name}", flush=True)
            else:
                print("[gpu] WARNING: CUDA unavailable; will still pass --gpu (may fall back to CPU).", flush=True)
        except Exception as e:
            print(f"[gpu] check failed: {e}", flush=True)

    # Optional clean-text regen for MUTAG
    text_dir = root / "data" / "texts" / "MUTAG"
    if (not args.skip_clean_text) and text_dir.is_dir():
        leaky = False
        for p in text_dir.glob("node_*.txt"):
            try:
                s = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "classified as mutagenic" in s or "classified as non-mutagenic" in s:
                leaky = True
                break
        if leaky:
            backup = root / "data" / "texts" / "MUTAG_leaky_backup"
            print("[info] Leaky MUTAG texts detected; regenerating clean texts...", flush=True)
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(text_dir, backup)
            code = run(
                [sys.executable, "scripts/prepare_mutag.py", "--overwrite", "--label-free-text"],
                logdir / "prepare_mutag_clean.txt",
            )
            if code != 0:
                print(f"[warn] prepare_mutag exit={code}", flush=True)
            sample = text_dir / "node_0.txt"
            if sample.exists():
                print("[info] sample clean text:", sample.read_text(encoding="utf-8", errors="ignore")[:200], flush=True)
        else:
            print("[info] MUTAG texts already look label-free; skip regen.", flush=True)

    def eval_one(dataset: str, ckpt: str, tau: float, tag: str) -> int:
        out = logdir / f"eval_{tag}_tau{tau}.txt"
        print("=" * 60, flush=True)
        print(f"[run] dataset={dataset} tau={tau} ckpt={ckpt}", flush=True)
        print(f"      log -> {out}", flush=True)
        ckpt_path = root / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
        if not ckpt_path.is_file():
            msg = f"[skip] missing ckpt: {ckpt_path}\n"
            print(msg, end="", flush=True)
            out.write_text(msg, encoding="utf-8")
            return 0
        cmd = [
            sys.executable,
            "evaluate_explainer_gating.py",
            "--dataset",
            dataset,
            "--ckpt-path",
            str(ckpt_path),
            "--text-tau",
            str(tau),
            "--explainer-epochs",
            str(args.epochs),
        ]
        if not args.cpu:
            cmd += ["--gpu", "--cuda", str(args.cuda)]
        return run(cmd, out)

    jobs = []
    if args.smoke:
        jobs = [("mutag", args.ckpt_mutag, args.tau, "mutag")]
    else:
        jobs = [
            ("mutag", args.ckpt_mutag, args.tau, "mutag"),
            ("nlm_chem", args.ckpt_nlm, args.tau, "nlm_chem"),
            ("pqaref", args.ckpt_pqa, args.tau, "pqaref"),
        ]
        if args.with_tau:
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                jobs += [
                    ("mutag", args.ckpt_mutag, t, "mutag"),
                    ("nlm_chem", args.ckpt_nlm, t, "nlm_chem"),
                    ("pqaref", args.ckpt_pqa, t, "pqaref"),
                ]

    rc = 0
    for ds, ckpt, tau, tag in jobs:
        code = eval_one(ds, ckpt, tau, tag)
        if code != 0:
            rc = code
            print(f"[error] eval failed dataset={ds} tau={tau} exit={code}", flush=True)
            # continue other jobs

    print(f"[done] logs: {logdir}", flush=True)
    print("Collect Always Full / Drop Text / Variant A / B / CG-Net / Bypass from eval_*_tau0.7.txt", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

