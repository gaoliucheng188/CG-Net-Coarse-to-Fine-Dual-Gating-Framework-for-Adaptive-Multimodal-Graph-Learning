#!/usr/bin/env python
"""Batch rerun canonical numbers for the CG-Net ESWA revision (honest rewrite).

Runs evaluate_explainer_gating.py over datasets x tau values, then a 5-seed
multiseed pass, all serially (8GB GPU). Writes a JSON of parsed results that the
paper tables can be rebuilt from.

Usage:
    conda run -n pytorch python -u rerun_canonical.py --datasets mutag nlm_chem pqaref --tau 0.1 0.3 0.5 0.7 0.9 --seeds 42 123 456 789 1024
"""
import argparse, os, re, json, subprocess, sys
from pathlib import Path
from datetime import datetime

CKPT_TEMPLATE = "ckpt/{ds}_base_h128_o128_full_modal.pth.tar"
LINE_KEYS = [
    "baseline_full", "baseline_notext", "hard_only", "soft_only", "ours",
    "bypass_count", "bypass_total", "avg_text_attention",
]

def run_eval(ds, ckpt, tau, seed=None, explainer_epochs=20):
    cmd = ["python", "-u", "evaluate_explainer_gating.py",
           "--dataset", ds, "--ckpt-path", ckpt,
           "--text-tau", str(tau), "--explainer-epochs", str(explainer_epochs),
           "--gpu", "--cuda", "0"]
    env = dict(os.environ)
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
    out = r.stdout + r.stderr
    res = {"dataset": ds, "tau": tau, "seed": seed, "exit": r.returncode}
    for key in LINE_KEYS:
        res[key] = None
    pats = {
        "baseline_full": r"Baseline \(Always Full\)\s*:\s*([\d.]+)",
        "baseline_notext": r"Ablation \(Always Drop Text\)\s*:\s*([\d.]+)",
        "hard_only": r"Variant A \(Hard Router Only\)\s*:\s*([\d.]+)",
        "soft_only": r"Variant B \(Soft Fusion Only\)\s*:\s*([\d.]+)",
        "ours": r"OURS \(Dual Gating Framework\)\s*:\s*([\d.]+)",
        "bypass_count_total": r"Hard Bypassed Text \(O\(1\)\)\s*:\s*(\d+) / (\d+)",
        "avg_text_attention": r"Avg Soft Text Attention \(Rest\)\s*:\s*([\d.]+)",
    }
    for k, pat in pats.items():
        m = re.search(pat, out)
        if m:
            if k == "bypass_count_total":
                res["bypass_count"] = int(m.group(1)); res["bypass_total"] = int(m.group(2))
            else:
                res[k] = float(m.group(1))
    # tail for debugging on failure
    if r.returncode != 0:
        res["error_tail"] = out[-1500:]
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mutag", "nlm_chem", "pqaref"])
    ap.add_argument("--tau", nargs="+", type=float, default=[0.1,0.3,0.5,0.7,0.9])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,1024])
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-seeds", action="store_true")
    args = ap.parse_args()

    outdir = Path("analysis_results/canonical")
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1) Threshold sweep (seed fixed 42) per dataset x tau
    if not args.skip_sweep:
        sweep = []
        for ds in args.datasets:
            ckpt = CKPT_TEMPLATE.format(ds=ds)
            if not os.path.exists(ckpt):
                print(f"[skip] missing {ckpt}", flush=True); continue
            for tau in args.tau:
                r = run_eval(ds, ckpt, tau, seed=42)
                print(f"[sweep] {ds} tau={tau} -> full={r['baseline_full']} notext={r['baseline_notext']} "
                      f"hard={r['hard_only']} soft={r['soft_only']} ours={r['ours']} "
                      f"bypass={r['bypass_count']}/{r['bypass_total']}", flush=True)
                sweep.append(r)
        results["sweep"] = sweep

    # 2) Multiseed (tau=0.7) per dataset x seeds
    if not args.skip_seeds:
        multiseed = {}
        for ds in args.datasets:
            ckpt = CKPT_TEMPLATE.format(ds=ds)
            if not os.path.exists(ckpt):
                continue
            rows = []
            for seed in args.seeds:
                r = run_eval(ds, ckpt, 0.7, seed=seed)
                print(f"[seed] {ds} seed={seed} -> ours={r['ours']} full={r['baseline_full']} "
                      f"notext={r['baseline_notext']} hard={r['hard_only']} soft={r['soft_only']} "
                      f"bypass={r['bypass_count']}/{r['bypass_total']}", flush=True)
                rows.append(r)
            multiseed[ds] = rows
        results["multiseed_tau0.7"] = multiseed

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = outdir / f"canonical_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDONE -> {path}", flush=True)

if __name__ == "__main__":
    main()