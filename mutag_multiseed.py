#!/usr/bin/env python
"""Reproducible 5-seed MUTAG multiseed at tau=0.7 using now-seeded evaluate.
Each seed now controls torch/numpy/cudnn (not just PYTHONHASHSEED), so per-seed
values are reproducible. Writes results to analysis_results/canonical.
"""
import os, re, json, subprocess
from pathlib import Path
from datetime import datetime

CKPT = "ckpt/mutag_base_h128_o128_full_modal.pth.tar"
SEEDS = [42, 123, 456, 789, 1024]
LINES = ["baseline_full", "baseline_notext", "hard_only", "soft_only", "ours",
         "bypass_count", "bypass_total", "avg_text_attention"]

def run(seed):
    cmd = ["python", "-u", "evaluate_explainer_gating.py",
           "--dataset", "mutag", "--ckpt-path", CKPT,
           "--text-tau", "0.7", "--explainer-epochs", "20",
           "--seed", str(seed), "--gpu", "--cuda", "0"]
    env = dict(os.environ); env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    out = r.stdout + r.stderr
    res = {"seed": seed, "exit": r.returncode}
    pats = {
        "baseline_full": r"Baseline \(Always Full\)\s*:\s*([\d.]+)",
        "baseline_notext": r"Ablation \(Always Drop Text\)\s*:\s*([\d.]+)",
        "hard_only": r"Variant A \(Hard Router Only\)\s*:\s*([\d.]+)",
        "soft_only": r"Variant B \(Soft Fusion Only\)\s*:\s*([\d.]+)",
        "ours": r"OURS \(Dual Gating Framework\)\s*:\s*([\d.]+)",
        "bypass_ct": r"Hard Bypassed Text \(O\(1\)\)\s*:\s*(\d+) / (\d+)",
        "avg_text_attention": r"Avg Soft Text Attention \(Rest\)\s*:\s*([\d.]+)",
    }
    for k, p in pats.items():
        m = re.search(p, out)
        if m:
            if k == "bypass_ct":
                res["bypass_count"] = int(m.group(1)); res["bypass_total"] = int(m.group(2))
            else:
                res[k] = float(m.group(1))
    if r.returncode != 0:
        res["error_tail"] = out[-1200:]
    print(f"[seed {seed}] full={res.get('baseline_full')} notext={res.get('baseline_notext')} "
          f"hard={res.get('hard_only')} soft={res.get('soft_only')} ours={res.get('ours')} "
          f"bypass={res.get('bypass_count')}/{res.get('bypass_total')} "
          f"attn={res.get('avg_text_attention')}", flush=True)
    return res

def main():
    rows = [run(s) for s in SEEDS]
    out = Path("analysis_results/canonical"); out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = out / f"mutag_multiseed_seeded_{ts}.json"
    with open(p, "w") as f:
        json.dump({"dataset": "mutag", "tau": 0.7, "seeds": SEEDS, "rows": rows}, f, indent=2)
    print(f"DONE -> {p}", flush=True)

if __name__ == "__main__":
    main()