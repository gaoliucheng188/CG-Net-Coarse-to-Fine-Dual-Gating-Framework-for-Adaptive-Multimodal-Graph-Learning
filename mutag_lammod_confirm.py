#!/usr/bin/env python
"""Quick confirm: modality-safe prior (source=graph + lambda_mod) on MUTAG tau=0.7."""
import os, re, json, subprocess
from pathlib import Path
from datetime import datetime

CKPT = "ckpt/mutag_base_h128_o128_full_modal.pth.tar"
SEEDS = [42, 123, 456]

def run(seed):
    cmd = ["python", "-u", "evaluate_explainer_gating.py",
           "--dataset", "mutag", "--ckpt-path", CKPT, "--text-tau", "0.7",
           "--explainer-epochs", "20", "--seed", str(seed), "--gpu", "--cuda", "0"]
    env = dict(os.environ); env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    out = r.stdout + r.stderr
    pats = {
        "baseline_full": r"Baseline \(Always Full\)\s*:\s*([\d.]+)",
        "baseline_notext": r"Ablation \(Always Drop Text\)\s*:\s*([\d.]+)",
        "ours": r"OURS \(Dual Gating Framework\)\s*:\s*([\d.]+)",
        "attn": r"Avg Soft Text Attention \(Rest\)\s*:\s*([\d.]+)",
    }
    res = {"seed": seed, "exit": r.returncode}
    for k, p in pats.items():
        m = re.search(p, out)
        if m: res[k] = float(m.group(1))
    if r.returncode != 0: res["err"] = out[-600:]
    print(f"[seed {seed}] full={res.get('baseline_full')} notext={res.get('baseline_notext')} "
          f"ours={res.get('ours')} attn={res.get('attn')}", flush=True)
    return res

def main():
    rows = [run(s) for s in SEEDS]
    out = Path("analysis_results/canonical"); out.mkdir(parents=True, exist_ok=True)
    p = out / f"mutag_lammod_confirm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w") as f: json.dump({"dataset":"mutag","tau":0.7,"lambda_mod":0.5,"rows":rows}, f, indent=2)
    print("DONE ->", p, flush=True)

if __name__ == "__main__":
    main()