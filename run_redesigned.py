#!/usr/bin/env python
"""Redesigned (modality-safe) method: MUTAG 789/1024 to complete 5; NLM, PQARef 2 seeds."""
import os, re, json, subprocess
from pathlib import Path
from datetime import datetime

PLAN = {
    "mutag":     (["mutag_base_h128_o128_full_modal.pth.tar"], [789, 1024]),
    "nlm_chem":  (["nlm_chem_base_h128_o128_full_modal.pth.tar"], [42, 123]),
    "pqaref":    (["pqaref_base_h128_o128_full_modal.pth.tar"], [42, 123]),
}
PAT = {
    "baseline_full": r"Baseline \(Always Full\)\s*:\s*([\d.]+)",
    "baseline_notext": r"Ablation \(Always Drop Text\)\s*:\s*([\d.]+)",
    "hard_only": r"Variant A \(Hard Router Only\)\s*:\s*([\d.]+)",
    "soft_only": r"Variant B \(Soft Fusion Only\)\s*:\s*([\d.]+)",
    "ours": r"OURS \(Dual Gating Framework\)\s*:\s*([\d.]+)",
    "bypass_ct": r"Hard Bypassed Text \(O\(1\)\)\s*:\s*(\d+) / (\d+)",
    "attn": r"Avg Soft Text Attention \(Rest\)\s*:\s*([\d.]+)",
}

def run(ds, ckpt, seed):
    cmd = ["python", "-u", "evaluate_explainer_gating.py", "--dataset", ds,
           "--ckpt-path", "ckpt/" + ckpt, "--text-tau", "0.7", "--explainer-epochs", "20",
           "--seed", str(seed), "--gpu", "--cuda", "0"]
    env = dict(os.environ); env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    out = r.stdout + r.stderr
    res = {"dataset": ds, "seed": seed, "exit": r.returncode}
    for k, p in PAT.items():
        m = re.search(p, out)
        if m:
            if k == "bypass_ct":
                res["bypass_count"] = int(m.group(1)); res["bypass_total"] = int(m.group(2))
            else:
                res[k] = float(m.group(1))
    if r.returncode != 0: res["err"] = out[-500:]
    print(f"[{ds} seed {seed}] full={res.get('baseline_full')} notext={res.get('baseline_notext')} "
          f"hard={res.get('hard_only')} soft={res.get('soft_only')} ours={res.get('ours')} "
          f"bypass={res.get('bypass_count')}/{res.get('bypass_total')} attn={res.get('attn')}", flush=True)
    return res

def main():
    rows = []
    for ds, (ckpts, seeds) in PLAN.items():
        for ckpt in ckpts:
            for s in seeds:
                rows.append(run(ds, ckpt, s))
    out = Path("analysis_results/canonical"); out.mkdir(parents=True, exist_ok=True)
    p = out / f"redesigned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w") as f: json.dump({"method":"modality-safe","tau":0.7,"rows":rows}, f, indent=2)
    print("DONE ->", p, flush=True)

if __name__ == "__main__":
    main()