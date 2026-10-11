"""Second-round surface-noise rate sweep: higher rates to establish a
rate-response curve. Honest goal: show Always Full degrades monotonically with
noise while CG-Net stays flat (modality-safe fallback).

Run:  conda run -n pytorch python -u run_noise_sweep2.py
"""
import subprocess
import sys

PLAN = [
    # (dataset, ckpt, noise_mode, noise_rate, seed)
    ("nlm_chem", "nlm_chem_base_h128_o128_full_modal.pth.tar", "typo", 0.3, 42),
    ("nlm_chem", "nlm_chem_base_h128_o128_full_modal.pth.tar", "typo", 0.5, 42),
    ("nlm_chem", "nlm_chem_base_h128_o128_full_modal.pth.tar", "ocr", 0.3, 42),
    ("nlm_chem", "nlm_chem_base_h128_o128_full_modal.pth.tar", "ocr", 0.5, 42),
    ("pqaref",   "pqaref_base_h128_o128_full_modal.pth.tar",   "typo", 0.3, 42),
    ("pqaref",   "pqaref_base_h128_o128_full_modal.pth.tar",   "typo", 0.5, 42),
    ("pqaref",   "pqaref_base_h128_o128_full_modal.pth.tar",   "ocr", 0.3, 42),
    ("pqaref",   "pqaref_base_h128_o128_full_modal.pth.tar",   "ocr", 0.5, 42),
]

def run(ds, ckpt, mode, rate, seed):
    cmd = [sys.executable, "-u", "evaluate_explainer_gating.py",
           "--dataset", ds, "--ckpt-path", "ckpt/" + ckpt,
           "--text-tau", "0.7", "--explainer-epochs", "20",
           "--seed", str(seed), "--gpu", "--cuda", "0"]
    cmd += ["--text-noise", mode, "--noise-rate", str(rate)]
    print(f"\n===== {ds} | {mode} rate={rate} seed={seed} =====", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!!! FAIL {ds} {mode} {rate}", flush=True)
        return False
    return True

ok = all(run(ds, ckpt, mode, rate, seed) for ds, ckpt, mode, rate, seed in PLAN)
print(f"\nALL DONE ok={ok}", flush=True)
sys.exit(0 if ok else 1)
