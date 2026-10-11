"""Sequential surface-noise robustness sweep for CG-Net (honest, no retraining).

Each job corrupts the loader's text_map BEFORE evaluation, so both the Always-Full
baseline pass and the per-sample Soft Masker see degraded text. Reproduces the
canonical no-noise numbers when --text-noise is omitted (sanity), and quantifies
whether CG-Net's modality-safe fallback survives typo/OCR surface noise.

Run:  conda run -n pytorch python -u run_noise_sweep.py
"""
import subprocess
import sys

PLAN = [
    # (dataset, ckpt, noise_mode, noise_rate, seed)
    ("nlm_chem", "nlm_chem_base_h128_o128_full_modal.pth.tar", "typo", 0.15, 42),
    ("nlm_chem", "nlm_chem_base_h128_o128_full_modal.pth.tar", "ocr", 0.15, 42),
    ("pqaref",   "pqaref_base_h128_o128_full_modal.pth.tar",   "typo", 0.15, 42),
    ("pqaref",   "pqaref_base_h128_o128_full_modal.pth.tar",   "ocr", 0.15, 42),
]

def run(ds, ckpt, mode, rate, seed):
    cmd = [sys.executable, "-u", "evaluate_explainer_gating.py",
           "--dataset", ds, "--ckpt-path", "ckpt/" + ckpt,
           "--text-tau", "0.7", "--explainer-epochs", "20",
           "--seed", str(seed), "--gpu", "--cuda", "0"]
    if mode:
        cmd += ["--text-noise", mode, "--noise-rate", str(rate)]
    print(f"\n===== {ds} | {mode or 'none'} rate={rate} seed={seed} =====", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!!! FAIL {ds} {mode}", flush=True)
        return False
    return True

ok = all(run(ds, ckpt, mode, rate, seed) for ds, ckpt, mode, rate, seed in PLAN)
print(f"\nALL DONE ok={ok}", flush=True)
sys.exit(0 if ok else 1)
