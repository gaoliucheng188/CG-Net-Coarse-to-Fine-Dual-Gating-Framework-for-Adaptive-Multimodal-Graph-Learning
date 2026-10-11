#!/usr/bin/env bash
# 6-hour plan: hard-split + timing for motivation/framework claims
# Usage (on server, one line also OK):
#   bash run_claim_ab_6h.sh 0
#   bash run_claim_ab_6h.sh 1    # use GPU 1 if GPU 0 is full
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_ID="${1:-0}"
OUTDIR="logs_claim_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "[info] logs -> $OUTDIR  cuda=$CUDA_ID"

echo "==== GPU ===="
nvidia-smi || true

run_one () {
  local ds="$1"
  local ckpt="$2"
  local tau="$3"
  local tag="$4"
  local out="$OUTDIR/${tag}.out"
  echo "============================================================"
  echo "[run] $tag  dataset=$ds tau=$tau"
  echo "      log -> $out"
  echo "============================================================"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] missing ckpt: $ckpt" | tee "$out"
    return 0
  fi
  python evaluate_explainer_gating.py \
    --dataset "$ds" \
    --ckpt-path "$ckpt" \
    --text-tau "$tau" \
    --explainer-epochs 20 \
    --gpu --cuda "$CUDA_ID" \
    2>&1 | tee "$out"
}

# Key operating points only (save time)
run_one mutag   ckpt/mutag_base_h128_o128_clean.pth.tar 0.3 mutag_tau0.3
run_one pqaref  ckpt/pqaref_base_h128_o128_mm.pth.tar   0.7 pqa_tau0.7

# Optional second MUTAG point for hard-set Soft check
run_one mutag   ckpt/mutag_base_h128_o128_clean.pth.tar 0.7 mutag_tau0.7

echo
echo "[done] Parse results:"
echo "  grep -E 'SPLIT ANALYSIS|B-claim|A-claim|TIMING|OURS|Hard Bypassed|Always Full|DropText' $OUTDIR/*.out"
echo "Decision:"
echo "  If Soft on HARD > Full on HARD  => weak support for claim B"
echo "  If Soft on HARD <= Full on HARD => claim B NOT supported; sell claim A only"
echo "  Bypass high + CG-Net All ~ Full => claim A supported"

