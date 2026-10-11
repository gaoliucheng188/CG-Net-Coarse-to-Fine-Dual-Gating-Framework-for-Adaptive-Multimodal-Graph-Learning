#!/bin/bash

echo "========================================================"
echo "Running Multi-seed Experiments (Task 4)"
echo "========================================================"

DATASETS=("mutag" "nlm_chem" "pqaref")
SEEDS=(42 123 2025 888 777)

for D in "${DATASETS[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo "Training $D with seed $S"
        # 注意: 你的项目代码可能需要修改 train.py 或 explainer_main.py 
        # 来支持 --seed 参数保存多份模型。当前仅作格式展示。
        python train.py --dataset "$D" --seed "$S" --batch_size 32
        
        echo "Evaluating Double Gating for $D with seed $S"
        python evaluate_explainer_gating.py --dataset "$D" --ckpt-path "ckpt/${D}_base_h128_o128_full_modal.pth.tar"
    done
done

echo "Multi-seed Experiments Completed."

