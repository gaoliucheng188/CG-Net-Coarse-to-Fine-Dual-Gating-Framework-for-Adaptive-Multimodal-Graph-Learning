#!/bin/bash
# 运行 PQAREF A2 实验 (结构 + 文本, 禁用图像)
echo "Starting PQAREF A2 Experiment..."
python train.py --epochs 1000 --hidden-dim 128 --output-dim 128 --num-gc-layers 3 --dropout 0.2 --weight-decay 0.0005 --train-ratio 0.7 --gpu --cuda 0 --dataset pqaref --multimodal --fusion-method attention --disable-image > pqaref_a2.log 2>&1

echo "Experiment finished. Generating plot..."
# 生成可视化图表
python plot_training_log.py pqaref_a2.log pqaref_a2_plot.png

echo "Done. Check pqaref_a2.log and pqaref_a2_plot.png"
