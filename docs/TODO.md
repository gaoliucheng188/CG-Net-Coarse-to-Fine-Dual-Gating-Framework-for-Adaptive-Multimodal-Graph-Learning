# Task Plan

- [ ] Review existing project structure and dataset handling for new `pqaref` dataset support.
- [ ] Download PQAref dataset into `data/pqaref/raw/` and convert to required format.
- [ ] Extend preprocessing/loading pipeline to ingest `pqaref` data.
- [ ] Update training configuration to recognize `pqaref` multimodal scenario.
- [ ] Run `python train.py --dataset pqaref --multimodal --fusion-method concat --epochs 1000 --hidden-dim 128 --output-dim 128 --num-gc-layers 3 --dropout 0.2 --weight-decay 0.0005 --train-ratio 0.7 --gpu --cuda 0` and verify execution.
- [ ] Document usage instructions and any caveats.
