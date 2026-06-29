#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --model glf_base \
  --dataset cifar100 \
  --data-dir data \
  --img-size 224 \
  --epochs 100 \
  --batch-size 128 \
  --workers 8 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  --mixup 0.2 \
  --cutmix 1.0 \
  --label-smoothing 0.1 \
  --random-erasing 0.25 \
  --gram-weight 0.01 \
  --amp \
  --channels-last \
  --run-name glf_base_cifar100_224_gram
