#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --dataset cifar100
  --data-dir data
  --img-size 224
  --epochs 100
  --batch-size 128
  --workers 8
  --lr 5e-4
  --weight-decay 0.05
  --warmup-epochs 10
  --mixup 0.2
  --cutmix 1.0
  --label-smoothing 0.1
  --random-erasing 0.25
  --amp
  --channels-last
)

python train.py --model cnn_only "${COMMON[@]}" --run-name cnn_only
python train.py --model mobilevit_lite --attention mha "${COMMON[@]}" --run-name mobilevit_lite
python train.py --model mocovit_lite "${COMMON[@]}" --run-name mocovit_lite
python train.py --model glf_base "${COMMON[@]}" --gram-weight 0.01 --run-name glf_full_gram
python train.py --model glf_base "${COMMON[@]}" --no-large-kernel --gram-weight 0.01 --run-name ablate_no_large_kernel
python train.py --model glf_base "${COMMON[@]}" --attention mha --gram-weight 0.01 --run-name ablate_mha
python train.py --model glf_base "${COMMON[@]}" --no-grn --gram-weight 0.01 --run-name ablate_no_grn
python train.py --model glf_base "${COMMON[@]}" --no-gate --gram-weight 0.01 --run-name ablate_no_gate
python train.py --model glf_base "${COMMON[@]}" --gram-weight 0.0 --run-name ablate_no_gram
