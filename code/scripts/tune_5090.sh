#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/tune_5090.sh proxy
#   bash scripts/tune_5090.sh regularization
#   bash scripts/tune_5090.sh confirm
#   bash scripts/tune_5090.sh all
#
# The script runs a staged tuning protocol:
# 1. proxy: cheaper 160px search for lr / weight decay / drop path.
# 2. regularization: check augmentation and Gram consistency around the default recipe.
# 3. confirm: full 224px, 100-epoch, 2-seed confirmation for reliable conclusions.

MODE="${1:-proxy}"

run_train() {
  local run_name="$1"
  shift
  if [[ -f "runs/${run_name}/best.pt" ]]; then
    echo "[skip] ${run_name} already has runs/${run_name}/best.pt"
    return 0
  fi
  echo "[run] ${run_name}"
  python train.py "$@" --run-name "${run_name}"
}

COMMON_PROXY=(
  --dataset cifar100
  --data-dir data
  --model glf_small
  --img-size 160
  --epochs 30
  --batch-size 256
  --workers 8
  --warmup-epochs 5
  --mixup 0.2
  --cutmix 1.0
  --label-smoothing 0.1
  --random-erasing 0.25
  --amp
  --channels-last
  --eval-interval 1
)

COMMON_FULL=(
  --dataset cifar100
  --data-dir data
  --model glf_base
  --img-size 224
  --epochs 100
  --batch-size 128
  --workers 8
  --warmup-epochs 10
  --label-smoothing 0.1
  --random-erasing 0.25
  --amp
  --channels-last
  --eval-interval 1
)

run_proxy_search() {
  for lr in 3e-4 5e-4 8e-4; do
    for wd in 0.03 0.05; do
      for dp in 0.12 0.18; do
        run_train "tune_proxy_lr${lr}_wd${wd}_dp${dp}" \
          "${COMMON_PROXY[@]}" \
          --lr "${lr}" \
          --weight-decay "${wd}" \
          --drop-path "${dp}" \
          --seed 42
      done
    done
  done
}

run_regularization_search() {
  run_train "tune_reg_no_mix_no_gram" \
    "${COMMON_FULL[@]}" \
    --epochs 60 \
    --lr 5e-4 \
    --weight-decay 0.05 \
    --drop-path 0.18 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --gram-weight 0.0 \
    --seed 42

  run_train "tune_reg_mix_cutmix_no_gram" \
    "${COMMON_FULL[@]}" \
    --epochs 60 \
    --lr 5e-4 \
    --weight-decay 0.05 \
    --drop-path 0.18 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --gram-weight 0.0 \
    --seed 42

  run_train "tune_reg_mix_cutmix_gram001" \
    "${COMMON_FULL[@]}" \
    --epochs 60 \
    --lr 5e-4 \
    --weight-decay 0.05 \
    --drop-path 0.18 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --gram-weight 0.01 \
    --seed 42

  run_train "tune_reg_strong_mix_gram001" \
    "${COMMON_FULL[@]}" \
    --epochs 60 \
    --lr 5e-4 \
    --weight-decay 0.05 \
    --drop-path 0.18 \
    --mixup 0.4 \
    --cutmix 1.0 \
    --gram-weight 0.01 \
    --seed 42
}

run_confirmation() {
  for seed in 42 43; do
    run_train "confirm_glf_base_seed${seed}" \
      "${COMMON_FULL[@]}" \
      --lr 5e-4 \
      --weight-decay 0.05 \
      --drop-path 0.18 \
      --mixup 0.2 \
      --cutmix 1.0 \
      --gram-weight 0.01 \
      --seed "${seed}"
  done
}

case "${MODE}" in
  proxy)
    run_proxy_search
    ;;
  regularization)
    run_regularization_search
    ;;
  confirm)
    run_confirmation
    ;;
  all)
    run_proxy_search
    run_regularization_search
    run_confirmation
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use proxy, regularization, confirm, or all." >&2
    exit 2
    ;;
esac

python scripts/summarize_results.py --runs-dir runs --output-dir results
