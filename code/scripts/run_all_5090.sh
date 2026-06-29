#!/usr/bin/env bash
set -euo pipefail

bash scripts/tune_5090.sh all
bash scripts/run_ablations_5090.sh
python scripts/summarize_results.py --runs-dir runs --output-dir results

