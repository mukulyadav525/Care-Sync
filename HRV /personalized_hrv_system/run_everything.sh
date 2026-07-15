#!/usr/bin/env bash
# ============================================================
# run_everything.sh — one command, the WHOLE project, both datasets.
#
#   1) Pooled multi-dataset model (Dataset3 + WESAD together)  <- best forecaster
#   2) Per-subject pipeline on Dataset3   (global pretrain -> finetune -> infer -> plots)
#   3) Per-subject pipeline on WESAD
#   4) Synthetic anomaly-scenario scorecard + plots
#
# Usage:  bash run_everything.sh
# From:   the personalized_hrv_system/ directory.
# Each stage is independent: a failure in one is logged and the rest continue.
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
log() { echo ""; echo "############## $* ##############"; }

# ---- 1) POOLED model across BOTH datasets (the headline result) ----
log "STAGE 1/4: Pooled multi-dataset training (Dataset3 + WESAD)"
for M in tcn xgboost; do
    python3 scripts/run_multi_dataset.py --model "$M" --config configs/config_multi.yaml \
        || echo "  [warn] pooled $M failed (see console above)"
done
# cross-dataset generalisation check (train on one, test on the other)
python3 scripts/run_multi_dataset.py --model tcn --lodo --config configs/config_multi.yaml \
    || echo "  [warn] leave-one-dataset-out failed"

# ---- 2) Per-subject pipeline on Dataset3 ----
log "STAGE 2/4: Per-subject pipeline on Dataset3"
bash run_all.sh configs/config.yaml || echo "  [warn] Dataset3 run_all had failures"

# ---- 3) Per-subject pipeline on WESAD ----
log "STAGE 3/4: Per-subject pipeline on WESAD"
bash run_all.sh configs/config_wesad.yaml || echo "  [warn] WESAD run_all had failures"

# ---- 4) Anomaly-scenario validation (synthetic; no dataset needed) ----
log "STAGE 4/4: Anomaly-scenario scorecard + plots"
python3 scripts/run_simulation.py  || echo "  [warn] run_simulation failed"
python3 scripts/plot_simulation.py || echo "  [warn] plot_simulation failed"

log "DONE. Pooled model -> models/multi/   per-subject -> models/, plots/, processed/"
