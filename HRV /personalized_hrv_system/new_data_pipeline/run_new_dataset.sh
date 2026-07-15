#!/usr/bin/env bash
# ============================================================================
# run_new_dataset.sh — evaluate a trained model on a NEW dataset, then print a
# couple of "next-N-minutes" forecasts as a sanity check.
#
# Usage:
#   bash new_data_pipeline/run_new_dataset.sh <MODEL_DIR> [MODEL] [CONFIG]
#
#   <MODEL_DIR>  trained model dir containing meta.joblib
#                  e.g. models/multi/tcn   (pooled, recommended)
#                       models/global/tcn  (global pretrained)
#   [MODEL]      tcn|lstm|gru|transformer|xgboost   (default: tcn)
#   [CONFIG]     config yaml                         (default: config_newdata.yaml)
#
# Set the new dataset path in the config's data.raw_root (or edit RAW_ROOT below).
# Run from the personalized_hrv_system/ directory.
# ============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${1:?usage: run_new_dataset.sh <MODEL_DIR> [MODEL] [CONFIG]}"
MODEL="${2:-tcn}"
CFG="${3:-new_data_pipeline/config_newdata.yaml}"
# Optional: uncomment to override the dataset root from the CLI instead of the config.
# RAW_ROOT_FLAG=(--raw-root /absolute/path/to/NEW_DATASET)
RAW_ROOT_FLAG=()

echo "############## Evaluate $MODEL on new dataset ##############"
python3 new_data_pipeline/evaluate_model.py \
    --config "$CFG" --model "$MODEL" --model-dir "$MODEL_DIR" "${RAW_ROOT_FLAG[@]}" \
    || echo "  [warn] evaluation failed (see output above)"

echo ""
echo "############## Example: forecast the next minutes (first subject) ##############"
# Predict for the first discovered subject as a demo of the live forecasting path.
FIRST_SUBJ="$(python3 - "$CFG" <<'PY'
import sys, yaml
sys.path.insert(0, "new_data_pipeline")
from eval_utils import resolve_raw_root, discover_subjects
cfg = yaml.safe_load(open(sys.argv[1]))
subs = discover_subjects(resolve_raw_root(cfg, None))
print(subs[0] if subs else "")
PY
)"
if [ -n "$FIRST_SUBJ" ]; then
    python3 new_data_pipeline/predict_next.py \
        --config "$CFG" --model "$MODEL" --model-dir "$MODEL_DIR" \
        --subject "$FIRST_SUBJ" "${RAW_ROOT_FLAG[@]}" \
        || echo "  [warn] forecast failed"
else
    echo "  (no subjects discovered — set data.raw_root in $CFG)"
fi

echo ""
echo "DONE. Reports -> new_data_pipeline/results/$(basename "$MODEL_DIR")/"
