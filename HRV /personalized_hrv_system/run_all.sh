#!/usr/bin/env bash
# ============================================================
# run_all.sh — Full per-subject pipeline (global pretrain -> finetune -> infer)
# Usage:  bash run_all.sh [configs/config.yaml]      # default = Dataset3
#         bash run_all.sh configs/config_wesad.yaml  # WESAD instead
# From:   the personalized_hrv_system/ directory.
# ============================================================
set -euo pipefail

# ── config ───────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${1:-$ROOT/configs/config.yaml}"
[[ "$CFG" = /* ]] || CFG="$ROOT/$CFG"   # allow a relative config path
RAW_ROOT="$(dirname "$ROOT")/$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['raw_root'])" "$CFG")"
LOG_DIR="$ROOT/logs"
DEEP_MODELS=(tcn lstm gru transformer)
ALL_MODELS=(tcn lstm gru transformer xgboost)

mkdir -p "$LOG_DIR"

# ── helpers ──────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() {
    local label="$1"; shift
    log "START  $label"
    if python3 "$@" >> "$LOG_DIR/${label//\//_}.log" 2>&1; then
        log "OK     $label"
    else
        log "FAIL   $label  (see $LOG_DIR/${label//\//_}.log)"
        # continue — don't abort the whole run for one subject/model failure
    fi
}

# ── discover subjects ─────────────────────────────────────────
# Use a while-read loop instead of mapfile (mapfile requires bash 4+;
# macOS ships bash 3.2 and does not have it).  basename is called inside
# the loop to avoid the xargs SIGPIPE race that caused "terminated with
# signal 13".
SUBJECTS=()
while IFS= read -r _d; do
    SUBJECTS+=("$(basename "$_d")")
done < <(find "$RAW_ROOT" -maxdepth 1 -type d -name 'S[0-9][0-9]' | sort)

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "ERROR: No subject folders (S01, S02, …) found under $RAW_ROOT"
    exit 1
fi
log "Found ${#SUBJECTS[@]} subjects: ${SUBJECTS[*]}"

# ════════════════════════════════════════════════════════════
# STEP 1 — Preprocessing (feature-build preview, per subject)
# ════════════════════════════════════════════════════════════
log "═══ STEP 1: Preprocessing ═══"
for S in "${SUBJECTS[@]}"; do
    run "step1_preprocess_$S" scripts/run_preprocessing.py --subject "$S" --config "$CFG"
done

# ════════════════════════════════════════════════════════════
# STEP 2 — Build Personal Digital Twin (per subject)
# ════════════════════════════════════════════════════════════
log "═══ STEP 2: Digital Twin ═══"
for S in "${SUBJECTS[@]}"; do
    run "step2_twin_$S" scripts/build_digital_twin.py --subject "$S" --config "$CFG"
done

# ════════════════════════════════════════════════════════════
# STEP 3 — GLOBAL pretraining (deep models) + xgboost per-subject baseline
#   A single ~1hr session is FAR too little data to train a deep net from
#   scratch (it collapses / overfits). Instead we train ONE global model per
#   architecture pooled across all subjects — this generalises to new people.
#   xgboost is kept as a fast per-subject baseline.
# ════════════════════════════════════════════════════════════
log "═══ STEP 3: Global pretraining + xgboost baseline ═══"
for M in "${DEEP_MODELS[@]}"; do
    run "step3_pretrain_$M" scripts/run_pretraining.py --all --model "$M" --config "$CFG"
done
for S in "${SUBJECTS[@]}"; do
    run "step3_xgb_${S}" scripts/run_training.py --subject "$S" --model xgboost --config "$CFG"
done

# ════════════════════════════════════════════════════════════
# STEP 4 — Per-subject PERSONALIZATION: fine-tune the global model's output
#   head on each subject (frozen backbone -> can't overfit the tiny session).
#   This is the "digital twin" personal-calibration layer.
# ════════════════════════════════════════════════════════════
log "═══ STEP 4: Per-subject fine-tune (personalization) ═══"
for M in "${DEEP_MODELS[@]}"; do
    [ -f "$ROOT/models/global/$M/meta.joblib" ] || { log "SKIP   finetune $M (no global model)"; continue; }
    for S in "${SUBJECTS[@]}"; do
        run "step4_finetune_${S}_${M}" scripts/run_finetune.py \
            --subject "$S" \
            --pretrained "$ROOT/models/global/$M" \
            --out "$ROOT/models/$S/finetuned_$M" \
            --freeze-backbone \
            --config "$CFG"
    done
done

# ════════════════════════════════════════════════════════════
# STEP 5 — Inference + anomaly scoring, using the best available model per
#   subject: the personalized fine-tuned model, falling back to the global
#   model if fine-tuning was skipped (very short subject).
# ════════════════════════════════════════════════════════════
log "═══ STEP 5: Inference ═══"
for S in "${SUBJECTS[@]}"; do
    for M in "${DEEP_MODELS[@]}"; do
        MDIR="$ROOT/models/$S/finetuned_$M"
        [ -f "$MDIR/meta.joblib" ] || MDIR="$ROOT/models/global/$M"
        if [ -f "$MDIR/meta.joblib" ]; then
            run "step5_infer_${S}_${M}" scripts/run_inference.py \
                --subject "$S" --model "$M" --model-dir "$MDIR" \
                --out "$ROOT/processed/${S}_${M}_inference.csv" --config "$CFG"
        else
            log "SKIP   step5_infer_${S}_${M}  (no global or finetuned model)"
        fi
    done
done

# ════════════════════════════════════════════════════════════
# STEP 6 — Plot inference results (per subject × deep models)
# ════════════════════════════════════════════════════════════
log "═══ STEP 6: Plot inference ═══"
for S in "${SUBJECTS[@]}"; do
    for M in "${DEEP_MODELS[@]}"; do
        CSVPATH="$ROOT/processed/${S}_${M}_inference.csv"
        if [ -f "$CSVPATH" ]; then
            run "step6_plot_${S}_${M}" scripts/plot_inference.py \
                --subject "$S" --model "$M" --csv "$CSVPATH" \
                --out "$ROOT/plots/${S}_${M}_inference.png"
        fi
    done
done

# ════════════════════════════════════════════════════════════
# STEP 7 — Online / incremental update (all deep personal models)
#           Adapts the model to the subject's most recent data.
#           Overwrites models/<S>/<M>/ in place.
# ════════════════════════════════════════════════════════════
log "═══ STEP 7: Online update (on personalized models) ═══"
for S in "${SUBJECTS[@]}"; do
    for M in "${DEEP_MODELS[@]}"; do
        MDIR="$ROOT/models/$S/finetuned_$M"
        if [ -f "$MDIR/meta.joblib" ]; then
            run "step7_online_${S}_${M}" scripts/run_online_update.py \
                --subject "$S" --model-dir "$MDIR" --config "$CFG"
        else
            log "SKIP   step7_online_${S}_${M}  (no personalized model)"
        fi
    done
done

# ════════════════════════════════════════════════════════════
# STEP 8 — Simulator validation (Scenarios A–G) + plots
# ════════════════════════════════════════════════════════════
log "═══ STEP 8: Simulation ═══"
run "step8_simulation"      scripts/run_simulation.py  --config "$CFG"
run "step8_plot_simulation" scripts/plot_simulation.py --config "$CFG" --out-dir "$ROOT/plots"

# ════════════════════════════════════════════════════════════
log "All steps complete. Logs in $LOG_DIR/"
log "Models  -> $ROOT/models/"
log "Results -> $ROOT/processed/"
log "Plots   -> $ROOT/plots/"
