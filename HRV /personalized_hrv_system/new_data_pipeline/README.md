# new_data_pipeline — run a trained model on a NEW dataset

Test a model that was already trained (on WESAD / Stress-Predict / the pooled set)
against a **different** dataset, and get a live **"predict the next N minutes"**
forecast for any user. Nothing is retrained here — these tools only *load* a model
directory and score it on new data.

The model directory's `meta.joblib` is the source of truth for which features /
targets / input-window length the model expects, so this works with any of:

| Model dir                     | What it is                                  |
|-------------------------------|---------------------------------------------|
| `models/multi/<m>`            | pooled multi-dataset model *(recommended)*  |
| `models/global/<m>`           | global pretrained model                     |
| `models/<S>/finetuned_<m>`    | a personalized per-subject model            |

where `<m>` ∈ `tcn | lstm | gru | transformer | xgboost`.

> The dataset must be an **Empatica-E4 export** — the same `HR/IBI/ACC/BVP/EDA/TEMP`
> CSV format as WESAD / Stress-Predict, either flat (`<SID>/HR.csv`) or nested
> (`<SID>/<SID>_E4_Data/HR.csv`). Both layouts are auto-detected. A new dataset that
> is missing some sensors still works: the missing feature columns are back-filled
> with the model's training mean (a warning lists them).

---

## 0. Point the config at your dataset

Edit [`config_newdata.yaml`](config_newdata.yaml) → `data.raw_root`. The path is
**relative to the parent of the project dir** (the `HRV` folder), e.g. `raw_root: "NEW_DATASET"`
resolves to `../NEW_DATASET`. An absolute path also works, or override per-command
with `--raw-root` / `--data`.

The `cleaning` / `features` / `model.horizons_s` blocks are copied from the
training config and **must stay matched to it** — don't change them unless the
model was trained differently.

## 1. Evaluate on the whole dataset (metrics)

```bash
# from the personalized_hrv_system/ directory
python new_data_pipeline/evaluate_model.py \
    --config new_data_pipeline/config_newdata.yaml \
    --model tcn --model-dir models/multi/tcn
```

Reports two metric families per horizon (1 / 5 / 10 min):

* **Regression** — MAE, RMSE, MAPE, R², bias, Pearson r, and 90/95% prediction-
  interval coverage, for HR and RMSSD (overall + per-subject distribution).
* **Classification** — **precision / recall / F1**, specificity, accuracy and MCC,
  by framing the forecast as an **"elevated-HR event"** detector: an event is
  HR crossing a threshold at that horizon. Threshold is either person-relative
  (`mean + k·std`, default) or absolute bpm.

```bash
# absolute tachycardia threshold instead of person-relative:
python new_data_pipeline/evaluate_model.py --model tcn --model-dir models/multi/tcn \
    --event-mode absolute --event-bpm 100

# only some subjects, score every 5th second (faster):
python new_data_pipeline/evaluate_model.py --model tcn --model-dir models/multi/tcn \
    --subjects S2 S5 --stride 5
```

Outputs land in `new_data_pipeline/results/<model_dir_name>/`:
`metrics_report.json`, `metrics_summary.txt`, and a per-subject
`<S>_<model>_inference.csv` (same format as the main pipeline, so you can plot it
with `scripts/plot_inference.py`).

## 2. Predict the next few minutes for one user

```bash
python new_data_pipeline/predict_next.py \
    --config new_data_pipeline/config_newdata.yaml \
    --model tcn --model-dir models/multi/tcn --subject S2

# or point straight at a folder of E4 CSVs:
python new_data_pipeline/predict_next.py \
    --model tcn --model-dir models/global/tcn --data /home/me/somebody/E4 \
    --out forecast.json
```

Takes the most recent clean input window and prints the forecast at each horizon
with 95% prediction intervals and the absolute time each prediction is *for*.

## 3. One-shot convenience runner

```bash
bash new_data_pipeline/run_new_dataset.sh models/multi/tcn          # tcn
bash new_data_pipeline/run_new_dataset.sh models/multi/xgboost xgboost
```

Runs the full evaluation, then a demo forecast for the first subject.

---

### What "elevated-HR event" precision/recall/F1 means

A forecaster outputs a number, not a class — so to get precision/recall/F1 we
define a binary clinical-style event and grade the model on whether it *anticipates*
it at each horizon:

* **event (truth)** = the true future HR at horizon *h* exceeds the threshold,
* **event (pred)**  = the model's predicted HR at horizon *h* exceeds the *same* threshold,
* then precision / recall / F1 / specificity are computed over all windows.

`--event-mode personal_sigma --event-k 1.0` (default) makes the threshold
person-relative (`mean + 1·std` of that recording's true HR). Use
`--event-mode absolute --event-bpm 100` for a fixed tachycardia line. Increase
`--event-k` (e.g. 1.5–2.0) for rarer, more extreme events.
