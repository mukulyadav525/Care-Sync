# HRV model integration

How the research system in `HRV /personalized_hrv_system` connects to the
rest of Care-Sync, what's real vs. mocked today, and how to deploy it.

## Where it lives

| Piece | Location | Role |
|---|---|---|
| Research pipeline | `HRV /personalized_hrv_system/src` | Data loading, feature engineering, forecasting models (TCN/LSTM/GRU/Transformer/XGBoost), anomaly scoring, digital twin. |
| API surface | `ai/hrv_forecast/` + `ai/routes/hrv.py` | FastAPI endpoints (`/hrv/forecast`, `/hrv/anomaly`, `/hrv/digital-twin`, `/hrv/status`) the frontend/backend call. Stable schemas regardless of what's behind them. |
| Serving app | `ai/app.py` | Existing FastAPI service (chat, trends). `/hrv/*` is mounted here, port 8001. |

## What's real vs. mock, as of this pass

| Endpoint | Status | Needs a trained checkpoint? |
|---|---|---|
| `POST /hrv/anomaly` | **Real.** Runs the actual feature-engineering pipeline (`src/features/build_features.py`) — HR/HRV rolling features, circadian baseline, physiological-state classification, subject-relative baselines — then the real `src/anomaly/explain.py` illness scoring and `src/personalization/digital_twin.py` expected-HR comparison. | No — pure numpy/pandas/scipy. |
| `POST /hrv/digital-twin` | **Real.** Calls `src/personalization/digital_twin.build_digital_twin()` directly on the engineered feature table. | No |
| `POST /hrv/forecast` | **Mock** (persistence heuristic) until `HRV_MODEL_DIR` points at a trained checkpoint. The full adapter (`ai/hrv_forecast/real_engine.py::forecast`) is written and will serve real predictions automatically once one loads. | Yes |
| `GET /hrv/status` | Reports which mode forecast is in (`trained`/`mock`) and why. | — |

Every anomaly/digital-twin response also carries `model_status`: `"pipeline"`
when the real feature engineering ran, `"mock"` only if it fell back (e.g.
fewer than 2 samples sent). Forecast responses use `"trained"` or `"mock"`.
If the real path throws for any reason, `ai/hrv_forecast/service.py` logs it
and falls back to mock rather than erroring the request — so `/hrv/*` never
500s just because a particular subject's data is thin.

One data-shape note: the pipeline computes RMSSD from raw IBI beats
(`samples[].ibi`). If callers only send a precomputed `samples[].rmssd`
instead, `feature_adapter.py` adds a fallback `RMSSD_provided` column so
anomaly/digital-twin still use real HRV numbers either way.

## Personalization: how a model runs per user

`HRV_MODEL_DIR` is a **base directory**, not a single model. The loader
(`ai/hrv_forecast/model_loader.py`) resolves a checkpoint per request using
this order, and caches each directory it loads (so many users sharing the
global model only pay the load cost once):

```
HRV_MODEL_DIR/<subject_id>/     <- that user's personally fine-tuned model (used if it exists)
HRV_MODEL_DIR/global/           <- population baseline (used for every user without a personal model yet)
HRV_MODEL_DIR/  (flat)          <- back-compat: a single checkpoint straight in HRV_MODEL_DIR, used for everyone
```

`subject_id` is whatever the frontend sends as `owner`/username on
`/hrv/forecast|anomaly|digital-twin`. `GET /hrv/status?subject_id=<user>`
tells you which tier is currently serving that specific user
(`personalized: true/false` + `checkpoint_dir`).

### 1. Train the baseline (global/population) model — do this once

The research pipeline in `HRV /personalized_hrv_system` expects Empatica-E4
style CSVs (`HR.csv`, `BVP.csv`, `IBI.csv`, `ACC.csv`, `EDA.csv`, `TEMP.csv`,
`tags*.csv`) inside one folder per subject. `configs/config.yaml` points at
the open-source demo dataset (`data.raw_root: Dataset3/Raw_data`) by default
— that's fine for the baseline, since its whole point is generic population
signal, not any one person's data.

```bash
cd "HRV /personalized_hrv_system"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, joblib, xgboost, scikit-learn etc.

# Trains on every S## subject folder found under the configured raw_root
python scripts/run_pretraining.py --all --model tcn --out ../../hrv_models/global
```

That writes `meta.joblib`, `scaler.joblib`, `tcn_model.pt` into
`hrv_models/global/` (adjust `--out` to wherever you want `HRV_MODEL_DIR` to
live — it doesn't have to be inside the repo).

### 2. Point the ai/ service at it

```bash
# ai/.env
HRV_MODEL_DIR=/absolute/path/to/hrv_models
```

Restart the service (`systemctl --user restart ai-service` or however you're
running uvicorn). `GET /hrv/status` now reports `"model_status": "trained"`,
`"personalized": false` (everyone's on the shared global model) — that's the
expected state right after step 1, for every user.

### 3. Personalize per user — fine-tune on their real Care-Sync sessions

Care-Sync stores each recorded session at `backend/Users/<username>/<session_name>/`
with exactly the same file layout the research pipeline reads (`HR.csv`,
`IBI.csv`, `ACC.csv`, ...). So a real user's own session can be fed straight
in — no format conversion needed. Point `--raw-root` at the backend's Users
folder and use `<username>/<session_name>` as the subject:

```bash
cd "HRV /personalized_hrv_system"
source .venv/bin/activate

python scripts/run_finetune.py \
  --subject mukul23/session_demo \
  --raw-root ../../backend/Users \
  --pretrained ../../hrv_models/global \
  --out ../../hrv_models/mukul23
```

That fine-tunes the global weights on just that user's recording (small
learning rate, few epochs — see `run_finetune.py --help` for
`--epochs`/`--lr`/`--freeze-backbone`) and saves the result to
`hrv_models/mukul23/`. No restart or env change needed — `model_loader.py`
resolves `HRV_MODEL_DIR/mukul23/` automatically on the next request for that
user, and only that user gets served the personal model; everyone else keeps
using `global/`. `GET /hrv/status?subject_id=mukul23` should now report
`"personalized": true`.

**Requirements**: fine-tuning needs at least ~20 training windows once
features are built (`input_seq_len_s=120`, `stride_s=5` in `config.yaml`) —
in practice that means at least a few tens of minutes of continuous
recording per session; longer/more sessions personalize better. If a user
doesn't have enough data yet, `run_finetune.py` raises a clear
`ValueError` rather than saving a bad model — just keep serving them the
global model until they've recorded more.

**Current limitation**: fine-tuning reads one subject *folder* at a time.
If a user has several sessions and you want to combine them into one
fine-tune run rather than just using their longest single session, that
needs a small merge step (concatenating each signal's real timestamps across
sessions) that isn't built yet — flagged here rather than silently skipped.
For now, pick the user's richest single session as `--subject
<username>/<best_session>`, or re-run fine-tuning against a newer session as
it becomes available (each run overwrites `hrv_models/<username>/`).

### 4. Keep it current — re-run fine-tuning as data accumulates

There's no cron job wired up yet. The simplest path: re-run step 3's command
periodically (e.g. weekly, or after a user logs a long new session) — it's
idempotent and just overwrites that user's directory. Automating this via
the `online.*` config block (`scripts/run_online_update.py`, incremental
nightly updates on recent windows) is a documented but not yet wired-up next
step — see that script's docstring.

## Deploying today — checklist

**Server (Django backend + AI service):**
- [ ] `backend/deploy/gunicorn.service` → Django on `127.0.0.1:8000` (existing).
- [ ] `backend/deploy/ai-service.service` → uvicorn on `127.0.0.1:8001` (new — copy to `/etc/systemd/system/`, `systemctl enable --now ai-service`).
- [ ] Reverse proxy: either `backend/deploy/Caddyfile` / `nginx.conf` (now has both `api.yourdomain.com` → 8000 and `ai.yourdomain.com` → 8001 blocks) if you own a domain, or `backend/deploy/cloudflare-tunnel.yml` (now has both hostnames) for a no-domain setup.
- [ ] `backend/.env` and `ai/.env` created from their `.env.example` files with real values (`DEBUG=False`, `DJANGO_SECRET_KEY`, `ALLOWED_HOSTS`, `CORS_ALLOWED_ORIGINS` on the Django side; `XAI_API_KEY`, `CORS_ALLOWED_ORIGINS`, optionally `HRV_MODEL_DIR` on the ai side).
- [ ] `CORS_ALLOWED_ORIGINS` on **both** services includes your real Netlify URL (e.g. `https://caresync.netlify.app`) — without this the frontend's requests will be blocked by the browser even though the server is up.

**Netlify (frontend):**
- [ ] `netlify.toml` → `NEXT_PUBLIC_API_URL` set to your Django URL + `/api`.
- [ ] `netlify.toml` → `NEXT_PUBLIC_AI_API_URL` (new) set to your ai service URL — currently a placeholder, must be filled in before deploying or the dashboard's chat/trends widgets will try to hit `127.0.0.1:8001`.
- [ ] Trigger a Netlify deploy after changing `netlify.toml` env vars (build-time env, not runtime).

The frontend calls `/hrv/forecast`, `/hrv/anomaly`, and `/hrv/digital-twin`
from `frontend/src/components/HRVInsights.tsx`, mounted on the session detail
page (`/portal/[owner]/[session]`) — forecast chart, resting/sleep/walking
vitals, and anomaly alert banner (with buzz + email) all come from that
component.

## Repo hygiene notes

- The `HRV ` folder name has a trailing space (pre-existing). Nothing in the
  integration code depends on changing it, so it wasn't renamed, but it's
  worth fixing eventually.
- `.venv/` directories (`ai/`, `backend/`, `HRV /personalized_hrv_system/`) are
  gitignored.
- Datasets, trained checkpoints, and notebook checkpoint folders under
  `HRV /personalized_hrv_system` are gitignored — large/binary/environment-specific,
  shouldn't be committed. Your server-trained checkpoints (`hrv_models/`)
  stay on the server and are referenced by path via `HRV_MODEL_DIR`, not
  committed to git.
- `device-ingestion/user_session_server.py` and `start_session_server.py`
  were removed — they were leftover duplicate Flask scripts from an earlier
  hardware setup (hardcoded to a different machine's paths,
  `/home/megha21337/...`), never called by anything in this repo. The only
  ingestion script actually wired to the backend is `heartbeat.py` (posts to
  `/api/devices/heartbeat/`), which stays.
