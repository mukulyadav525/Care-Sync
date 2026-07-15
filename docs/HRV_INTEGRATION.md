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

## Deploying the trained forecaster (what's left)

You said you're training a baseline model on your own server — once that's
done:

1. Confirm the checkpoint directory has what `src/pipeline/inference_pipeline.load_model()`
   expects: `meta.joblib`, `scaler.joblib`, and either `<model_type>_model.pt`
   (torch models) or an `xgb/` directory (XGBoost).
2. `pip install torch joblib xgboost scikit-learn` inside `ai/.venv` (commented
   out in `ai/requirements.txt` today so the mock-only path stays light).
3. Set `HRV_MODEL_DIR=/absolute/path/to/checkpoint` in `ai/.env`.
4. Restart the `ai` service (`sudo systemctl restart ai-service` if using the
   systemd unit below). `GET /hrv/status` should flip to `"model_status": "trained"`.
5. Nothing else changes — `/hrv/forecast`'s request/response shape is
   identical in mock and trained mode.

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

**Known gap:** the frontend doesn't call `/hrv/forecast`, `/hrv/anomaly`, or
`/hrv/digital-twin` yet — there's no dashboard widget for them. The API is
live and real (see table above); wiring it into a UI is a separate follow-up
if you want it visualized.

## Repo hygiene notes

- The `HRV ` folder name has a trailing space (pre-existing). Nothing in the
  integration code depends on changing it, so it wasn't renamed, but it's
  worth fixing eventually.
- `.venv/` directories (`ai/`, `backend/`, `HRV /personalized_hrv_system/`) are
  gitignored.
- Datasets, trained checkpoints, and notebook checkpoint folders under
  `HRV /personalized_hrv_system` are gitignored — large/binary/environment-specific,
  shouldn't be committed. Your server-trained checkpoint stays on the server
  and is referenced by path via `HRV_MODEL_DIR`, not committed to git.
