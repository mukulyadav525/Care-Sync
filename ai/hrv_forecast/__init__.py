"""HRV forecasting/anomaly/digital-twin service layer for the ai/ FastAPI app.

Wraps the research pipeline in `HRV /personalized_hrv_system` behind a stable
API contract (see schemas.py). Currently serves a deterministic mock engine
because no trained checkpoint exists yet; model_loader.py will pick up a
real one automatically once HRV_MODEL_DIR is configured. See
docs/HRV_INTEGRATION.md for the full picture.
"""
