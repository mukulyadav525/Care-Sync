import os
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")

MODEL_NAME = "grok-4"

TEMPERATURE = 0.2

# ----------------- Deployment / CORS -----------------
# Comma-separated list of origins allowed to call this API (your Netlify
# frontend URL, custom domain, etc). Localhost is always allowed so local
# dev keeps working. Example:
#   CORS_ALLOWED_ORIGINS=https://caresync.netlify.app,https://caresync.yourdomain.com
_DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
CORS_ALLOWED_ORIGINS = _DEFAULT_ORIGINS + [
    o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
]

# Set to the directory containing a trained HRV checkpoint (meta.joblib,
# scaler.joblib, <model>_model.pt or xgb/) to serve real forecasts instead
# of the mock engine. See ai/hrv_forecast/model_loader.py and
# docs/HRV_INTEGRATION.md.
HRV_MODEL_DIR = os.environ.get("HRV_MODEL_DIR", "")
