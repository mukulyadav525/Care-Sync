from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai.routes.trends import router as trends_router
from ai.routes.hrv import router as hrv_router
import pandas as pd
from fastapi.responses import JSONResponse

from langchain_core.messages import HumanMessage

from ai import config
from ai.workflow import graph
from ai.services.session_memory import get_history
from ai.rate_limit import RateLimitMiddleware


app = FastAPI(
    title="CareSync AI",
    version="1.0"
)
app.include_router(trends_router)
app.include_router(hrv_router)
# Rate limiting first (innermost concerns run last in Starlette's middleware
# stack, so adding this after CORS below means CORS headers still get added
# to 429 responses too).
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    # Localhost + whatever's in CORS_ALLOWED_ORIGINS (set this to your
    # Netlify URL / custom domain in production — see ai/.env.example).
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "CareSync AI"
    }


@app.get("/health")
def health():
    """Liveness/readiness probe for the process manager / load balancer.
    Deliberately dependency-free (no HRV model or LLM calls) so it stays
    fast and never false-fails because a downstream service is degraded."""
    return {"status": "ok"}
@app.get("/graph-data")
async def graph_data():
    df = pd.read_csv("ai/data/sample_ppg.csv")

    return JSONResponse(
        content=df.to_dict(orient="records")
    )


@app.post("/chat")
async def chat(req: ChatRequest):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=req.question)
            ],
            "session_id": req.session_id,
            "history": get_history(req.session_id),
            "intent": "",
            "analysis": {},
            "response": ""
        }
    )

    return {
        "response": result["response"],
        "analysis": result["analysis"],
        "intent": result["intent"]
    }