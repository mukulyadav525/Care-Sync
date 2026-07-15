"""Minimal dependency-free rate limiting for the ai/ FastAPI service.

There's no auth layer in front of this service (it's a sibling of the Django
backend, not behind its JWT check), so without this, `/chat` (an LLM call —
real cost per request) and the `/hrv/*` compute endpoints are wide open to
anyone who can reach the port. This adds a simple fixed-window, per-client-IP
limiter — no extra dependency (no slowapi/redis), fine for a single-process
deployment. If you scale to multiple ai/ workers behind a load balancer,
replace the in-memory store with Redis so limits are shared across processes.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# path prefix -> (max requests, window seconds). First match (by prefix) wins;
# falls back to _DEFAULT_LIMIT if nothing matches.
_LIMITS: list[tuple[str, int, int]] = [
    ("/chat", 15, 60),          # LLM calls are the most expensive path
    ("/hrv/forecast", 30, 60),
    ("/hrv/anomaly", 30, 60),
    ("/hrv/digital-twin", 30, 60),
    ("/hrv/status", 60, 60),
    ("/trends", 60, 60),
    ("/graph-data", 60, 60),
]
_DEFAULT_LIMIT = (120, 60)  # generous default for anything unlisted (e.g. /health)

# client_key -> deque of request timestamps within the current window
_hits: dict[str, deque] = defaultdict(deque)


def _limit_for(path: str) -> tuple[int, int]:
    for prefix, count, window in _LIMITS:
        if path.startswith(prefix):
            return count, window
    return _DEFAULT_LIMIT


def _client_key(request: Request) -> str:
    # Respect a reverse proxy's forwarded IP (Caddy/nginx/Cloudflare tunnel all
    # sit in front of this in production); fall back to the raw peer address.
    fwd = request.headers.get("x-forwarded-for")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    return f"{ip}:{request.url.path}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        limit, window = _limit_for(request.url.path)
        key = _client_key(request)
        now = time.monotonic()
        q = _hits[key]

        while q and now - q[0] > window:
            q.popleft()

        if len(q) >= limit:
            retry_after = max(1, int(window - (now - q[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests, slow down.", "retry_after_s": retry_after},
                headers={"Retry-After": str(retry_after)},
            )

        q.append(now)
        return await call_next(request)
