"""Optional HTTP API (extra ``api``): ``GET /health``, ``GET /score?lat=..&lon=..``.

The handlers are plain functions in ``handle_score`` so that they are testable without FastAPI.
"""

from __future__ import annotations

import hmac

from ..config import Settings, load_settings
from ..severity import load_bundle
from .scoring import ObservationError, score
from .weather import FixtureWeatherProvider, OpenWeatherClient


def make_provider(settings: Settings):
    if settings.openweather_api_key:
        return OpenWeatherClient(settings.openweather_api_key, settings.openweather_url, settings.timeout_s)
    return FixtureWeatherProvider()


def check_token(settings: Settings, header: str | None) -> bool:
    if not settings.api_token:
        return True
    given = (header or "").removeprefix("Bearer ").strip()
    return hmac.compare_digest(given, settings.api_token)


def handle_score(bundle: dict, provider, lat: float, lon: float) -> tuple[int, dict]:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return 422, {"error": "lat must be in [-90, 90] and lon in [-180, 180]"}
    try:
        return 200, score(bundle, provider.current(lat, lon))
    except ObservationError as exc:
        return 422, {"error": str(exc)}


def create_app(settings: Settings | None = None):
    from fastapi import FastAPI, Header, HTTPException

    settings = settings or load_settings()
    bundle = load_bundle(settings.model_dir)
    provider = make_provider(settings)
    app = FastAPI(title="roadrisk-live", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": bundle["model"], "live_weather": bool(settings.openweather_api_key)}

    @app.get("/score")
    def score_endpoint(lat: float, lon: float, authorization: str | None = Header(default=None)):
        if not check_token(settings, authorization):
            raise HTTPException(status_code=401, detail="invalid token")
        status, body = handle_score(bundle, provider, lat, lon)
        if status != 200:
            raise HTTPException(status_code=status, detail=body["error"])
        return body

    return app
