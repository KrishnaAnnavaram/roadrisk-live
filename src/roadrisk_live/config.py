"""Settings from environment variables. The API key is read from the environment only, never from code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models_out")
    seed: int = 13
    train_end: date = date(2021, 12, 31)
    valid_end: date = date(2022, 12, 31)
    openweather_api_key: str | None = field(default=None, repr=False)
    openweather_url: str = "https://api.openweathermap.org"
    timeout_s: float = 10.0
    api_token: str | None = field(default=None, repr=False)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)
    try:
        seed = int(env.get("ROADRISK_SEED") or 13)
        timeout = float(env.get("ROADRISK_TIMEOUT_S") or 10.0)
    except ValueError as exc:
        raise ValueError("ROADRISK_SEED must be an integer and ROADRISK_TIMEOUT_S a number") from exc
    train_end = date.fromisoformat(env.get("ROADRISK_TRAIN_END") or "2021-12-31")
    valid_end = date.fromisoformat(env.get("ROADRISK_VALID_END") or "2022-12-31")
    if not train_end < valid_end:
        raise ValueError("ROADRISK_TRAIN_END must be before ROADRISK_VALID_END")
    return Settings(
        data_dir=Path(env.get("ROADRISK_DATA_DIR") or "data"),
        model_dir=Path(env.get("ROADRISK_MODEL_DIR") or "models_out"),
        seed=seed,
        train_end=train_end,
        valid_end=valid_end,
        openweather_api_key=env.get("OPENWEATHER_API_KEY") or None,
        openweather_url=(env.get("ROADRISK_OPENWEATHER_URL") or "https://api.openweathermap.org").rstrip("/"),
        timeout_s=timeout,
        api_token=env.get("ROADRISK_API_TOKEN") or None,
    )
