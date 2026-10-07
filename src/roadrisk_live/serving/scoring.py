"""Observation -> feature row -> severity probabilities, with the SAME feature functions as training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build import FEATURES, FLAGS, geo_cell, time_features
from ..features.weather import RANGES, VOCAB
from .weather import Observation

ROAD_FLAGS = [f for f in FLAGS if f not in ("is_weekend", "is_rush_hour", "is_night")]


class ObservationError(ValueError):
    """Raised when a live value is outside its physical range (for example a Kelvin value read as Fahrenheit)."""


def feature_row(obs: Observation, road: dict[str, int] | None = None) -> pd.DataFrame:
    """One feature row. Road flags are unknown for a live point unless the caller gives them (default 0)."""
    values = {"temperature_f": obs.temperature_f, "humidity_pct": obs.humidity_pct, "pressure_in": obs.pressure_in,
              "visibility_mi": obs.visibility_mi, "wind_speed_mph": obs.wind_speed_mph,
              "precipitation_in": obs.precipitation_in}
    for col, value in values.items():
        lo, hi = RANGES[col]
        if value is not None and not lo <= value <= hi:
            raise ObservationError(f"{col} = {value:.1f} is outside [{lo}, {hi}]: check the units")
    if obs.weather_cat not in VOCAB:
        raise ObservationError(f"unknown weather category {obs.weather_cat!r}")
    t = time_features(pd.Series([obs.local_time]), night=pd.Series([obs.is_night]))
    row = {**{k: (np.nan if v is None else v) for k, v in values.items()}, **t.iloc[0].to_dict(),
           "weather_cat": obs.weather_cat, "state": obs.state or "unknown", "geo_cell": geo_cell(obs.lat, obs.lng)}
    for flag in ROAD_FLAGS:
        row[flag] = int((road or {}).get(flag, 0))
    return pd.DataFrame([row])[FEATURES]


def score(bundle: dict, obs: Observation, road: dict[str, int] | None = None) -> dict:
    row = feature_row(obs, road)
    raw = bundle["pipeline"].predict_proba(row)[0]
    probs = {int(c): float(p) for c, p in zip(bundle["pipeline"].classes_, raw)}
    probs = {c: probs.get(c, 0.0) for c in bundle["classes"]}
    return {
        "place": obs.place, "state": obs.state, "observed_utc": obs.observed_utc.isoformat(),
        "local_time": obs.local_time.isoformat(), "weather_cat": obs.weather_cat,
        "temperature_f": None if obs.temperature_f is None else round(obs.temperature_f, 1),
        "predicted_severity": max(probs, key=probs.get),
        "expected_severity": round(sum(c * p for c, p in probs.items()), 3),
        "probabilities": {str(c): round(p, 4) for c, p in probs.items()},
        "model": bundle["model"],
        "note": "Severity of a reported accident under these conditions, not the chance of an accident.",
    }
