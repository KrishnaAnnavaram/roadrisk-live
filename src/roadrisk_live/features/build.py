"""The feature table and the preprocessing pipeline. Training and live scoring call the same functions.

Model input columns (``FEATURES``):

* weather: temperature_f, humidity_pct, pressure_in, visibility_mi, wind_speed_mph, precipitation_in,
  weather_cat (shared vocabulary)
* time (LOCAL observation time): hour, weekday, month, is_weekend, is_rush_hour, is_night
* location: state, geo_cell (0.5-degree grid cell from latitude and longitude)
* road: the 13 point-of-interest flags of US-Accidents (junction, traffic signal, crossing, ...)

City is not a feature. The prototype set City = 0 at inference, which is a real city code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .weather import RANGES

NUMERIC = ["temperature_f", "humidity_pct", "pressure_in", "visibility_mi", "wind_speed_mph", "precipitation_in",
           "hour", "weekday", "month"]
FLAGS = ["is_weekend", "is_rush_hour", "is_night", "junction", "traffic_signal", "crossing", "station", "stop",
         "amenity", "bump", "give_way", "no_exit", "railway", "roundabout", "traffic_calming"]
CATEGORICAL = ["weather_cat", "state", "geo_cell"]
FEATURES = NUMERIC + FLAGS + CATEGORICAL


def geo_cell(lat: float, lng: float, size: float = 0.5) -> str:
    if lat is None or lng is None or pd.isna(lat) or pd.isna(lng):
        return "unknown"
    return f"{np.floor(lat / size) * size:.1f}_{np.floor(lng / size) * size:.1f}"


def time_features(local_time: pd.Series, night: pd.Series | None = None) -> pd.DataFrame:
    t = pd.to_datetime(local_time)
    out = pd.DataFrame({"hour": t.dt.hour, "weekday": t.dt.weekday, "month": t.dt.month})
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["is_rush_hour"] = (out["hour"].isin([7, 8, 9, 16, 17, 18]) & (out["is_weekend"] == 0)).astype(int)
    if night is None:
        out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 20)).astype(int)
    else:
        out["is_night"] = night.astype(int).to_numpy()
    return out


def clip_to_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Set physically impossible values to missing. Return the count per column."""
    out = df.copy()
    counts = {}
    for col, (lo, hi) in RANGES.items():
        if col in out:
            bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
            counts[col] = int(bad.sum())
            out.loc[bad, col] = np.nan
    return out, counts


def check_feature_frame(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame misses columns: {missing}")


def make_preprocessor(min_frequency: int = 20) -> ColumnTransformer:
    """Imputation, scaling and one-hot encoding. Unknown or rare categories go to one 'infrequent' column."""
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("flag", SimpleImputer(strategy="constant", fill_value=0), FLAGS),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                                   sparse_output=False))]), CATEGORICAL),
    ])
