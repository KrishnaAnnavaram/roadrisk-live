"""Load US-Accidents into the feature table, validate it and split it by TIME.

* All rows are used by default (chunked reading). ``max_rows`` is only for quick trials.
* ``Start_Time`` in US-Accidents is the local time of the accident, so the time features are local.
* The split is by date: train up to ``train_end``, valid up to ``valid_end``, test after it.
  A random split mixes years, and the severity distribution of US-Accidents changes by year and source.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..features.build import FEATURES, clip_to_ranges, geo_cell, time_features
from ..features.weather import from_text

RAW_COLUMNS = {
    "ID": "id", "Source": "source", "Severity": "severity", "Start_Time": "start_time", "Start_Lat": "lat",
    "Start_Lng": "lng", "State": "state", "Temperature(F)": "temperature_f", "Humidity(%)": "humidity_pct",
    "Pressure(in)": "pressure_in", "Visibility(mi)": "visibility_mi", "Wind_Speed(mph)": "wind_speed_mph",
    "Precipitation(in)": "precipitation_in", "Weather_Condition": "weather_condition",
    "Sunrise_Sunset": "sunrise_sunset", "Amenity": "amenity", "Bump": "bump", "Crossing": "crossing",
    "Give_Way": "give_way", "Junction": "junction", "No_Exit": "no_exit", "Railway": "railway",
    "Roundabout": "roundabout", "Station": "station", "Stop": "stop", "Traffic_Calming": "traffic_calming",
    "Traffic_Signal": "traffic_signal",
}
SPLITS = ("train", "valid", "test")


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)


def to_feature_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename, validate and derive the features. Returns the table and a validation report."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"input misses columns: {missing}")
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    report = {"rows_in": len(df)}
    df["start_time"] = pd.to_datetime(df["start_time"].astype(str).str.slice(0, 19), errors="coerce")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    bad = df["start_time"].isna() | ~df["severity"].isin([1, 2, 3, 4])
    report["dropped_bad_time_or_severity"] = int(bad.sum())
    df = df[~bad].copy()
    df["severity"] = df["severity"].astype(int)
    for col in ("temperature_f", "humidity_pct", "pressure_in", "visibility_mi", "wind_speed_mph",
                "precipitation_in", "lat", "lng"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df, report["set_missing_out_of_range"] = clip_to_ranges(df)
    night = df["sunrise_sunset"].astype(str).str.lower().eq("night")
    has_night = df["sunrise_sunset"].notna()
    t = time_features(df["start_time"])
    t.loc[has_night.to_numpy(), "is_night"] = night[has_night].astype(int).to_numpy()
    for col in t.columns:
        df[col] = t[col].to_numpy()
    for col in ("amenity", "bump", "crossing", "give_way", "junction", "no_exit", "railway", "roundabout", "station",
                "stop", "traffic_calming", "traffic_signal"):
        df[col] = _flag(df[col])
    df["weather_cat"] = df["weather_condition"].map(from_text)
    df["geo_cell"] = [geo_cell(a, b) for a, b in zip(df["lat"], df["lng"])]
    df["state"] = df["state"].fillna("unknown").astype(str)
    report["rows_out"] = len(df)
    keep = ["id", "source", "severity", "start_time"] + FEATURES
    return df[keep].reset_index(drop=True), report


def load_accidents(path: str | Path, max_rows: int | None = None, chunksize: int = 500_000):
    """Read the Kaggle CSV (or a parquet file with the ``parquet`` extra) in chunks."""
    path = Path(path)
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path, columns=list(RAW_COLUMNS))
        return to_feature_table(raw.head(max_rows) if max_rows else raw)
    tables, reports, read = [], [], 0
    for chunk in pd.read_csv(path, usecols=list(RAW_COLUMNS), chunksize=chunksize, low_memory=False):
        if max_rows is not None:
            chunk = chunk.head(max_rows - read)
        table, rep = to_feature_table(chunk)
        tables.append(table)
        reports.append(rep)
        read += len(chunk)
        if max_rows is not None and read >= max_rows:
            break
    merged = {k: sum(r[k] for r in reports) for k in ("rows_in", "dropped_bad_time_or_severity", "rows_out")}
    merged["set_missing_out_of_range"] = {
        c: sum(r["set_missing_out_of_range"].get(c, 0) for r in reports) for c in reports[0]["set_missing_out_of_range"]}
    return pd.concat(tables, ignore_index=True), merged


def time_split(df: pd.DataFrame, train_end: date, valid_end: date) -> pd.DataFrame:
    out = df.copy()
    day = out["start_time"].dt.date
    out["split"] = "test"
    out.loc[day <= valid_end, "split"] = "valid"
    out.loc[day <= train_end, "split"] = "train"
    return out


def describe_splits(df: pd.DataFrame) -> dict:
    """Rows, date range, severity shares and source shares per split (source and year drift is visible)."""
    out = {}
    for name in SPLITS:
        part = df[df["split"] == name]
        if part.empty:
            out[name] = {"rows": 0}
            continue
        out[name] = {
            "rows": len(part),
            "from": str(part["start_time"].min().date()), "to": str(part["start_time"].max().date()),
            "severity_share": part["severity"].value_counts(normalize=True).sort_index().round(4).to_dict(),
            "source_share": part["source"].value_counts(normalize=True).round(4).to_dict(),
        }
    return out


def daily_counts(df: pd.DataFrame, state: str | None = None) -> pd.Series:
    """Accidents per calendar day (days with no row count as 0)."""
    part = df if state is None else df[df["state"] == state]
    counts = part.groupby(part["start_time"].dt.normalize()).size()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full, fill_value=0).rename("accidents")
