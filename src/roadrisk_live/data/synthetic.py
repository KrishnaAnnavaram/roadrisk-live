"""Synthetic US-Accidents-style rows and daily counts (no download, no key).

Built-in structure, on purpose:

* Severity depends on weather (snow, ice, heavy rain, low visibility), night, speed-like road context
  (no traffic signal) and the state. Severity 2 dominates (about 70 %), as in the real data.
* Two sources with different severity mixes, and a drift by year (more severity 2 in later years).
* Daily counts with a weekly cycle, a trend, winter peaks and random holidays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["CA", "TX", "FL", "NY", "PA", "MN", "CO", "WA"]
CENTRES = {"CA": (36.5, -119.5), "TX": (31.0, -99.0), "FL": (28.0, -82.0), "NY": (42.5, -75.5),
           "PA": (40.9, -77.6), "MN": (46.0, -94.0), "CO": (39.0, -105.5), "WA": (47.4, -120.5)}
CONDITIONS = ["Fair", "Clear", "Mostly Cloudy", "Overcast", "Light Rain", "Rain", "Heavy Rain", "Light Snow",
              "Snow", "Light Freezing Rain", "Fog", "Haze", "T-Storm", "Light Drizzle"]
_COND_P = np.array([30, 8, 14, 10, 9, 4, 2, 4, 2, 1, 3, 3, 2, 3], dtype=float)
_RISK = {"Heavy Rain": 0.9, "Snow": 1.0, "Light Freezing Rain": 1.3, "Fog": 0.8, "T-Storm": 0.7, "Light Snow": 0.6,
         "Rain": 0.4, "Light Rain": 0.2}
FLAG_COLS = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station",
             "Stop", "Traffic_Calming", "Traffic_Signal"]


def make_accidents(n: int = 20000, seed: int = 13, start: str = "2018-01-01", end: str = "2023-03-31") -> pd.DataFrame:
    """Rows with the raw US-Accidents column names that ``to_feature_table`` reads."""
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    times = t0 + pd.to_timedelta(rng.random(n) * (t1 - t0).total_seconds(), unit="s")
    times = pd.DatetimeIndex(times).floor("min")
    state = rng.choice(STATES, n)
    cond = rng.choice(CONDITIONS, n, p=_COND_P / _COND_P.sum())
    winter = np.isin(times.month, [12, 1, 2])
    cold_state = np.isin(state, ["MN", "NY", "PA", "CO", "WA"])
    temp = rng.normal(62, 15, n) - 25 * (winter & cold_state)
    snowy = np.isin(cond, ["Light Snow", "Snow", "Light Freezing Rain"])
    temp = np.where(snowy, np.minimum(temp, rng.normal(26, 5, n)), temp)
    vis = np.where(np.isin(cond, ["Fog", "Haze", "Heavy Rain", "Snow"]), rng.uniform(0.2, 3, n), rng.uniform(6, 10, n))
    hour = times.hour
    night = (hour < 6) | (hour >= 20)
    signal = rng.random(n) < 0.18
    junction = rng.random(n) < 0.08
    source = np.where(rng.random(n) < 0.6, "Source1", "Source2")
    year = times.year
    risk = (np.array([_RISK.get(c, 0.0) for c in cond]) + 0.6 * night - 0.8 * signal + 0.3 * junction
            + 0.4 * (vis < 2) + 0.3 * np.isin(state, ["MN", "CO"]) + rng.normal(0, 0.6, n))
    risk = risk + np.where(source == "Source2", 0.5, 0.0) - 0.08 * (year - 2018)
    severity = np.select([risk > 2.2, risk > 1.2, risk < -1.2], [4, 3, 1], default=2)
    lat = np.array([CENTRES[s][0] for s in state]) + rng.normal(0, 1.2, n)
    lng = np.array([CENTRES[s][1] for s in state]) + rng.normal(0, 1.5, n)
    df = pd.DataFrame({
        "ID": [f"A-{i}" for i in range(n)], "Source": source, "Severity": severity,
        "Start_Time": times.strftime("%Y-%m-%d %H:%M:%S"), "Start_Lat": lat, "Start_Lng": lng, "State": state,
        "Temperature(F)": temp.round(1), "Humidity(%)": rng.uniform(20, 100, n).round(0),
        "Pressure(in)": rng.normal(29.9, 0.3, n).round(2), "Visibility(mi)": vis.round(1),
        "Wind_Speed(mph)": np.abs(rng.normal(8, 6, n)).round(1),
        "Precipitation(in)": np.where(np.isin(cond, ["Rain", "Heavy Rain", "Light Rain"]), rng.uniform(0, 0.4, n), 0),
        "Weather_Condition": cond, "Sunrise_Sunset": np.where(night, "Night", "Day"),
    })
    for col in FLAG_COLS:
        df[col] = rng.random(n) < 0.05
    df["Traffic_Signal"] = signal
    df["Junction"] = junction
    # a few broken rows, as in real exports
    bad = rng.choice(n, size=max(1, n // 500), replace=False)
    df.loc[bad[: len(bad) // 2], "Temperature(F)"] = 520.0  # a Kelvin-like unit error
    df.loc[bad[len(bad) // 2 :], "Severity"] = 0
    return df


def make_daily_counts(start: str = "2019-01-01", days: int = 1200, seed: int = 13) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    weekly = np.array([1.08, 1.10, 1.10, 1.12, 1.18, 0.78, 0.64])[idx.weekday]
    trend = 1500 + 0.6 * np.arange(days)
    winter = 1 + 0.25 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365.25)
    level = trend * weekly * winter
    holidays = rng.random(days) < 0.015
    level = np.where(holidays, level * 0.6, level)
    counts = rng.poisson(level)
    return pd.Series(counts, index=idx, name="accidents")
