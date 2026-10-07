"""Rolling-origin evaluation: at each origin, fit on the past only and score the next ``horizon`` days.

Metrics are out-of-sample: MAE, RMSE, sMAPE and MASE (scaled by the in-sample seasonal-naive MAE of the
same history). The prototype reported in-sample errors on its training windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import make_forecaster


def _mase_scale(history: np.ndarray, m: int = 7) -> float:
    diffs = np.abs(history[m:] - history[:-m])
    return float(diffs.mean()) if len(diffs) else float("nan")


def backtest(series: pd.Series, models=("seasonal_naive", "holt_winters", "lag_regression", "naive"),
             horizon: int = 7, initial: int = 365, step: int = 28) -> pd.DataFrame:
    """One row per (model, origin, step ahead) with the forecast and the actual value."""
    series = series.asfreq("D")
    if series.isna().any():
        series = series.interpolate(limit_direction="both")
    rows = []
    for origin in range(initial, len(series) - horizon + 1, step):
        history = series.iloc[:origin]  # nothing after the origin is visible
        actual = series.iloc[origin : origin + horizon].to_numpy(dtype=float)
        scale = _mase_scale(history.to_numpy(dtype=float))
        for name in models:
            forecast = make_forecaster(name).fit(history.copy()).predict(horizon)
            for h in range(horizon):
                rows.append({"model": name, "origin": series.index[origin].date().isoformat(), "h": h + 1,
                             "actual": actual[h], "forecast": float(forecast[h]), "scale": scale})
    if not rows:
        raise ValueError("the series is too short for this initial window and horizon")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    err = results["forecast"] - results["actual"]
    work = results.assign(abs_err=err.abs(), sq_err=err**2,
                          smape=2 * err.abs() / (results["actual"].abs() + results["forecast"].abs()).clip(lower=1e-9),
                          scaled=err.abs() / results["scale"])
    table = work.groupby("model").agg(origins=("origin", "nunique"), mae=("abs_err", "mean"),
                                      rmse=("sq_err", lambda s: float(np.sqrt(s.mean()))),
                                      smape_pct=("smape", lambda s: 100 * float(s.mean())), mase=("scaled", "mean"))
    if "seasonal_naive" in table.index:
        table["mae_vs_seasonal_naive"] = table["mae"] / table.loc["seasonal_naive", "mae"]
    return table.sort_values("mae").round(4).reset_index()
