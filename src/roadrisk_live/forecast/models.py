"""Forecasters with one interface: ``fit(history)`` then ``predict(horizon)``.

``history`` is a daily ``pd.Series`` that ends at the forecast origin. A forecaster sees nothing after
the origin. It never trains on its own forecasts: the prototype appended 30 ARIMA forecast days to the
history and used them as LSTM training targets.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Naive:
    name = "naive"

    def fit(self, history: pd.Series) -> "Naive":
        self.last, self.end = float(history.iloc[-1]), history.index[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last)


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, m: int = 7):
        self.m = m

    def fit(self, history: pd.Series) -> "SeasonalNaive":
        self.season = history.iloc[-self.m :].to_numpy(dtype=float)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.array([self.season[i % self.m] for i in range(horizon)])


class HoltWinters:
    """Additive Holt-Winters (level, trend, weekly season). Smoothing values come from a grid search on
    one-step errors INSIDE the history."""

    name = "holt_winters"
    GRID = (0.1, 0.3, 0.5)

    def __init__(self, m: int = 7):
        self.m = m

    def _run(self, y: np.ndarray, a: float, b: float, g: float):
        m = self.m
        level, trend = y[:m].mean(), (y[m : 2 * m].mean() - y[:m].mean()) / m
        season = list(y[:m] - level)
        sse = 0.0
        for t in range(m, len(y)):
            s = season[t - m]
            pred = level + trend + s
            sse += (y[t] - pred) ** 2
            new_level = a * (y[t] - s) + (1 - a) * (level + trend)
            trend = b * (new_level - level) + (1 - b) * trend
            season.append(g * (y[t] - new_level) + (1 - g) * s)
            level = new_level
        return sse, level, trend, season

    def fit(self, history: pd.Series) -> "HoltWinters":
        y = history.to_numpy(dtype=float)
        if len(y) < 3 * self.m:
            raise ValueError("Holt-Winters needs at least three seasons of history")
        best = min(itertools.product(self.GRID, (0.01, 0.05), self.GRID), key=lambda p: self._run(y, *p)[0])
        _, self.level, self.trend, season = self._run(y, *best)
        self.season = season[-self.m :]
        self.params = best
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.array([self.level + (h + 1) * self.trend + self.season[h % self.m] for h in range(horizon)])


class LagRegression:
    """A global regression on calendar features and lags of 7, 14, 21 and 28 days (horizon <= 7).

    The scaler and the model are fit on the history only.
    """

    name = "lag_regression"
    LAGS = (7, 14, 21, 28)

    def _features(self, index: pd.DatetimeIndex, values: pd.Series) -> pd.DataFrame:
        f = pd.DataFrame(index=index)
        for d in range(7):
            f[f"dow_{d}"] = (index.weekday == d).astype(float)
        doy = index.dayofyear.to_numpy()
        f["sin_doy"], f["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        f["t"] = (index - self.start).days.to_numpy(dtype=float)
        for lag in self.LAGS:
            f[f"lag_{lag}"] = values.reindex(index - pd.Timedelta(days=lag)).to_numpy()
        return f

    def fit(self, history: pd.Series) -> "LagRegression":
        self.start = history.index[0]
        self.history = history.astype(float)
        x = self._features(history.index, self.history).iloc[max(self.LAGS) :]
        y = self.history.iloc[max(self.LAGS) :]
        self.model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(x, y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon > min(self.LAGS):
            raise ValueError(f"lag_regression supports a horizon up to {min(self.LAGS)} days")
        future = pd.date_range(self.history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return self.model.predict(self._features(future, self.history))


FORECASTERS = {"naive": Naive, "seasonal_naive": SeasonalNaive, "holt_winters": HoltWinters,
               "lag_regression": LagRegression}


def make_forecaster(name: str):
    try:
        return FORECASTERS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown forecaster {name!r}; choose from {sorted(FORECASTERS)}") from exc
