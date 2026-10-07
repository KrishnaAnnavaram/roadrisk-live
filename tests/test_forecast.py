import numpy as np
import pandas as pd
import pytest

from roadrisk_live.cli import main
from roadrisk_live.data.synthetic import make_daily_counts
from roadrisk_live.forecast import backtest, make_forecaster, summarize


@pytest.fixture(scope="module")
def series():
    return make_daily_counts(days=600, seed=4)


@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "holt_winters", "lag_regression"])
def test_forecasters_never_see_the_future(name, series):
    # reference problem 6: the prototype trained its LSTM on its own ARIMA forecasts and future days
    history = series.iloc[:400]
    poisoned = pd.concat([history, pd.Series(1e9, index=series.index[400:450])])
    a = make_forecaster(name).fit(history).predict(7)
    b = make_forecaster(name).fit(poisoned.iloc[:400]).predict(7)
    assert np.allclose(a, b) and len(a) == 7 and np.isfinite(a).all()


def test_seasonal_naive_repeats_the_last_week(series):
    f = make_forecaster("seasonal_naive").fit(series.iloc[:100]).predict(10)
    assert np.array_equal(f[:7], series.iloc[93:100].to_numpy()) and f[7] == f[0]


def test_lag_regression_horizon_limit(series):
    with pytest.raises(ValueError):
        make_forecaster("lag_regression").fit(series.iloc[:200]).predict(8)
    with pytest.raises(ValueError):
        make_forecaster("arima")


def test_backtest_is_out_of_sample_and_has_baselines(series):
    # reference problem 7: no baseline and in-sample errors
    res = backtest(series, horizon=7, initial=365, step=56)
    origins = pd.to_datetime(res["origin"].unique())
    assert origins.min() == series.index[365]
    table = summarize(res).set_index("model")
    assert table.loc["seasonal_naive", "mae_vs_seasonal_naive"] == 1.0
    assert table.loc["naive", "mae"] > table.loc["seasonal_naive", "mae"]  # the weekly cycle matters
    assert (table["origins"] == len(origins)).all()
    with pytest.raises(ValueError):
        backtest(series.iloc[:100], initial=365)


def test_cli_forecast(series, tmp_path, monkeypatch):
    monkeypatch.setenv("ROADRISK_MODEL_DIR", str(tmp_path))
    path = tmp_path / "counts.csv"
    series.to_csv(path)
    assert main(["forecast", "--counts", str(path), "--step", "56"]) == 0
    assert (tmp_path / "forecast_summary.csv").exists()
