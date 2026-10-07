import json

import numpy as np
import pytest

from roadrisk_live.cli import main
from roadrisk_live.features.build import NUMERIC
from roadrisk_live.serving.api import check_token, handle_score
from roadrisk_live.serving.scoring import ObservationError, feature_row, score
from roadrisk_live.serving.weather import FixtureWeatherProvider, OpenWeatherClient, parse_current
from roadrisk_live.config import Settings
from roadrisk_live.severity import fit, load_bundle, metrics, proba, save_bundle, train_and_select

PAYLOAD = {"coord": {"lon": -93.27, "lat": 44.98}, "weather": [{"id": 501}], "main": {"temp": 263.15,
           "humidity": 80, "pressure": 1000}, "visibility": 1609, "wind": {"speed": 10.0}, "rain": {"1h": 25.4},
           "dt": 1700000000, "timezone": -21600, "name": "Minneapolis"}


@pytest.fixture(scope="module")
def trained(table):
    return train_and_select(table, names=("majority", "logreg"), log=lambda m: None)


def test_preprocessing_is_fit_on_train_only(table):
    # reference problem 1: SMOTE and the scaler saw the test rows
    train = table[table.split == "train"]
    pipe = fit("logreg", train)
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    expected = imputer.transform(train[NUMERIC])[:, : len(NUMERIC)].mean(axis=0)
    assert np.allclose(scaler.mean_[: len(NUMERIC)], expected)
    assert len(train) < len(table)


def test_selection_on_valid_and_test_metrics(trained, table):
    assert trained["best"] == max(trained["valid"], key=lambda n: trained["valid"][n]["macro_f1"])
    assert trained["test"]["n"] == int((table.split == "test").sum())
    assert set(trained["test"]["per_class_recall"]) == {"1", "2", "3", "4"}
    assert set(trained["test_per_source"]) == {"Source1", "Source2"}
    assert trained["valid"]["logreg"]["macro_f1"] > trained["valid"]["majority"]["macro_f1"]


def test_metrics_are_macro_and_per_class():
    y = np.array([1, 2, 2, 2, 3, 4])
    p = np.eye(4)[[0, 1, 1, 1, 1, 3]]
    m = metrics(y, p)
    assert m["per_class_recall"]["3"] == 0.0 and m["per_class_recall"]["2"] == 1.0
    assert m["accuracy"] == pytest.approx(5 / 6) and m["macro_f1"] < m["accuracy"]


def test_parse_current_converts_every_unit_system():
    obs = parse_current(PAYLOAD, "standard", "MN")
    assert obs.temperature_f == pytest.approx(14.0)
    assert obs.wind_speed_mph == pytest.approx(22.37, abs=0.01)
    assert obs.visibility_mi == pytest.approx(1.0, abs=0.001) and obs.precipitation_in == pytest.approx(1.0)
    assert obs.weather_cat == "rain" and obs.state == "MN"
    assert obs.local_time.hour == 16  # observation time in the local zone, not the server clock
    metric = parse_current({**PAYLOAD, "main": {**PAYLOAD["main"], "temp": -10.0}}, "metric")
    assert metric.temperature_f == pytest.approx(14.0)
    imperial = parse_current({**PAYLOAD, "main": {**PAYLOAD["main"], "temp": 14.0}, "wind": {"speed": 22.0}},
                             "imperial")
    assert imperial.temperature_f == 14.0 and imperial.wind_speed_mph == 22.0
    with pytest.raises(ValueError):
        parse_current(PAYLOAD, "kelvin")


def test_a_unit_error_is_refused_not_scored(trained):
    # a Kelvin value read as Fahrenheit must not reach the model
    obs = parse_current(PAYLOAD, "imperial")  # 263.15 "F"
    with pytest.raises(ObservationError):
        feature_row(obs)
    bundle = {"pipeline": trained["model"], "classes": (1, 2, 3, 4), "model": trained["best"]}

    class Provider:
        def current(self, lat, lon):
            return obs

    status, body = handle_score(bundle, Provider(), 44.9, -93.2)
    assert status == 422 and "units" in body["error"]
    assert handle_score(bundle, Provider(), 95, 0)[0] == 422


def test_live_row_uses_the_training_columns_and_unknown_state(trained):
    obs = parse_current(PAYLOAD, "standard", state="ZZ")  # a state that training never saw
    row = feature_row(obs)
    assert list(row.columns) == list(trained["model"].feature_names_in_)
    bundle = {"pipeline": trained["model"], "classes": (1, 2, 3, 4), "model": trained["best"]}
    out = score(bundle, obs)
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["predicted_severity"] in (1, 2, 3, 4)


def test_fixture_provider_and_bundle_roundtrip(trained, tmp_path):
    provider = FixtureWeatherProvider()
    assert provider.current(25.7, -80.2).place == "Miami"
    assert provider.current(44.9, -93.3).weather_cat == "snow"
    save_bundle(trained, tmp_path, {"train_end": "2021-12-31"})
    bundle = load_bundle(tmp_path)
    assert bundle["model"] == trained["best"]
    assert json.loads((tmp_path / "severity_report.json").read_text(encoding="utf-8"))["best"] == trained["best"]
    assert np.allclose(proba(bundle["pipeline"], trained_rows := _rows(trained)), proba(trained["model"], trained_rows))
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "none")


def _rows(trained):
    obs = FixtureWeatherProvider().current(39.7, -105.0)
    return feature_row(obs)


def test_token_check_and_client_needs_a_key():
    assert check_token(Settings(api_token=None), None)
    assert check_token(Settings(api_token="t"), "Bearer t") and not check_token(Settings(api_token="t"), "Bearer x")
    with pytest.raises(ValueError):
        OpenWeatherClient("")


def test_cli_prepare_train_score(raw, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROADRISK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ROADRISK_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    path = tmp_path / "acc.csv"
    raw.to_csv(path, index=False)
    assert main(["prepare", str(path)]) == 0
    assert main(["train", "--model", "majority", "--model", "logreg"]) == 0
    capsys.readouterr()
    assert main(["score", "39.74", "-104.99"]) == 0
    assert json.loads(capsys.readouterr().out)["place"] == "Denver"
