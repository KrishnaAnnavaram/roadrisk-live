from datetime import date

import pandas as pd
import pytest

from roadrisk_live.config import load_settings
from roadrisk_live.data.accidents import daily_counts, describe_splits, load_accidents, to_feature_table
from roadrisk_live.features.build import FEATURES, geo_cell, time_features
from roadrisk_live.features.weather import (
    VOCAB,
    celsius_to_f,
    from_openweather_id,
    from_text,
    hpa_to_inhg,
    kelvin_to_f,
    meters_to_miles,
    mps_to_mph,
)


def test_unit_conversions():
    # reference problem 2: a Kelvin value was converted with the Celsius formula (about 530 F)
    assert kelvin_to_f(273.15) == pytest.approx(32.0)
    assert kelvin_to_f(300.0) == pytest.approx(80.33, abs=0.01)
    assert celsius_to_f(100) == pytest.approx(212.0)
    assert hpa_to_inhg(1013.25) == pytest.approx(29.92, abs=0.01)
    assert meters_to_miles(1609.344) == pytest.approx(1.0)
    assert mps_to_mph(10) == pytest.approx(22.37, abs=0.01)


@pytest.mark.parametrize("text, code, cat", [
    ("Light Rain", 500, "rain"), ("Heavy Rain", 502, "heavy_rain"), ("Light Snow", 600, "snow"),
    ("Light Freezing Rain", 511, "ice"), ("Fog", 741, "fog"), ("Haze", 721, "haze_smoke_dust"),
    ("Thunder in the Vicinity", 211, "thunderstorm"), ("Fair", 800, "clear"), ("Mostly Cloudy", 803, "clouds"),
    ("Light Drizzle", 300, "drizzle"),
])
def test_training_text_and_live_ids_share_one_vocabulary(text, code, cat):
    # reference problem 3: OpenWeather "Rain" never matched the training label "Light Rain"
    assert from_text(text) == from_openweather_id(code) == cat


def test_vocabulary_edges():
    assert from_text(None) == "other" and from_text("Volcanic Ash") == "haze_smoke_dust"
    assert from_openweather_id(781) == "other"
    assert set(VOCAB) >= {from_openweather_id(c) for c in range(200, 900)}


def test_time_and_location_features():
    t = time_features(pd.Series(["2023-03-06 08:15:00", "2023-03-11 23:00:00"]))
    assert t["hour"].tolist() == [8, 23] and t["is_rush_hour"].tolist() == [1, 0]
    assert t["is_weekend"].tolist() == [0, 1] and t["is_night"].tolist() == [0, 1]
    assert geo_cell(44.98, -93.27) == "44.5_-93.5" and geo_cell(None, 1) == "unknown"


def test_feature_table_validates_rows(raw):
    table, report = to_feature_table(raw)
    assert report["dropped_bad_time_or_severity"] == 4  # severity 0 rows
    assert report["set_missing_out_of_range"]["temperature_f"] == 4  # the 520 F values
    assert table["temperature_f"].max() < 135
    assert set(FEATURES) <= set(table.columns) and table["severity"].isin([1, 2, 3, 4]).all()
    assert "city" not in table.columns
    with pytest.raises(ValueError):
        to_feature_table(raw.drop(columns=["Severity"]))


def test_time_split_is_by_date(table):
    # reference problem 5: a random split mixes years
    tr, va, te = (table[table.split == s] for s in ("train", "valid", "test"))
    assert tr["start_time"].max() < va["start_time"].min() and va["start_time"].max() < te["start_time"].min()
    assert te["start_time"].min().date() >= date(2023, 1, 1)
    desc = describe_splits(table)
    assert set(desc) == {"train", "valid", "test"} and "source_share" in desc["test"]


def test_loader_reads_all_rows_in_chunks(raw, tmp_path):
    # reference problem 4: the prototype trained on a 10 000-row sample
    path = tmp_path / "acc.csv"
    raw.to_csv(path, index=False)
    table, report = load_accidents(path, chunksize=700)
    assert report["rows_in"] == len(raw) and len(table) == report["rows_out"]
    small, _ = load_accidents(path, max_rows=1000, chunksize=700)
    assert len(small) <= 1000


def test_daily_counts_fill_missing_days(table):
    counts = daily_counts(table)
    assert counts.sum() == len(table) and counts.index.is_monotonic_increasing
    assert (counts.index[1:] - counts.index[:-1]).max() == pd.Timedelta(days=1)


def test_settings_keep_the_key_out_of_repr():
    s = load_settings({"OPENWEATHER_API_KEY": "secret-value", "ROADRISK_TRAIN_END": "2020-12-31"})
    assert s.openweather_api_key == "secret-value" and "secret-value" not in repr(s)
    with pytest.raises(ValueError):
        load_settings({"ROADRISK_TRAIN_END": "2023-01-01", "ROADRISK_VALID_END": "2022-01-01"})
