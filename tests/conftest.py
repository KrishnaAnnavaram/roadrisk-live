import os
from datetime import date

import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from roadrisk_live.data.accidents import time_split, to_feature_table  # noqa: E402
from roadrisk_live.data.synthetic import make_accidents  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return make_accidents(4000, seed=3)


@pytest.fixture(scope="session")
def table(raw):
    t, _ = to_feature_table(raw)
    return time_split(t, date(2021, 12, 31), date(2022, 12, 31))
