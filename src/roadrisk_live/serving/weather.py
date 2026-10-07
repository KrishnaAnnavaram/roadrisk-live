"""Weather providers behind one interface, and the parser of OpenWeather "current weather" responses.

Fixes against the prototype:

* Units: the parser reads the ``units`` that the request used. ``standard`` gives Kelvin and m/s,
  ``metric`` gives Celsius and m/s, ``imperial`` gives Fahrenheit and mph. Visibility is always metres
  and rain is always millimetres. Every value becomes the training unit. The prototype used
  ``temp * 9/5 + 32`` on a Kelvin value (about 530 F).
* Time: the features use the OBSERVATION time (``dt``) shifted by the location's ``timezone`` offset,
  not the clock of the server.
* Condition: the OpenWeather condition id maps to the shared vocabulary.
* Location: latitude and longitude give the grid cell. The state comes from reverse geocoding.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib import resources
from typing import Protocol

from ..features.weather import (
    celsius_to_f,
    hpa_to_inhg,
    from_openweather_id,
    kelvin_to_f,
    meters_to_miles,
    mm_to_inches,
    mps_to_mph,
)

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


@dataclass(frozen=True)
class Observation:
    lat: float
    lng: float
    observed_utc: datetime
    local_time: datetime
    is_night: bool
    temperature_f: float | None
    humidity_pct: float | None
    pressure_in: float | None
    visibility_mi: float | None
    wind_speed_mph: float | None
    precipitation_in: float
    condition_id: int
    weather_cat: str
    state: str
    place: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["observed_utc"] = self.observed_utc.isoformat()
        d["local_time"] = self.local_time.isoformat()
        return d


def parse_current(payload: dict, units: str, state: str = "unknown") -> Observation:
    """Turn an OpenWeather ``/data/2.5/weather`` JSON response into training units."""
    if units not in ("standard", "metric", "imperial"):
        raise ValueError("units must be standard, metric or imperial")
    main, wind = payload.get("main", {}), payload.get("wind", {})
    temp = main.get("temp")
    if temp is not None:
        temp = {"standard": kelvin_to_f, "metric": celsius_to_f, "imperial": float}[units](float(temp))
    speed = wind.get("speed")
    if speed is not None:
        speed = float(speed) if units == "imperial" else mps_to_mph(float(speed))
    rain = (payload.get("rain") or {}).get("1h", 0.0) + (payload.get("snow") or {}).get("1h", 0.0)
    observed = datetime.fromtimestamp(int(payload["dt"]), tz=timezone.utc)
    offset = timedelta(seconds=int(payload.get("timezone", 0)))
    sys_ = payload.get("sys", {})
    sunrise, sunset = sys_.get("sunrise"), sys_.get("sunset")
    if sunrise and sunset:
        is_night = not (int(sunrise) <= int(payload["dt"]) < int(sunset))
    else:
        local_hour = (observed + offset).hour
        is_night = local_hour < 6 or local_hour >= 20
    cond = int((payload.get("weather") or [{"id": 0}])[0].get("id", 0))
    vis = payload.get("visibility")
    return Observation(
        lat=float(payload["coord"]["lat"]), lng=float(payload["coord"]["lon"]), observed_utc=observed,
        local_time=(observed + offset).replace(tzinfo=None), is_night=is_night, temperature_f=temp,
        humidity_pct=main.get("humidity"),
        pressure_in=None if main.get("pressure") is None else hpa_to_inhg(float(main["pressure"])),
        visibility_mi=None if vis is None else meters_to_miles(float(vis)), wind_speed_mph=speed,
        precipitation_in=mm_to_inches(float(rain)), condition_id=cond, weather_cat=from_openweather_id(cond),
        state=state, place=str(payload.get("name", "")),
    )


class WeatherProvider(Protocol):
    def current(self, lat: float, lng: float) -> Observation: ...


class OpenWeatherClient:
    """Live client (standard library HTTP). The key comes from ``OPENWEATHER_API_KEY``."""

    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org", timeout_s: float = 10.0,
                 units: str = "imperial"):
        if not api_key:
            raise ValueError("OPENWEATHER_API_KEY is not set")
        self._key, self.base_url, self.timeout_s, self.units = api_key, base_url.rstrip("/"), timeout_s, units

    def _get(self, path: str, params: dict):
        query = urllib.parse.urlencode({**params, "appid": self._key})
        with urllib.request.urlopen(f"{self.base_url}{path}?{query}", timeout=self.timeout_s) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def reverse_state(self, lat: float, lng: float) -> str:
        rows = self._get("/geo/1.0/reverse", {"lat": lat, "lon": lng, "limit": 1})
        if rows and rows[0].get("country") == "US":
            return US_STATES.get(rows[0].get("state", ""), "unknown")
        return "unknown"

    def current(self, lat: float, lng: float) -> Observation:
        payload = self._get("/data/2.5/weather", {"lat": lat, "lon": lng, "units": self.units})
        return parse_current(payload, self.units, self.reverse_state(lat, lng))


class FixtureWeatherProvider:
    """Offline provider: recorded responses shipped with the package (``serving/fixtures``)."""

    def __init__(self):
        folder = resources.files("roadrisk_live.serving").joinpath("fixtures")
        self.fixtures = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.iterdir())
                         if f.name.endswith(".json")]

    def current(self, lat: float, lng: float) -> Observation:
        nearest = min(self.fixtures, key=lambda f: (f["response"]["coord"]["lat"] - lat) ** 2
                      + (f["response"]["coord"]["lon"] - lng) ** 2)
        return parse_current(nearest["response"], nearest["units"], nearest["state"])
