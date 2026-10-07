"""One weather vocabulary for training and live scoring, and unit conversions.

The training data (US-Accidents) has about 140 free-text conditions ("Light Rain", "Mostly Cloudy",
"Heavy T-Storm / Windy"). OpenWeather gives numeric condition ids (500 = light rain). The prototype
fed OpenWeather's "Rain" into a label encoder fit on "Light Rain", so live rows fell back to code 0.
Here BOTH sides map to the same eleven categories.
"""

from __future__ import annotations

import math

VOCAB = ("clear", "clouds", "drizzle", "rain", "heavy_rain", "thunderstorm", "snow", "ice", "fog",
         "haze_smoke_dust", "other")

# order matters: the first matching rule wins
_TEXT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("t-storm", "thunder", "tstorm"), "thunderstorm"),
    (("freezing", "sleet", "ice pellets", "hail", "wintry mix", "ice"), "ice"),
    (("snow", "blowing snow", "squalls"), "snow"),
    (("heavy rain", "heavy drizzle", "rain shower", "heavy showers"), "heavy_rain"),
    (("drizzle",), "drizzle"),
    (("rain", "showers"), "rain"),
    (("fog", "mist"), "fog"),
    (("haze", "smoke", "dust", "sand", "ash"), "haze_smoke_dust"),
    (("cloud", "overcast"), "clouds"),
    (("fair", "clear", "sunny"), "clear"),
]


def from_text(condition: object) -> str:
    """Map a US-Accidents ``Weather_Condition`` string to the vocabulary. Missing gives ``other``."""
    if condition is None or (isinstance(condition, float) and math.isnan(condition)):
        return "other"
    text = str(condition).strip().lower()
    for keys, cat in _TEXT_RULES:
        if any(k in text for k in keys):
            return cat
    return "other"


def from_openweather_id(code: int) -> str:
    """Map an OpenWeather condition id (https://openweathermap.org/weather-conditions) to the vocabulary."""
    code = int(code)
    if 200 <= code < 300:
        return "thunderstorm"
    if 300 <= code < 400:
        return "drizzle"
    if code == 511:
        return "ice"  # freezing rain
    if code in (502, 503, 504, 522, 531):
        return "heavy_rain"
    if 500 <= code < 600:
        return "rain"
    if code in (611, 612, 613, 615, 616):
        return "ice"  # sleet and rain-and-snow
    if 600 <= code < 700:
        return "snow"
    if code in (701, 741):
        return "fog"
    if code in (711, 721, 731, 751, 761, 762):
        return "haze_smoke_dust"
    if code == 800:
        return "clear"
    if 801 <= code <= 804:
        return "clouds"
    return "other"


def kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def celsius_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.0295299830714


def meters_to_miles(m: float) -> float:
    return m / 1609.344


def mps_to_mph(v: float) -> float:
    return v * 2.2369362921


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


# physical ranges in the training units. A value outside is a unit error or a sensor error.
RANGES = {
    "temperature_f": (-60.0, 135.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_in": (25.0, 32.5),
    "visibility_mi": (0.0, 100.0),
    "wind_speed_mph": (0.0, 150.0),
    "precipitation_in": (0.0, 10.0),
}
