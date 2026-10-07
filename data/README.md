# data/

Git ignores everything in this folder except this file. Do not commit the accident file, weather
responses, prepared tables or models. The offline demo and the tests use synthetic rows
(`roadrisk synth`) and recorded example weather responses in `src/roadrisk_live/serving/fixtures/`.

## Sources

| File | Source | License and terms | Notes |
|---|---|---|---|
| `US_Accidents_March23.csv` (about 3 GB, 7.7 M rows, 2016-2023) | Kaggle "US Accidents (2016 - 2023)", `sobhanmoosavi/us-accidents` | CC BY-NC-SA 4.0 (non-commercial) | Cite the dataset papers of the authors |
| Live weather | OpenWeather "Current weather data" (`/data/2.5/weather`) and "Reverse geocoding" (`/geo/1.0/reverse`) | OpenWeather terms of service | Needs your own key in `OPENWEATHER_API_KEY` |

## Columns that `prepare` reads

`ID, Source, Severity, Start_Time, Start_Lat, Start_Lng, State, Temperature(F), Humidity(%), Pressure(in),
Visibility(mi), Wind_Speed(mph), Precipitation(in), Weather_Condition, Sunrise_Sunset` and the road flags
`Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop, Traffic_Calming,
Traffic_Signal`.

`Start_Time` is the local time of the accident. `Severity` (1 to 4) describes the impact on traffic
(delay), not the injuries. Its distribution changes by source and by year.

## Download and prepare

```bash
kaggle datasets download -d sobhanmoosavi/us-accidents -p data/raw --unzip
roadrisk prepare data/raw/US_Accidents_March23.csv
```

## Files that the commands write

| File | Contents |
|---|---|
| `data/prepared/accidents.pkl` | The feature table with `severity`, `source`, `start_time` and `split` (local file, do not share) |
| `data/prepared/accidents.json` | The validation report and the rows, dates, severity shares and source shares per split |
| `models_out/severity_model.joblib` | The pipeline, the feature list, the classes and the training window |
| `models_out/severity_report.json` | Valid scores of each model, test report, per-source test scores |
| `models_out/forecast_backtest.csv`, `forecast_summary.csv` | Rolling-origin forecasts and the summary |
