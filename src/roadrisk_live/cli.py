"""The ``roadrisk`` command."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings
from .severity import MODELS


def _prepared_path(args, settings: Settings) -> Path:
    return Path(args.prepared or settings.data_dir / "prepared" / "accidents.pkl")


def cmd_synth(args, settings) -> int:
    from .data.synthetic import make_accidents, make_daily_counts

    out = Path(args.out or settings.data_dir / "raw")
    out.mkdir(parents=True, exist_ok=True)
    make_accidents(args.n, seed=settings.seed).to_csv(out / "synthetic_accidents.csv", index=False)
    make_daily_counts(seed=settings.seed).to_csv(out / "synthetic_daily_counts.csv")
    print(f"wrote {out / 'synthetic_accidents.csv'} and {out / 'synthetic_daily_counts.csv'}")
    return 0


def cmd_prepare(args, settings) -> int:
    from .data.accidents import describe_splits, load_accidents, time_split

    table, report = load_accidents(args.input, max_rows=args.max_rows)
    if args.source:
        table = table[table["source"].isin(args.source)]
        report["kept_sources"] = args.source
    table = time_split(table, settings.train_end, settings.valid_end)
    out = _prepared_path(args, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(out)
    summary = {"validation": report, "splits": describe_splits(table)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_train(args, settings) -> int:
    from .severity import save_bundle, train_and_select

    table = pd.read_pickle(_prepared_path(args, settings))
    result = train_and_select(table, names=args.model or MODELS, seed=settings.seed)
    save_bundle(result, settings.model_dir, {"train_end": settings.train_end, "valid_end": settings.valid_end})
    print(json.dumps({"best": result["best"], "valid": result["valid"], "test": result["test"],
                      "test_per_source": result["test_per_source"]}, indent=2, default=str))
    return 0


def cmd_score(args, settings) -> int:
    from .serving.api import handle_score, make_provider
    from .severity import load_bundle

    provider = make_provider(settings)
    if not settings.openweather_api_key:
        print("OPENWEATHER_API_KEY is not set: using the recorded offline responses", file=sys.stderr)
    status, body = handle_score(load_bundle(settings.model_dir), provider, args.lat, args.lon)
    print(json.dumps(body, indent=2))
    return 0 if status == 200 else 1


def cmd_forecast(args, settings) -> int:
    from .forecast import backtest, summarize

    if args.counts:
        series = pd.read_csv(args.counts, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        from .data.accidents import daily_counts

        series = daily_counts(pd.read_pickle(_prepared_path(args, settings)), state=args.state)
    results = backtest(series, horizon=args.horizon, initial=args.initial, step=args.step)
    table = summarize(results)
    out = Path(settings.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "forecast_backtest.csv", index=False)
    table.to_csv(out / "forecast_summary.csv", index=False)
    print(table.to_string(index=False))
    return 0


def cmd_serve(args, settings) -> int:  # pragma: no cover - needs the api extra and a server
    import uvicorn

    from .serving.api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def cmd_demo(args, settings) -> int:
    from .data.accidents import describe_splits, time_split, to_feature_table
    from .data.synthetic import make_accidents, make_daily_counts
    from .forecast import backtest, summarize
    from .serving.api import handle_score
    from .serving.weather import FixtureWeatherProvider
    from .severity import save_bundle, train_and_select

    out = Path(args.out or settings.model_dir / "demo")
    table, report = to_feature_table(make_accidents(args.n, seed=settings.seed))
    table = time_split(table, settings.train_end, settings.valid_end)
    print("validation:", json.dumps(report))
    print("rows per split:", {k: v["rows"] for k, v in describe_splits(table).items()})
    result = train_and_select(table, seed=settings.seed)
    save_bundle(result, out, {"train_end": settings.train_end, "valid_end": settings.valid_end})
    t = result["test"]
    print(f"selected on valid: {result['best']}. Test macro-F1 {t['macro_f1']:.3f}, balanced accuracy "
          f"{t['balanced_accuracy']:.3f}, per-class recall {t['per_class_recall']}")
    provider = FixtureWeatherProvider()
    bundle = {"pipeline": result["model"], "classes": (1, 2, 3, 4), "model": result["best"]}
    for lat, lon in ((44.98, -93.27), (25.77, -80.19), (39.74, -104.99)):
        _, body = handle_score(bundle, provider, lat, lon)
        print(json.dumps({k: body[k] for k in ("place", "weather_cat", "temperature_f", "local_time",
                                                "predicted_severity", "expected_severity")}))
    summary = summarize(backtest(make_daily_counts(seed=settings.seed), horizon=7, initial=365, step=28))
    summary.to_csv(out / "forecast_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roadrisk", description="Accident severity with live weather.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write synthetic accidents and daily counts")
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("prepare", help="validate US-Accidents, build features, split by time")
    p.add_argument("input", help="US_Accidents_March23.csv (or .parquet with the parquet extra)")
    p.add_argument("--max-rows", type=int, help="only for quick trials; default is all rows")
    p.add_argument("--source", action="append", help="keep only this source (repeatable)")
    p.add_argument("--prepared")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="fit on train, select on valid, score test once, save the bundle")
    p.add_argument("--model", action="append", choices=MODELS)
    p.add_argument("--prepared")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("score", help="score the current weather at a point")
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("forecast", help="rolling-origin backtest of daily accident counts")
    p.add_argument("--counts", help="CSV with a date index and one count column")
    p.add_argument("--state")
    p.add_argument("--prepared")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--initial", type=int, default=365)
    p.add_argument("--step", type=int, default=28)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("serve", help="run the HTTP API (api extra)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="offline end-to-end demo on synthetic data and recorded weather")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    # joblib asks the OS for the number of physical cores and warns with a traceback on some Windows hosts
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args, load_settings())


if __name__ == "__main__":
    raise SystemExit(main())
