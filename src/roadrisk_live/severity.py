"""Severity models: one scikit-learn Pipeline (preprocessing + classifier), trained on the train split only.

* No SMOTE. The prototype oversampled before the split, so synthetic near-copies of test rows sat in
  train. Here the imbalance is handled with balanced sample weights computed on the train split.
* The preprocessing is inside the pipeline, so imputation and scaling statistics come from train only.
* The model family is chosen on the VALID split (macro-F1). The TEST split is scored one time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .features.build import FEATURES, check_feature_frame, make_preprocessor

CLASSES = (1, 2, 3, 4)
MODELS = ("majority", "logreg", "hgb")
BUNDLE_FILE = "severity_model.joblib"


def make_model(name: str, seed: int = 13) -> Pipeline:
    if name == "majority":
        clf = DummyClassifier(strategy="most_frequent")
    elif name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                             early_stopping=False, random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    return Pipeline([("prep", make_preprocessor()), ("clf", clf)])


def fit(name: str, train: pd.DataFrame, seed: int = 13) -> Pipeline:
    check_feature_frame(train)
    pipe = make_model(name, seed)
    if name == "majority":  # the floor: the most frequent severity, with no weights
        return pipe.fit(train[FEATURES], train["severity"])
    weights = compute_sample_weight("balanced", train["severity"])
    pipe.fit(train[FEATURES], train["severity"], clf__sample_weight=weights)
    return pipe


def proba(pipe: Pipeline, df: pd.DataFrame) -> np.ndarray:
    raw = pipe.predict_proba(df[FEATURES])
    out = np.zeros((len(df), len(CLASSES)))
    for j, c in enumerate(pipe.classes_):
        out[:, CLASSES.index(int(c))] = raw[:, j]
    return out


def metrics(y_true, p: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.array(CLASSES)[p.argmax(1)]
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.full(4, np.nan), where=cm.sum(1) > 0)
    conf, correct = p.max(1), y_pred == y_true
    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        in_bin = (conf > lo) & (conf <= lo + 0.1)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(correct.mean()),
        "per_class_recall": {str(c): (None if np.isnan(r) else round(float(r), 4)) for c, r in zip(CLASSES, recall)},
        "confusion": cm.tolist(),
        "ece": float(ece),
    }


def per_source(df: pd.DataFrame, p: np.ndarray) -> dict:
    out = {}
    for src in sorted(df["source"].unique()):
        m = (df["source"] == src).to_numpy()
        y_pred = np.array(CLASSES)[p[m].argmax(1)]
        out[src] = {"n": int(m.sum()), "macro_f1": float(f1_score(df["severity"][m], y_pred, labels=list(CLASSES),
                                                                  average="macro", zero_division=0))}
    return out


def train_and_select(df: pd.DataFrame, names=MODELS, seed: int = 13, log=print) -> dict:
    """Fit each model on train, score valid, select by valid macro-F1, refit nothing, score test once."""
    parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    for s, part in parts.items():
        if part.empty:
            raise ValueError(f"the {s} split is empty: check ROADRISK_TRAIN_END and ROADRISK_VALID_END")
    fitted, valid_scores = {}, {}
    for name in names:
        start = time.time()
        fitted[name] = fit(name, parts["train"], seed)
        valid_scores[name] = metrics(parts["valid"]["severity"], proba(fitted[name], parts["valid"]))
        log(f"{name}: valid macro-F1 {valid_scores[name]['macro_f1']:.4f} ({time.time() - start:.1f} s)")
    best = max(valid_scores, key=lambda n: valid_scores[n]["macro_f1"])
    p_test = proba(fitted[best], parts["test"])
    return {
        "best": best,
        "model": fitted[best],
        "valid": {n: {k: v for k, v in m.items() if k != "confusion"} for n, m in valid_scores.items()},
        "test": metrics(parts["test"]["severity"], p_test),
        "test_per_source": per_source(parts["test"], p_test),
        "rows": {s: len(p) for s, p in parts.items()},
    }


def save_bundle(result: dict, folder: str | Path, extra: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result["model"], "features": FEATURES, "classes": CLASSES, "model": result["best"],
                 **extra}, folder / BUNDLE_FILE)
    report = {k: v for k, v in result.items() if k != "model"}
    (folder / "severity_report.json").write_text(json.dumps({**report, **extra}, indent=2, default=str),
                                                 encoding="utf-8")
    return folder / BUNDLE_FILE


def load_bundle(folder: str | Path) -> dict:
    path = Path(folder) / BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}: run 'roadrisk train' first")
    return joblib.load(path)
