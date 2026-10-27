"""Chronological model comparison with a fixed, auditable feature contract."""
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["initial_temperature", "heating_seconds", "active_energy_proxy",
            "reactive_energy_proxy", "bulk_total", "wire_total"]
TARGET = "final_temperature"
META = ["batch_id", "prediction_at"]
SEED = 42


def validate_frame(frame, training=True):
    expected = META + FEATURES + ([TARGET] if training else [])
    if frame.columns.duplicated().any() or set(frame.columns) != set(expected):
        raise ValueError("CSV must contain exactly: " + ", ".join(expected))
    frame = frame[expected].copy()
    if frame.empty:
        raise ValueError("CSV cannot be empty")
    if frame["batch_id"].isna().any():
        raise ValueError("batch_id cannot be missing")
    frame["batch_id"] = frame["batch_id"].astype(str).str.strip()
    if frame["batch_id"].eq("").any() or frame["batch_id"].duplicated().any():
        raise ValueError("Each batch_id must be nonempty and unique")
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True, errors="raise", format="ISO8601")
    if frame["prediction_at"].isna().any():
        raise ValueError("prediction_at cannot be missing")
    for column in FEATURES + ([TARGET] if training else []):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite with no missing values")
        minimum = 0 if column not in ("initial_temperature", TARGET) else np.nextafter(0.0, 1.0)
        if (frame[column] < minimum).any():
            raise ValueError(f"{column} contains a negative or invalid temperature value")
    return frame


def read_frame(path, training=True):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        for line, cells in enumerate(reader, 2):
            if cells and len(cells) != len(header):
                raise ValueError(f"CSV row {line} has the wrong cell count")
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV header names")
    return validate_frame(pd.read_csv(path, dtype={"batch_id": str}), training)


def temporal_partitions(frame):
    times = np.sort(frame["prediction_at"].unique())
    if len(frame) < 30 or len(times) < 12:
        raise ValueError("Training needs at least 30 batches and 12 distinct prediction times")
    split = int(len(times) * 0.8)
    boundary = times[split]
    development = frame[frame["prediction_at"] < boundary].reset_index(drop=True)
    holdout = frame[frame["prediction_at"] >= boundary].reset_index(drop=True)
    return development, holdout


def temporal_folds(frame):
    times = np.sort(frame["prediction_at"].unique())
    for train, valid in TimeSeriesSplit(n_splits=3).split(times):
        yield (np.flatnonzero(frame["prediction_at"].isin(times[train])),
               np.flatnonzero(frame["prediction_at"].isin(times[valid])))


def candidate_models():
    return {
        "median_baseline": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "random_forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                              max_depth=12, random_state=SEED, n_jobs=1),
    }


def train_model(csv_path, output_dir, synthetic=False):
    frame = read_frame(csv_path).sort_values(["prediction_at", "batch_id"]).reset_index(drop=True)
    development, holdout = temporal_partitions(frame)
    X, y = development[FEATURES], development[TARGET]
    folds = list(temporal_folds(development))
    scores = {}
    models = candidate_models()
    for name, estimator in models.items():
        fold_mae = []
        for train, valid in folds:
            fitted = clone(estimator).fit(X.iloc[train], y.iloc[train])
            fold_mae.append(float(mean_absolute_error(y.iloc[valid], fitted.predict(X.iloc[valid]))))
        scores[name] = {"mae": float(np.mean(fold_mae)), "fold_mae": fold_mae}
    selected = min(scores, key=lambda name: (scores[name]["mae"], name))
    fitted = clone(models[selected]).fit(X, y)
    prediction = fitted.predict(holdout[FEATURES])
    baseline = DummyRegressor(strategy="median").fit(X, y).predict(holdout[FEATURES])
    mae = float(mean_absolute_error(holdout[TARGET], prediction))
    report = {
        "schema_version": 1, "data_kind": "synthetic" if synthetic else "local_user_data",
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "features": FEATURES, "selected_model": selected, "cv": scores,
        "split": {"strategy": "chronological, tied timestamps kept together",
                  "development_rows": len(development), "holdout_rows": len(holdout),
                  "development_end": development["prediction_at"].max().isoformat(),
                  "holdout_start": holdout["prediction_at"].min().isoformat()},
        "cv_windows": [{"train_end": development.iloc[train]["prediction_at"].max().isoformat(),
                        "validation_start": development.iloc[valid]["prediction_at"].min().isoformat(),
                        "validation_end": development.iloc[valid]["prediction_at"].max().isoformat()}
                       for train, valid in folds],
        "holdout": {"mae": mae, "rmse": float(math.sqrt(mean_squared_error(holdout[TARGET], prediction))),
                    "median_baseline_mae": float(mean_absolute_error(holdout[TARGET], baseline))},
        "business_target_mae": 6.8, "business_target_met": None if synthetic else mae <= 6.8,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "features": FEATURES, "sklearn_version": sklearn.__version__,
                 "estimator": fitted}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(report), encoding="utf-8")
    predictions = holdout[META + [TARGET]].copy()
    predictions["predicted_temperature"] = prediction
    predictions["absolute_error"] = abs(predictions[TARGET] - prediction)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    pd.concat([development[META].assign(partition="development"),
               holdout[META].assign(partition="holdout")]).to_csv(output / "splits.csv", index=False)
    return report


def render_report(report):
    lines = ["# Steel temperature model evaluation", "",
             "Data: **" + report["data_kind"] + "**", "",
             "Synthetic metrics demonstrate software behavior only." if report["data_kind"] == "synthetic"
             else "Local-data evaluation; raw records are excluded from the public repository.", "",
             "| Model | Development CV MAE (C) |", "|---|---:|"]
    for name, score in sorted(report["cv"].items(), key=lambda item: item[1]["mae"]):
        lines.append(f"| {name} | {score['mae']:.3f} |")
    lines += ["", "Selected using development CV only: **" + report["selected_model"] + "**.", "",
              f"Holdout MAE: **{report['holdout']['mae']:.3f} C**; "
              f"RMSE: {report['holdout']['rmse']:.3f} C; "
              f"median baseline MAE: {report['holdout']['median_baseline_mae']:.3f} C.", "",
              f"Development: {report['split']['development_rows']} batches; "
              f"holdout: {report['split']['holdout_rows']} batches.",
              f"Development ends: {report['split']['development_end']}.",
              f"Holdout starts: {report['split']['holdout_start']}.", "",
              "Business threshold MAE <= 6.8: " + (
                  "not assessed on synthetic data." if report["business_target_met"] is None
                  else str(report["business_target_met"]) + " on this local holdout only."), "",
              "The saved model is fitted on development rows only. No refit on holdout.",
              "Batch timestamps are grouped in chronological folds; fold durations can differ.",
              "No process-control deployment or energy savings are established by this evaluation.", "",
              "Dataset SHA-256: " + report["dataset_sha256"], "", "Package versions:", "",
              *[f"- {key}: {value}" for key, value in report["versions"].items()]]
    return "\n".join(lines) + "\n"


def predict_file(model_path, csv_path, output_path):
    """Only load artifacts created by a trusted training run: joblib can execute code."""
    frame = read_frame(csv_path, training=False)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1 or bundle.get("features") != FEATURES:
        raise ValueError("Unsupported model artifact or feature contract")
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model needs the same scikit-learn version used for training")
    values = np.asarray(bundle["estimator"].predict(frame[FEATURES]))
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Invalid model predictions")
    result = frame[META].copy()
    result["predicted_temperature"] = values
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
