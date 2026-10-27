"""Construct one row per heat from timestamped process records."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from steel_model import FEATURES, META, TARGET, validate_frame

TEMP_TIME = "Время замера"
TEMP_VALUE = "Температура"
ARC_START = "Начало нагрева дугой"
ARC_END = "Конец нагрева дугой"
ACTIVE = "Активная мощность"
REACTIVE = "Реактивная мощность"
TABLES = ("temp", "arc", "bulk", "bulk_time", "wire", "wire_time")


def _material_total(values, times, key, start, cutoff):
    volume = values.loc[key].fillna(0)
    if (volume < 0).any() or not np.isfinite(volume.to_numpy(dtype=float)).all():
        raise ValueError("invalid_material_volume")
    added_at = pd.to_datetime(times.loc[key], utc=True, errors="coerce", format="ISO8601")
    if ((volume > 0) & added_at.isna()).any():
        raise ValueError("missing_material_time")
    return float(volume[(added_at >= start) & (added_at <= cutoff)].sum())


def prepare_tables(tables):
    frames = {name: value.copy() for name, value in tables.items()}
    missing = set(TABLES) - set(frames)
    if missing:
        raise ValueError("Missing tables: " + ", ".join(sorted(missing)))
    for name, frame in frames.items():
        if "key" not in frame or frame["key"].isna().any():
            raise ValueError(f"{name}: missing batch key")
        frame["key"] = frame["key"].astype(str).str.strip()
        if frame["key"].eq("").any():
            raise ValueError(f"{name}: empty batch key")
    temp, arc = frames["temp"], frames["arc"]
    for column in (TEMP_TIME, TEMP_VALUE):
        if column not in temp:
            raise ValueError("Missing temperature column: " + column)
    for column in (ARC_START, ARC_END, ACTIVE, REACTIVE):
        if column not in arc:
            raise ValueError("Missing arc column: " + column)
    temp[TEMP_TIME] = pd.to_datetime(temp[TEMP_TIME], utc=True, errors="coerce", format="ISO8601")
    temp[TEMP_VALUE] = pd.to_numeric(temp[TEMP_VALUE], errors="raise")
    for column in (ARC_START, ARC_END):
        arc[column] = pd.to_datetime(arc[column], utc=True, errors="coerce", format="ISO8601")
    for column in (ACTIVE, REACTIVE):
        arc[column] = pd.to_numeric(arc[column], errors="raise")
    for name in ("bulk", "wire"):
        values, times = frames[name], frames[name + "_time"]
        if values["key"].duplicated().any() or times["key"].duplicated().any():
            raise ValueError(name + ": duplicate material batch key")
        columns = [column for column in values if column.startswith(name.title() + " ")]
        if not columns or set(times.columns) != set(values.columns):
            raise ValueError(name + ": volume/time columns must match")
        for column in columns:
            values[column] = pd.to_numeric(values[column], errors="raise")
        frames[name] = values.set_index("key")[columns]
        frames[name + "_time"] = times.set_index("key")[columns]
    arc_groups = dict(tuple(arc.groupby("key", sort=False)))
    rows, dropped = [], Counter()
    for key, history in temp.groupby("key", sort=False):
        if (history[TEMP_TIME].isna().any() or history[TEMP_TIME].duplicated().any()
                or not np.isfinite(history[TEMP_VALUE]).all() or (history[TEMP_VALUE] <= 0).any()):
            dropped["invalid_temperature_history"] += 1
            continue
        history = history.sort_values(TEMP_TIME)
        if len(history) < 2:
            dropped["single_temperature_measurement"] += 1
            continue
        if key not in arc_groups or any(key not in frames[name].index
                                        for name in ("bulk", "wire", "bulk_time", "wire_time")):
            dropped["missing_process_source"] += 1
            continue
        start, cutoff = history[TEMP_TIME].iloc[0], history[TEMP_TIME].iloc[-1]
        heating = arc_groups[key].copy()
        duration = (heating[ARC_END] - heating[ARC_START]).dt.total_seconds()
        if (heating[[ARC_START, ARC_END]].isna().any().any() or (duration <= 0).any()
                or not np.isfinite(heating[[ACTIVE, REACTIVE]]).all().all()
                or (heating[[ACTIVE, REACTIVE]] < 0).any().any()):
            dropped["invalid_arc_history"] += 1
            continue
        if (((heating[ARC_START] < start) & (heating[ARC_END] > start))
                | ((heating[ARC_START] < cutoff) & (heating[ARC_END] > cutoff))).any():
            dropped["arc_crosses_measurement_boundary"] += 1
            continue
        eligible = (heating[ARC_START] >= start) & (heating[ARC_END] <= cutoff)
        heating = heating[eligible]
        duration = duration[eligible]
        if heating.empty:
            dropped["no_completed_heating_in_window"] += 1
            continue
        try:
            bulk = _material_total(frames["bulk"], frames["bulk_time"], key, start, cutoff)
            wire = _material_total(frames["wire"], frames["wire_time"], key, start, cutoff)
        except ValueError as exc:
            dropped[str(exc)] += 1
            continue
        rows.append({"batch_id": key, "prediction_at": cutoff.isoformat(),
                     "initial_temperature": float(history[TEMP_VALUE].iloc[0]),
                     "heating_seconds": float(duration.sum()),
                     "active_energy_proxy": float((heating[ACTIVE] * duration).sum()),
                     "reactive_energy_proxy": float((heating[REACTIVE] * duration).sum()),
                     "bulk_total": bulk, "wire_total": wire,
                     TARGET: float(history[TEMP_VALUE].iloc[-1])})
    if not rows:
        raise ValueError("No eligible batches after source and timestamp checks")
    prepared = validate_frame(pd.DataFrame(rows)).sort_values(["prediction_at", "batch_id"]).reset_index(drop=True)
    audit = {"input_temperature_batches": int(temp["key"].nunique()), "retained_batches": len(prepared),
             "dropped_batches": dict(dropped), "features": FEATURES,
             "prediction_contract": "late-process snapshot, completed events between first and final measurement",
             "excluded_sources": ["untimestamped gas totals"], "target": TARGET}
    return prepared[META + FEATURES + [TARGET]], audit


def prepare_raw(folder, output_csv):
    folder = Path(folder)
    tables, hashes = {}, {}
    for name in TABLES:
        path = folder / f"data_{name}_new.csv"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tables[name] = pd.read_csv(path, dtype={"key": str})
    prepared, audit = prepare_tables(tables)
    audit["source_sha256"] = hashes
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    audit["prepared_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return prepared, audit
