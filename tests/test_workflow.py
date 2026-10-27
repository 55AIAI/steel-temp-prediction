import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import (ACTIVE, ARC_END, ARC_START, REACTIVE, TEMP_TIME, TEMP_VALUE,
                          prepare_tables)
from steel_model import (FEATURES, META, TARGET, predict_file, read_frame, temporal_folds,
                         temporal_partitions, train_model, validate_frame)

BASE = Path(__file__).resolve().parents[1]


def raw_fixture():
    return {
        "temp": pd.DataFrame({"key": ["A", "A", "A"], TEMP_TIME: [
            "2025-01-01 10:00:00", "2025-01-01 10:20:00", "2025-01-01 11:00:00"],
            TEMP_VALUE: [1500, 1520, 1540]}),
        "arc": pd.DataFrame({"key": ["A"], ARC_START: ["2025-01-01 10:05:00"],
                              ARC_END: ["2025-01-01 10:15:00"], ACTIVE: [2.0], REACTIVE: [1.0]}),
        "bulk": pd.DataFrame({"key": ["A"], "Bulk 1": [20.0]}),
        "bulk_time": pd.DataFrame({"key": ["A"], "Bulk 1": ["2025-01-01 10:10:00"]}),
        "wire": pd.DataFrame({"key": ["A"], "Wire 1": [5.0]}),
        "wire_time": pd.DataFrame({"key": ["A"], "Wire 1": ["2025-01-01 10:10:00"]}),
    }


class Preparation(unittest.TestCase):
    def test_known_process_features(self):
        frame, audit = prepare_tables(raw_fixture())
        self.assertEqual(frame.loc[0, "heating_seconds"], 600)
        self.assertEqual(frame.loc[0, "active_energy_proxy"], 1200)
        self.assertEqual(frame.loc[0, "reactive_energy_proxy"], 600)
        self.assertEqual(frame.loc[0, "bulk_total"], 20)
        self.assertEqual(audit["retained_batches"], 1)

    def test_target_and_intermediate_measurements_do_not_change_features(self):
        tables = raw_fixture()
        before, _ = prepare_tables(tables)
        tables["temp"].loc[1, TEMP_VALUE] = 1700
        tables["temp"].loc[2, TEMP_VALUE] = 1800
        after, _ = prepare_tables(tables)
        pd.testing.assert_frame_equal(before[META + FEATURES], after[META + FEATURES])
        self.assertNotEqual(before[TARGET].iloc[0], after[TARGET].iloc[0])

    def test_material_after_cutoff_does_not_enter_features(self):
        tables = raw_fixture()
        tables["bulk_time"].loc[0, "Bulk 1"] = "2025-01-01 11:10:00"
        frame, _ = prepare_tables(tables)
        self.assertEqual(frame.loc[0, "bulk_total"], 0)

    def test_heating_after_cutoff_does_not_enter_features(self):
        tables = raw_fixture()
        extra = tables["arc"].copy()
        extra[ARC_START], extra[ARC_END], extra[ACTIVE] = (
            "2025-01-01 11:05:00", "2025-01-01 11:15:00", 9000)
        tables["arc"] = pd.concat([tables["arc"], extra], ignore_index=True)
        frame, _ = prepare_tables(tables)
        self.assertEqual(frame.loc[0, "active_energy_proxy"], 1200)

    def test_missing_final_temperature_does_not_become_previous_target(self):
        tables = raw_fixture()
        tables["temp"].loc[2, TEMP_VALUE] = np.nan
        with self.assertRaisesRegex(ValueError, "No eligible"):
            prepare_tables(tables)

    def test_missing_nonzero_material_time_is_rejected(self):
        tables = raw_fixture()
        tables["wire_time"].loc[0, "Wire 1"] = None
        with self.assertRaisesRegex(ValueError, "No eligible"):
            prepare_tables(tables)

    def test_invalid_arc_records_are_rejected(self):
        for column, value in ((ACTIVE, -1), (REACTIVE, np.inf), (ARC_END, "bad-time")):
            tables = raw_fixture()
            tables["arc"].loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "No eligible"):
                prepare_tables(tables)

    def test_arc_crossing_prediction_boundary_is_rejected(self):
        tables = raw_fixture()
        tables["arc"].loc[0, ARC_END] = "2025-01-01 11:05:00"
        with self.assertRaisesRegex(ValueError, "No eligible"):
            prepare_tables(tables)

    def test_duplicate_material_batch_is_not_summed_silently(self):
        tables = raw_fixture()
        tables["bulk"] = pd.concat([tables["bulk"], tables["bulk"]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_tables(tables)

    def test_drop_audit_accounts_for_every_batch(self):
        tables = raw_fixture()
        missing = tables["temp"].copy()
        missing["key"] = "B"
        tables["temp"] = pd.concat([tables["temp"], missing], ignore_index=True)
        frame, audit = prepare_tables(tables)
        self.assertEqual(len(frame), 1)
        self.assertEqual(audit["dropped_batches"], {"missing_process_source": 1})
        self.assertEqual(audit["input_temperature_batches"],
                         audit["retained_batches"] + sum(audit["dropped_batches"].values()))


class SchemaAndSplits(unittest.TestCase):
    def frame(self):
        return read_frame(BASE / "demo/batches.csv")

    def test_target_derived_features_are_not_accepted(self):
        frame = self.frame()
        for column in ("temp_mean", "temp_max", "temp_min", "temp_last"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "exactly"):
                validate_frame(frame.assign(**{column: 42}))

    def test_invalid_values_and_duplicate_batches_are_rejected(self):
        for column, value in (("initial_temperature", 0), ("heating_seconds", -1),
                               ("bulk_total", np.nan), ("wire_total", np.inf),
                               ("prediction_at", None), ("batch_id", "")):
            frame = self.frame()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_frame(frame)
        frame = self.frame()
        frame.loc[1, "batch_id"] = frame.loc[0, "batch_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_frame(frame)

    def test_inference_requires_features_without_target(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_frame(frame, training=False)
        self.assertEqual(len(validate_frame(frame.drop(columns=TARGET), training=False)), len(frame))

    def test_temporal_folds_never_train_on_future_or_tied_timestamps(self):
        frame = self.frame()
        frame.loc[1, "prediction_at"] = frame.loc[0, "prediction_at"]
        development, holdout = temporal_partitions(frame)
        self.assertLess(development.prediction_at.max(), holdout.prediction_at.min())
        self.assertTrue(set(development.batch_id).isdisjoint(set(holdout.batch_id)))
        for train, valid in temporal_folds(development):
            self.assertLess(development.iloc[train].prediction_at.max(),
                            development.iloc[valid].prediction_at.min())

    def test_short_datasets_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "30 batches"):
            temporal_partitions(self.frame().head(10))

    def test_duplicate_headers_and_wrong_cell_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("batch_id,batch_id\nA,A\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_frame(path)
            path.write_text("batch_id,prediction_at\nA\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row 2"):
                read_frame(path)


class ModelWorkflow(unittest.TestCase):
    def test_holdout_labels_cannot_influence_training_or_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = train_model(BASE / "demo/batches.csv", root / "first", synthetic=True)
            frame = read_frame(BASE / "demo/batches.csv")
            _, holdout = temporal_partitions(frame)
            frame.loc[frame.batch_id.isin(holdout.batch_id), TARGET] += 500
            altered = root / "altered.csv"
            frame.to_csv(altered, index=False)
            second = train_model(altered, root / "second", synthetic=True)
            self.assertEqual(first["cv"], second["cv"])
            self.assertEqual(first["selected_model"], second["selected_model"])
            self.assertGreater(second["holdout"]["mae"], first["holdout"]["mae"] + 400)
            self.assertIsNone(first["business_target_met"])

    def test_scaler_fits_development_only_and_artifact_roundtrip_works(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("steel_model.candidate_models", return_value={
                    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10))}):
                report = train_model(BASE / "demo/batches.csv", root, synthetic=True)
            bundle = joblib.load(root / "model.joblib")
            development, _ = temporal_partitions(read_frame(BASE / "demo/batches.csv"))
            np.testing.assert_allclose(bundle["estimator"][0].mean_, development[FEATURES].mean())
            predicted = predict_file(root / "model.joblib", BASE / "demo/predict.csv", root / "predicted.csv")
            self.assertEqual(len(predicted), 5)
            self.assertTrue(np.isfinite(predicted.predicted_temperature).all())
            self.assertEqual(report["features"], FEATURES)
            pd.testing.assert_frame_equal(predicted, predict_file(
                root / "model.joblib", BASE / "demo/predict.csv", root / "again.csv"))

    def test_cli_demo_and_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            result = subprocess.run([sys.executable, str(BASE / "main.py"), "train", "--demo",
                                     "--output-dir", str(output)], check=False,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "holdout_predictions.csv").exists())
            self.assertTrue((output / "splits.csv").exists())
            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["data_kind"], "synthetic")
            self.assertEqual(len(report["cv"]), 3)
            result = subprocess.run([sys.executable, str(BASE / "main.py"), "predict",
                                     "--model", str(output / "model.joblib"), "--csv",
                                     str(BASE / "demo/predict.csv"), "--output", str(output / "predict.csv")],
                                    check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(pd.read_csv(output / "predict.csv")), 5)

    def test_incompatible_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            joblib.dump({"schema_version": 0}, root / "bad.joblib")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                predict_file(root / "bad.joblib", BASE / "demo/predict.csv", root / "pred.csv")


if __name__ == "__main__":
    unittest.main()
