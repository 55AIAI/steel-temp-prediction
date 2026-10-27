# Steel Temperature Prediction

Predict the final measured temperature of a steel heat from an initial temperature
and timestamped process events available by the prediction cutoff.

The project includes raw-data preparation, chronological model comparison,
a saved-model prediction command, tests, Docker and a small synthetic demo.

## Evaluation status

The repository includes a runnable synthetic example and [its sample report](docs/demo-report.md).
Synthetic metrics demonstrate software behavior and do not assess the industrial
MAE <= 6.8 requirement. No verified industrial score is published here.

Raw industrial records, their hashes, real-data evaluation metadata and trained
local-data artifacts are excluded from the public repository.

The earlier notebook used temperature aggregates containing the target and scaled
data before cross-validation. Its historical scores are not valid quality claims.
The notebook now walks through the corrected workflow.

## Quick start

Python 3.11–3.13 is checked in CI. The demo uses only synthetic data; no API key,
database or model download is required after installing dependencies.

```bash
git clone https://github.com/55AIAI/steel-temp-prediction.git
cd steel-temp-prediction
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py train --demo --output-dir output/demo
python main.py predict --model output/demo/model.joblib --csv demo/predict.csv --output output/demo/new_predictions.csv
```

Training writes `metrics.json`, `report.md`, `model.joblib`,
`holdout_predictions.csv` and `splits.csv`. The saved estimator is fitted on
development rows only. Synthetic scores do not assess the industrial MAE target.

Open [Final_Y.ipynb](Final_Y.ipynb) for a short walkthrough.
Jupyter is optional and can be installed separately.

## Prepare your own raw data

Provide these six files in one local directory:

- `data_temp_new.csv`
- `data_arc_new.csv`
- `data_bulk_new.csv` and `data_bulk_time_new.csv`
- `data_wire_new.csv` and `data_wire_time_new.csv`

```bash
python main.py prepare --raw-dir path/to/raw --output data/prepared.csv
python main.py train --csv data/prepared.csv --output-dir output/local
```

Preparation writes an audit beside the CSV and accounts for every retained or
excluded temperature batch. The original Russian source-column names are supported.
See [the data contract](docs/data-contract.md) for schemas, cutoff rules and assumptions.

The model receives only six fixed features: initial temperature, completed heating
seconds, active/reactive energy proxies, and bulk/wire totals within the observation
window. The target, batch ID, timestamps and all-temperature aggregates are excluded.
Untimestamped gas totals are not used.

For prediction, provide the same feature columns plus `batch_id` and `prediction_at`,
without `final_temperature`. Unexpected columns, duplicate batch IDs, invalid dates,
missing/non-finite values and negative process amounts are rejected.

## Evaluation and limits

The latest 20% of distinct prediction timestamps form the holdout. Tied timestamps
stay together. Models are selected using three expanding chronological folds on the
development subset; scaling for Ridge is fitted inside each fold's Pipeline.

Candidates are a median baseline, Ridge and a random forest with fixed parameters.
The holdout is evaluated after model selection, with no refit on its labels.
Folds contain time-ordered groups of batches, not equal-duration calendar windows.

This is a retrospective late-process temperature estimate, not a forecast at the
start of treatment. Production use needs an explicit scheduled cutoff, source
availability checks, process review and a new untouched evaluation dataset.
No energy savings, safe process control or production readiness is established.
[Model card and limitations](docs/model-card.md) explain the scope.

Joblib artifacts can execute code when loaded. Only load models from trusted
training runs, using the same dependency versions as training.

## Tests and Docker

```bash
python -m unittest discover -s tests -v
docker build -t steel-temperature .
docker run --rm steel-temperature train --demo
```

CI runs regression tests on Python 3.11–3.13, the synthetic train/predict workflow,
and a Docker build/demo check. Tests verify target-value invariance of features,
cutoff handling, missing final targets, chronological boundaries, holdout isolation
and artifact round-trips.
