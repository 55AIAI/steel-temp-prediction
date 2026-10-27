# 🏭 Steel Temperature Prediction

Industrial regression experiment to predict the final measured temperature of a steel batch.

## Methodology correction (October 2026)

**Important:** An earlier notebook version aggregated all temperature measurements, including the final target, into features such as `temp_mean`, `temp_min`, `temp_max` and `temp_std`. This caused **target leakage**. That feature construction has been fixed: only the measurements **before** the final measurement contribute to temperature features.

The previously saved test MAE of 2.594 is **invalid for reporting** until the corrected notebook is run with the original data and evaluated again. No replacement score is claimed.

## Dataset requirements

The notebook expects the seven original CSV files in its working directory: `data_arc_new.csv`, `data_bulk_new.csv`, `data_bulk_time_new.csv`, `data_gas_new.csv`, `data_temp_new.csv`, `data_wire_new.csv`, and `data_wire_time_new.csv`. These files are not included in the public repository; obtain them from an authorized source before executing.

## Run

1. Create a clean Python environment with Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy and XGBoost.
2. Place the authorized CSVs alongside `Final_Y.ipynb`.
3. Restart the notebook kernel and run every cell from top to bottom.
4. Confirm that temperature features exclude the target measurement and report the *new* held-out MAE.

**Requirement:** MAE ≤ 6.8 (to be reassessed after leakage correction).

## Safety of evaluation

Do not select hyperparameters using held-out test results. Re-evaluate features that may not be available at the intended inference time. Reproducibility depends on access to the original source data.
