# Steel temperature model evaluation

Data: **synthetic**

Synthetic metrics demonstrate software behavior only.

| Model | Development CV MAE (C) |
|---|---:|
| ridge | 3.403 |
| random_forest | 6.093 |
| median_baseline | 13.405 |

Selected using development CV only: **ridge**.

Holdout MAE: **2.098 C**; RMSE: 2.596 C; median baseline MAE: 16.157 C.

Development: 144 batches; holdout: 36 batches.
Development ends: 2025-01-12T22:00:00+00:00.
Holdout starts: 2025-01-13T00:00:00+00:00.

Business threshold MAE <= 6.8: not assessed on synthetic data.

The saved model is fitted on development rows only. No refit on holdout.
Batch timestamps are grouped in chronological folds; fold durations can differ.
No process-control deployment or energy savings are established by this evaluation.

Dataset SHA-256: f138bdc924805a667b0c04b833127db1c72fb4854493896523d804c42f989400

Package versions:

- python: 3.13.7
- scikit_learn: 1.7.1
- pandas: 2.3.2
- numpy: 2.5.1
- joblib: 1.5.1
