# Model card

## Task and population

A regression workflow estimates the final observed steel temperature in a late-process
snapshot. One row is one heat. This project does not select furnace settings, dispatch
actuators or establish reductions in energy use.

Raw process data is supplied locally by the operator. Raw records, input hashes,
real-data metrics, individual heat predictions and trained local artifacts are
excluded from the public repository. Only the explicitly synthetic fixture and
its evaluation are distributed.

## Corrections to the historical notebook

The original temperature mean/min/max/std/range aggregates contained the final
temperature label. A model could therefore use information unavailable before the
target measurement. StandardScaler was also fitted before cross-validation.
The old reported scores must not be compared directly with the corrected pipeline.

The current model uses a fixed six-feature allowlist, one temperature value measured
at the beginning of the observation window, and eligible completed process events.
Changing the final or intermediate temperature values leaves the features unchanged.

## Evaluation

The latest 20% of distinct cutoff timestamps form one holdout. Earlier timestamps
are used for three expanding-window validation folds. All records with the same
timestamp remain together. Model selection uses the unweighted mean fold MAE;
the final holdout is not used to pick a model or fit preprocessing.

Candidates and hyperparameters are fixed in source. The selected estimator is fitted
on development records only, then evaluated once on holdout. Ridge scaling is inside
its Pipeline. Batch sampling is irregular, so folds need not span equal clock durations.

No industrial evaluation metric is published. The MAE <= 6.8 requirement
remains an unevaluated business criterion in the public example. Synthetic scores
cannot establish that criterion or demonstrate deployment performance.

Each operator should use an untouched final dataset for their own assessment.
Once a holdout result has been inspected, further tuning should reserve new data
for another final assessment.

## Limits

- Operator-dependent measurement schedules may bias a retrospective cutoff.
- Upstream sensor reporting latency is not represented.
- Material timestamps must describe availability of the whole associated amount.
- Aggregating across material types loses chemical composition information.
- Raw source units, clock timezone, sensor calibration and completeness need domain review.
- A single site/time period does not establish transfer to other plants.
- No uncertainty intervals or out-of-distribution detection are implemented.
- Runtime package versions are recorded; exact fitted-model portability is not promised.

Only load trusted joblib artifacts. Use the same package versions as training.

## References

- [scikit-learn: data leakage and Pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn: model persistence](https://scikit-learn.org/stable/model_persistence.html)
