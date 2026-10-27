# Data contract and prediction cutoff

## Prepared CSV

Each row is one distinct heat. Column names are exact; extra columns are rejected.

| Column | Meaning |
|---|---|
| batch_id | Nonempty unique heat identifier; never a model feature |
| prediction_at | ISO timestamp for the prediction cutoff; never a model feature |
| initial_temperature | First recorded temperature, C |
| heating_seconds | Sum of completed arc intervals inside the observation window |
| active_energy_proxy | Sum of active power times interval seconds |
| reactive_energy_proxy | Sum of reactive power times interval seconds |
| bulk_total | Sum of timestamp-eligible bulk quantities |
| wire_total | Sum of timestamp-eligible wire quantities |
| final_temperature | Last recorded temperature, C; training label only |

Prediction CSVs contain every column above except final_temperature. Row order is
preserved in prediction exports. All numeric values must be finite and present.
Temperatures must be positive; amounts and durations must be nonnegative.

The power and material units are inherited from the source. Energy proxies are not
claimed to be kWh, and bulk/wire totals combine material types. Verify vendor units
and whether aggregation across types is appropriate before applying the model.

Timestamps are parsed onto a common UTC time base. Naive source timestamps are
interpreted as UTC for ordering; the actual plant timezone is not independently known.

## Raw tables

All raw tables require a nonempty key column.

- temp: key, Время замера, Температура.
- arc: key, Начало нагрева дугой, Конец нагрева дугой, Активная мощность,
  Реактивная мощность.
- bulk and bulk_time: key plus the same Bulk N columns.
- wire and wire_time: key plus the same Wire N columns.

A material volume/time table has at most one row per key. Missing volume cells mean
no recorded addition and become zero. A positive volume needs a valid timestamp.
A missing material-source row does not mean zero: that heat is excluded.

The table timestamp is assumed to describe availability of its corresponding
quantity. If the source aggregates quantities delivered across multiple times,
this assumption must be checked with the data owner before operational use.

## Feature and label construction

For each heat, sort temperature observations by time. The first temperature is the
only temperature feature; the final observation supplies only the label and cutoff.
Intermediate temperature values do not enter the feature vector.

Exclude the entire heat if a temperature observation is missing/non-finite, time is
missing/duplicated, or there is only one measurement. In particular, a missing final
temperature must not silently be replaced by an earlier available value.

Arc durations and power must be valid. Only complete arc intervals from the initial
measurement through the cutoff are summed. Intervals crossing either boundary cause
exclusion because partial-interval energy cannot be reconstructed. Completed events
wholly after the cutoff do not contribute.

Bulk and wire additions contribute only when their recorded timestamps are between
the initial measurement and cutoff, inclusively. A positive untimed quantity makes
the heat unusable. Untimestamped gas totals are excluded from the feature contract.

The actual final-measurement timestamp sets a retrospective cutoff. It is not itself
a model feature, but its schedule may depend on operator decisions. Consequently
this evaluation does not establish performance at an independently scheduled online
cutoff or at the beginning of treatment.

Preparation audits record heat counts by exclusion reason and SHA-256 hashes.
Quality gates inspect complete heat histories; this is an offline preparation
routine, not a live streaming-data admission policy.
