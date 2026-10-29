# Dataset Description and Preprocessing Documentation

**Project:** Adaptive Cloud-Based Big Data Analytics for Urban Taxi Demand Prediction Using Streaming Data and Concept Drift Detection
**Module:** Member 1: data collection, cleaning, feature engineering, Spark processing
**Code:** `preprocess_local.ipynb` and `spark_view_and_ml_prep.ipynb` (PySpark). **Reports:** `dataset_profile.json`, `quality_report.json`

---

## 1. Dataset

| Item | Value |
|---|---|
| Source | Kaggle, *Newyork Yellow Taxi Trip Data 2020-2019* (`microize/newyork-yellow-taxi-trip-data-2020-2019`), derived from NYC TLC Trip Record Data |
| License | see the dataset page on Kaggle |
| Files | 18 monthly CSV files, 2019-01 to 2020-06 (about 1.8 GB zipped) |
| Period used | 2019-01-01 to 2020-06-30 (18 months) |
| Raw rows | 101,246,797 |
| Distinct pickup zone IDs | 263 (including unknown/invalid IDs) |
| Zone lookup | `taxi+_zone_lookup.csv` (LocationID, Borough, Zone) |

The files follow the standard TLC yellow-taxi schema (vendor, timestamps, passengers, distance, fares, zones, payment). Only five columns are used:

| Raw column | Internal name | Meaning |
|---|---|---|
| `tpep_pickup_datetime` | `pickup_ts` | Pickup time (local wall-clock time) |
| `tpep_dropoff_datetime` | `dropoff_ts` | Drop-off time |
| `PULocationID` | `zone_id` | Pickup taxi zone (1 to 263; 264/265 = unknown) |
| `passenger_count` | `passenger_count` | Passengers (driver-entered) |
| `trip_distance` | `trip_distance` | Trip distance in miles |

### 1.1 Raw data profile

- Missing values: `passenger_count` 527,484 nulls (0.52%); no nulls in pickup time, drop-off time, zone or distance.
- Typical trip (1st / 50th / 99th percentile, 2% sample): distance 0.02 / 1.62 / 19.2 miles; duration 57 s / 669 s / 3,730 s; passengers 0 / 1 / 6.
- **Bad timestamps:** pickup times range from 2001 to 2090. 1,135 rows fall outside the 18-month period (for example 2001, 2008, 2018-12, 2026, 2090).
- **Monthly volume** (rows in the period):

| Month | Rows | Month | Rows | Month | Rows |
|---|---:|---|---:|---|---:|
| 2019-01 | 7,667,565 | 2019-07 | 6,310,376 | 2020-01 | 6,405,286 |
| 2019-02 | 7,019,184 | 2019-08 | 6,073,089 | 2020-02 | 6,299,567 |
| 2019-03 | 7,832,324 | 2019-09 | 6,567,628 | 2020-03 | 3,006,988 |
| 2019-04 | 7,433,149 | 2019-10 | 7,214,110 | 2020-04 | 237,922 |
| 2019-05 | 7,565,360 | 2019-11 | 6,877,811 | 2020-05 | 348,403 |
| 2019-06 | 6,940,798 | 2019-12 | 6,896,314 | 2020-06 | 549,788 |

Demand falls by about 96% between February and April 2020 (COVID-19). The drop is real, so it was **kept**: it is a genuine distribution shift and a natural test for concept-drift detection.

---

## 2. Preprocessing pipeline (Apache Spark)

Environment: PySpark 4.x in local mode, Java 17+, Spark session timezone set to UTC.

| Step | What happens |
|---|---|
| 1. Staging | Each CSV is parsed once, reduced to the 5 columns above, cast to proper types and stored as Parquet (`staging_raw/`). Later steps read this compact copy. |
| 2. Profiling | Row counts, nulls, ranges, quantiles and monthly volume are saved to `dataset_profile.json`. |
| 3. Cleaning | Invalid rows are flagged by rule, counted, removed (Section 3) and duplicates dropped. |
| 4. Trip features | Year, month, day, hour, weekday and weekend flag are extracted; zone name and borough are joined from the lookup. Saved as `trips_clean/`. |
| 5. Demand aggregation | Pickups are counted per zone and 15-minute interval on a complete grid; lag and rolling features and calendar features are added. Saved as `demand_15min/`. |
| 6. Validation | Asserts there are no duplicate (interval, zone) keys and that total demand equals the number of clean trips. Results go to `quality_report.json`. |
| 7. Spark view | `spark_view_and_ml_prep.ipynb` reloads the Parquet outputs, shows schemas, samples and Spark SQL summaries (monthly trips, top zones, hourly pattern, monthly demand) and re-checks the demand table. |
| 8. ML preparation | Removes the warm-up week of each zone, adds cyclic time features, renames the target to `label`, splits by time into train and test, and saves `ml_ready_15min/`. |

Timestamps are NYC local wall-clock time. Spark treats them as UTC, which avoids missing or repeated hours at daylight-saving changes.

---

## 3. Cleaning rules and results

| Rule | Condition for removal | Rows affected | % of raw |
|---|---|---:|---:|
| Missing required fields | pickup time, drop-off time or zone is null | 0 | 0.00% |
| Pickup outside period | pickup before 2019-01-01 or on/after 2020-07-01 | 1,135 | 0.001% |
| Invalid duration | duration under 60 s, over 6 h, or drop-off not after pickup | 1,293,899 | 1.28% |
| Invalid distance | distance is 0 or negative, or over 100 miles | 947,351 | 0.94% |
| Invalid zone | zone ID outside 1 to 263 (264/265 = unknown) | 955,661 | 0.94% |
| **Removed (union of the rules above)** | a row can break several rules | **2,414,034** | **2.38%** |
| Duplicates | identical pickup, drop-off, zone, distance and passengers | 171,965 | 0.17% |
| **Clean trips** | | **98,660,798** | **97.45%** |

**Passenger count (not a removal rule).** 1,857,673 rows have a passenger count of 0 or above 6, and 527,484 are null. These trips are real pickups, and demand is a count of pickups, so they are **kept**. Out-of-range values are set to null (a missing value is left as null). The field is not used as a demand feature. Setting `DROP_BAD_PASSENGERS = True` in the notebook removes such rows instead.

Check: 101,246,797 − 2,414,034 − 171,965 = 98,660,798.

---

## 4. Output datasets

Stored as Parquet. Not committed to GitHub because of size; see Section 6.

### 4.1 `trips_clean/` (98,660,798 rows)
Partitioned by `pickup_year` and `pickup_month`.

| Column | Description |
|---|---|
| `pickup_ts`, `dropoff_ts` | pickup and drop-off timestamps |
| `zone_id`, `borough`, `zone_name` | pickup zone (name `Unknown` if not in the lookup) |
| `passenger_count`, `trip_distance`, `duration_s` | passengers (nullable), miles, seconds |
| `pickup_day`, `pickup_hour` | day of month, hour 0 to 23 |
| `pickup_weekday`, `is_weekend` | weekday (Mon=0 to Sun=6), 1 for Sat/Sun |

### 4.2 `demand_15min/` (13,705,632 rows)
One row per zone per 15-minute interval: 52,512 intervals × 261 zones. Intervals with no pickups are included with `demand = 0`, so the series have no gaps and lag offsets are exact. The raw data has 263 distinct zone IDs, including unknown/invalid ones; after cleaning, 261 of the valid zone IDs (1 to 263) have at least one pickup in the period.

| Column | Description |
|---|---|
| `interval_start`, `zone_id` | key |
| `demand` | number of pickups in the interval (**prediction target**) |
| `lag_1`, `lag_2`, `lag_4` | demand 1, 2 and 4 intervals earlier (15, 30, 60 min) |
| `lag_1d`, `lag_1w` | demand at the same time 1 day and 1 week earlier |
| `roll_mean_4`, `roll_mean_1d` | mean of the previous 4 intervals / previous 96 intervals |
| `hour`, `slot_of_day`, `day`, `weekday`, `month`, `is_weekend` | calendar features (`slot_of_day` = 0 to 95) |

### 4.3 `ml_ready_15min/` (about 13.5 million rows)
Partitioned by `split` (`train` / `test`). Built from `demand_15min` by `spark_view_and_ml_prep.ipynb`.

| Part | Description |
|---|---|
| `interval_start`, `label` | interval key and the target (= `demand`, as a double) |
| `split` | `train` before `TEST_START` (default 2020-01-01), `test` from that date; split by time, no overlap |
| 18 features | `zone_id`, `hour`, `slot_of_day`, `weekday`, `is_weekend`, `month`, `day`, `lag_1`, `lag_2`, `lag_4`, `lag_1d`, `lag_1w`, `roll_mean_4`, `roll_mean_1d`, plus `slot_sin`, `slot_cos`, `weekday_sin`, `weekday_cos` (cyclic encodings, so 23:45 and 00:00 are close) |

Differences from `demand_15min`: the first week of every zone (672 intervals, where `lag_1w` has no history) is removed, so no feature is null; the notebook asserts this. The default split gives training data from 2019 and test data from 2020, which includes the COVID shift.

A naive reference (predict the demand of the same time one week earlier) is computed in the notebook so that Member 2 has a baseline MAE and RMSE to beat.

---

## 5. Notes for Members 2 and 3

- **Nulls in lag features:** `demand_15min` has null lags at the start of every zone (`lag_1w` for the first 672 intervals). `ml_ready_15min` already removes that first week; use it for training.
- **Split by time, never randomly.** Sort by `interval_start`.
- **COVID shift:** March to June 2020 differs strongly from 2019. A static model trained on 2019 will degrade there, which is the intended setting for the drift experiments.
- **Streaming replay (Member 3):** replay `trips_clean` ordered by `pickup_ts`; re-aggregate with the same 15-minute rule so that streaming demand matches `demand_15min`.
- **Time zone:** all timestamps are local NYC time stored as-is.

---

## 6. Reproducing the results

1. Install Java 17+, then `pip install pyspark pyarrow fsspec`. On Windows also install `winutils.exe` and `hadoop.dll` and set `HADOOP_HOME`.
2. Place the 18 CSVs and the zone lookup in one folder; set `RAW_DIR`, `OUT_DIR`, `DOCS_DIR` in the config cell.
3. Run `preprocess_local.ipynb` top to bottom (about 10 GB of disk for the extracted CSVs, a few GB for the outputs), then `spark_view_and_ml_prep.ipynb`.
4. Data is shared through cloud storage, not Git; the repository contains code and the two JSON reports only.
