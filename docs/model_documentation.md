# Model Documentation (Member 2): Taxi Demand Prediction

**Module:** exploratory analysis, feature selection, model development, static vs periodic retraining
**Code:** `ml/common.py`, `ml/01_eda.py`, `ml/02_train.py`, `ml/03_evaluate.py`
**Input:** `ml_ready_15min/` (Member 1). Train = 2019, test = 2020-01 to 2020-06 (8,970,048 / 4,560,192 rows).

---

## 1. Prediction task

Predict `label` (number of pickups) for one zone in one 15-minute interval, using 18 features
(calendar features, cyclic encodings, lags `lag_1/2/4/1d/1w`, rolling means). Lag features use only past data, so there is no leakage.
Metrics: MAE, RMSE and WAPE (sum of absolute errors / sum of actual demand).
WAPE is reported because MAE falls in April 2020 only because demand is near zero.

## 2. Exploratory data analysis

Figures are in `ml/outputs/figures/`.

| Figure | Finding |
|---|---|
| `eda_monthly_demand.png` | TODO: e.g. demand falls sharply in March-April 2020 (COVID-19) |
| `eda_heatmap_hour_weekday.png` | TODO: peak hours and weekday vs weekend pattern |
| `eda_top_zones.png` | TODO: busiest zone IDs and what share of demand they cover |
| `eda_sparsity.png` | TODO: overall share of zero-demand intervals (printed by `01_eda.py`) |
| `eda_acf.png` | TODO: autocorrelation peaks at 1 day and 1 week, which justifies `lag_1d` and `lag_1w` |
| `eda_feature_corr.png` | TODO: which features correlate most with demand |
| `eda_daily_demand.png` | TODO: the level shift around mid-March 2020 |

## 3. Features and importance

All 18 features from Member 1 were kept for the main model (`FEATURES` in `common.py`; fixed order).
`zone_id` is used as a categorical feature in the saved models.

Feature importance (gain) is in `outputs/feature_importance.csv` and `outputs/figures/feature_importance.png`.

TODO: list the top 5 features and one sentence on each (usually the lag and rolling features dominate).
TODO: mention features that are redundant (`hour`, `slot_of_day` and the sin/cos pairs overlap) and that they were kept / removed.

## 4. Models and tuning

| Model | Description |
|---|---|
| Naive baseline | prediction = demand at the same time one week earlier (`lag_1w`) |
| Linear regression | baseline without `zone_id`, trained on 2019 |
| XGBoost static | trained once on all of 2019, frozen, evaluated on all of 2020 |
| XGBoost periodic (expanding) | retrained at the start of each month on all data before that month |
| XGBoost periodic (sliding) | retrained at the start of each month on the previous 8 weeks only |

**Tuning** used a time-based split (train Jan-Oct 2019, validate Nov-Dec 2019) with early stopping.
The test set was never used for tuning.

| max_depth | best iteration | validation RMSE |
|---|---:|---:|
| 6 | 645 | 4.3830 |
| 8 | 222 | 4.3843 |
| 10 | 99 | 4.3917 |

Chosen: `max_depth = 6`, `n_estimators = 710` (best iteration + 10%). Depths 6 and 8 are nearly equal, so the choice is not sensitive.
Other settings: learning rate 0.1, subsample 0.8, colsample_bytree 0.8, min_child_weight 5 (see `common.py` and `best_hyperparams.json`).

## 5. Results (test period 2020-01 to 2020-06)

### 5.1 Overall

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naive (lag_1w) | 1.338 | 5.943 | 0.372 |
| Linear regression | 0.911 | 3.174 | 0.254 |
| XGBoost static | 0.899 | 2.880 | 0.250 |
| XGBoost periodic, expanding | **0.792** | **2.788** | **0.220** |
| XGBoost periodic, sliding | 0.819 | 2.971 | 0.228 |

### 5.2 MAE per month

| Month | Naive | Linear | Static | Periodic expanding | Periodic sliding |
|---|---:|---:|---:|---:|---:|
| 2020-01 | 2.376 | 1.623 | 1.409 | 1.409 | 1.445 |
| 2020-02 | 2.116 | 1.650 | 1.438 | 1.431 | 1.464 |
| 2020-03 | 2.508 | 1.105 | 0.970 | 0.974 | 1.106 |
| 2020-04 | 0.251 | 0.302 | 0.525 | 0.256 | **0.229** |
| 2020-05 | 0.331 | 0.360 | 0.524 | 0.311 | **0.283** |
| 2020-06 | 0.429 | 0.441 | 0.539 | **0.380** | 0.398 |

TODO: add the per-month WAPE table from `outputs/results_monthly.csv`.

### 5.3 Training cost

| Strategy | Retrains | Rows per retrain | Time per retrain |
|---|---:|---:|---:|
| Static | 1 | 8.97 M | about 830 s |
| Periodic expanding | 6 | 9.0 M to 12.8 M | about 830-1050 s |
| Periodic sliding | 6 | 1.4 M | about 120-130 s |

TODO: verify these against `outputs/training_log.csv` (the file and the console output gave different totals for the sliding model).

### 5.4 Interpretation

- Jan-Mar 2020: static and periodic models perform almost the same, because the data still resembles 2019.
- From April 2020 the static model degrades (MAE about 0.52 vs 0.23-0.31 for periodic) and is worse than the naive baseline in April-May. This is concept drift caused by COVID-19.
- Expanding retraining gives the best overall MAE, RMSE and WAPE. The sliding window adapts best in April-May and costs about 7 times less per retrain.
- The static model still follows the shape of the crash because lag features carry the recent demand level, but it is biased upwards at low demand (see `compare_citywide.png`).
- Between mid-March and April 1 no model has been retrained, and all predictions overshoot for about two weeks. A fixed monthly schedule reacts too late, which motivates drift-triggered retraining (Member 3).

## 6. Limitations

- Predictions are one step ahead (15 minutes) and use true past demand as lag inputs. Multi-step forecasting would be less accurate.
- A single test period (Jan-Jun 2020) with one major drift event, so the conclusions about retraining strategy come from one scenario.
- Tuning was a small grid (3 depths) with a fixed learning rate; more tuning might improve all models.
- The periodic schedule is fixed to monthly retraining; weekly retraining was not tested.
- The unknown-zone rows and 0/7+ passenger counts were handled by Member 1 (see dataset documentation).
- Metrics are averaged over all zones, and most zone-intervals are small or zero, so per-zone behaviour is shown separately (`results_top_zones_mae.csv`).

## 7. notes to 3rd member

- `ml/common.py` with fixed `FEATURES`, `train_model()`, `predict()`, `load_model()`.
- `outputs/models/static_xgb.json` and `outputs/best_hyperparams.json`.
- `outputs/predictions/*.parquet`: `interval_start, zone_id, y_true, y_pred, abs_error`.
  The `abs_error` stream of the static model can be used as the drift signal for ADWIN / Page-Hinkley.
- Periodic schedule: monthly retraining, expanding vs 8-week sliding window. Training times are in `outputs/training_log.csv`.
