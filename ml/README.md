# Member 2 - ML & Taxi Demand Prediction

## Task definition (share with Member 3)
Predict `label` (pickups) for one zone in one 15-min interval, from 18 features
(see `FEATURES` in `common.py`; lag features only use the past). Metrics: MAE, RMSE (+ WAPE).

## Run
pip install -r requirements.txt
# put Member 1 ml_ready_15min/ into <repo root>/data/ (run these scripts from inside ml/)
python 01_eda.py        # figures in outputs/figures/eda_*.png
python 02_train.py      # baselines, tuning, static + periodic models
python 03_evaluate.py   # tables + comparison charts

## Hand-off to Member 3
- `common.py`: `FEATURES` (fixed order), `train_model(df, n_estimators, max_depth)`, `predict(model, df)`, `load_model(path)`
- `outputs/models/static_xgb.json`, `outputs/best_hyperparams.json`
- `outputs/predictions/*.parquet`: interval_start, zone_id, y_true, y_pred, abs_error
- Periodic schedule: monthly retrain, expanding vs 8-week sliding window
