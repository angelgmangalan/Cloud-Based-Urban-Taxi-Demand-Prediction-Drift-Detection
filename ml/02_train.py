"""Baselines, tuning, STATIC model, PERIODIC models (expanding + sliding).
Outputs: outputs/models/*.json, outputs/predictions/*.parquet, outputs/training_log.csv"""
import json, time
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

# ---- periodic retraining settings ----
RETRAIN_OFFSET = pd.DateOffset(months=1)   # use pd.DateOffset(weeks=1) for weekly
SLIDING_WEEKS = 8                          # window for the sliding strategy

df = load_data()
ts = df["interval_start"].values
train_end = np.searchsorted(ts, TEST_START.to_datetime64())
val_start = np.searchsorted(ts, VAL_START.to_datetime64())
train, test = df.iloc[:train_end], df.iloc[train_end:]
assert (df["split"].iloc[:train_end] == "train").all() and (test["split"] == "test").all()
print(f"train {len(train):,} rows | test {len(test):,} rows")

log = []


def save_preds(name, sub, p):
    out = pd.DataFrame({"interval_start": sub["interval_start"].values,
                        "zone_id": sub["zone_id"].values,
                        "y_true": sub[TARGET].values, "y_pred": np.asarray(p, dtype="float32")})
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    out.to_parquet(PRED / f"{name}.parquet", index=False)
    print(f"{name:22s}", {k: round(v, 3) for k, v in metrics(out.y_true, out.y_pred).items()})


# ---------- 1. Naive baseline: same time last week ----------
save_preds("naive_lag1w", test, test["lag_1w"].values)

# ---------- 2. Linear regression baseline (no zone_id) ----------
lin_feats = [f for f in FEATURES if f != "zone_id"]
t0 = time.time()
lin = LinearRegression().fit(train[lin_feats], train[TARGET])
log.append(dict(strategy="linear", retrain_start=str(TEST_START.date()), train_rows=len(train), seconds=time.time() - t0))
save_preds("linear", test, np.clip(lin.predict(test[lin_feats]), 0, None))

# ---------- 3. Tuning on Nov-Dec 2019 (test set never touched) ----------
tr_a, val = df.iloc[:val_start], df.iloc[val_start:train_end]
Xa, Xv = make_X(tr_a), make_X(val)
best = None
for depth in [6, 8, 10]:
    m = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, **xgb_params(depth))
    m.fit(Xa, tr_a[TARGET], eval_set=[(Xv, val[TARGET])], verbose=False)
    rmse = m.best_score
    print(f"depth={depth} best_iter={m.best_iteration} val_rmse={rmse:.4f}")
    if best is None or rmse < best["val_rmse"]:
        best = dict(max_depth=depth, n_estimators=int(m.best_iteration * 1.1) + 1, val_rmse=float(rmse))
json.dump(best, open(OUT / "best_hyperparams.json", "w"), indent=2)
print("best:", best)
del Xa, Xv

# ---------- 4. STATIC model: train once on all 2019, freeze ----------
static, sec = train_model(train, best["n_estimators"], best["max_depth"])
static.save_model(str(MODELS / "static_xgb.json"))
log.append(dict(strategy="static", retrain_start=str(TEST_START.date()), train_rows=len(train), seconds=sec))
save_preds("static", test, predict(static, test))

# feature importance plot (feature-selection evidence)
imp = pd.Series(static.get_booster().get_score(importance_type="gain")).sort_values()
imp.plot(kind="barh", figsize=(7, 6), title="XGBoost feature importance (gain)")
plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).to_csv(OUT / "feature_importance.csv")

# ---------- 5. PERIODIC retraining ----------
def run_periodic(name, window_weeks=None):
    end_ts = df["interval_start"].max() + pd.Timedelta(minutes=15)
    starts, s = [], TEST_START
    while s < end_ts:
        starts.append(s); s = s + RETRAIN_OFFSET
    bounds = starts + [end_ts]
    parts = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        hi = np.searchsorted(ts, s.to_datetime64())
        lo = 0 if window_weeks is None else np.searchsorted(ts, (s - pd.Timedelta(weeks=window_weeks)).to_datetime64())
        chunk_hi = np.searchsorted(ts, e.to_datetime64())
        tr, te = df.iloc[lo:hi], df.iloc[hi:chunk_hi]      # only data BEFORE s is used for training
        model, sec = train_model(tr, best["n_estimators"], best["max_depth"])
        log.append(dict(strategy=name, retrain_start=str(s.date()), train_rows=len(tr), seconds=sec))
        parts.append(pd.DataFrame({"interval_start": te["interval_start"].values, "zone_id": te["zone_id"].values,
                                   "y_true": te[TARGET].values, "y_pred": predict(model, te)}))
        print(f"  {name} retrain @ {s.date()}  rows={len(tr):,}  {sec:.0f}s")
        model.save_model(str(MODELS / f"{name}_{s:%Y%m%d}.json"))
    out = pd.concat(parts, ignore_index=True)
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    out.to_parquet(PRED / f"{name}.parquet", index=False)
    print(name, {k: round(v, 3) for k, v in metrics(out.y_true, out.y_pred).items()})

run_periodic("periodic_expanding", None)
run_periodic("periodic_sliding", SLIDING_WEEKS)

pd.DataFrame(log).to_csv(OUT / "training_log.csv", index=False)
print("All done.")
