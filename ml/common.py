"""Shared config + helpers for Member 2 (ML). Also the hand-off API for Member 3."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

# ---------------- CONFIG: edit these ----------------
DATA_DIR = Path("../data/ml_ready_15min")     # repo-root data/ folder (not committed)
OUT = Path("outputs")
MODELS, PRED, FIG = OUT / "models", OUT / "predictions", OUT / "figures"
for d in (MODELS, PRED, FIG):
    d.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2020-01-01")       # must match Member 1's TEST_START
VAL_START = pd.Timestamp("2019-11-01")        # validation = Nov-Dec 2019 (for tuning only)
USE_GPU = False
# -----------------------------------------------------

# Fixed feature schema (order matters!) - give this to Member 3.
FEATURES = ["zone_id", "hour", "slot_of_day", "weekday", "is_weekend", "month", "day",
            "lag_1", "lag_2", "lag_4", "lag_1d", "lag_1w", "roll_mean_4", "roll_mean_1d",
            "slot_sin", "slot_cos", "weekday_sin", "weekday_cos"]
TARGET = "label"
ZONE_CATS = list(range(1, 264))               # valid TLC zone IDs


def load_data() -> pd.DataFrame:
    """Load ml_ready_15min (train+test), downcast to save RAM, sort by time."""
    df = pd.read_parquet(DATA_DIR)
    df["interval_start"] = pd.to_datetime(df["interval_start"])
    df["split"] = df["split"].astype(str)
    for c in FEATURES + [TARGET]:
        df[c] = df[c].astype("int16" if c == "zone_id" else "float32")
    df = df.sort_values(["interval_start", "zone_id"]).reset_index(drop=True)
    return df


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix with fixed column order; zone_id as categorical."""
    X = df[FEATURES].copy()
    X["zone_id"] = pd.Categorical(X["zone_id"], categories=ZONE_CATS)
    return X


def metrics(y, p) -> dict:
    y, p = np.asarray(y, dtype="float64"), np.asarray(p, dtype="float64")
    err = p - y
    return {"MAE": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "WAPE": np.abs(err).sum() / max(y.sum(), 1e-9)}


def xgb_params(max_depth=8, lr=0.1):
    return dict(objective="reg:squarederror", tree_method="hist",
                device="cuda" if USE_GPU else "cpu",
                max_depth=max_depth, learning_rate=lr, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, enable_categorical=True,
                n_jobs=-1, random_state=42)


def train_model(df: pd.DataFrame, n_estimators: int, max_depth=8, lr=0.1):
    """Hand-off function for Member 3: train an XGBoost model on a dataframe
    that has FEATURES + 'label'. Returns (model, seconds)."""
    import xgboost as xgb
    t0 = time.time()
    m = xgb.XGBRegressor(n_estimators=n_estimators, **xgb_params(max_depth, lr))
    m.fit(make_X(df), df[TARGET])
    return m, time.time() - t0


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Hand-off function for Member 3: demand prediction, clipped at 0."""
    return np.clip(model.predict(make_X(df)), 0, None)


def load_model(path):
    import xgboost as xgb
    m = xgb.XGBRegressor()
    m.load_model(str(path))
    return m
