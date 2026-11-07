"""Compare all strategies -> results tables (csv) + charts (png)."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

names = ["naive_lag1w", "linear", "static", "periodic_expanding", "periodic_sliding"]
preds = {n: pd.read_parquet(PRED / f"{n}.parquet") for n in names}

# overall
overall = pd.DataFrame({n: metrics(p.y_true, p.y_pred) for n, p in preds.items()}).T.round(4)
overall.to_csv(OUT / "results_overall.csv"); print(overall, "\n")

# per month
rows = []
for n, p in preds.items():
    p = p.assign(month=p["interval_start"].dt.to_period("M").astype(str))
    for m, g in p.groupby("month"):
        rows.append(dict(strategy=n, month=m, **metrics(g.y_true, g.y_pred)))
monthly = pd.DataFrame(rows)
monthly.to_csv(OUT / "results_monthly.csv", index=False)
print(monthly.pivot(index="month", columns="strategy", values="MAE").round(3), "\n")

# charts: MAE / RMSE / WAPE per month
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for ax, met in zip(axes, ["MAE", "RMSE", "WAPE"]):
    pv = monthly.pivot(index="month", columns="strategy", values=met)[names]
    pv.plot(ax=ax, marker="o"); ax.set_title(f"{met} per month"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "compare_monthly.png", dpi=150); plt.close()

# per-zone error for top 10 zones (by 2020 demand), static vs periodic
ref = preds["static"]
top = ref.groupby("zone_id")["y_true"].sum().nlargest(10).index
zr = pd.DataFrame({n: preds[n][preds[n].zone_id.isin(top)].groupby("zone_id")["abs_error"].mean()
                   for n in ["naive_lag1w", "static", "periodic_expanding", "periodic_sliding"]})
zr.to_csv(OUT / "results_top_zones_mae.csv")
zr.plot(kind="bar", figsize=(11, 4.5), title="MAE in top-10 zones (2020)"); plt.ylabel("MAE")
plt.tight_layout(); plt.savefig(FIG / "compare_top_zones.png", dpi=150); plt.close()

# training cost
log = pd.read_csv(OUT / "training_log.csv")
cost = log.groupby("strategy").agg(retrains=("seconds", "size"), total_sec=("seconds", "sum"),
                                   mean_sec=("seconds", "mean")).round(1)
cost.to_csv(OUT / "results_training_cost.csv"); print(cost)

# citywide actual vs predicted (daily) - shows how static degrades
fig, ax = plt.subplots(figsize=(11, 4.5))
for n in ["static", "periodic_expanding", "periodic_sliding"]:
    p = preds[n].set_index("interval_start")["y_pred"].groupby(pd.Grouper(freq="D")).sum()
    p.plot(ax=ax, label=n)
preds["static"].set_index("interval_start")["y_true"].groupby(pd.Grouper(freq="D")).sum().plot(ax=ax, label="actual", color="black", lw=2)
ax.legend(); ax.set_title("Citywide daily demand: actual vs predicted (2020)")
plt.tight_layout(); plt.savefig(FIG / "compare_citywide.png", dpi=150); plt.close()
print("Saved tables + figures in", OUT)
