"""Exploratory data analysis -> outputs/figures/eda_*.png"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

df = load_data()
df["month_p"] = df["interval_start"].dt.to_period("M")

# 1. Monthly total demand (COVID collapse)
monthly = df.groupby("month_p")[TARGET].sum()
ax = (monthly / 1e6).plot(kind="bar", figsize=(11, 4), color="steelblue")
ax.set_ylabel("pickups (millions)"); ax.set_title("Monthly total demand")
plt.tight_layout(); plt.savefig(FIG / "eda_monthly_demand.png", dpi=150); plt.close()

# 2. Hour x weekday heatmap (2019, mean demand per zone-interval)
d19 = df[df["split"] == "train"]
heat = d19.groupby(["weekday", "hour"])[TARGET].mean().unstack()
plt.figure(figsize=(11, 4)); plt.imshow(heat.values, aspect="auto", cmap="YlOrRd")
plt.colorbar(label="mean demand"); plt.yticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
plt.xticks(range(24)); plt.xlabel("hour"); plt.title("Demand by weekday and hour (2019)")
plt.tight_layout(); plt.savefig(FIG / "eda_heatmap_hour_weekday.png", dpi=150); plt.close()

# 3. Top 15 zones
top = df.groupby("zone_id")[TARGET].sum().sort_values(ascending=False).head(15)
ax = top.sort_values().plot(kind="barh", figsize=(7, 5), color="darkorange")
ax.set_title("Top 15 pickup zones (total demand)")
plt.tight_layout(); plt.savefig(FIG / "eda_top_zones.png", dpi=150); plt.close()

# 4. Sparsity: share of zero-demand intervals
zero_share = (df[TARGET] == 0).mean()
per_zone_zero = df.groupby("zone_id")[TARGET].apply(lambda s: (s == 0).mean())
print(f"Overall share of zero-demand rows: {zero_share:.1%}")
plt.figure(figsize=(7, 4)); plt.hist(per_zone_zero, bins=30, color="gray")
plt.xlabel("share of zero intervals in zone"); plt.ylabel("# zones")
plt.title(f"Sparsity (overall {zero_share:.0%} zeros)")
plt.tight_layout(); plt.savefig(FIG / "eda_sparsity.png", dpi=150); plt.close()

# 5. Autocorrelation of citywide demand (justifies lag_1, lag_1d, lag_1w)
city = df.groupby("interval_start")[TARGET].sum()
city = city[city.index < TEST_START]
lags = np.arange(1, 673)
acf = [city.autocorr(int(l)) for l in lags]
plt.figure(figsize=(11, 4)); plt.plot(lags * 15 / 60, acf)
for h, name in [(24, "1 day"), (168, "1 week")]:
    plt.axvline(h, color="red", ls="--", alpha=.5); plt.text(h, 0.9, name, color="red")
plt.xlabel("lag (hours)"); plt.ylabel("autocorrelation"); plt.title("Citywide demand ACF (2019)")
plt.tight_layout(); plt.savefig(FIG / "eda_acf.png", dpi=150); plt.close()

# 6. Feature-target correlations (1M row sample)
s = d19.sample(1_000_000, random_state=0)
corr = s[FEATURES[1:] + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
ax = corr.plot(kind="barh", figsize=(7, 6)); ax.set_title("Correlation with demand")
plt.tight_layout(); plt.savefig(FIG / "eda_feature_corr.png", dpi=150); plt.close()

# 7. Citywide daily demand for 2020 (drift visual)
daily = city.resample("D").sum()
plt.figure(figsize=(11, 4)); daily.plot(); plt.title("Citywide daily demand"); plt.ylabel("pickups")
plt.axvline(pd.Timestamp("2020-03-15"), color="red", ls="--"); plt.text(pd.Timestamp("2020-03-16"), daily.max()*.9, "COVID", color="red")
plt.tight_layout(); plt.savefig(FIG / "eda_daily_demand.png", dpi=150); plt.close()
print("EDA done ->", FIG)
