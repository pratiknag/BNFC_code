# Created by Pratik Nag, November 2026 (approximate)
"""Posterior summaries of the ERA5 BNFC (nonstat) fit that need the pSGLD weight samples
(Section 5.3 of the manuscript); writes results_application/figures/bayes_insights.json."""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src/python_scripts"))
os.chdir(ROOT)
from plot_application import factor_maps, load_geometry

data_path = "application_precipitation_v2/"
grid, dates, meta = load_geometry(data_path)
share, skew, cov, gc = factor_maps(data_path, grid[["x1", "x2"]].values.astype(np.float32), 2000)
S = share.shape[0]
out = {"weight_samples": S}

# skewness: posterior probability of a negative sign at each grid location
p_neg = (skew < 0).mean(0)
out["skew_sign_determined_frac"] = float(np.mean((p_neg >= 0.95) | (p_neg <= 0.05)))
out["skew_neg_determined_frac"] = float(np.mean(p_neg >= 0.95))
out["skew_pos_determined_frac"] = float(np.mean(p_neg <= 0.05))
out["skew_signchange_in_all_samples"] = float(np.mean((skew.min(1) < 0) & (skew.max(1) > 0)))
lo, hi = np.quantile(skew, [0.025, 0.975], axis=0)
out["skew_band_width_mean"] = float((hi - lo).mean())
# factor share: 95% band over weight samples
lo, hi = np.quantile(share, [0.025, 0.975], axis=0)
out["share_band_width_mean"] = float((hi - lo).mean())
out["share_band_width_max"] = float((hi - lo).max())
# lowland vs orographic contrast: probability that the share at the location with the
# largest posterior mean share exceeds that at the location with the smallest
m = share.mean(0)
out["P_share_max_loc_gt_min_loc"] = float(np.mean(share[:, m.argmax()] > share[:, m.argmin()]))

# factor posterior over the test days: law of total variance, weight samples x draws
p = np.load("results_application/application_precipitation_v2/nonstat/predictions.npz")
F = p["factor_samples"].astype(np.float64)                  # (days, R, n)
D, R, n = F.shape
r = R // S
for name, X in (("domain_mean", F.mean(-1, keepdims=True)), ("location", F)):
    X = X.reshape(D, S, r, X.shape[-1])
    within = X.var(2, ddof=1).mean(1)                       # E_w var(g | data, w)
    between = X.mean(2).var(1, ddof=1) - within / r         # var_w E(g | data, w), unbiased
    between = np.clip(between, 0, None)
    frac = between / (between + within)
    out[f"{name}_between_weight_frac_mean"] = float(frac.mean())
    out[f"{name}_between_weight_frac_median"] = float(np.median(frac))
    out[f"{name}_between_weight_frac_q90"] = float(np.quantile(frac, 0.9))

# 14 September 2024 and the probability of a positive domain-wide state on each test day
ts = pd.read_csv("results_application/figures/inference_factor_timeseries.csv")
f_dom = F.mean(-1)                                          # (days, R)
p_pos = (f_dom > 0).mean(1)
out["frac_days_P_pos_gt_0.95_or_lt_0.05"] = float(np.mean((p_pos > 0.95) | (p_pos < 0.05)))
for d in ("2024-09-12", "2024-09-14", "2024-09-15"):
    i = int(np.where(ts.date.values == d)[0][0])
    out[f"P_pos_{d}"] = float(p_pos[i])
print(json.dumps(out, indent=1))
json.dump(out, open("results_application/figures/bayes_insights.json", "w"), indent=1)
