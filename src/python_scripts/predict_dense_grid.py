#!/usr/bin/env python
# coding: utf-8

"""
Posterior predictive maps of BNFC on a dense grid, given all locations of the
data grid (the complete-data fit of the application, Train_BNFC.py --vecchia).

For each selected test day, conditions on the copula data at all n locations of
the data grid (sample_copula_full-<i>.csv) and draws the field on a regular
G x G grid over the same domain, with the sampler of Copula_BNFC.py (exact
Gaussian conditional, factor posterior by self-normalised weights, pSGLD weight
samples). Writes
    <out_dir>/dense_predictions.npz   data-scale draws (n_days, R, G*G)
    <out_dir>/dense_timing.json       prediction time per day
    <out_dir>/map_dense_<date>.png    ERA5 field, predictive mean, 2.5% and 97.5%
                                      quantiles
There is no truth on the dense grid, so no scores are computed.
"""

# =========================
# Imports
# =========================
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Copula_BNFC import (load_csv_fields, load_models, load_weight_samples, load_margin,
                         kriging_operators, posterior_predictive, device)
import plot_application as pa
from plot_BNFC_results import COLORBAR_TICKSIZE


def dense_grid(G, meta):
    """G x G grid on [0, 1]^2 (x1 fastest, as grid.csv) with its longitudes and latitudes."""
    x = np.linspace(0.0, 1.0, G)
    x1, x2 = np.meshgrid(x, x)
    (lon0, lon1), (lat0, lat1) = meta["lon_range"], meta["lat_range"]
    grid = pd.DataFrame({"x1": x1.ravel(), "x2": x2.ravel()})
    grid["lon"] = lon0 + (lon1 - lon0) * grid["x1"]
    grid["lat"] = lat0 + (lat1 - lat0) * grid["x2"]
    return grid


def plot_dense(y_obs, grid_obs, draws, grid_dense, meta, date, out_dir):
    """2 x 2 maps on one colour scale: ERA5 field | predictive mean / 2.5% | 97.5% quantile."""
    mean = draws.mean(0)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    norm = pa.value_norm(y_obs, mean, lo, hi)
    n_side = int(round(np.sqrt(len(grid_dense))))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.8), constrained_layout=True,
                             subplot_kw={"projection": pa.PROJ})
    pa.map_field(axes[0, 0], y_obs, grid_obs, meta, pa.CMAP_VALUE, norm,
                 "ERA5 field (conditioning data)", True, False)
    pa.map_field(axes[0, 1], mean, grid_dense, meta, pa.CMAP_VALUE, norm,
                 f"Predictive mean ({n_side}$\\times${n_side})", False, False)
    pa.map_field(axes[1, 0], lo, grid_dense, meta, pa.CMAP_VALUE, norm,
                 "2.5% quantile", True, True)
    im = pa.map_field(axes[1, 1], hi, grid_dense, meta, pa.CMAP_VALUE, norm,
                      "97.5% quantile", False, True)
    cb = fig.colorbar(im, ax=axes[1, :], location="bottom", shrink=0.7, pad=0.02, format="%g",
                      ticks=pa.sqrt_ticks(norm), label=f"Precipitation ({pa.UNITS})")
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)
    path = os.path.join(out_dir, f"map_dense_{date}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="BNFC predictions on a dense grid")
    parser.add_argument("--data_path", type=str, default="application_precipitation_full/")
    parser.add_argument("--factor", choices=["stat", "nonstat"], default="nonstat")
    parser.add_argument("--out_dir", type=str, default="results_application_full/dense/")
    parser.add_argument("--dates", nargs="+", default=["2024-07-02", "2024-09-14"])
    parser.add_argument("--G", type=int, default=80, help="dense grid is G x G")
    parser.add_argument("--R", type=int, default=100, help="predictive draws per day")
    parser.add_argument("--L", type=int, default=200, help="factor draws per predictive draw")
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    grid_obs, dates, meta = pa.load_geometry(args.data_path)
    index_of = {d: int(i) for i, d in dates.items()}
    days = [index_of[d] for d in args.dates]
    u_obs, coords_obs = load_csv_fields(
        os.path.join(args.data_path, "sample_copula_full-{}.csv"), days)
    y_obs, _ = load_csv_fields(os.path.join(args.data_path, "sample_full-{}.csv"), days)
    assert np.allclose(coords_obs, grid_obs[["x1", "x2"]].values, atol=1e-5)
    grid_dense = dense_grid(args.G, meta)

    neural_sampler, _, loc, log_scale, cov_params = load_models(args.data_path, args.factor)
    nets = load_weight_samples(args.data_path, args.factor, neural_sampler)
    if args.R % len(nets):
        raise SystemExit(f"R={args.R} is not divisible by {len(nets)} weight samples")
    ct = torch.tensor(coords_obs, device=device)
    cs = torch.tensor(grid_dense[["x1", "x2"]].values, dtype=torch.float32, device=device)
    F_inv = load_margin(args.data_path, "nonstat", cs)
    print(f"conditioning on {ct.shape[0]} locations, predicting {cs.shape[0]} locations, "
          f"{len(nets)} weight sample(s)")

    torch.cuda.synchronize()
    t0 = time.time()
    kriging = kriging_operators(ct, cs, cov_params=cov_params)
    torch.cuda.synchronize()
    t_kriging = time.time() - t0

    Y, t_days = [], []
    for k, date in enumerate(args.dates):
        t0 = time.time()
        u_b = torch.tensor(u_obs[k:k + 1], device=device)
        draws = [posterior_predictive(u_b, ct, cs, loc, log_scale, net, F_inv, args.factor,
                                      kriging, R=args.R // len(nets), L=args.L)[2]
                 for net in nets]
        y = torch.cat(draws, 1)[0].cpu().numpy()
        torch.cuda.synchronize()
        t_days.append(time.time() - t0)
        y = np.maximum(y, args.y_min)
        Y.append(y)
        print(f"{date}: {t_days[-1]:.1f} s")
        print("wrote", plot_dense(y_obs[k], grid_obs, y, grid_dense, meta, date, args.out_dir))

    np.savez_compressed(os.path.join(args.out_dir, "dense_predictions.npz"),
                        data_samples=np.array(Y, dtype=np.float32), y_obs=y_obs,
                        coords_obs=coords_obs, coords_dense=grid_dense[["x1", "x2"]].values,
                        days=np.array(days), dates=np.array(args.dates))
    timing = {"n_cond": int(ct.shape[0]), "n_pred": int(cs.shape[0]), "R": args.R, "L": args.L,
              "weight_samples": len(nets), "kriging_seconds": t_kriging,
              "seconds_per_day": dict(zip(args.dates, t_days)),
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    with open(os.path.join(args.out_dir, "dense_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
