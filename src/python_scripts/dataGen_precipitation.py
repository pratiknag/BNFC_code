#!/usr/bin/env python
# coding: utf-8

"""
Application data: ERA5 daily precipitation over central Europe, in the same
file layout as the simulation scenarios (dataGen_Gamma.py, dataGen_SignChange.py).

The ERA5 fields are prepared as in the precipitation application of the
FNO-DSTM-EnKF project (precipitation_reanalysis_data_preprocessing.py):
hourly total precipitation over 15E-25E, 45N-55N (0.25 deg) is aggregated to
daily values and linearly interpolated onto a regular S x S grid. Here the
daily values are kept on the data scale (daily total, mm/day) and are not
standardised or smoothed, since the copula and the marginal quantile network
take care of the margins. Each replicate is ONE day.

Data scale:   Y(s, t) = daily precipitation (mm/day), optionally averaged over
              the last --smooth_days days (2 = the 2-day moving average of
              the FNO-DSTM-EnKF preprocessing), which reduces the dry days.
Copula scale: U(s, t) = F_hat_s{Y(s, t)}, the randomised empirical PIT of the
              training days at grid cell s,
                  U = (#{train < y} + V (#{train = y} + 1)) / (N_train + 1),
              V ~ U(0, 1). The randomisation breaks the ties of dry days
              (exact zeros), so that U is continuous and lies in (0, 1).
              With --tie_noise shared, one V per day is used at all sites,
              so that neighbouring dry cells get comonotone, not
              independent, copula values.

Coordinates are rescaled to [0, 1]^2, x1 = (lon - 15) / 10, x2 = (lat - 45) / 10,
so that the Gaussian component (SE correlation, range 0.1) is the one used in
the simulations.

Training days: 1 Jan 1994 - 31 Dec 2022. Test days: the last N_test days of
the record (by default 15 Jun 2024 - 31 Dec 2024, the test period of the
FNO-DSTM-EnKF application). The days in between are not used.

Files written to <out_dir>/ for replicate i (same layout as the simulations):
    sample_copula-{i}.csv, sample-{i}.csv                (all replicates)
    sample_copula_full-{i}.csv, sample_full-{i}.csv      (test replicates)
plus
    obs_index.txt   observed grid cells (indices into the full grid)
    grid.csv        full grid: x1, x2, lon, lat
    dates.csv       replicate index -> date, split
    meta.json       N_train, N_test, test_start and settings
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kstest
from tqdm import tqdm

LON_MIN, LON_MAX = 15.0, 25.0
LAT_MIN, LAT_MAX = 45.0, 55.0


def load_daily_precipitation(era5_dir, years, S, smooth_days=1):
    """Daily precipitation (T, S, S) in mm/day on an ascending lat x lon grid."""
    lat = np.linspace(LAT_MIN, LAT_MAX, S)
    lon = np.linspace(LON_MIN, LON_MAX, S)
    fields, dates = [], []
    for year in tqdm(years, desc="ERA5"):
        ds = xr.open_dataset(
            os.path.join(era5_dir, f"era5_precipitation_{year}",
                         "data_stream-oper_stepType-accum.nc"),
            engine="netcdf4",
        )
        tp = ds["tp"] * 1000.0                                   # m -> mm per hour
        daily = tp.resample(valid_time="1D").mean() * 24.0       # mm/day
        daily = daily.interp(latitude=lat, longitude=lon, method="linear")
        daily = daily.transpose("valid_time", "latitude", "longitude")
        fields.append(np.clip(daily.values, 0.0, None))
        dates.append(pd.DatetimeIndex(daily.valid_time.values))
        ds.close()
    Y = np.concatenate(fields)
    dates = dates[0].append(dates[1:])
    if smooth_days > 1:
        # moving average over the last smooth_days days (across year boundaries);
        # the first days, without a complete window, are dropped
        c = np.cumsum(np.concatenate((np.zeros((1,) + Y.shape[1:]), Y)), axis=0)
        Y = (c[smooth_days:] - c[:-smooth_days]) / smooth_days
        dates = dates[smooth_days - 1:]
    return Y, dates, lon, lat


def randomised_pit(train, values, rng, shared=False):
    """
    train: (N, n) training values, values: (T, n). Returns U (T, n) in (0, 1).
    shared: one tie-breaking draw per day (row) for all sites.
    """
    N, n = train.shape
    sorted_train = np.sort(train, axis=0)
    U = np.empty_like(values, dtype=float)
    V_day = rng.uniform(size=values.shape[0])
    for j in range(n):
        below = np.searchsorted(sorted_train[:, j], values[:, j], side="left")
        upto = np.searchsorted(sorted_train[:, j], values[:, j], side="right")
        V = V_day if shared else rng.uniform(size=values.shape[0])
        U[:, j] = (below + V * (upto - below + 1)) / (N + 1)
    return U


def write_csv(path, coords, values):
    pd.DataFrame({"x1": coords[:, 0], "x2": coords[:, 1], "y": values}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="ERA5 precipitation copula data")
    parser.add_argument("--era5_dir", type=str,
                        default="../FNO-DSTM-EnKF/datasets/precip_reanalysis_datasets",
                        help="folder with era5_precipitation_<year>/ subfolders")
    parser.add_argument("--out_dir", type=str, default="application_precipitation/")
    parser.add_argument("--first_year", type=int, default=1994)
    parser.add_argument("--last_year", type=int, default=2024)
    parser.add_argument("--train_end", type=str, default="2022-12-31")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--N_test", type=int, default=200)
    parser.add_argument("--smooth_days", type=int, default=1,
                        help="moving-average window in days (1 = daily values)")
    parser.add_argument("--tie_noise", choices=["independent", "shared"], default="independent",
                        help="tie-breaking draws per site and day, or one per day")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train_all_sites", action="store_true",
                        help="write the training days at all grid locations (complete data, for "
                             "fitting with the Vecchia approximation); the observed locations, the "
                             "probability integral transform and the test days are unchanged")
    args = parser.parse_args()
    print(args)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    years = range(args.first_year, args.last_year + 1)
    Y, dates, lon, lat = load_daily_precipitation(args.era5_dir, years, args.grid,
                                                  args.smooth_days)
    Y = Y.reshape(len(Y), -1)                                    # (T, S*S), lon fastest
    print(f"daily fields: {Y.shape}, {dates[0].date()} - {dates[-1].date()}")

    train = np.where(dates <= pd.Timestamp(args.train_end))[0]
    test = np.arange(len(dates) - args.N_test, len(dates))
    assert train[-1] < test[0], "test days overlap the training period"

    # full grid in the simulation ordering: x1 varies fastest
    lon_g, lat_g = np.meshgrid(lon, lat)
    lon_g, lat_g = lon_g.ravel(), lat_g.ravel()
    coords = np.column_stack(((lon_g - LON_MIN) / (LON_MAX - LON_MIN),
                              (lat_g - LAT_MIN) / (LAT_MAX - LAT_MIN)))
    pd.DataFrame({"x1": coords[:, 0], "x2": coords[:, 1], "lon": lon_g, "lat": lat_g}) \
        .to_csv(os.path.join(args.out_dir, "grid.csv"), index=False)

    idx = np.sort(rng.choice(len(coords), args.n_obs, replace=False))
    np.savetxt(os.path.join(args.out_dir, "obs_index.txt"), idx, fmt="%d")

    days = np.concatenate((train, test))
    U = randomised_pit(Y[train], Y[days], rng, shared=args.tie_noise == "shared")
    Yd = Y[days]
    N_train = len(train)

    for i in tqdm(range(len(days)), desc="write"):
        # --train_all_sites: the training days at every grid location (the complete
        # data), the test days at the same n_obs locations as without the flag
        sites = np.arange(len(coords)) if (args.train_all_sites and i < N_train) else idx
        write_csv(os.path.join(args.out_dir, f"sample_copula-{i}.csv"), coords[sites], U[i, sites])
        write_csv(os.path.join(args.out_dir, f"sample-{i}.csv"), coords[sites], Yd[i, sites])
        if i >= N_train:
            write_csv(os.path.join(args.out_dir, f"sample_copula_full-{i}.csv"), coords, U[i])
            write_csv(os.path.join(args.out_dir, f"sample_full-{i}.csv"), coords, Yd[i])

    pd.DataFrame({
        "index": np.arange(len(days)),
        "date": dates[days].strftime("%Y-%m-%d"),
        "split": ["train"] * N_train + ["test"] * len(test),
    }).to_csv(os.path.join(args.out_dir, "dates.csv"), index=False)

    meta = {
        "N_train": int(N_train), "N_test": int(len(test)), "test_start": int(N_train),
        "n_obs": args.n_obs, "grid": args.grid, "train_all_sites": bool(args.train_all_sites),
        "train_period": [str(dates[train[0]].date()), str(dates[train[-1]].date())],
        "test_period": [str(dates[test[0]].date()), str(dates[test[-1]].date())],
        "lon_range": [LON_MIN, LON_MAX], "lat_range": [LAT_MIN, LAT_MAX],
        "units": "mm/day", "dry_fraction_train": float((Y[train] == 0).mean()),
        "config": vars(args),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2))

    # U at a fixed site over the training days should be U(0, 1)
    ks = kstest(U[:N_train, idx[0]], "uniform")
    print(f"KS test of U(s_1) over training days against U(0,1): "
          f"stat={ks.statistic:.4f}, p={ks.pvalue:.3f}")


if __name__ == "__main__":
    main()
