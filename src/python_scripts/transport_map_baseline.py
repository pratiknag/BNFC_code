#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, November 2026

"""
Bayesian transport map (Katzfuss and Schafer, batram package) as a nonparametric benchmark for
the held-out copula log score at the observed locations of the simulations.

The map is fitted to the normal scores x = Phi^{-1}(u) of the training replicates at the n observed
locations (maximin ordering, up to --max_nn previous nearest neighbors). For a test replicate, the
log density of u on [0, 1]^n is log f_X(x) - sum_j log phi(x_j), which is comparable with the
"Log score (observed)" of the copula models in <results_dir>/heldout_loglik.csv. The map is defined
only at the locations with training replicates, so there is no full-grid score and no prediction
at unobserved locations.

Runs on CPU in its own environment (batram needs Python >= 3.10):

    env_batram/bin/python src/python_scripts/transport_map_baseline.py

Output: <out_dir>/heldout_loglik.csv (model "tmap") and <out_dir>/comparison.csv (paired
differences of every model to BNFC (nonstat) at the observed locations).
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from batram.legmods import Data, SimpleTM
from veccs.orderings2 import find_prev_nearest_neighbors, maximin_ordering


def load_fields(path_fmt, indices):
    values, coords = [], None
    for i in indices:
        df = pd.read_csv(path_fmt.format(i))
        values.append(df["y"].values)
        if coords is None:
            coords = np.column_stack((df["x1"].values, df["x2"].values))
    return np.array(values, dtype=np.float64), coords.astype(np.float64)


def normal_scores(u):
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))


def fit_and_score(data_path, args):
    fmt = os.path.join(data_path, "sample_copula-{}.csv")
    u_train, coords = load_fields(fmt, range(args.train_start, args.train_start + args.N))
    u_test, coords_test = load_fields(fmt, range(args.test_start, args.test_start + args.n_test_reps))
    assert np.allclose(coords, coords_test)
    x_train, x_test = normal_scores(u_train), normal_scores(u_test)

    order = maximin_ordering(coords, 10)
    locs = coords[order]
    nn = find_prev_nearest_neighbors(locs, np.arange(locs.shape[0]), max_nn=args.max_nn)
    data = Data.new(torch.as_tensor(locs), torch.as_tensor(x_train[:, order]), torch.as_tensor(nn))

    t0 = time.time()
    tm = SimpleTM(data, theta_init=None, linear=False, smooth=1.5, nug_mult=4.0)
    opt = torch.optim.Adam(tm.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.nsteps)
    res = tm.fit(args.nsteps, args.lr, optimizer=opt, scheduler=sched, batch_size=args.batch_size,
                 silent=False)
    elapsed = time.time() - t0
    losses = np.asarray(res.losses)
    print(f"  fit {elapsed:.0f}s, loss/N {losses[0] / args.N:.3f} -> {losses[-1] / args.N:.3f} "
          f"(change over the last 10% of steps {(losses[-1] - losses[int(0.9 * len(losses))]) / args.N:.4f})")

    log_fx = tm.score(torch.as_tensor(x_test[:, order])).numpy()
    log_u = log_fx - norm.logpdf(x_test).sum(1)
    scen = os.path.basename(os.path.normpath(data_path))
    return pd.DataFrame({"scenario": scen, "set": "observed", "n_sites": x_test.shape[1], "model": "tmap",
                         "replicate": np.arange(args.test_start, args.test_start + args.n_test_reps),
                         "log_score": log_u}), elapsed


def main():
    parser = argparse.ArgumentParser(description="Bayesian transport map benchmark (observed locations)")
    parser.add_argument("--scenarios", nargs="+",
                        default=["simulations_gamma/", "simulations_signchange/", "simulations_mixgamma/"])
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="heldout_loglik.csv of the copula models, for the paired comparison")
    parser.add_argument("--out_dir", type=str, default="results_M1_M2/transport_map/")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--train_start", type=int, default=0,
                        help="first training replicate; the map is fitted to replicates train_start..train_start+N-1")
    parser.add_argument("--test_start", type=int, default=2000)
    parser.add_argument("--n_test_reps", type=int, default=100)
    parser.add_argument("--max_nn", type=int, default=30)
    parser.add_argument("--nsteps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rows, times = [], {}
    for d in args.scenarios:
        print(d)
        r, t = fit_and_score(d, args)
        rows.append(r)
        times[r.scenario.iloc[0]] = t
    tm = pd.concat(rows)
    tm.to_csv(os.path.join(args.out_dir, "heldout_loglik.csv"), index=False)

    ls = pd.read_csv(os.path.join(args.results_dir, "heldout_loglik.csv"))
    ls = pd.concat([ls[ls.set == "observed"], tm])
    out = []
    for scen, d in ls.groupby("scenario"):
        if scen not in times:
            continue
        wide = d.pivot(index="replicate", columns="model", values="log_score")
        for m in wide.columns:
            diff = wide[m] - wide["nonstat"]
            out.append({"scenario": scen, "model": m, "mean": wide[m].mean(),
                        "se": wide[m].std(ddof=1) / math.sqrt(len(wide)),
                        "diff_to_nonstat": diff.mean(), "diff_se": diff.std(ddof=1) / math.sqrt(len(wide)),
                        "fit_seconds": times[scen] if m == "tmap" else np.nan})
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(args.out_dir, "comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
