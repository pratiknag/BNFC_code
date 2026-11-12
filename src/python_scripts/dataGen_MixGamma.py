#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026

"""
Simulation scenario (iii) of the manuscript: data from a location-specific
factor whose family changes over space,

    W(s) = Z(s) + T(U, s),   Z ~ GP(0, SE correlation, range 0.1),   U ~ U(0, 1),
    T(u, s) = {1 - r(s)} H_A^{-1}(u) + r(s) H_B^{-1}(u),
    r(s) = 1 / [1 + exp{-(s1 - 0.5) / 0.1}],

    H_A = 0.6 Gamma(shape 2, scale 1) + 0.4 Gamma(shape 20, scale 0.5)   (west, bimodal)
    H_B = LogNormal(1.2, 0.6^2)                                          (east, right skewed)

on a 30 x 30 grid on [0, 1]^2. Each replicate is ONE draw of (U, Z). T(., s) is
a convex combination of two quantile functions, so it is nondecreasing in u
and T(U, s) is a location-specific factor in the sense of Lemma 1, but it is
not produced by a neural network and no single parametric family matches it.

The copula data are U(s) = F_1(W(s); s), with the location-specific marginal
F_1(w; s) = int_0^1 Phi{w - T(u, s)} du evaluated by midpoint quadrature over
u. The data scale is the latent field itself, Y(s) = W(s).

Files written to <out_dir>/ for replicate i (same layout as dataGen_Gamma.py):
    sample_copula-{i}.csv, sample-{i}.csv                (all replicates)
    sample_copula_full-{i}.csv, sample_full-{i}.csv      (test replicates)
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gamma, kstest, lognorm, norm
from tqdm import tqdm

MIX_WEIGHT = 0.6
MIX_SHAPES = (2.0, 20.0)
MIX_SCALES = (1.0, 0.5)
LN_MU, LN_SIGMA = 1.2, 0.6


def se_correlation(coords, lengthscale=0.1, nugget=1e-4):
    """Squared exponential correlation + nugget, as build_covariance in Train_BNFC."""
    sqdist = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-0.5 * sqdist / lengthscale ** 2) + nugget * np.eye(len(coords))


def mixture_quantile(u, x_max=40.0, n_grid=200001):
    """H_A^{-1}(u) for the two-component gamma mixture, by inverting its cdf on a fine grid."""
    x = np.linspace(0.0, x_max, n_grid)
    cdf = (MIX_WEIGHT * gamma.cdf(x, MIX_SHAPES[0], scale=MIX_SCALES[0])
           + (1 - MIX_WEIGHT) * gamma.cdf(x, MIX_SHAPES[1], scale=MIX_SCALES[1]))
    return np.interp(u, cdf, x)


def lognormal_quantile(u):
    return lognorm.ppf(u, s=LN_SIGMA, scale=np.exp(LN_MU))


def region_weight(coords):
    """r(s): 0 in the west, 1 in the east, smooth transition around s1 = 0.5."""
    return 1.0 / (1.0 + np.exp(-(coords[:, 0] - 0.5) / 0.1))


def target_function(u, coords):
    """T(u, s). u: (...,) array, coords: (n, 2). Returns (..., n)."""
    u = np.asarray(u)[..., None]
    r = region_weight(coords)
    return (1 - r) * mixture_quantile(u) + r * lognormal_quantile(u)


def F1(w, t_nodes):
    """F_1(w_j; s_j) = mean_k Phi(w_j - T(u_k, s_j)). w: (n,), t_nodes: (K, n)."""
    return norm.cdf(w[None, :] - t_nodes).mean(0)


def write_csv(path, coords, values):
    pd.DataFrame({"x1": coords[:, 0], "x2": coords[:, 1], "y": values}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Regional gamma-mixture / lognormal factor copula data")
    parser.add_argument("--out_dir", type=str, default="simulations_mixgamma/")
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--N_train", type=int, default=2000)
    parser.add_argument("--N_test", type=int, default=100)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=2000,
                        help="quadrature nodes for F_1")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    print(args)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    x = np.linspace(0, 1, args.grid)
    xv, yv = np.meshgrid(x, x)
    coords = np.column_stack((xv.ravel(), yv.ravel()))
    idx = np.sort(rng.choice(len(coords), args.n_obs, replace=False))
    np.savetxt(os.path.join(args.out_dir, "obs_index.txt"), idx, fmt="%d")

    L = np.linalg.cholesky(se_correlation(coords, args.lengthscale))

    # midpoint quadrature nodes of U ~ U(0, 1) for the location-specific margins
    u_nodes = (np.arange(args.K) + 0.5) / args.K
    t_nodes = target_function(u_nodes, coords)                      # (K, n_grid)

    u_pool = []
    for i in tqdm(range(args.N_train + args.N_test)):
        u = rng.uniform()
        W = target_function(u, coords) + L @ rng.standard_normal(len(coords))
        U = F1(W, t_nodes)

        write_csv(os.path.join(args.out_dir, f"sample_copula-{i}.csv"), coords[idx], U[idx])
        write_csv(os.path.join(args.out_dir, f"sample-{i}.csv"), coords[idx], W[idx])
        if i >= args.N_train:
            write_csv(os.path.join(args.out_dir, f"sample_copula_full-{i}.csv"), coords, U)
            write_csv(os.path.join(args.out_dir, f"sample_full-{i}.csv"), coords, W)
        u_pool.append(U[idx[0]])

    # U at a fixed site across replicates should be U(0, 1)
    ks = kstest(np.array(u_pool), "uniform")
    print(f"KS test of U(s_1) across replicates against U(0,1): "
          f"stat={ks.statistic:.4f}, p={ks.pvalue:.3f}")


if __name__ == "__main__":
    main()
