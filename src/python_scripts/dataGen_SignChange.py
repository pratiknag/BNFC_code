#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
Simulation scenario (ii) of the manuscript: a location-dependent factor whose
effect changes sign over space, as for the factor networks of Figure S2 of the supplement,

    W(s) = Z(s) + g(U, s),   Z ~ GP(0, SE correlation, range 0.1),
    U ~ U(0, 1),
    g(u, s) = c b(s) H^{-1}(u),   b(s) = tanh{(s2 - 0.5) / 0.15},

with H the right-skewed Gamma(shape 2, scale 1) distribution and c = 1.5, on a
30 x 30 grid on [0, 1]^2. In the north (b > 0), the factor raises the field and
is right skewed (stronger upper-tail dependence); in the south (b < 0), it
lowers the field and is left skewed (stronger lower-tail dependence); the two
halves are negatively dependent, and near s2 = 0.5 the factor vanishes. A
common factor, or any Gaussian scale mixture, adds positive dependence to every
pair of sites and cannot represent this. Each replicate is ONE draw of (U, Z).

The copula data are U(s) = F_1(W(s); s), with F_1(w; s) = int_0^1 Phi{w - g(u, s)} du
evaluated by midpoint quadrature over u. The data scale is Y(s) = W(s).

Files written to <out_dir>/ for replicate i (same layout as dataGen_Gamma.py):
    sample_copula-{i}.csv, sample-{i}.csv                (all replicates)
    sample_copula_full-{i}.csv, sample_full-{i}.csv      (test replicates)
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gamma as sp_gamma, kstest, norm
from tqdm import tqdm

GAMMA_SHAPE = 2.0
AMPLITUDE = 1.5
BOUNDARY_WIDTH = 0.15


def se_correlation(coords, lengthscale=0.1, nugget=1e-4):
    """Squared exponential correlation + nugget, as build_covariance in Train_BNFC."""
    sqdist = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-0.5 * sqdist / lengthscale ** 2) + nugget * np.eye(len(coords))


def F1(w, g_nodes):
    """F_1(w_j; s_j) = mean_k Phi(w_j - g(v_k, s_j)). w: (n,), g_nodes: (K, n)."""
    return norm.cdf(w[None, :] - g_nodes).mean(0)


def write_csv(path, coords, values):
    pd.DataFrame({"x1": coords[:, 0], "x2": coords[:, 1], "y": values}).to_csv(path, index=False)


def loading(coords):
    """b(s) = tanh{(s2 - 0.5) / 0.15}: positive in the north, negative in the south."""
    return np.tanh((coords[:, 1] - 0.5) / BOUNDARY_WIDTH)


def target_function(u, coords):
    """g(u, s). u: (...,) array in (0, 1), coords: (n, 2). Returns (..., n)."""
    q = sp_gamma.ppf(np.asarray(u), a=GAMMA_SHAPE)[..., None]
    return AMPLITUDE * loading(coords) * q


def main():
    parser = argparse.ArgumentParser(description="Sign-change factor copula data")
    parser.add_argument("--out_dir", type=str, default="simulations_signchange/")
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--N_train", type=int, default=2000)
    parser.add_argument("--N_test", type=int, default=100)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=2000,
                        help="quadrature nodes for F_1")
    parser.add_argument("--seed", type=int, default=2026)
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
    g_nodes = target_function(u_nodes, coords)                      # (K, n_grid)

    u_pool, W_all = [], []
    for i in tqdm(range(args.N_train + args.N_test)):
        u = rng.uniform()
        W = target_function(u, coords) + L @ rng.standard_normal(len(coords))
        U = F1(W, g_nodes)

        write_csv(os.path.join(args.out_dir, f"sample_copula-{i}.csv"), coords[idx], U[idx])
        write_csv(os.path.join(args.out_dir, f"sample-{i}.csv"), coords[idx], W[idx])
        if i >= args.N_train:
            write_csv(os.path.join(args.out_dir, f"sample_copula_full-{i}.csv"), coords, U)
            write_csv(os.path.join(args.out_dir, f"sample_full-{i}.csv"), coords, W)
        u_pool.append(U[idx[0]])
        W_all.append(W[idx])

    # U at a fixed site across replicates should be U(0, 1)
    ks = kstest(np.array(u_pool), "uniform")
    print(f"KS test of U(s_1) across replicates against U(0,1): "
          f"stat={ks.statistic:.4f}, p={ks.pvalue:.3f}")
    # correlation between the most northern and most southern observed sites
    c = coords[idx]
    north, south = np.argmax(c[:, 1]), np.argmin(c[:, 1])
    W_all = np.array(W_all)
    print(f"cor(W) between northernmost and southernmost observed sites: "
          f"{np.corrcoef(W_all[:, north], W_all[:, south])[0, 1]:.3f}")


if __name__ == "__main__":
    main()
