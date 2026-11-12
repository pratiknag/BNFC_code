#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
Simulation scenario (i) of the manuscript: gamma-factor data from the common
factor copula model of Krupskii et al. (2018),

    W(s) = Z(s) + V0,   Z ~ GP(0, SE correlation, range 0.1),
    V0 ~ Gamma(shape 6, scale 3),

on a 30 x 30 grid on [0, 1]^2. Each replicate is ONE draw of (V0, Z): the
integral over the factor in F_n(w) = int Phi_Sigma(w - v 1) dF_V0(v) is the
distribution of a single draw, not an average of draws.

The copula data are U(s) = F_1(W(s)), with the marginal distribution function
F_1(w) = int Phi(w - v) dF_V0(v) evaluated by quantile quadrature. The data
scale is the latent field itself, Y(s) = W(s), so G = F_1.

Files written to <out_dir>/ for replicate i:
    sample_copula-{i}.csv       U at the n observed sites   (all replicates)
    sample-{i}.csv              Y at the n observed sites   (all replicates)
    sample_copula_full-{i}.csv  U at all grid sites         (test replicates)
    sample_full-{i}.csv         Y at all grid sites         (test replicates)
Replicates 0..N_train-1 are for training, N_train..N_train+N_test-1 for testing.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gamma, kstest, norm
from tqdm import tqdm


def se_correlation(coords, lengthscale=0.1, nugget=1e-4):
    """Squared exponential correlation + nugget, as build_covariance in Train_BNFC."""
    sqdist = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-0.5 * sqdist / lengthscale ** 2) + nugget * np.eye(len(coords))


def F1(w, v_nodes):
    """F_1(w) = mean_k Phi(w - v_k) over factor quantile nodes v_k. w: (n,)."""
    return norm.cdf(w[:, None] - v_nodes[None, :]).mean(1)


def write_csv(path, coords, values):
    pd.DataFrame({"x1": coords[:, 0], "x2": coords[:, 1], "y": values}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Gamma-factor copula data")
    parser.add_argument("--out_dir", type=str, default="simulations_gamma/")
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--N_train", type=int, default=2000)
    parser.add_argument("--N_test", type=int, default=100)
    parser.add_argument("--shape", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=2000,
                        help="quadrature nodes for F_1")
    parser.add_argument("--seed", type=int, default=2025)
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

    # quantile nodes of V0 for the marginal F_1
    p = (np.arange(args.K) + 0.5) / args.K
    v_nodes = gamma.ppf(p, a=args.shape, scale=args.scale)

    u_pool = []
    for i in tqdm(range(args.N_train + args.N_test)):
        v0 = rng.gamma(args.shape, args.scale)
        W = v0 + L @ rng.standard_normal(len(coords))
        U = F1(W, v_nodes)

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
