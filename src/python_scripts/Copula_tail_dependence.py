#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, August 2025

"""
Tail flexibility of the BNFC copula (Section 3.3 of the manuscript, Section S3 of the supplement).

Uses the same building blocks as Train_BNFC.py / Copula_BNFC.py
(NNFactorWithLocations, sampler_means, build_covariance, HalfNormal prior on
theta) and simulates from the location-dependent BNFC model

    W(s) = Z(s) + g(V, s),   V ~ U(theta_1, theta_2),
    U(s) = F_1(W(s); s),     F_1(w; s) = E_v Phi{w - g(v, s)},

where F_1 is applied through the exact pseudo-observations
U_j = rank(W_j) / (N + 1) of a large sample, which are uniform by construction
even when g(., s) is steep and a grid approximation of F_1 would be inaccurate.

for many randomly drawn factor networks g (weights from a Gaussian prior,
theta from the HalfNormal prior used in training).

For each network instance:
  * bivariate copula C_2 at two sites with cor(Z_1, Z_2) = rho:
      zeta_1 = E{(U_1 + U_2 - 1)^3}, lambda_L(q), lambda_U(q)
  * the three instances with strong, medium and weak tail asymmetry
    (largest |zeta_1|, about half of it, and zeta_1 closest to 0) are re-simulated on a regular grid with the squared
    exponential correlation of build_covariance, and the correlation
    matrix of the copula field U is plotted.

Outputs (Figures S1 and S2 of the supplement):
  <fig_dir>/zeta_plot.png
  <fig_dir>/correlation_matrices.png
and the raw values in <out_dir>/tail_dependence.{npz,json}.
"""

# =========================
# Imports
# =========================
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.distributions as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import NNFactorWithLocations
from Train_BNFC import build_covariance, sampler_means

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Plot style (matches plot_BNFC_results.py)
# =========================
COLOR = "#2a78d6"
HIGHLIGHT = "#eb6834"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e1"
CMAP_COV = "RdBu_r"          # diverging: g(v, s) may decrease in v at some sites

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 18,
})

# =========================
# Random BNFC factor networks
# =========================
def make_network(output):
    """
    g(v, s) with the architecture of the location-dependent BNFC
    (NNFactorWithLocations: 3 x 64 GELU). output = "linear" leaves the factor
    unrestricted, "positive" keeps the positive output activation (g >= 0).
    """
    g = NNFactorWithLocations(64).to(device)
    if output == "linear":
        g.net2[-1] = torch.nn.Identity()
    return g


def draw_factor_network(weight_scale, output):
    """Random g(v, s): weights ~ N(0, weight_scale^2 / fan_in), biases ~ N(0, 1)."""
    g = make_network(output)
    with torch.no_grad():
        for m in g.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(0., weight_scale / math.sqrt(m.in_features))
                m.bias.normal_(0., 1.)
    return g.eval()


def draw_theta(prior_sigma):
    """theta = (theta_1, theta_2) from the HalfNormal prior of Train_BNFC, shape (1, 1, 2)."""
    return dist.HalfNormal(prior_sigma).sample((1, 1, 2)).to(device)


# =========================
# Simulation from the BNFC copula
# =========================
@torch.no_grad()
def factor_means(coords, theta, g, u):
    """G(v) at coords for v = theta_1 + |theta_2 - theta_1| u. u: (M,), returns (M, n)."""
    mc_u = u.reshape(1, -1, 1, 1)
    return sampler_means(coords.unsqueeze(0), theta, g, "nonstat", mc_u)[0, :, 0]


@torch.no_grad()
def simulate_copula(coords, L_Z, theta, g, N):
    """
    N draws from the BNFC copula at coords: W = G(V) + Z, then
    U_j = rank(W_j) / (N + 1), the empirical version of F_{1,j}(W_j).

    coords: (n, 2), L_Z: Cholesky factor of cor(Z), (n, n).
    """
    n = coords.shape[0]
    mu = factor_means(coords, theta, g, torch.rand(N, device=device))   # (N, n)
    W = mu + torch.randn(N, n, device=device) @ L_Z.T
    ranks = W.argsort(0).argsort(0).float() + 1.
    return (ranks / (N + 1)).cpu().numpy()


# =========================
# Dependence measures
# =========================
def zeta_1(U):
    """Skewness measure of a bivariate copula, E{(U_1 + U_2 - 1)^3}."""
    return float(np.mean((U[:, 0] + U[:, 1] - 1.) ** 3))


def lambda_q(U, q):
    """Empirical lambda_L(q) = C(q,q)/q and lambda_U(q) = P(U_1>1-q, U_2>1-q)/q."""
    lam_l = np.mean((U[:, 0] <= q) & (U[:, 1] <= q)) / q
    lam_u = np.mean((U[:, 0] > 1 - q) & (U[:, 1] > 1 - q)) / q
    return float(lam_l), float(lam_u)


# =========================
# Figures
# =========================
def plot_zeta(zeta, picks, path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    idx = np.arange(1, len(zeta) + 1)

    ax.axhline(0., color=TEXT_MUTED, lw=1)

    ax.plot(idx, zeta, color=COLOR, lw=1, alpha=0.35, zorder=1)
    ax.scatter(idx, zeta, s=36, color=COLOR, edgecolor="white", lw=1, zorder=2)

    for name, k in picks.items():
        ax.scatter(idx[k], zeta[k], s=90, facecolor="none", edgecolor=HIGHLIGHT,
                   lw=2, zorder=3)
        ax.annotate(name, (idx[k], zeta[k]), xytext=(0, 12 if zeta[k] >= 0 else -12),
                    textcoords="offset points", ha="center",
                    va="bottom" if zeta[k] >= 0 else "top", color=TEXT_MUTED,
                    fontsize=16, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    ax.set_xlabel("Simulation")
    ax.set_ylabel(r"$\zeta_1$")
    ax.set_xlim(0, len(zeta) + 1)
    lim = 1.3 * np.abs(zeta).max()
    ax.set_ylim(-lim, lim)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_correlations(covs, titles, path):
    vmax = 1.
    fig, axes = plt.subplots(1, len(covs), figsize=(5.2 * len(covs), 5.0),
                             constrained_layout=True)
    for ax, cov, title in zip(axes, covs, titles):
        im = ax.imshow(cov, cmap=CMAP_COV, vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"$\mathrm{cor}\{U(\mathbf{s}_j), U(\mathbf{s}_k)\}$")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Tail flexibility of the BNFC copula")
    parser.add_argument("--n_instances", type=int, default=100,
                        help="number of random factor networks")
    parser.add_argument("--N", type=int, default=100000,
                        help="bivariate copula draws per instance")
    parser.add_argument("--N_grid", type=int, default=5000,
                        help="copula field draws for the correlation plots")
    parser.add_argument("--rho", type=float, default=0.5,
                        help="cor(Z_1, Z_2) of the bivariate copula")
    parser.add_argument("--q", type=float, default=0.05,
                        help="level for lambda_L(q), lambda_U(q)")
    parser.add_argument("--grid_size", type=int, default=8,
                        help="grid_size x grid_size sites for the correlation plots")
    parser.add_argument("--weight_scale", type=float, default=3.0)
    parser.add_argument("--output", choices=["linear", "positive"], default="linear",
                        help="output layer of g (application-dependent choice)")
    parser.add_argument("--prior_sigma", type=float, default=4.0,
                        help="HalfNormal scale of the theta prior (as in Train_BNFC)")
    parser.add_argument("--out_dir", type=str, default="results/tail_dependence/")
    parser.add_argument("--fig_dir", type=str, default="results/figures/")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- bivariate setting: two sites, cor(Z_1, Z_2) = rho ----
    pts = torch.tensor([[0.25, 0.25], [0.25, 0.35]], device=device)
    L_biv = torch.linalg.cholesky(torch.tensor(
        [[1., args.rho], [args.rho, 1.]], device=device))

    # ---- grid setting: squared exponential correlation from build_covariance ----
    x = torch.linspace(0., 1., args.grid_size)
    xv, yv = torch.meshgrid(x, x, indexing="xy")
    grid = torch.stack((xv.reshape(-1), yv.reshape(-1)), dim=1).to(device)
    L_grid = torch.linalg.cholesky(
        build_covariance(torch.ones(1, 2, device=device), grid.unsqueeze(0))[0])

    states, thetas = [], []
    zeta, lam_l, lam_u = [], [], []
    for _ in tqdm(range(args.n_instances)):
        g = draw_factor_network(args.weight_scale, args.output)
        theta = draw_theta(args.prior_sigma)
        U = simulate_copula(pts, L_biv, theta, g, args.N)

        zeta.append(zeta_1(U))
        ll, lu = lambda_q(U, args.q)
        lam_l.append(ll)
        lam_u.append(lu)
        states.append({k: v.cpu() for k, v in g.state_dict().items()})
        thetas.append(theta.cpu())

    zeta, lam_l, lam_u = map(np.array, (zeta, lam_l, lam_u))

    # ---- strong / medium / weak tail asymmetry, by zeta_1: largest |zeta_1|,
    #      zeta_1 closest to half of it (same sign), and zeta_1 closest to 0
    #      among the quarter of networks with the weakest tails (so that the weak
    #      case is not a near-comonotone copula) ----
    k_strong = int(np.argmax(np.abs(zeta)))
    strength = np.maximum(lam_l, lam_u)
    moderate = np.where(strength <= np.quantile(strength, 0.25))[0]
    picks = {"strong": k_strong,
             "medium": int(np.argmin(np.abs(zeta - zeta[k_strong] / 2))),
             "weak": int(moderate[np.argmin(np.abs(zeta[moderate]))])}

    covs, titles = [], []
    for name, k in picks.items():
        g = make_network(args.output)
        g.load_state_dict(states[k])
        g.eval()
        U = simulate_copula(grid, L_grid, thetas[k].to(device), g, args.N_grid)
        covs.append(np.corrcoef(U.T))
        z_show = 0. if abs(zeta[k]) < 5e-5 else zeta[k]       # avoid "-0.0000"
        titles.append(f"{name.capitalize()}: " rf"$\zeta_1={z_show:.4f}$" "\n"
                      rf"$\lambda_L(q)={lam_l[k]:.2f}$, $\lambda_U(q)={lam_u[k]:.2f}$")

    plot_zeta(zeta, picks, os.path.join(args.fig_dir, "zeta_plot.png"))
    plot_correlations(covs, titles, os.path.join(args.fig_dir, "correlation_matrices.png"))

    # Gaussian copula reference at the same rho and q
    Zg = np.random.multivariate_normal([0., 0.], [[1., args.rho], [args.rho, 1.]], args.N)
    from scipy.stats import norm
    gauss_l, gauss_u = lambda_q(norm.cdf(Zg), args.q)

    np.savez_compressed(os.path.join(args.out_dir, "tail_dependence.npz"),
                        zeta=zeta, lambda_L=lam_l, lambda_U=lam_u,
                        theta=torch.cat(thetas).reshape(-1, 2).numpy(),
                        correlations=np.stack(covs), grid=grid.cpu().numpy())
    summary = {
        "config": vars(args),
        "picks": picks,
        "zeta": {"min": float(zeta.min()), "median": float(np.median(zeta)),
                 "max": float(zeta.max()),
                 "n_positive": int((zeta > 0).sum()), "n_negative": int((zeta < 0).sum())},
        "lambda_L_q": {"min": float(lam_l.min()), "median": float(np.median(lam_l)),
                       "max": float(lam_l.max())},
        "lambda_U_q": {"min": float(lam_u.min()), "median": float(np.median(lam_u)),
                       "max": float(lam_u.max())},
        "gaussian_reference": {"lambda_L_q": gauss_l, "lambda_U_q": gauss_u},
        "picked_values": {name: {"zeta": float(zeta[k]), "lambda_L_q": float(lam_l[k]),
                                 "lambda_U_q": float(lam_u[k])}
                          for name, k in picks.items()},
    }
    with open(os.path.join(args.out_dir, "tail_dependence.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: summary[k] for k in
                      ("picks", "zeta", "lambda_L_q", "lambda_U_q",
                       "gaussian_reference", "picked_values")}, indent=2))
    print(f"figures written to {args.fig_dir}")


if __name__ == "__main__":
    main()
