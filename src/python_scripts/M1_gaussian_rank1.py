#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026

"""
Gaussian loading copula (M1): Gaussian copula with a rank-one,
spatially varying loading,

    W(s) = Z(s) + b(s) V,   V ~ N(0, 1),   Z ~ GP(0, rho_alpha),   Z independent of V,
    b(s) = sum_k beta_k psi_k(s),   psi_k(s) = exp(-||s - c_k||^2 / (2 h^2)),

with K = g^2 centres c_k on a g x g grid over [0, 1]^2 and bandwidth h equal to
the grid spacing (--basis_grid 5: K = 25; --basis_grid 7: K = 49). rho_alpha is
the Gaussian component of BNFC, (1 - nugget) K_SE(range) + nugget I, with the
range and nugget estimated jointly with the loading.

M1 has the second-order flexibility of BNFC (nonstat): Sigma_alpha + b b'
covers long-range correlation that varies over space and changes sign. But it
is Gaussian, so lambda_L = lambda_U = 0 and zeta_1 = 0, and any advantage of
BNFC over M1 is due to non-Gaussian dependence.

The copula is the Gaussian copula with correlation
R = D^{-1/2} (Sigma_alpha + b b') D^{-1/2}, D = diag(1 + b_j^2), so with
x_i = Phi^{-1}(u_i) the log-likelihood depends on the data only through
S = sum_i x_i x_i':

    l(beta, alpha) = -N/2 log|R| - 1/2 tr{(R^{-1} - I) S}.

Prior: beta ~ N(0, sigma_beta^2 I), sigma_beta ~ HalfNormal(2); log range and
logit nugget ~ N(0, 3^2) (weak; BNFC has point estimates of both). Fitted with
NUTS (dense mass matrix), started from the posterior mode. Only b b' is
identified, so b and -b give the same model.

Prediction given x at the observed sites is exact Gaussian kriging with R
(on the correlation scale), U_0 = Phi(X_0), and the
data scale through the marginal network T_psi shared by every model.

    fit:      python M1_gaussian_rank1.py fit --data_path application_precipitation_v2/ --N 10591
    score:    python M1_gaussian_rank1.py score --data_path ... --test_start 10591 --n_test_reps 200
    predict:  python M1_gaussian_rank1.py predict --data_path ... --out_dir ... --y_min 0

Files: <data_path>/models/posterior-m1[_K<K>].{pth,json};
<out_dir>/<scenario>/m1/{predictions.npz,metrics.json,heldout_loglik.csv}.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from pyro.infer import MCMC, NUTS
from scipy.stats import norm
from tqdm import tqdm

from Copula_BNFC import (load_csv_fields, load_scenario, load_margin, subset_conditioning,
                         replicate_metrics, summarise, device)

DT = torch.float64
CPU = torch.device("cpu")


def model_name(basis_grid):
    return "m1" if basis_grid == 5 else f"m1_K{basis_grid ** 2}"


def posterior_path(data_path, basis_grid, ext):
    return os.path.join(data_path, "models", f"posterior-{model_name(basis_grid)}.{ext}")


# =========================
# Loading basis and correlation
# =========================
def basis(coords, basis_grid):
    """(n, K) Gaussian bumps on a basis_grid x basis_grid grid of centres over [0, 1]^2."""
    g = np.linspace(0, 1, basis_grid)
    cx, cy = np.meshgrid(g, g)
    centres = np.column_stack((cx.ravel(), cy.ravel()))
    h = g[1] - g[0]
    d2 = ((coords[:, None, :] - centres[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * h ** 2))


def correlation(coords, Psi, beta, lengthscale, nugget):
    """Copula correlation R (n, n) and the loading b (n,), in torch."""
    n = coords.shape[0]
    d2 = torch.cdist(coords, coords).pow(2)
    Sigma = (1 - nugget) * torch.exp(-0.5 * d2 / lengthscale ** 2) \
        + nugget * torch.eye(n, dtype=coords.dtype, device=coords.device)
    b = Psi @ beta
    SigmaW = Sigma + b[:, None] * b[None, :]
    dinv = torch.rsqrt(1 + b ** 2)
    return SigmaW * dinv[:, None] * dinv[None, :], b


# =========================
# Unconstrained parameters: z = (beta (K), log sigma_beta, log range, logit nugget)
# =========================
def unpack(z, K):
    return z[:K], torch.exp(z[K]), torch.exp(z[K + 1]), torch.sigmoid(z[K + 2])


def make_potential(S, N, coords, Psi, fixed=None):
    """Negative log posterior of the unconstrained parameters (with Jacobians). fixed =
    (range, nugget) holds the Gaussian component fixed; their coordinates of z then only
    carry their prior."""
    K = Psi.shape[1]

    def potential(params):
        z = params["z"]
        beta, sb, ls, nug = unpack(z, K)
        if fixed is not None:
            ls, nug = fixed
        R, _ = correlation(coords, Psi, beta, ls, nug)
        L = torch.linalg.cholesky(R)
        logdet = 2 * torch.log(torch.diagonal(L)).sum()
        Rinv = torch.cholesky_inverse(L)
        loglik = -0.5 * N * logdet - 0.5 * ((Rinv * S).sum() - torch.trace(S))
        logprior = (-0.5 * (beta ** 2).sum() / sb ** 2 - K * torch.log(sb)
                    - 0.5 * sb ** 2 / 4.0 + z[K]                       # HalfNormal(2), log Jacobian
                    - 0.5 * (z[K + 1] ** 2 + z[K + 2] ** 2) / 9.0)
        return -(loglik + logprior)

    return potential


# =========================
# Fit: posterior mode, then NUTS
# =========================
def fit(args):
    data, coords = load_csv_fields(os.path.join(args.data_path, "sample_copula-{}.csv"),
                                   range(args.N))
    x = norm.ppf(np.clip(data.astype(np.float64), 1e-6, 1 - 1e-6))
    S = torch.tensor(x.T @ x, dtype=DT)
    N = x.shape[0]
    c = torch.tensor(coords, dtype=DT)
    Psi = torch.tensor(basis(coords.astype(np.float64), args.basis_grid), dtype=DT)
    K = Psi.shape[1]
    fixed = ((torch.tensor(args.init_lengthscale, dtype=DT), torch.tensor(args.init_nugget, dtype=DT))
             if args.fix_cov else None)
    potential = make_potential(S, N, c, Psi, fixed)
    print(f"{args.data_path}: N={N}, n={c.shape[0]}, K={K}"
          + (f", Gaussian component fixed (range {args.init_lengthscale}, nugget {args.init_nugget})"
             if fixed else ""))

    t0 = time.time()
    # b = 0 is a saddle point (b enters through b b'), so start from a constant loading
    z = torch.cat([torch.full((K,), 0.5, dtype=DT), torch.tensor(
        [0.0, math.log(args.init_lengthscale), math.log(args.init_nugget / (1 - args.init_nugget))],
        dtype=DT)]).requires_grad_(True)
    opt = torch.optim.LBFGS([z], lr=1, max_iter=500, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = potential({"z": z})
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)
    z0 = z.detach().clone()
    _, sb0, ls0, nug0 = unpack(z0, K)
    print(f"mode: -log post/N {potential({'z': z0}).item() / N:.4f}, range {ls0:.4f}, "
          f"nugget {nug0:.4f}, sigma_beta {sb0:.3f}, time {time.time() - t0:.1f}s")

    kernel = NUTS(potential_fn=potential, full_mass=True, target_accept_prob=0.8,
                  max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.warmup,
                initial_params={"z": z0}, num_chains=1)
    mcmc.run()
    elapsed = time.time() - t0
    zs = mcmc.get_samples()["z"]
    beta, sb, ls, nug = zs[:, :K], torch.exp(zs[:, K]), torch.exp(zs[:, K + 1]), \
        torch.sigmoid(zs[:, K + 2])
    if fixed is not None:
        ls, nug = torch.full_like(ls, args.init_lengthscale), torch.full_like(nug, args.init_nugget)
    diag = mcmc.diagnostics()["z"]

    os.makedirs(os.path.join(args.data_path, "models"), exist_ok=True)
    torch.save({"beta": beta, "sigma_beta": sb, "lengthscale": ls, "nugget": nug,
                "basis_grid": args.basis_grid, "mode": z0},
               posterior_path(args.data_path, args.basis_grid, "pth"))
    summ = {"time_seconds": elapsed, "K": K, "N": N,
            "min_ess": float(diag["n_eff"].min()), "max_rhat": float(diag["r_hat"].max()),
            **{f"{k}_{s}": float(f(v)) for k, v in (("lengthscale", ls), ("nugget", nug),
                                                    ("sigma_beta", sb))
               for s, f in (("mean", torch.mean), ("sd", torch.std))}}
    with open(posterior_path(args.data_path, args.basis_grid, "json"), "w") as f:
        json.dump({"config": vars(args), **summ}, f, indent=2)
    print(json.dumps(summ, indent=2))


def load_posterior(data_path, basis_grid):
    path = posterior_path(data_path, basis_grid, "pth")
    return torch.load(path) if os.path.exists(path) else None


def draw_correlations(post, coords, idx):
    """Correlation matrices R (numpy) of the posterior draws idx at coords."""
    c = torch.tensor(coords, dtype=DT)
    Psi = torch.tensor(basis(coords.astype(np.float64), post["basis_grid"]), dtype=DT)
    return [correlation(c, Psi, post["beta"][i], post["lengthscale"][i], post["nugget"][i])[0]
            .numpy() for i in idx]


# =========================
# Held-out log score
# =========================
def log_copula(u, R):
    """u: (N, n). Gaussian copula log-densities (N,) with correlation R."""
    x = norm.ppf(u)
    L = np.linalg.cholesky(R)
    white = np.linalg.solve(L, x.T)
    return -np.log(np.diag(L)).sum() - 0.5 * ((white ** 2).sum(0) - (x ** 2).sum(1))


def score(args):
    """log {(1/R) sum_r c_n(u | eta_r)} per test replicate, as in heldout_loglik.py."""
    post = load_posterior(args.data_path, args.basis_grid)
    name = model_name(args.basis_grid)
    scen = os.path.basename(os.path.normpath(args.data_path))
    test = list(range(args.test_start, args.test_start + args.n_test_reps))
    S = len(post["beta"])
    idx = np.linspace(0, S - 1, args.R).round().astype(int)
    rows = []
    for set_name, fname in (("observed", "sample_copula-{}.csv"),
                            ("full", "sample_copula_full-{}.csv")):
        u, coords = load_csv_fields(os.path.join(args.data_path, fname), test)
        u = np.clip(u.astype(np.float64), 1e-6, 1 - 1e-6)
        lc = np.stack([log_copula(u, R) for R in draw_correlations(post, coords, idx)])
        m = lc.max(0)
        sc = m + np.log(np.exp(lc - m).mean(0))
        print(f"{scen:22s} {set_name:8s} {name:8s} n={u.shape[1]:4d}  mean log score "
              f"{sc.mean():10.3f}  (se {sc.std(ddof=1) / math.sqrt(len(sc)):.3f})")
        rows += [{"scenario": scen, "set": set_name, "n_sites": u.shape[1], "model": name,
                  "replicate": t, "log_score": float(s)} for t, s in zip(test, sc)]
    out = os.path.join(args.out_dir, scen, name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "heldout_loglik.csv"), index=False)


# =========================
# Posterior predictive simulation
# =========================
def kriging(R, n):
    """Kriging operators of the test sites (after the first n) given the first n sites."""
    A = np.linalg.solve(R[:n, :n], R[:n, n:]).T
    cond = R[n:, n:] - A @ R[:n, n:]
    cond = 0.5 * (cond + cond.T)
    ev = np.linalg.eigvalsh(cond).min()
    if ev < 1e-6:
        cond += (1e-6 - ev) * np.eye(len(cond))
    return A, np.linalg.cholesky(cond)


def factor_posteriors(post, coords_train, coords_test, idx):
    """Per posterior draw: D^{1/2} at the conditioning sites, g = Sigma_W^{-1} b, the
    posterior variance 1 - b' Sigma_W^{-1} b of V given the latent data w, and b at the
    test sites, on the scale W = Z + b V of the model."""
    c = torch.tensor(np.vstack((coords_train, coords_test)), dtype=DT)
    Psi = torch.tensor(basis(c.numpy(), post["basis_grid"]), dtype=DT)
    n, out = len(coords_train), []
    for i in idx:
        R, b = correlation(c, Psi, post["beta"][i], post["lengthscale"][i], post["nugget"][i])
        R, b = R.numpy(), b.numpy()
        d = np.sqrt(1 + b ** 2)
        SW = R[:n, :n] * d[:n, None] * d[None, :n]                          # Sigma + b b'
        g = np.linalg.solve(SW, b[:n])
        out.append((d[:n], g, max(1 - b[:n] @ g, 1e-12), b[n:]))
    return out


def predict(args):
    post = load_posterior(args.data_path, args.basis_grid)
    name = model_name(args.basis_grid)
    scen = os.path.basename(os.path.normpath(args.data_path))
    test_indices = list(range(args.test_start, args.test_start + args.n_test_reps))
    u_obs, coords_train, u_true, y_true, coords_test = load_scenario(args.data_path, test_indices)
    u_obs, coords_train = subset_conditioning(u_obs, coords_train, args.n_cond)
    F_inv = load_margin(args.data_path, args.margin, torch.tensor(coords_test, device=device))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(post["beta"]), args.R, replace=len(post["beta"]) < args.R)
    n = coords_train.shape[0]
    t0 = time.time()
    krigs = [kriging(R, n) for R in
             draw_correlations(post, np.vstack((coords_train, coords_test)), idx)]
    facs = factor_posteriors(post, coords_train, coords_test, idx)
    x_obs = norm.ppf(np.clip(u_obs.astype(np.float64), 1e-6, 1 - 1e-6))

    U, X, Y, F = [], [], [], []
    for s in tqdm(range(0, len(test_indices), args.batch_size)):
        xb = x_obs[s:s + args.batch_size]
        x0 = np.stack([xb @ A.T + rng.standard_normal((len(xb), A.shape[0])) @ Lc.T
                       for A, Lc in krigs], 1)                               # (B, R, n_test)
        # centred factor contribution b(s0) V, V | w ~ N(g'w, v) with w = D^{1/2} x
        F.append(np.stack([np.outer((xb * d_obs) @ g + np.sqrt(v) * rng.standard_normal(len(xb)), b_te)
                           for d_obs, g, v, b_te in facs], 1).astype(np.float32))
        u0 = norm.cdf(x0)
        U.append(u0.astype(np.float32))
        X.append(x0.astype(np.float32))
        if F_inv is not None:
            with torch.no_grad():
                ut = torch.tensor(u0, dtype=torch.float32, device=device)
                Y.append(torch.stack([F_inv(ut[:, r]) for r in range(ut.shape[1])], 1)
                         .cpu().numpy())
    elapsed = time.time() - t0
    U, X = np.concatenate(U), np.concatenate(X)
    Y = np.concatenate(Y) if Y else np.full_like(U, np.nan)
    if args.y_min is not None:
        Y = np.maximum(Y, args.y_min)

    per_rep_copula = [replicate_metrics(U[k], u_true[k]) for k in range(len(test_indices))]
    per_rep_data = [replicate_metrics(Y[k], y_true[k]) for k in range(len(test_indices))]
    metrics = {
        "scenario": scen, "factor": name, "config": vars(args), "margin": args.margin,
        "n_cond": n, "test_indices": test_indices, "time_seconds": elapsed,
        "summary": {"copula": summarise(per_rep_copula), "data": summarise(per_rep_data)},
        "per_replicate": {"copula": per_rep_copula, "data": per_rep_data},
    }
    out = os.path.join(args.out_dir, scen, name)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        latent_samples=X, copula_samples=U, data_samples=Y,
                        u_obs=u_obs, u_true=u_true, y_true=y_true,
                        coords_train=coords_train, coords_test=coords_test,
                        test_indices=np.array(test_indices), factor_samples=np.concatenate(F))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    s = metrics["summary"]["copula"]
    print("[copula] " + ", ".join(f"{k}={s[k]['mean']:.4f}({s[k]['se']:.4f})"
                                  for k in ("mae", "crps", "energy_score", "coverage95")))


def main():
    parser = argparse.ArgumentParser(description="M1: Gaussian copula with a rank-one varying loading")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--N", type=int, default=2000)
    f.add_argument("--warmup", type=int, default=500)
    f.add_argument("--num_samples", type=int, default=1000)
    f.add_argument("--max_tree_depth", type=int, default=8)
    f.add_argument("--init_lengthscale", type=float, default=0.15)
    f.add_argument("--init_nugget", type=float, default=0.05)
    f.add_argument("--fix_cov", action="store_true",
                   help="hold the range and nugget at --init_lengthscale/--init_nugget "
                        "(the simulations: 0.1 and 1e-4, as for the other models)")
    sc = sub.add_parser("score")
    p = sub.add_parser("predict")
    for q in (sc, p):
        q.add_argument("--out_dir", type=str, default="results_M1_M2/")
        q.add_argument("--test_start", type=int, default=2000)
        q.add_argument("--n_test_reps", type=int, default=100)
    sc.add_argument("--R", type=int, default=20, help="posterior draws (as in heldout_loglik.py)")
    p.add_argument("--R", type=int, default=100, help="posterior predictive draws per replicate")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--margin", choices=["nonstat", "stat"], default="nonstat")
    p.add_argument("--n_cond", type=int, default=None)
    p.add_argument("--y_min", type=float, default=None)
    for q in (f, sc, p):
        q.add_argument("--data_path", type=str, default="simulations_gamma/")
        q.add_argument("--basis_grid", type=int, default=5, help="K = basis_grid^2 loading bumps")
        q.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    {"fit": fit, "score": score, "predict": predict}[args.cmd](args)


if __name__ == "__main__":
    main()
