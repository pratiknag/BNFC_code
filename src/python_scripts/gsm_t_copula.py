#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
External baseline: the Student-t copula, i.e. the Gaussian scale mixture

    W(s) = Z(s) / sqrt(G),   Z ~ GP(0, Sigma_beta),   G ~ Gamma(nu / 2, rate nu / 2),
    Sigma_beta = (1 - beta) R + beta 1 1',

where R is the Gaussian component shared by every model in the comparison (the
SE correlation with range 0.1 in the simulations; in the application, the range
and nugget estimated by BNFC (stat), --kernel/--lengthscale/--nugget), and beta in [0, 1) adds the
constant long-range correlation that a common factor adds to R. The t copula
thus has the second-order structure of the common factor copula, and the two
differ only in how the dependence is induced: by a multiplicative scale
mixture instead of an additive factor. The t copula is tail dependent for every
finite nu, with lambda_U = lambda_L, and it is radially symmetric.

The copula density is available in closed form,

    log c_n(u) = log Gamma((nu + n)/2) + (n - 1) log Gamma(nu/2) - n log Gamma((nu + 1)/2)
                 - log|Sigma|/2 - (nu + n)/2 log(1 + x' Sigma^{-1} x / nu)
                 + (nu + 1)/2 sum_j log(1 + x_j^2 / nu),     x_j = t_nu^{-1}(u_j),

so the posterior of (beta, nu) is computed exactly on a grid of (logit beta,
log nu) (two passes: a coarse grid and a fine grid around the mode), with the
prior nu ~ Gamma(2, rate 0.1) of Juarez and Steel (2010) and beta ~ U(0, 1).
Posterior draws are taken from the fine grid.

Conditional simulation is exact: given x at the observed sites, x_0 at the test
sites is multivariate t with nu + n degrees of freedom, location A x and scale
matrix (nu + x' Sigma^{-1} x) / (nu + n) times the kriging covariance.

    fit:      python gsm_t_copula.py fit --data_path simulations_gamma/
    predict:  python gsm_t_copula.py predict --scenarios ... [--n_cond 25 --out_dir ...]

Files: <data_path>/models/posterior-tcopula.{pth,json}; predictions in the
format of Copula_BNFC.py under <out_dir>/<scenario>/tcopula/.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln
from scipy.stats import t as sp_t
from tqdm import tqdm

from Copula_BNFC import (load_csv_fields, load_scenario, load_margin, subset_conditioning,
                         replicate_metrics, summarise, device)
from dataGen_SignChange import se_correlation

NAME = "tcopula"
PRIOR_SHAPE, PRIOR_RATE = 2.0, 0.1


def posterior_path(data_path, ext):
    return os.path.join(data_path, "models", f"posterior-{NAME}.{ext}")


# =========================
# Copula density
# =========================
def log_copula(u, Sigma_chol, logdet, nu):
    """u: (N, n) in (0, 1). Returns (N,) log t-copula densities with nu degrees of freedom."""
    n = u.shape[1]
    x = sp_t.ppf(u, nu)
    white = np.linalg.solve(Sigma_chol, x.T).T                      # L^{-1} x
    quad = (white ** 2).sum(1)
    return (gammaln((nu + n) / 2) + (n - 1) * gammaln(nu / 2) - n * gammaln((nu + 1) / 2)
            - 0.5 * logdet - 0.5 * (nu + n) * np.log1p(quad / nu)
            + 0.5 * (nu + 1) * np.log1p(x ** 2 / nu).sum(1))


FIXED = {"kernel": "fixed"}


def base_correlation(coords, gc=FIXED):
    """
    Gaussian component R shared with the other models: the SE correlation with
    range 0.1 of the simulations (kernel "fixed"), or (1 - nugget) K + nugget I with
    the squared exponential ("se") or exponential kernel K and the given range,
    as estimated by Train_BNFC --learn_cov.
    """
    if gc["kernel"] == "fixed":
        return se_correlation(coords)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    K = np.exp(-d / gc["lengthscale"]) if gc["kernel"] == "exponential" \
        else np.exp(-0.5 * d ** 2 / gc["lengthscale"] ** 2)
    return (1 - gc["nugget"]) * K + gc["nugget"] * np.eye(len(coords))


def correlation(coords, beta, gc=FIXED):
    return (1 - beta) * base_correlation(coords, gc) + beta


def gaussian_component(coords, beta=0.0, gc=FIXED):
    L = np.linalg.cholesky(correlation(coords, beta, gc))
    return L, 2 * np.log(np.diag(L)).sum()


def log_prior(nu):
    return (PRIOR_SHAPE - 1) * np.log(nu) - PRIOR_RATE * nu


# =========================
# Fit: exact grid posterior of nu
# =========================
def fit(args):
    data, coords = load_csv_fields(os.path.join(args.data_path, "sample_copula-{}.csv"),
                                   range(args.N))
    u = np.clip(data.astype(np.float64), 1e-6, 1 - 1e-6)
    gc = FIXED if args.kernel == "fixed" else \
        {"kernel": args.kernel, "lengthscale": args.lengthscale, "nugget": args.nugget}

    n = u.shape[1]
    t0 = time.time()

    def log_post(z_beta, log_nu):
        """(len(z_beta), len(log_nu)) log posterior on the grid, uniform in (logit beta, log nu)."""
        beta = 1 / (1 + np.exp(-z_beta))
        chol = [gaussian_component(coords.astype(np.float64), b, gc) for b in beta]
        out = np.empty((len(beta), len(log_nu)))
        for j, nu in enumerate(tqdm(np.exp(log_nu), leave=False)):
            x = sp_t.ppf(u, nu)
            const = (gammaln((nu + n) / 2) + (n - 1) * gammaln(nu / 2) - n * gammaln((nu + 1) / 2))
            marg = 0.5 * (nu + 1) * np.log1p(x ** 2 / nu).sum()
            for i, (L, logdet) in enumerate(chol):
                quad = (np.linalg.solve(L, x.T) ** 2).sum(0)
                out[i, j] = (len(u) * (const - 0.5 * logdet) - 0.5 * (nu + n) * np.log1p(quad / nu).sum()
                             + marg)
            # priors, with the Jacobians of log(nu) and logit(beta)
            out[:, j] += log_prior(nu) + np.log(nu) + np.log(beta * (1 - beta))
        return out

    zb = np.linspace(-14.0, 7.0, args.n_coarse)
    ln = np.linspace(math.log(0.5), math.log(2000.0), args.n_coarse)
    lp = log_post(zb, ln)
    i, j = np.unravel_index(np.argmax(lp), lp.shape)
    if i in (0, len(zb) - 1) or j in (0, len(ln) - 1):
        print(f"[warn] posterior mode at the boundary of the grid: beta = "
              f"{1 / (1 + np.exp(-zb[i])):.3g}, nu = {np.exp(ln[j]):.3g}")
    zb_f = np.linspace(zb[max(i - 1, 0)], zb[min(i + 1, len(zb) - 1)], args.n_fine)
    ln_f = np.linspace(ln[max(j - 1, 0)], ln[min(j + 1, len(ln) - 1)], args.n_fine)
    lp_f = log_post(zb_f, ln_f)
    w = np.exp(lp_f - lp_f.max())
    w /= w.sum()
    elapsed = time.time() - t0

    rng = np.random.default_rng(args.seed)
    # draws from the piecewise-constant posterior on the fine grid
    k = rng.choice(w.size, args.num_samples, p=w.ravel())
    ki, kj = np.unravel_index(k, w.shape)
    jit = rng.uniform(size=(2, args.num_samples)) - 0.5
    beta = 1 / (1 + np.exp(-(zb_f[ki] + jit[0] * (zb_f[1] - zb_f[0]))))
    nu = np.exp(ln_f[kj] + jit[1] * (ln_f[1] - ln_f[0]))
    edge = w[[0, -1], :].sum() + w[:, [0, -1]].sum()
    if edge > 0.01:
        print(f"[warn] {edge:.3f} of the posterior mass is on the edge of the fine grid")

    os.makedirs(os.path.join(args.data_path, "models"), exist_ok=True)
    torch.save({"nu": torch.tensor(nu), "beta": torch.tensor(beta), "gaussian_component": gc},
               posterior_path(args.data_path, "pth"))
    summ = {f"{k}_{s}": float(f(v)) for k, v in (("nu", nu), ("beta", beta))
            for s, f in (("mean", np.mean), ("sd", np.std),
                         ("q025", lambda a: np.quantile(a, 0.025)),
                         ("q975", lambda a: np.quantile(a, 0.975)))}
    with open(posterior_path(args.data_path, "json"), "w") as f:
        json.dump({"config": vars(args), "time_seconds": elapsed, **summ}, f, indent=2)
    print(f"{args.data_path}: posterior nu mean {summ['nu_mean']:.3f} (sd {summ['nu_sd']:.3f}), "
          f"beta mean {summ['beta_mean']:.4f} (sd {summ['beta_sd']:.4f}), time {elapsed:.1f}s")


def load_posterior(data_path):
    """Posterior draws (nu, beta), each (S,), and the Gaussian component, or None if not fitted."""
    path = posterior_path(data_path, "pth")
    if not os.path.exists(path):
        return None
    ck = torch.load(path)
    return ck["nu"].numpy(), ck["beta"].numpy(), ck.get("gaussian_component", FIXED)


# =========================
# Held-out log score
# =========================
def heldout_log_score(data_path, u, coords, R):
    """log { (1/R) sum_r c_n(u | nu_r) } for each row of u (N, n), or None if not fitted."""
    post = load_posterior(data_path)
    if post is None:
        return None
    nu, beta, gc = post
    idx = np.linspace(0, len(nu) - 1, R).round().astype(int)
    lc = np.stack([log_copula(u, *gaussian_component(coords.astype(np.float64), beta[i], gc), nu[i])
                   for i in idx])                                    # (R, N)
    m = lc.max(0)
    return m + np.log(np.exp(lc - m).mean(0))


# =========================
# Posterior predictive simulation
# =========================
def kriging(coords_train, coords_test, beta, gc=FIXED):
    C = correlation(np.vstack((coords_train, coords_test)).astype(np.float64), beta, gc)
    n = coords_train.shape[0]
    L_tr = np.linalg.cholesky(C[:n, :n])
    A = np.linalg.solve(C[:n, :n], C[:n, n:]).T                      # (n_test, n)
    cond = C[n:, n:] - A @ C[:n, n:]
    cond = 0.5 * (cond + cond.T)
    ev = np.linalg.eigvalsh(cond).min()
    if ev < 1e-6:
        cond += (1e-6 - ev) * np.eye(len(cond))
    return L_tr, A, np.linalg.cholesky(cond)


def posterior_predictive(u_obs, nu, krigs, rng):
    """
    u_obs: (B, n); nu: (R,) posterior draws, with kriging operators krigs[r]
    under the matching beta draws. Returns copula and latent draws (B, R, n_test).
    """
    B, n = u_obs.shape
    U, X = [], []
    for v, (L_tr, A, L_cond) in zip(nu, krigs):
        x = sp_t.ppf(np.clip(u_obs, 1e-6, 1 - 1e-6), v)               # (B, n)
        quad = (np.linalg.solve(L_tr, x.T) ** 2).sum(0)               # (B,)
        df = v + n
        scale = np.sqrt((v + quad) / df) / np.sqrt(rng.chisquare(df, size=B) / df)
        eps = rng.standard_normal((B, A.shape[0])) @ L_cond.T
        x0 = x @ A.T + scale[:, None] * eps
        U.append(sp_t.cdf(x0, v))
        X.append(x0)
    return np.stack(U, 1), np.stack(X, 1)


def predict_scenario(data_path, args):
    name = os.path.basename(os.path.normpath(data_path))
    post = load_posterior(data_path)
    if post is None:
        print(f"[skip] {name}/{NAME}: missing {posterior_path(data_path, 'pth')}")
        return None
    test_indices = list(range(args.test_start, args.test_start + args.n_test_reps))
    u_obs, coords_train, u_true, y_true, coords_test = load_scenario(data_path, test_indices)
    u_obs, coords_train = subset_conditioning(u_obs, coords_train, args.n_cond)
    margin = "stat" if args.margin == "own" else args.margin
    cs = torch.tensor(coords_test, device=device)
    F_inv = load_margin(data_path, margin, cs)
    rng = np.random.default_rng(args.seed)
    # R posterior draws of (nu, beta), shared by the test replicates
    idx = rng.choice(len(post[0]), args.R, replace=len(post[0]) < args.R)
    nu, beta = post[0][idx], post[1][idx]
    krigs = [kriging(coords_train, coords_test, b, post[2]) for b in beta]
    print(f"\n=== {name} / {NAME} === posterior means nu {post[0].mean():.3f}, "
          f"beta {post[1].mean():.4f}")

    U, W, Y = [], [], []
    t0 = time.time()
    for s in tqdm(range(0, len(test_indices), args.batch_size)):
        u_d, x_d = posterior_predictive(u_obs[s:s + args.batch_size], nu, krigs, rng)
        U.append(u_d.astype(np.float32))
        W.append(x_d.astype(np.float32))
        if F_inv is not None:
            with torch.no_grad():
                ut = torch.tensor(u_d, dtype=torch.float32, device=device)
                Y.append(torch.stack([F_inv(ut[:, r]) for r in range(ut.shape[1])], 1)
                         .cpu().numpy())
    elapsed = time.time() - t0
    U, W = np.concatenate(U), np.concatenate(W)
    Y = np.concatenate(Y) if Y else np.full_like(U, np.nan)
    if args.y_min is not None:                   # e.g. 0 for precipitation
        Y = np.maximum(Y, args.y_min)

    per_rep_copula = [replicate_metrics(U[k], u_true[k]) for k in range(len(test_indices))]
    per_rep_data = [replicate_metrics(Y[k], y_true[k]) for k in range(len(test_indices))]
    metrics = {
        "scenario": name, "factor": NAME, "config": vars(args), "margin": margin,
        "n_cond": int(coords_train.shape[0]), "test_indices": test_indices,
        "time_seconds": elapsed, "nu_posterior_mean": float(post[0].mean()),
        "beta_posterior_mean": float(post[1].mean()),
        "summary": {"copula": summarise(per_rep_copula), "data": summarise(per_rep_data)},
        "per_replicate": {"copula": per_rep_copula, "data": per_rep_data},
    }
    out = os.path.join(args.out_dir, name, NAME)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        latent_samples=W, copula_samples=U, data_samples=Y,
                        u_obs=u_obs, u_true=u_true, y_true=y_true,
                        coords_train=coords_train, coords_test=coords_test,
                        test_indices=np.array(test_indices))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    s = metrics["summary"]["copula"]
    print("[copula] " + ", ".join(f"{k}={s[k]['mean']:.4f}({s[k]['se']:.4f})"
                                  for k in ("mae", "crps", "energy_score", "coverage95")))
    return metrics


def predict(args):
    rows = []
    for data_path in args.scenarios:
        m = predict_scenario(data_path, args)
        if m is None:
            continue
        row = {"scenario": m["scenario"], "factor": NAME, "time_seconds": m["time_seconds"]}
        for scale in ("copula", "data"):
            for k, v in m["summary"][scale].items():
                row[f"{scale}_{k}"] = v["mean"]
                row[f"{scale}_{k}_se"] = v["se"]
        rows.append(row)
    if rows:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "summary.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(path):
            old = pd.read_csv(path)
            keep = ~old.set_index(["scenario", "factor"]).index.isin(
                new.set_index(["scenario", "factor"]).index)
            new = pd.concat([old[keep], new], ignore_index=True)
        new.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Student-t copula (Gaussian scale mixture)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data_path", type=str, default="simulations_gamma/")
    f.add_argument("--N", type=int, default=2000)
    f.add_argument("--n_coarse", type=int, default=40)
    f.add_argument("--n_fine", type=int, default=40)
    f.add_argument("--num_samples", type=int, default=500)
    f.add_argument("--seed", type=int, default=123)
    f.add_argument("--kernel", choices=["fixed", "se", "exponential"], default="fixed",
                   help="Gaussian component: the fixed SE correlation of the simulations, or the "
                        "given range and nugget (e.g. those estimated by Train_BNFC --learn_cov)")
    f.add_argument("--lengthscale", type=float, default=0.1)
    f.add_argument("--nugget", type=float, default=1e-4)
    p = sub.add_parser("predict")
    p.add_argument("--scenarios", nargs="+",
                   default=["simulations_gamma/", "simulations_signchange/", "simulations_mixgamma/"])
    p.add_argument("--out_dir", type=str, default="results/")
    p.add_argument("--test_start", type=int, default=2000)
    p.add_argument("--n_test_reps", type=int, default=100)
    p.add_argument("--R", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--margin", choices=["nonstat", "stat", "own"], default="nonstat")
    p.add_argument("--n_cond", type=int, default=None)
    p.add_argument("--y_min", type=float, default=None,
                   help="lower bound of the data scale (predictions are floored at it)")
    p.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(args)
    fit(args) if args.cmd == "fit" else predict(args)


if __name__ == "__main__":
    main()
