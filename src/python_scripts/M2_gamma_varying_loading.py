#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
Gamma loading copula (M2): parametric factor with a spatially varying loading,

    W(s) = Z(s) + b(s) H_kappa^{-1}(V),   V ~ U(0, 1),   Z ~ GP(0, rho_alpha),

with H_kappa the Ga(kappa, 1) distribution (shape estimated; the scale is absorbed by b),
the loading b(s) = sum_k beta_k psi_k(s) of M1 (M1_gaussian_rank1.basis) and the
Gaussian component of BNFC, (1 - nugget) K_SE(range) + nugget I. M2 is the special case
g(v, s) = b(s) H_kappa^{-1}(v) of BNFC (nonstat): a parametric factor shape entering
through a separable, signed loading. The sign of b is identified, because the gamma
factor is skewed.

The factor is discretised with the deterministic quantile nodes
q_m = H_kappa^{-1}{(m - 0.5)/M}, so that the approximation is itself a factor copula:

    F_1j(w) = (1/M) sum_m Phi(w - b_j q_m),   f_1j(w) = (1/M) sum_m phi(w - b_j q_m),
    f_n(w) = (1/M) sum_m phi_Sigma(w - b q_m),

and (w - b q_m)' Sigma^{-1} (w - b q_m) = w'S^{-1}w - 2 q_m w'S^{-1}b + q_m^2 b'S^{-1}b, so the
joint term costs O(n^2) per replicate. w_j = F_1j^{-1}(u_j) is found by bisection with
Newton refinement and carries the implicit-function gradient, as in
train_model_single_factor_HMC_NUTS.py. The nodes depend on kappa through the gamma
quantile function, differentiated by central differences (scipy).

Prior: beta ~ N(0, sigma_beta^2 I), sigma_beta ~ HalfNormal(2), kappa ~ Ga(2, rate 0.5);
log range and logit nugget ~ N(0, 3^2), as for M1. Fitted by NUTS (dense mass matrix),
started from the posterior mode.

    fit:      python M2_gamma_varying_loading.py fit --data_path application_precipitation_v2/ --N 10591
    score:    python M2_gamma_varying_loading.py score --data_path ... --test_start 10591 --n_test_reps 200
    predict:  python M2_gamma_varying_loading.py predict --data_path ... --out_dir ... --y_min 0

Files: <data_path>/models/posterior-m2.{pth,json}; <out_dir>/<scenario>/m2/.
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
from scipy.stats import gamma as sp_gamma
from tqdm import tqdm

from Copula_BNFC import (load_csv_fields, load_scenario, load_margin, subset_conditioning,
                         replicate_metrics, summarise)
from M1_gaussian_rank1 import basis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
NAME = "m2"
normal = torch.distributions.Normal(torch.tensor(0., dtype=DT, device=device),
                                    torch.tensor(1., dtype=DT, device=device))


def posterior_path(data_path, ext):
    return os.path.join(data_path, "models", f"posterior-{NAME}.{ext}")


# =========================
# Gamma quantile nodes, differentiable in the shape
# =========================
class GammaNodes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kappa, p):
        k = kappa.item()
        pn = p.cpu().numpy()
        h = 1e-4 * max(k, 1e-2)
        ctx.save_for_backward(torch.tensor((sp_gamma.ppf(pn, k + h) - sp_gamma.ppf(pn, k - h)) / (2 * h),
                                           dtype=kappa.dtype, device=kappa.device))
        return torch.tensor(sp_gamma.ppf(pn, k), dtype=kappa.dtype, device=kappa.device)

    @staticmethod
    def backward(ctx, grad):
        dq, = ctx.saved_tensors
        return (grad * dq).sum(), None


def nodes(kappa, M):
    p = (torch.arange(M, dtype=DT) + 0.5) / M
    return GammaNodes.apply(kappa, p)


# =========================
# Copula log-density
# =========================
def gaussian_component(coords, lengthscale, nugget):
    n = coords.shape[0]
    d2 = torch.cdist(coords, coords).pow(2)
    return (1 - nugget) * torch.exp(-0.5 * d2 / lengthscale ** 2) \
        + nugget * torch.eye(n, dtype=coords.dtype, device=coords.device)


@torch.no_grad()
def F1_inv(u, G, tol=1e-6, max_iter=60, n_newton=2):
    """u: (N, n); G: (n, M) nodes b_j q_m. Returns w (N, n) with mean_m Phi(w - G) = u."""
    low = (G.min(1).values - 10).expand_as(u).clone()
    high = (G.max(1).values + 10).expand_as(u).clone()
    for _ in range(max_iter):
        mid = (low + high) / 2
        val = normal.cdf(mid.unsqueeze(-1) - G).mean(-1)
        low = torch.where(val < u, mid, low)
        high = torch.where(val >= u, mid, high)
        if (high - low).max() < tol:
            break
    w = (low + high) / 2
    for _ in range(n_newton):
        r = w.unsqueeze(-1) - G
        step = (normal.cdf(r).mean(-1) - u) / normal.log_prob(r).exp().mean(-1).clamp_min(1e-300)
        w = w - torch.where(step.abs() < 10 * tol, step, torch.zeros_like(step))
    return w


def log_copula(u, b, q, Sigma, chunk=2000):
    """u: (N, n); b: (n,); q: (M,); Sigma: (n, n). Returns (N,) log copula densities."""
    G = b[:, None] * q[None, :]                                           # (n, M)
    L = torch.linalg.cholesky(Sigma)
    logdet = 2 * torch.log(torch.diagonal(L)).sum()
    Si_b = torch.cholesky_solve(b[:, None], L).squeeze(-1)
    c = (b * Si_b).sum()
    out = []
    for s in range(0, u.shape[0], chunk):
        ub = u[s:s + chunk]
        w0 = F1_inv(ub, G.detach())
        r0 = w0.unsqueeze(-1) - G
        F1 = normal.cdf(r0).mean(-1)
        f1 = normal.log_prob(r0).exp().mean(-1)
        w = w0 - (F1 - F1.detach()) / f1.detach().clamp_min(1e-300)       # implicit gradient
        r = w.unsqueeze(-1) - G
        log_f1 = (torch.logsumexp(normal.log_prob(r), -1) - math.log(len(q))).sum(-1)
        Si_w = torch.cholesky_solve(w.T, L).T
        quad = (w * Si_w).sum(-1, keepdim=True) - 2 * q * (w @ Si_b)[:, None] + q ** 2 * c
        log_fn = torch.logsumexp(-0.5 * quad, -1) - math.log(len(q)) - 0.5 * logdet \
            - 0.5 * w.shape[1] * math.log(2 * math.pi)
        out.append(log_fn - log_f1)
    return torch.cat(out)


# =========================
# Unconstrained parameters: z = (beta (K), log sigma_beta, log kappa, log range, logit nugget)
# =========================
def unpack(z, K):
    return (z[:K], torch.exp(z[K]), torch.exp(z[K + 1]), torch.exp(z[K + 2]),
            torch.sigmoid(z[K + 3]))


def make_potential(u, coords, Psi, M, fixed=None):
    """fixed = (range, nugget) holds the Gaussian component fixed; their coordinates of z
    then only carry their prior."""
    K = Psi.shape[1]

    def potential(params):
        z = params["z"]
        beta, sb, kappa, ls, nug = unpack(z, K)
        if fixed is not None:
            ls, nug = fixed
        loglik = log_copula(u, Psi @ beta, nodes(kappa, M), gaussian_component(coords, ls, nug)).sum()
        logprior = (-0.5 * (beta ** 2).sum() / sb ** 2 - K * torch.log(sb)
                    - 0.5 * sb ** 2 / 4.0 + z[K]                          # HalfNormal(2) + Jacobian
                    + torch.log(kappa) - 0.5 * kappa + z[K + 1]           # Ga(2, 0.5) + Jacobian
                    - 0.5 * (z[K + 2] ** 2 + z[K + 3] ** 2) / 9.0)
        return -(loglik + logprior)

    return potential


def fit(args):
    data, coords = load_csv_fields(os.path.join(args.data_path, "sample_copula-{}.csv"),
                                   range(args.N))
    u = torch.tensor(np.clip(data.astype(np.float64), 1e-6, 1 - 1e-6), dtype=DT, device=device)
    c = torch.tensor(coords, dtype=DT, device=device)
    Psi = torch.tensor(basis(coords.astype(np.float64), args.basis_grid), dtype=DT, device=device)
    K = Psi.shape[1]
    fixed = ((torch.tensor(args.init_lengthscale, dtype=DT, device=device),
              torch.tensor(args.init_nugget, dtype=DT, device=device)) if args.fix_cov else None)
    potential = make_potential(u, c, Psi, args.M, fixed)
    N = u.shape[0]
    print(f"{args.data_path}: N={N}, n={c.shape[0]}, K={K}, M={args.M}"
          + (f", Gaussian component fixed (range {args.init_lengthscale}, nugget {args.init_nugget})"
             if fixed else ""))

    t0 = time.time()
    # start from a positive constant loading (b = 0 is a stationary point)
    z = torch.cat([torch.full((K,), 0.3, dtype=DT), torch.tensor(
        [0.0, math.log(2.0), math.log(args.init_lengthscale),
         math.log(args.init_nugget / (1 - args.init_nugget))], dtype=DT)]).to(device)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.02)
    for it in range(args.map_steps):
        opt.zero_grad()
        loss = potential({"z": z})
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            _, sb, kp, ls, ng = unpack(z.detach(), K)
            print(f"  mode search {it + 1}: -log post/N {loss.item() / N:.4f}, kappa {kp:.3f}, "
                  f"range {ls:.4f}, nugget {ng:.4f}")
    z0 = z.detach().clone()
    print(f"mode found in {time.time() - t0:.0f}s")

    kernel = NUTS(potential_fn=potential, full_mass=True, target_accept_prob=0.8,
                  max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.warmup,
                initial_params={"z": z0}, num_chains=1)
    mcmc.run()
    elapsed = time.time() - t0
    zs = mcmc.get_samples()["z"].cpu()
    diag = mcmc.diagnostics()["z"]
    beta = zs[:, :K]
    sb, kappa, ls, nug = (torch.exp(zs[:, K]), torch.exp(zs[:, K + 1]), torch.exp(zs[:, K + 2]),
                          torch.sigmoid(zs[:, K + 3]))
    if fixed is not None:
        ls, nug = torch.full_like(ls, args.init_lengthscale), torch.full_like(nug, args.init_nugget)
    os.makedirs(os.path.join(args.data_path, "models"), exist_ok=True)
    torch.save({"beta": beta, "sigma_beta": sb, "kappa": kappa, "lengthscale": ls, "nugget": nug,
                "basis_grid": args.basis_grid, "M": args.M, "mode": z0.cpu()},
               posterior_path(args.data_path, "pth"))
    summ = {"time_seconds": elapsed, "K": K, "N": N, "M": args.M,
            "min_ess": float(diag["n_eff"].min()), "max_rhat": float(diag["r_hat"].max()),
            **{f"{k}_{s}": float(f(v)) for k, v in (("kappa", kappa), ("lengthscale", ls),
                                                    ("nugget", nug), ("sigma_beta", sb))
               for s, f in (("mean", torch.mean), ("sd", torch.std))}}
    with open(posterior_path(args.data_path, "json"), "w") as f:
        json.dump({"config": vars(args), **summ}, f, indent=2)
    print(json.dumps(summ, indent=2))


def load_posterior(data_path):
    path = posterior_path(data_path, "pth")
    return torch.load(path) if os.path.exists(path) else None


def draw_params(post, coords, i):
    """Loading b (n,), nodes q (M,) and Sigma (n, n) of posterior draw i at coords."""
    c = torch.tensor(coords, dtype=DT, device=device)
    Psi = torch.tensor(basis(coords.astype(np.float64), post["basis_grid"]), dtype=DT, device=device)
    b = Psi @ post["beta"][i].to(DT).to(device)
    q = nodes(post["kappa"][i].to(DT), post["M"]).to(device)
    return b, q, gaussian_component(c, post["lengthscale"][i].to(DT).to(device),
                                    post["nugget"][i].to(DT).to(device))


# =========================
# Held-out log score
# =========================
@torch.no_grad()
def score(args):
    post = load_posterior(args.data_path)
    scen = os.path.basename(os.path.normpath(args.data_path))
    test = list(range(args.test_start, args.test_start + args.n_test_reps))
    idx = np.linspace(0, len(post["kappa"]) - 1, args.R).round().astype(int)
    rows = []
    for set_name, fname in (("observed", "sample_copula-{}.csv"),
                            ("full", "sample_copula_full-{}.csv")):
        u, coords = load_csv_fields(os.path.join(args.data_path, fname), test)
        u = torch.tensor(np.clip(u.astype(np.float64), 1e-6, 1 - 1e-6), dtype=DT, device=device)
        lc = torch.stack([log_copula(u, *draw_params(post, coords, i), chunk=50) for i in idx])
        sc = (torch.logsumexp(lc, 0) - math.log(len(idx))).cpu().numpy()
        print(f"{scen:22s} {set_name:8s} {NAME:8s} n={u.shape[1]:4d}  mean log score "
              f"{sc.mean():10.3f}  (se {sc.std(ddof=1) / math.sqrt(len(sc)):.3f})")
        rows += [{"scenario": scen, "set": set_name, "n_sites": u.shape[1], "model": NAME,
                  "replicate": t, "log_score": float(s)} for t, s in zip(test, sc)]
    out = os.path.join(args.out_dir, scen, NAME)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "heldout_loglik.csv"), index=False)


# =========================
# Posterior predictive simulation (Section 4.4 with G(v) = b H^{-1}(v))
# =========================
@torch.no_grad()
def predict(args):
    post = load_posterior(args.data_path)
    scen = os.path.basename(os.path.normpath(args.data_path))
    test_indices = list(range(args.test_start, args.test_start + args.n_test_reps))
    u_obs, coords_train, u_true, y_true, coords_test = load_scenario(args.data_path, test_indices)
    u_obs, coords_train = subset_conditioning(u_obs, coords_train, args.n_cond)
    F_inv = load_margin(args.data_path, args.margin, torch.tensor(coords_test, device=device))
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    idx = rng.choice(len(post["kappa"]), args.R, replace=len(post["kappa"]) < args.R)
    n = coords_train.shape[0]
    t0 = time.time()
    ops = []
    for i in idx:
        b, q, S = draw_params(post, np.vstack((coords_train, coords_test)), i)
        A = torch.linalg.solve(S[:n, :n], S[:n, n:]).T
        cond = S[n:, n:] - A @ S[:n, n:]
        cond = 0.5 * (cond + cond.T)
        ev = torch.linalg.eigvalsh(cond).min()
        if ev < 1e-6:
            cond = cond + (1e-6 - ev) * torch.eye(len(cond), dtype=DT, device=device)
        ops.append((b[:n, None] * q, b[n:, None] * q, torch.linalg.cholesky(S[:n, :n]), A,
                    torch.linalg.cholesky(cond)))
    u_all = torch.tensor(np.clip(u_obs.astype(np.float64), 1e-6, 1 - 1e-6), dtype=DT, device=device)

    U, X, Y, F = [], [], [], []
    for s in tqdm(range(0, len(test_indices), args.batch_size)):
        ub = u_all[s:s + args.batch_size]
        B = ub.shape[0]
        u_d, x_d, f_d = [], [], []
        for G_tr, G_te, L_tr, A, L_c in ops:
            w = F1_inv(ub, G_tr)                                           # (B, n)
            resid = w.unsqueeze(-1) - G_tr                                 # (B, n, M)
            white = torch.linalg.solve_triangular(L_tr, resid.permute(1, 0, 2).reshape(n, -1),
                                                  upper=False)
            logw = -0.5 * (white ** 2).sum(0).reshape(B, -1)               # (B, M)
            m = torch.multinomial(torch.softmax(logw, -1), 1, generator=gen).squeeze(-1)
            mean = G_te[:, m].T + (w - G_tr[:, m].T) @ A.T                 # (B, n_test)
            x0 = mean + torch.randn(B, A.shape[0], generator=gen, dtype=DT, device=device) @ L_c.T
            u_d.append(normal.cdf(x0.unsqueeze(-1) - G_te).mean(-1))
            x_d.append(x0)
            f_d.append((G_te[:, m] - G_te.mean(1, keepdim=True)).T)       # centred factor contribution
        u_d, x_d = torch.stack(u_d, 1), torch.stack(x_d, 1)              # (B, R, n_test)
        U.append(u_d.float().cpu().numpy())
        X.append(x_d.float().cpu().numpy())
        F.append(torch.stack(f_d, 1).float().cpu().numpy())
        if F_inv is not None:
            ut = u_d.float()
            Y.append(torch.stack([F_inv(ut[:, r]) for r in range(ut.shape[1])], 1).cpu().numpy())
    elapsed = time.time() - t0
    U, X = np.concatenate(U), np.concatenate(X)
    Y = np.concatenate(Y) if Y else np.full_like(U, np.nan)
    if args.y_min is not None:
        Y = np.maximum(Y, args.y_min)

    per_rep_copula = [replicate_metrics(U[k], u_true[k]) for k in range(len(test_indices))]
    per_rep_data = [replicate_metrics(Y[k], y_true[k]) for k in range(len(test_indices))]
    metrics = {
        "scenario": scen, "factor": NAME, "config": vars(args), "margin": args.margin,
        "n_cond": n, "test_indices": test_indices, "time_seconds": elapsed,
        "summary": {"copula": summarise(per_rep_copula), "data": summarise(per_rep_data)},
        "per_replicate": {"copula": per_rep_copula, "data": per_rep_data},
    }
    out = os.path.join(args.out_dir, scen, NAME)
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
    parser = argparse.ArgumentParser(description="M2: gamma factor with a spatially varying loading")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--N", type=int, default=2000)
    f.add_argument("--warmup", type=int, default=300)
    f.add_argument("--num_samples", type=int, default=300)
    f.add_argument("--max_tree_depth", type=int, default=7)
    f.add_argument("--map_steps", type=int, default=600)
    f.add_argument("--init_lengthscale", type=float, default=0.135)
    f.add_argument("--init_nugget", type=float, default=0.057)
    f.add_argument("--M", type=int, default=100, help="factor quantile nodes")
    f.add_argument("--fix_cov", action="store_true",
                   help="hold the range and nugget at --init_lengthscale/--init_nugget "
                        "(the simulations: 0.1 and 1e-4, as for the other models)")
    sc = sub.add_parser("score")
    p = sub.add_parser("predict")
    for q in (sc, p):
        q.add_argument("--out_dir", type=str, default="results_M1_M2/")
        q.add_argument("--test_start", type=int, default=2000)
        q.add_argument("--n_test_reps", type=int, default=100)
    sc.add_argument("--R", type=int, default=20)
    p.add_argument("--R", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--margin", choices=["nonstat", "stat"], default="nonstat")
    p.add_argument("--n_cond", type=int, default=None)
    p.add_argument("--y_min", type=float, default=None)
    for q in (f, sc, p):
        q.add_argument("--data_path", type=str, default="simulations_gamma/")
        q.add_argument("--basis_grid", type=int, default=5)
        q.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    {"fit": fit, "score": score, "predict": predict}[args.cmd](args)


if __name__ == "__main__":
    main()
