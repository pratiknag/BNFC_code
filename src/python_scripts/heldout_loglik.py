#!/usr/bin/env python
# coding: utf-8

"""
Held-out copula log-likelihood (log score) of the test replicates under every
fitted model: BNFC (stat, nonstat) and the common factor copula models
(exponential, gamma, pareto, mixgamma). Nothing is refitted.

For a test replicate with copula data u at locations s_1..s_n,

    log c_n(u) = log f_n(w) - sum_j log f_{1,j}(w_j),   w_j = F_{1,j}^{-1}(u_j),

with the factor integrals evaluated on the same deterministic grid for every
model: K quantile nodes of the factor with weights p_k (for BNFC, the nodes
g(v_k, s), v_k the K midpoints of U(theta_1, theta_2)), so the approximation is
itself a factor copula (Section 4.2 of the manuscript). The score averages the copula density
over R posterior draws of the parameters,

    log score = log { (1/R) sum_r c_n(u | eta_r) },

which is the log posterior predictive density of the copula data. Higher is
better. For BNFC (stat, nonstat), the R draws are split evenly between the
pSGLD weight samples, so the score also averages over the posterior of the
network weights. The
Student-t copula (tcopula, gsm_t_copula.py) has a closed-form density and is
scored exactly. Two sets of locations are scored: the n observed sites
(sample_copula-<i>.csv) and the full grid (sample_copula_full-<i>.csv), which
also tests the copula at locations not used in fitting.

Output: <out_dir>/heldout_loglik.csv (one row per scenario, model, set and
replicate) and heldout_loglik_summary.csv.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_model_single_factor_HMC_NUTS as H
from utils import VecchiaFactor
import gsm_t_copula as T
from Copula_BNFC import load_csv_fields, load_models, load_weight_samples, device
from Train_BNFC import build_covariance, sampler_means

MODELS = ["stat", "nonstat", "exponential", "gamma", "pareto", "mixgamma", "tcopula"]
DT = torch.float64


# =========================
# Factor nodes per posterior draw: G (K, n) and log weights (K,)
# =========================
@torch.no_grad()
def bnfc_nodes(data_path, factor, coords, R, K, seed, weights="sgmcmc"):
    """
    R draws of theta, split evenly between the weight samples. Each entry is
    (G, logp, cov_params of the draw); the last is None unless the range is variational.
    """
    net, _, loc, log_scale, cov_params = load_models(data_path, factor)
    nets = load_weight_samples(data_path, factor, net, weights)
    if R % len(nets):
        raise SystemExit(f"R={R} is not divisible by {len(nets)} weight samples")
    g = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(R, loc.shape[0], generator=g, device=device)
    thetas = torch.exp(loc + torch.exp(log_scale) * eps)            # LogNormal q(theta)
    vb_range = loc.shape[0] >= 3
    mc_u = ((torch.arange(K, device=device) + 0.5) / K).view(1, K, 1, 1)
    c = torch.tensor(coords, dtype=torch.float32, device=device)[None]
    out = []
    for r in range(R):
        nk = nets[r % len(nets)]
        G = sampler_means(c, thetas[r].view(1, 1, -1), nk, factor, mc_u)[0, :, 0, :]
        cp = (thetas[r, 2], cov_params[1], cov_params[2]) if vb_range else None
        out.append((G.to(DT), torch.full((K,), -math.log(K), dtype=DT, device=device), cp))
    return out, cov_params


@torch.no_grad()
def hmc_nodes(data_path, factor, n, R, K):
    ck = torch.load(os.path.join(data_path, "models", f"hmc_posterior-{factor}.pth"),
                    map_location=device)
    draws = {k: ck[k].to(device).to(DT) for k in ("vartheta", "pi") if k in ck}
    S = draws["vartheta"].shape[0]
    q = H.base_quantiles(factor, K).to(DT)
    out = []
    for i in np.linspace(0, S - 1, R).round().astype(int):
        v, logp = H.factor_nodes(factor, q, {k: d[i] for k, d in draws.items()})
        out.append((v[:, None].expand(-1, n), logp, None))
    gc = ck.get("gaussian_component", {"kernel": "fixed"})
    return out, gc


# =========================
# Copula log-density on a factor grid
# =========================
@torch.no_grad()
def log_copula(u, G, logp, L, logdet, chunk=10, tol=1e-9, max_iter=80, vecchia=None):
    """
    u: (N, n), G: (K, n), logp: (K,), L: Cholesky of Sigma. Returns (N,).
    vecchia: None, or (VecchiaFactor, U) to replace L and logdet by the Vecchia
    approximation of the Gaussian component.
    """
    normal = torch.distributions.Normal(torch.zeros((), dtype=DT, device=device),
                                        torch.ones((), dtype=DT, device=device))
    p = logp.exp()
    n = u.shape[1]
    out = []
    for s in range(0, u.shape[0], chunk):
        ub = u[s:s + chunk]                                             # (B, n)
        low = (G.min(0).values - 10).expand_as(ub).clone()
        high = (G.max(0).values + 10).expand_as(ub).clone()
        for _ in range(max_iter):                                       # w = F_1^{-1}(u)
            mid = (low + high) / 2
            val = (normal.cdf(mid[:, :, None] - G.T[None]) * p).sum(-1)
            low = torch.where(val < ub, mid, low)
            high = torch.where(val >= ub, mid, high)
            if (high - low).max() < tol:
                break
        w = (low + high) / 2                                            # (B, n)
        r = w[:, :, None] - G.T[None]                                   # (B, n, K)
        log_f1 = torch.logsumexp(normal.log_prob(r) + logp, -1).sum(-1)  # (B,)
        if vecchia is None:
            resid = (w[:, None, :] - G[None]).reshape(-1, n).T          # (n, B*K)
            white = torch.linalg.solve_triangular(L, resid, upper=False)
            quad = (white ** 2).sum(0).reshape(ub.shape[0], -1)         # (B, K)
            ld = logdet
        else:
            quad, ld = vecchia[0].quad_logdet(w[:, None, :] - G[None], vecchia[1])
        log_fn = torch.logsumexp(-0.5 * quad + logp, -1) - 0.5 * ld \
            - 0.5 * n * math.log(2 * math.pi)
        out.append(log_fn - log_f1)
    return torch.cat(out)


def gaussian_component(coords, cov_params):
    c = torch.tensor(coords, dtype=torch.float32, device=device)[None]
    Sigma = build_covariance(torch.ones(1, 2, device=device), c, cov_params)[0].to(DT)
    L = torch.linalg.cholesky(Sigma)
    return L, 2 * torch.log(torch.diagonal(L)).sum()


def hmc_cov_params(gc):
    if gc["kernel"] == "fixed":
        return None
    return (torch.tensor(gc["lengthscale"], device=device),
            torch.tensor(gc["nugget"], device=device), gc["kernel"])


# =========================
# Vecchia approximation: accuracy and cost (online supplement)
# =========================
def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_vecchia(args):
    """
    Exact against Vecchia evaluation of the copula log-density of the fitted BNFC
    (nonstat) model of args.scenarios[0], with the posterior median of theta and
    its estimated Gaussian component:
      (a) on regular grids of n = g^2 locations, with replicates simulated from
          the fitted model, since the network can be evaluated at any location;
      (b) on the test replicates of the data at all grid locations.
    Writes vecchia_benchmark.csv and vecchia_benchmark.png to args.out_dir.
    """
    data_path = args.scenarios[0]
    scen = os.path.basename(os.path.normpath(data_path))
    net, _, loc, _, cov_params = load_models(data_path, "nonstat")
    theta = torch.exp(loc).view(1, 1, -1)                            # posterior median
    kern = ("fixed", 0.1, 1e-4) if cov_params is None else \
        (cov_params[2], float(cov_params[0]), float(cov_params[1]))
    K = args.K
    mc_u = ((torch.arange(K, device=device) + 0.5) / K).view(1, K, 1, 1)
    logp = torch.full((K,), -math.log(K), dtype=DT, device=device)
    normal = torch.distributions.Normal(torch.zeros((), dtype=DT, device=device),
                                        torch.ones((), dtype=DT, device=device))
    rng = torch.Generator(device=device).manual_seed(args.seed)

    def nodes(coords):
        c = torch.tensor(coords, dtype=torch.float32, device=device)[None]
        return sampler_means(c, theta, net, "nonstat", mc_u)[0, :, 0, :].to(DT)   # (K, n)

    def evaluate(u, coords, G, label, n_rep):
        rows = []
        sync(); t0 = time.time()
        L, logdet = gaussian_component(coords, cov_params)
        exact = log_copula(u, G, logp, L, logdet)
        sync(); t_exact = (time.time() - t0) / n_rep
        del L
        for m in args.vecchia_m:
            sync(); t0 = time.time()
            vf = VecchiaFactor(coords, m, device)
            build = time.time() - t0
            U = vf.factor(kern[0], kern[1], kern[2], dtype=DT)
            approx = log_copula(u, G, logp, None, None, vecchia=(vf, U))
            sync(); t_v = (time.time() - t0 - build) / n_rep
            d = (approx - exact) / u.shape[1]
            rows.append({"set": label, "n": u.shape[1], "m": m, "time_exact": t_exact,
                         "time_vecchia": t_v, "ordering_seconds": build,
                         "mean_diff_per_location": float(d.mean()),
                         "mean_abs_diff_per_location": float(d.abs().mean()),
                         "exact_log_score_per_location": float(exact.mean() / u.shape[1])})
            print(rows[-1])
        return rows

    rows = []
    # (a) simulated replicates on grids of increasing size
    for g in args.grids:
        x = np.linspace(0, 1, g)
        xv, yv = np.meshgrid(x, x)
        coords = np.column_stack((xv.ravel(), yv.ravel())).astype(np.float32)
        G = nodes(coords)
        L, _ = gaussian_component(coords, cov_params)
        k = torch.randint(0, K, (args.n_bench,), generator=rng, device=device)
        z = torch.randn(len(coords), args.n_bench, generator=rng, device=device, dtype=DT)
        W = G[k] + (L @ z).T                                           # (n_bench, n)
        del L
        u = (normal.cdf(W[:, :, None] - G.T[None]) * logp.exp()).sum(-1).clamp(1e-6, 1 - 1e-6)
        rows += evaluate(u, coords, G, "simulated", args.n_bench)
    # (b) the test replicates of the data at all grid locations
    test = list(range(args.test_start, args.test_start + args.n_test_reps))
    u, coords = load_csv_fields(os.path.join(data_path, "sample_copula_full-{}.csv"), test)
    u = torch.tensor(u, dtype=DT, device=device).clamp(1e-6, 1 - 1e-6)
    rows += evaluate(u, coords, nodes(coords), "data", len(test))

    # (c) one ELBO gradient step (forward and backward) of BNFC (nonstat) on grids of
    #     n locations, with the Monte Carlo sizes of the complete-data fit
    from Train_BNFC import nested_mc_loss
    from utils import NNFactorWithLocations
    step_rows = []
    for g in args.step_grids:
        x = np.linspace(0, 1, g)
        xv, yv = np.meshgrid(x, x)
        coords = np.column_stack((xv.ravel(), yv.ravel())).astype(np.float32)
        c = torch.tensor(coords, device=device)[None]
        u = torch.rand(args.step_batch, len(coords), generator=rng, device=device)
        for m in [0] + list(args.vecchia_m):
            netk = NNFactorWithLocations(64).to(device)
            ls = torch.tensor(np.log(0.15), device=device, requires_grad=True)
            nug = torch.tensor(0.0, device=device, requires_grad=True)
            loc_ = torch.tensor([0.0, math.log(2.0)], device=device, requires_grad=True)
            lsc = torch.full((2,), -5.0, device=device, requires_grad=True)
            vf = VecchiaFactor(coords, m, device) if m > 0 else None
            row = {"n": len(coords), "m": m}
            try:
                times = []
                torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
                for it in range(args.step_reps + 1):
                    sync(); t0 = time.time()
                    with torch.enable_grad():
                        loss = nested_mc_loss(u, c, loc_, lsc, netk, torch.tensor(4.0, device=device),
                                              "nonstat", M1=100, M2=args.step_M2, num_factor=1, N=10591,
                                              cov_params=(torch.exp(ls), torch.sigmoid(nug - 4.0), "se"),
                                              vecchia=vf)
                        loss.backward()
                    sync()
                    if it > 0:                         # the first step warms up the GPU
                        times.append(time.time() - t0)
                row["step_seconds"] = float(np.mean(times))
                row["peak_memory_gb"] = (torch.cuda.max_memory_allocated() / 1e9
                                         if device.type == "cuda" else float("nan"))
            except torch.OutOfMemoryError:
                row["step_seconds"], row["peak_memory_gb"] = float("nan"), float("nan")
                row["note"] = "out of memory"
            torch.cuda.empty_cache() if device.type == "cuda" else None
            print(row)
            step_rows.append(row)
    pd.DataFrame(step_rows).to_csv(os.path.join(args.out_dir, "vecchia_benchmark_training.csv"),
                                   index=False)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "vecchia_benchmark.csv"), index=False)

    sim = df[df.set == "simulated"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    ex = sim.groupby("n").time_exact.first()
    axes[0].plot(ex.index, ex.values, "o-", color="#52514e", lw=2.4, ms=8, label="Exact")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#7a5cc7"]
    for c, (m, d) in zip(colors, sim.groupby("m")):
        axes[0].plot(d.n, d.time_vecchia, "o-", color=c, lw=2.4, ms=8, label=f"Vecchia, $m={m}$")
        axes[1].plot(d.n, d.mean_abs_diff_per_location, "o-", color=c, lw=2.4, ms=8,
                     label=f"$m={m}$")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of locations $n$")
        ax.grid(color="#e6e5e1", which="both", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Seconds per replicate")
    axes[0].set_title("Copula log-density: cost")
    axes[1].set_ylabel("Mean absolute error per location")
    axes[1].set_title("Vecchia against exact")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    path = os.path.join(args.out_dir, "vecchia_benchmark.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)
    pd.set_option("display.width", 200)
    print(df.round(5).to_string(index=False))


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Held-out copula log-likelihood of the test replicates")
    parser.add_argument("--scenarios", nargs="+",
                        default=["simulations_gamma/", "simulations_signchange/", "simulations_mixgamma/"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sets", nargs="+", choices=["observed", "full"], default=["observed", "full"])
    parser.add_argument("--test_start", type=int, default=2000)
    parser.add_argument("--n_test_reps", type=int, default=100)
    parser.add_argument("--R", type=int, default=20, help="posterior draws per model")
    parser.add_argument("--weights", choices=["sgmcmc", "map"], default="sgmcmc",
                        help="BNFC: average over the pSGLD weight samples, or the fitted network")
    parser.add_argument("--K", type=int, default=1000, help="factor nodes")
    parser.add_argument("--out_dir", type=str, default="results/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vecchia", type=int, default=0,
                        help="score with the Vecchia approximation of the Gaussian component "
                             "with this many nearest neighbours (0 = exact)")
    parser.add_argument("--benchmark_vecchia", action="store_true",
                        help="compare exact and Vecchia evaluation of the BNFC (nonstat) copula "
                             "density of the first scenario (see benchmark_vecchia)")
    parser.add_argument("--vecchia_m", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--grids", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96],
                        help="benchmark: grids of g x g locations")
    parser.add_argument("--n_bench", type=int, default=5,
                        help="benchmark: simulated replicates per grid")
    parser.add_argument("--step_grids", type=int, nargs="+", default=[10, 16, 24, 32],
                        help="benchmark: grids of g x g locations for the ELBO step timing")
    parser.add_argument("--step_batch", type=int, default=4,
                        help="benchmark: replicates per mini-batch in the ELBO step timing")
    parser.add_argument("--step_M2", type=int, default=5,
                        help="benchmark: outer Monte Carlo draws in the ELBO step timing")
    parser.add_argument("--step_reps", type=int, default=3)
    args = parser.parse_args()
    print(args)
    if args.benchmark_vecchia:
        benchmark_vecchia(args)
        return

    test = list(range(args.test_start, args.test_start + args.n_test_reps))
    rows = []
    for data_path in args.scenarios:
        scen = os.path.basename(os.path.normpath(data_path))
        for set_name in args.sets:
            fname = "sample_copula-{}.csv" if set_name == "observed" else "sample_copula_full-{}.csv"
            u, coords = load_csv_fields(os.path.join(data_path, fname), test)
            u = torch.tensor(u, dtype=DT, device=device).clamp(1e-6, 1 - 1e-6)
            n = u.shape[1]
            for model in args.models:
                if model == "tcopula":
                    score = T.heldout_log_score(data_path, u.cpu().numpy(), coords, args.R)
                    if score is None:
                        print(f"[skip] {scen}/{model}")
                        continue
                elif model in ("stat", "nonstat"):
                    nodes, cov_params = bnfc_nodes(data_path, model, coords, args.R, args.K,
                                                   args.seed, args.weights)
                else:
                    if not os.path.exists(os.path.join(data_path, "models", f"hmc_posterior-{model}.pth")):
                        print(f"[skip] {scen}/{model}")
                        continue
                    nodes, gc = hmc_nodes(data_path, model, n, args.R, args.K)
                    cov_params = hmc_cov_params(gc)
                if model != "tcopula":
                    if args.vecchia > 0:
                        vf = VecchiaFactor(coords, args.vecchia, device)
                        U = (vf.factor("fixed", dtype=DT) if cov_params is None else
                             vf.factor(cov_params[2], cov_params[0], cov_params[1], dtype=DT))
                        L, logdet, vec = None, None, (vf, U)
                    else:
                        L, logdet = gaussian_component(coords, cov_params)
                        vec = None
                    if any(cp is not None for *_, cp in nodes) and args.vecchia > 0:
                        raise SystemExit("variational range: use --vecchia 0")
                    lc = torch.stack([log_copula(u, G, logp, L, logdet, vecchia=vec)
                                      if cp is None else
                                      log_copula(u, G, logp, *gaussian_component(coords, cp))
                                      for G, logp, cp in nodes])                  # (R, N)
                    score = (torch.logsumexp(lc, 0) - math.log(lc.shape[0])).cpu().numpy()
                print(f"{scen:22s} {set_name:8s} {model:12s} n={n:4d}  "
                      f"mean log score {score.mean():10.3f}  (se {score.std(ddof=1)/math.sqrt(len(score)):.3f})")
                rows += [{"scenario": scen, "set": set_name, "n_sites": n, "model": model,
                          "replicate": t, "log_score": float(sc)} for t, sc in zip(test, score)]

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "heldout_loglik.csv"), index=False)

    # paired differences to the best model per scenario and set
    summ = []
    for (scen, set_name), d in df.groupby(["scenario", "set"]):
        wide = d.pivot(index="replicate", columns="model", values="log_score")
        best = wide.mean().idxmax()
        for m in wide.columns:
            diff = wide[m] - wide[best]
            summ.append({"scenario": scen, "set": set_name, "n_sites": int(d.n_sites.iloc[0]),
                         "model": m, "mean_log_score": wide[m].mean(),
                         "se": wide[m].std(ddof=1) / math.sqrt(len(wide)),
                         "diff_to_best": diff.mean(),
                         "diff_se": diff.std(ddof=1) / math.sqrt(len(wide)) if m != best else 0.0,
                         "best": m == best})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out_dir, "heldout_loglik_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
