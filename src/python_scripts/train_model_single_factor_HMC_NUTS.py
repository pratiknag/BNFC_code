#!/usr/bin/env python
# coding: utf-8

"""
HMC/NUTS fit of the common factor copula model of Krupskii et al. (2018),

    W_j = Z_j + V0,   Z ~ N(0, Sigma),   V0 ~ F_V0(. ; vartheta),

with Sigma the squared exponential correlation (range 0.1, nugget 1e-4) used
by every model in the comparison, and one of four parametric factors:

    exponential : Exponential(rate = vartheta)
    gamma       : Gamma(shape = 6, rate = vartheta)
    pareto      : Pareto(scale = vartheta + 1, alpha = 2)
    mixgamma    : pi Gamma(shape = 2, rate = vartheta_1)
                  + (1 - pi) Gamma(shape = 20, rate = vartheta_2)

The first three are scale families, V0 = s(vartheta) Q with Q fixed, so the
factor integrals are evaluated with the K quantile nodes v_k = s(vartheta) q_k,
q_k = F_Q^{-1}((k - 0.5) / K), with equal weights 1/K. The gamma mixture uses
K/2 quantile nodes per component, v_k = q_k^(c) / vartheta_c, with weights
2 pi/K and 2 (1 - pi)/K, so that every factor is discretised with K nodes. The same nodes are used for F_1, f_1 and f_n (as with the
common random numbers of Section 4.2 of the manuscript), so the approximation is itself a valid
factor copula and the log-likelihood is a smooth deterministic function of the
factor parameters, as NUTS requires:

    log c_n(u) = log f_n(w) - sum_j log f_1(w_j),   w_j = F_1^{-1}(u_j),
    F_1(w) = sum_k p_k Phi(w - v_k),  f_1(w) = sum_k p_k phi(w - v_k),
    f_n(w) = sum_k p_k phi_Sigma(w - v_k 1).

w = F_1^{-1}(u) is found by bisection, refined by Newton steps so that the
log-likelihood is smooth in the parameters (NUTS rejects almost every proposal
if the value jitters with the bisection tolerance), and carries the
implicit-function gradient, as in Train_BNFC.nested_mc_loss.

Prior: vartheta ~ HalfNormal(prior_sigma) (each component for mixgamma), and
pi ~ U(0, 1) for mixgamma. Posterior draws are saved to
<data_path>/models/hmc_posterior-<factor>.pth.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as pdist
import torch
import torch.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_value
from scipy.stats import gamma as sp_gamma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA_SHAPE = 6.0
PARETO_ALPHA = 2.0
MIXGAMMA_SHAPES = (2.0, 20.0)


# ============================================================
# Factor quantile nodes: v_k = s(vartheta) * q_k
# ============================================================
def base_quantiles(factor, K):
    """(K,) unit-scale quantile nodes, or (2, K) for the two mixture components."""
    p = (np.arange(K) + 0.5) / K
    if factor == "exponential":
        q = -np.log1p(-p)                               # Exp(1)
    elif factor == "gamma":
        q = sp_gamma.ppf(p, a=GAMMA_SHAPE)              # Gamma(6, rate 1)
    elif factor == "pareto":
        q = (1 - p) ** (-1.0 / PARETO_ALPHA)            # Pareto(scale 1, alpha 2)
    elif factor == "mixgamma":
        p = (np.arange(K // 2) + 0.5) / (K // 2)        # K/2 nodes per component
        q = np.stack([sp_gamma.ppf(p, a=a) for a in MIXGAMMA_SHAPES])   # Gamma(a_c, rate 1)
    else:
        raise ValueError(f"Unknown factor: {factor}")
    return torch.tensor(q, dtype=torch.get_default_dtype(), device=device)


def factor_scale(factor, vartheta):
    if factor in ("exponential", "gamma"):
        return 1.0 / vartheta
    return vartheta + 1.0                               # pareto


def factor_nodes(factor, q, params):
    """
    Nodes v (K',) and log weights log p (K',) of the discretised factor.
    params: {"vartheta": scalar, or (2,) for mixgamma; "pi": scalar for mixgamma}.
    """
    if factor == "mixgamma":
        K = q.shape[-1]
        pi = params["pi"]
        v = (q / params["vartheta"][:, None]).reshape(-1)
        logp = torch.cat([torch.log(pi).expand(K), torch.log1p(-pi).expand(K)]) - math.log(K)
        return v, logp
    v = factor_scale(factor, params["vartheta"]) * q
    return v, torch.full_like(v, -math.log(q.shape[-1]))


# ============================================================
# Copula log-likelihood
# ============================================================
def correlation(coords, kernel, lengthscale, nugget):
    """
    Gaussian component. kernel = "fixed": the SE correlation (range 0.1,
    nugget 1e-4) of the simulations; "se" / "exponential": (1 - nugget) K + nugget I
    with the given range and nugget (e.g. those estimated by Train_BNFC --learn_cov).
    """
    if kernel == "fixed":
        return se_correlation(coords)
    d = torch.cdist(coords, coords)
    K = torch.exp(-d / lengthscale) if kernel == "exponential" \
        else torch.exp(-0.5 * d.pow(2) / lengthscale ** 2)
    return (1 - nugget) * K + nugget * torch.eye(coords.shape[0], device=coords.device)


def se_correlation(coords, lengthscale=0.1, nugget=1e-4):
    sqdist = torch.cdist(coords, coords).pow(2)
    K = torch.exp(-0.5 * sqdist / lengthscale ** 2)
    return K + nugget * torch.eye(coords.shape[0], device=coords.device)


class CopulaLik:
    """Precomputes the Gaussian-component quantities that do not depend on vartheta."""

    def __init__(self, u, coords, factor, K, cov=("fixed", 0.1, 1e-4)):
        eps = 1e-6
        self.u = u.clamp(eps, 1 - eps)                  # (N, n)
        self.factor = factor
        self.q = base_quantiles(factor, K)              # (K,) or (2, K)

        Sigma = correlation(coords, *cov)
        self.L = torch.linalg.cholesky(Sigma)
        self.logdet = 2 * torch.log(torch.diagonal(self.L)).sum()
        ones = torch.ones(coords.shape[0], 1, device=device)
        self.Si_1 = torch.cholesky_solve(ones, self.L).squeeze(-1)      # Sigma^{-1} 1
        self.c = (ones.squeeze(-1) * self.Si_1).sum()                   # 1' Sigma^{-1} 1

    @torch.no_grad()
    def F1_inv(self, v, p, tol=1e-5, max_iter=60, n_newton=2):
        normal = dist.Normal(0., 1.)
        low = torch.full_like(self.u, (v.min() - 10.).item())
        high = torch.full_like(self.u, (v.max() + 10.).item())
        for _ in range(max_iter):
            mid = (low + high) / 2
            val = (normal.cdf(mid.unsqueeze(-1) - v) * p).sum(-1)
            low = torch.where(val < self.u, mid, low)
            high = torch.where(val >= self.u, mid, high)
            if (high - low).max() < tol:
                break
        w = (low + high) / 2
        # Newton refinement: the error of w drops from tol to O(tol^2)
        for _ in range(n_newton):
            r = w.unsqueeze(-1) - v
            step = ((normal.cdf(r) * p).sum(-1) - self.u) / \
                (normal.log_prob(r).exp() * p).sum(-1).clamp_min(1e-30)
            w = w - torch.where(step.abs() < 10 * tol, step, torch.zeros_like(step))
        return w

    def log_lik(self, params):
        normal = dist.Normal(0., 1.)
        v, logp = factor_nodes(self.factor, self.q, params)            # (K,), (K,)
        p = logp.exp()

        # w = F1^{-1}(u) with the implicit-function gradient attached
        w0 = self.F1_inv(v.detach(), p.detach())
        r0 = w0.unsqueeze(-1) - v                                       # (N,n,K)
        F1 = (normal.cdf(r0) * p).sum(-1)
        f1 = (normal.log_prob(r0).exp() * p).sum(-1)
        w = w0 - (F1 - F1.detach()) / f1.detach().clamp_min(1e-12)

        # sum_j log f_1(w_j)
        resid = w.unsqueeze(-1) - v                                     # (N,n,K)
        log_f1 = torch.logsumexp(normal.log_prob(resid) + logp, -1).sum(-1)

        # log f_n(w): (w - v 1)' Sigma^{-1} (w - v 1) = a - 2 v b + v^2 c
        Si_w = torch.cholesky_solve(w.T, self.L).T                      # (N,n)
        a = (w * Si_w).sum(-1, keepdim=True)
        b = (w @ self.Si_1).unsqueeze(-1)
        quad = a - 2 * v * b + v ** 2 * self.c                          # (N,K)
        # the -n/2 log(2 pi) terms of f_n and of the n marginals cancel
        log_fn = torch.logsumexp(-0.5 * quad + logp, -1) - 0.5 * self.logdet
        log_f1 = log_f1 + 0.5 * w.shape[1] * math.log(2 * math.pi)

        # sum in double precision: float32 rounding over N replicates is visible to NUTS
        return (log_fn - log_f1).double().sum()


def map_init(lik, prior_sigma, steps=300, lr=0.05):
    """
    Posterior mode of the gamma mixture by Adam on the unconstrained scale
    (log vartheta, logit pi), used to initialise NUTS. The posterior is very
    concentrated, and NUTS started far from the mode spends most of the
    warmup with tiny step sizes and maximal trees.
    """
    z = torch.zeros(3, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    prior = pdist.HalfNormal(torch.tensor(prior_sigma, device=device))
    for _ in range(steps):
        opt.zero_grad()
        vartheta, pi = z[:2].exp(), torch.sigmoid(z[2])
        log_post = lik.log_lik({"vartheta": vartheta, "pi": pi}) \
            + prior.log_prob(vartheta).sum() + z[:2].sum() \
            + torch.log(pi) + torch.log1p(-pi)           # Jacobians of exp and sigmoid
        (-log_post).backward()
        opt.step()
    z = z.detach()
    return {"vartheta": z[:2].exp(), "pi": torch.sigmoid(z[2])}


def model(lik, prior_sigma):
    scale = torch.tensor(prior_sigma, device=device)
    if lik.factor == "mixgamma":
        params = {"vartheta": pyro.sample("vartheta", pdist.HalfNormal(scale).expand([2]).to_event(1)),
                  "pi": pyro.sample("pi", pdist.Uniform(torch.tensor(0., device=device),
                                                        torch.tensor(1., device=device)))}
    else:
        params = {"vartheta": pyro.sample("vartheta", pdist.HalfNormal(scale))}
    pyro.factor("log_copula", lik.log_lik(params))


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HMC/NUTS common factor copula")
    parser.add_argument("--data_path", type=str, default="simulations_gamma/",
                        help="directory with sample_copula-<i>.csv")
    parser.add_argument("--N", type=int, default=2000, help="training replicates")
    parser.add_argument("--factor", type=str, default="exponential",
                        choices=["exponential", "pareto", "gamma", "mixgamma"])
    parser.add_argument("--prior_sigma", type=float, default=4.0)
    parser.add_argument("--K", type=int, default=200, help="factor quantile nodes")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--kernel", choices=["fixed", "se", "exponential"], default="fixed",
                        help="Gaussian component: fixed SE (range 0.1) or the given range/nugget")
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--nugget", type=float, default=1e-4)
    parser.add_argument("--max_tree_depth", type=int, default=10)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                        help="precision of the log-likelihood; float32 rounding over N "
                             "replicates makes it too rough for NUTS for the gamma mixture")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(args)
    pyro.set_rng_seed(args.seed)
    torch.set_default_dtype(getattr(torch, args.dtype))

    data, coords = [], None
    for i in range(args.N):
        df = pd.read_csv(os.path.join(args.data_path, f"sample_copula-{i}.csv"))
        data.append(df["y"].values)
        if coords is None:
            coords = df[["x1", "x2"]].values
    u = torch.tensor(np.array(data), dtype=torch.get_default_dtype(), device=device)
    coords = torch.tensor(coords, dtype=torch.get_default_dtype(), device=device)
    print("Data shape:", tuple(u.shape))

    lik = CopulaLik(u, coords, args.factor, args.K,
                    cov=(args.kernel, args.lengthscale, args.nugget))

    t0 = time.time()
    if args.factor == "mixgamma":
        # three correlated parameters: start at the mode, dense mass matrix
        init = map_init(lik, args.prior_sigma)
        print("MAP initialisation:", {k: v.cpu().numpy().round(4) for k, v in init.items()})
        kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=args.max_tree_depth,
                      full_mass=True, init_strategy=init_to_value(values=init))
    else:
        kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.warmup)
    mcmc.run(lik, args.prior_sigma)
    elapsed = time.time() - t0
    mcmc.summary()

    draws = {k: v.detach().cpu() for k, v in mcmc.get_samples().items()}
    out_dir = os.path.join(args.data_path, "models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({**draws, "factor": args.factor, "K": args.K,
                "gaussian_component": {"kernel": args.kernel, "lengthscale": args.lengthscale,
                                       "nugget": args.nugget}},
               os.path.join(out_dir, f"hmc_posterior-{args.factor}.pth"))
    summary = {f"{k}_mean": v.mean(0).tolist() for k, v in draws.items()}
    summary.update({f"{k}_sd": v.std(0).tolist() for k, v in draws.items()})
    with open(os.path.join(out_dir, f"hmc_posterior-{args.factor}.json"), "w") as f:
        json.dump({"config": vars(args), "time_seconds": elapsed, **summary}, f, indent=2)
    print(f"{summary}, time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
