#!/usr/bin/env python
# coding: utf-8

"""
Posterior predictive (conditional) simulation for the parametric
single-factor copula models W(s) = Z(s) + V0 (exponential, gamma, pareto),
using the HMC posterior of the factor parameter.

Prediction follows exactly the same steps as Copula_BNFC.py so that the
models are compared on equal terms:

    vartheta ~ HMC posterior draws (train_model_single_factor_HMC_NUTS.py,
               fitted to the same scenario)
    v_l    ~ F_V0(gamma), l = 1..L               (common draws)
    w      = F1^{-1}(u_obs),  F1(w) = mean_l Phi(w - v_l)
    v*     ~ p(v | w)  via weights phi_Sigma(w - v_l 1)
    W_0    ~ N(kriging mean given v*, kriging covariance)
    U_0    = F1(W_0),   Y_0 = F_inv(U_0, s_0)     (data margin, see --margin)

The factor parameterisations are those of the HMC fit:
    exponential : Exponential(rate = vartheta)
    gamma       : Gamma(concentration = 6, rate = vartheta)
    pareto      : Pareto(scale = vartheta + 1, alpha = 2)
    mixgamma    : pi Gamma(2, rate = vartheta_1) + (1 - pi) Gamma(20, rate = vartheta_2)

Results are written in the same format as Copula_BNFC.py and plotted by
plot_BNFC_results.py.
"""

# =========================
# Imports
# =========================
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist

from tqdm import tqdm

from Copula_BNFC import (
    device,
    load_scenario,
    load_margin,
    subset_conditioning,
    kriging_operators,
    replicate_metrics,
    summarise,
)

from train_model_single_factor_HMC_NUTS import GAMMA_SHAPE, PARETO_ALPHA, MIXGAMMA_SHAPES

PARETO_SCALE_SHIFT = 1.0


# =========================
# Factor distributions
# =========================
def factor_distribution(factor, gamma):
    if factor == "exponential":
        return dist.Exponential(gamma)
    if factor == "gamma":
        return dist.Gamma(torch.full_like(gamma, GAMMA_SHAPE), gamma)
    if factor == "pareto":
        return dist.Pareto(gamma + PARETO_SCALE_SHIFT, torch.full_like(gamma, PARETO_ALPHA))
    raise ValueError(f"Unknown factor: {factor}")


def sample_factor(factor, posterior, idx, L):
    """L factor draws for each of the posterior draws idx (B,). Returns (L, B)."""
    vartheta = posterior["vartheta"][idx]
    if factor != "mixgamma":
        return factor_distribution(factor, vartheta).sample((L,))
    pi = posterior["pi"][idx]                                            # (B,)
    second = torch.rand(L, idx.shape[0], device=device) >= pi            # component 2 w.p. 1 - pi
    shape = torch.where(second, MIXGAMMA_SHAPES[1], MIXGAMMA_SHAPES[0])
    rate = torch.where(second, vartheta[:, 1], vartheta[:, 0])
    return dist.Gamma(shape, rate).sample()


@torch.no_grad()
def F1_inv_common(u, v, tol=1e-4, max_iter=60):
    """
    Bisection for F1(w) = u with F1(w) = mean_l Phi(w - v_l).

    u: (B, n), v: (L, B). The bracket [min v - 10, max v + 10] always
    contains the root.
    """
    normal = dist.Normal(0., 1.)
    low = (v.amin(0) - 10.).unsqueeze(-1).expand_as(u).clone()
    high = (v.amax(0) + 10.).unsqueeze(-1).expand_as(u).clone()
    for _ in range(max_iter):
        mid = (low + high) / 2
        val = normal.cdf(mid.unsqueeze(0) - v[:, :, None]).mean(0)
        low = torch.where(val < u, mid, low)
        high = torch.where(val >= u, mid, high)
        if (high - low).max() < tol:
            break
    return (low + high) / 2


# =========================
# Posterior predictive sampler
# =========================
@torch.no_grad()
def posterior_predictive(u_obs, factor, posterior, F_inv, kriging, R, L):
    """
    u_obs: (B, n). posterior: dict of HMC draws. F_inv: u_test (B, n_test) -> y_test.
    Returns latent, copula and data-scale draws, each (B, R, n_test), and the
    posterior draws of the centred factor v* - E V0, (B, R, 1), the same at every site.
    """
    L_train, A, L_cond = kriging
    B, n = u_obs.shape
    n_test = A.shape[0]
    normal = dist.Normal(0., 1.)

    w_draws = torch.empty(B, R, n_test, device=device)
    f_draws = torch.empty(B, R, 1, device=device)
    u_draws = torch.empty(B, R, n_test, device=device)
    y_draws = torch.empty(B, R, n_test, device=device)

    for r in range(R):
        # one posterior draw of the factor parameter per replicate
        idx = torch.randint(0, posterior["vartheta"].shape[0], (B,), device=device)
        v = sample_factor(factor, posterior, idx, L)                     # (L,B)

        w_obs = F1_inv_common(u_obs.clamp(1e-6, 1 - 1e-6), v)                                  # (B,n)

        # p(v | w): weights phi_Sigma(w - v 1)
        resid = w_obs.unsqueeze(0) - v[:, :, None]                       # (L,B,n)
        white = torch.linalg.solve_triangular(
            L_train, resid.reshape(-1, n).T, upper=False
        ).T.reshape(L, B, n)
        log_w = -0.5 * (white ** 2).sum(-1)                              # (L,B)
        star = torch.distributions.Categorical(logits=log_w.T).sample()  # (B,)
        v_star = v[star, torch.arange(B, device=device)]                 # (B,)

        cond_mean = v_star[:, None] + (w_obs - v_star[:, None]) @ A.T    # (B,n_test)
        w_test = cond_mean + torch.randn(B, n_test, device=device) @ L_cond.T

        u_test = normal.cdf(w_test.unsqueeze(0) - v[:, :, None]).mean(0)
        y_test = F_inv(u_test)

        w_draws[:, r] = w_test
        u_draws[:, r] = u_test
        y_draws[:, r] = y_test
        f_draws[:, r, 0] = v_star - v.mean(0)

    return w_draws, u_draws, y_draws, f_draws


# =========================
# One scenario / factor
# =========================
def run_scenario(data_path, factor, args):
    name = os.path.basename(os.path.normpath(data_path))
    post_path = os.path.join(data_path, "models", f"hmc_posterior-{factor}.pth")
    if not os.path.exists(post_path):
        print(f"[skip] {name}/{factor}: missing {post_path}")
        return None

    test_indices = list(range(args.test_start, args.test_start + args.n_test_reps))
    u_obs, coords_train, u_true, y_true, coords_test = load_scenario(data_path, test_indices)
    u_obs, coords_train = subset_conditioning(u_obs, coords_train, args.n_cond)

    ckpt = torch.load(post_path, map_location=device)
    posterior = {k: ckpt[k].to(device).float() for k in ("vartheta", "pi") if k in ckpt}

    print(f"\n=== {name} / {factor} ===")
    print(f"posterior: {post_path} ({posterior['vartheta'].shape[0]} draws, " + ", ".join(
        f"mean {k} = {v.mean(0).cpu().numpy().round(4)}" for k, v in posterior.items()) + ")")

    ct = torch.tensor(coords_train, device=device)
    cs = torch.tensor(coords_test, device=device)
    # data margin: the common marginal network (stationary by default for --margin own)
    margin = "stat" if args.margin == "own" else args.margin
    F_inv = load_margin(data_path, margin, cs)
    if F_inv is None:
        print(f"[skip] {name}/{factor}: missing F_inv-{margin}.pth")
        return None
    # Gaussian component used in the HMC fit (fixed SE correlation if not recorded)
    gc = ckpt.get("gaussian_component", {"kernel": "fixed"})
    cov_params = None if gc["kernel"] == "fixed" else (
        torch.tensor(gc["lengthscale"], device=device),
        torch.tensor(gc["nugget"], device=device), gc["kernel"])
    kriging = kriging_operators(ct, cs, cov_params=cov_params)

    W, U, Y, G = [], [], [], []
    t0 = time.time()
    for start in tqdm(range(0, len(test_indices), args.batch_size)):
        u_b = torch.tensor(u_obs[start:start + args.batch_size], device=device)
        w_d, u_d, y_d, g_d = posterior_predictive(u_b, factor, posterior, F_inv, kriging,
                                                  R=args.R, L=args.L)
        G.append(g_d.cpu().numpy())
        W.append(w_d.cpu().numpy())
        U.append(u_d.cpu().numpy())
        Y.append(y_d.cpu().numpy())
    elapsed = time.time() - t0

    W = np.concatenate(W).astype(np.float32)
    U = np.concatenate(U).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32)
    G = np.concatenate(G).astype(np.float32)
    if args.y_min is not None:                   # e.g. 0 for precipitation
        Y = np.maximum(Y, args.y_min)

    per_rep_copula = [replicate_metrics(U[k], u_true[k]) for k in range(len(test_indices))]
    per_rep_data = [replicate_metrics(Y[k], y_true[k]) for k in range(len(test_indices))]

    metrics = {
        "scenario": name,
        "factor": factor,
        "config": vars(args),
        "posterior_file": post_path,
        "margin": margin,
        "n_cond": int(coords_train.shape[0]),
        "factor_settings": {"gamma_shape": GAMMA_SHAPE, "pareto_alpha": PARETO_ALPHA,
                            "pareto_scale_shift": PARETO_SCALE_SHIFT,
                            "mixgamma_shapes": list(MIXGAMMA_SHAPES)},
        "test_indices": test_indices,
        "time_seconds": elapsed,
        "summary": {"copula": summarise(per_rep_copula), "data": summarise(per_rep_data)},
        "per_replicate": {"copula": per_rep_copula, "data": per_rep_data},
    }

    out = os.path.join(args.out_dir, name, factor)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "predictions.npz"),
        latent_samples=W, copula_samples=U, data_samples=Y, factor_samples=G,
        u_obs=u_obs, u_true=u_true, y_true=y_true,
        coords_train=coords_train, coords_test=coords_test,
        test_indices=np.array(test_indices),
    )
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    s = metrics["summary"]
    print(f"time: {elapsed:.1f}s")
    for scale in ("copula", "data"):
        print(f"[{scale}] " + ", ".join(
            f"{k}={s[scale][k]['mean']:.4f}({s[scale][k]['se']:.4f})"
            for k in ("mae", "rmse", "crps", "energy_score", "coverage95")))
    print(f"saved to {out}")
    return metrics


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Single-factor copula posterior predictive simulation")
    parser.add_argument("--scenarios", nargs="+",
                        default=["simulations_gamma/", "simulations_BNFC/"])
    parser.add_argument("--factors", nargs="+",
                        choices=["exponential", "gamma", "pareto", "mixgamma"],
                        default=["exponential", "gamma", "pareto", "mixgamma"])
    parser.add_argument("--out_dir", type=str, default="results/")
    parser.add_argument("--test_start", type=int, default=2000)
    parser.add_argument("--n_test_reps", type=int, default=100)
    parser.add_argument("--R", type=int, default=100,
                        help="posterior predictive draws per replicate")
    parser.add_argument("--L", type=int, default=200,
                        help="factor draws v per parameter draw")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--margin", choices=["nonstat", "stat", "own"], default="nonstat",
                        help="data margin: the location-dependent or stationary marginal network "
                             "for every model, or 'own' (stationary here)")
    parser.add_argument("--n_cond", type=int, default=None,
                        help="condition the prediction on a fixed subset of this many observed "
                             "sites (default: all); the models are not refitted")
    parser.add_argument("--y_min", type=float, default=None,
                        help="lower bound of the data scale (predictions are floored at it)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = []
    for data_path in args.scenarios:
        for factor in args.factors:
            m = run_scenario(data_path, factor, args)
            if m is None:
                continue
            row = {"scenario": m["scenario"], "factor": factor, "time_seconds": m["time_seconds"]}
            for scale in ("copula", "data"):
                for k, v in m["summary"][scale].items():
                    row[f"{scale}_{k}"] = v["mean"]
                    row[f"{scale}_{k}_se"] = v["se"]
            rows.append(row)

    if rows:
        os.makedirs(args.out_dir, exist_ok=True)
        summary_path = os.path.join(args.out_dir, "summary.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(summary_path):
            old = pd.read_csv(summary_path)
            keep = ~old.set_index(["scenario", "factor"]).index.isin(
                new.set_index(["scenario", "factor"]).index)
            new = pd.concat([old[keep], new], ignore_index=True)
        new.to_csv(summary_path, index=False)
        print(f"\nsummary written to {summary_path}")


if __name__ == "__main__":
    main()
