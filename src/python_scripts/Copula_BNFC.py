#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, February 2025

"""
Posterior predictive (conditional) simulation for BNFC.

Uses the same building blocks as Train_BNFC.py (sampler_means, F1_inv,
build_covariance), and draws from the posterior predictive distribution
of Section 4.4 of the manuscript:

    theta ~ q(theta)
    w     = F1^{-1}(u_obs)                       (bisection, common draws v)
    v*    ~ p(v | w)  via self-normalised weights phi_Sigma(w - G(v))
    W_0   ~ N(kriging mean given v*, kriging covariance)
    U_0   = F1_hat(W_0)                          (same draws v)
    Y_0   = F_inv(U_0, s_0)                      (data-scale margin, see --margin)

Uncertainty about the network weights (--weights sgmcmc, the default): the R
draws are split evenly between the S pSGLD weight samples of Train_BNFC.py
(models/sgmcmc-<factor>.pth), so that the predictive distribution averages over
the posterior of the weights. With --weights map, or if no samples exist (e.g.
in the application), the fitted network (point estimate) is used.

Results (samples + metrics) are saved per scenario/factor (with --label, under
that name instead); plotting is done separately by plot_BNFC_results.py.
"""

# =========================
# Imports
# =========================
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist

from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from utils import *   # NNFactor, NNFactorWithLocations, F_Inv, F_Inv_WithLocs
from Train_BNFC import build_covariance, sampler_means, F1_inv

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Data loading
# =========================
def load_csv_fields(path_fmt, indices):
    """Returns values (R, n) and coords (n, 2) from a list of csv files."""
    values, coords = [], None
    for i in indices:
        df = pd.read_csv(path_fmt.format(i))
        values.append(df["y"].values)
        if coords is None:
            coords = np.column_stack((df["x1"].values, df["x2"].values))
    return np.array(values, dtype=np.float32), coords.astype(np.float32)


def load_scenario(data_path, test_indices):
    u_obs, coords_train = load_csv_fields(
        os.path.join(data_path, "sample_copula-{}.csv"), test_indices)
    u_true, coords_test = load_csv_fields(
        os.path.join(data_path, "sample_copula_full-{}.csv"), test_indices)
    y_true, coords_test_y = load_csv_fields(
        os.path.join(data_path, "sample_full-{}.csv"), test_indices)
    assert np.allclose(coords_test, coords_test_y, atol=1e-6), \
        "copula and data truth files use different coordinates"
    return u_obs, coords_train, u_true, y_true, coords_test


def subset_conditioning(u_obs, coords_train, n_cond, seed=2026):
    """
    Keeps n_cond of the observed sites for conditioning (the same sites for every
    model and replicate). The models are not refitted: only the prediction
    conditions on fewer sites. n_cond=None keeps all of them.
    """
    if n_cond is None or n_cond >= coords_train.shape[0]:
        return u_obs, coords_train
    keep = np.sort(np.random.default_rng(seed).choice(coords_train.shape[0], n_cond, replace=False))
    return u_obs[:, keep], coords_train[keep]


# =========================
# Models
# =========================
def model_paths(data_path, factor):
    m = os.path.join(data_path, "models")
    return {
        "sampler": os.path.join(m, f"VaNFC-{factor}.pth"),
        "F_inv": os.path.join(m, f"F_inv-{factor}.pth"),
        "params": os.path.join(m, f"posterior_params_{factor}.pth"),
    }


def load_weight_samples(data_path, factor, net, weights="sgmcmc"):
    """
    Networks for the pSGLD weight samples (models/sgmcmc-<factor>.pth), or [net],
    the fitted network, for weights="map" or if there are no samples.
    """
    path = os.path.join(data_path, "models", f"sgmcmc-{factor}.pth")
    if weights == "map" or not os.path.exists(path):
        if weights != "map":
            print(f"[warn] {path} not found: using the fitted network only")
        return [net]
    nets = []
    for sd in torch.load(path, map_location=device):
        nk = copy.deepcopy(net)
        nk.load_state_dict(sd)
        nk.eval()
        nets.append(nk)
    return nets


def load_models(data_path, factor, num_factor=1):
    paths = model_paths(data_path, factor)
    ckpt = torch.load(paths["params"], map_location=device)

    if factor == "stat":
        neural_sampler = NNFactor(num_factor, 32, 32).to(device)
        F_inv = F_Inv(hidden=64, K=16, beta=2.0).to(device)
    else:
        neural_sampler = NNFactorWithLocations(**ckpt.get("arch", {"hidden2": 64})).to(device)
        F_inv = None

    neural_sampler.load_state_dict(torch.load(paths["sampler"], map_location=device))
    # the variant's own marginal network is optional: predictions use --margin
    if not os.path.exists(paths["F_inv"]):
        F_inv = None
    elif F_inv is None:
        F_inv = load_F_inv_withlocs(paths["F_inv"], device).eval()
    else:
        F_inv.load_state_dict(torch.load(paths["F_inv"], map_location=device))
        F_inv.eval()

    neural_sampler.eval()
    # estimated Gaussian component (Train_BNFC --learn_cov), else the fixed one
    # (with a variational range, "lengthscale" is the median of q; the draws come
    # from the third coordinate of q(theta), see range_draws)
    cov_params = ((ckpt["lengthscale"].to(device), ckpt["nugget"].to(device),
                   ckpt.get("kernel", "se"))
                  if "lengthscale" in ckpt else None)
    return (neural_sampler, F_inv, ckpt["loc"].to(device), ckpt["log_scale"].to(device),
            cov_params)


def load_margin(data_path, margin, coords_test):
    """
    Marginal quantile network T_psi of the data, fitted by Train_BNFC.py
    independently of the copula. Returns u_test (B, n_test) -> y_test, or None.
    margin = "nonstat": location-dependent F_inv-nonstat.pth; "stat": F_inv-stat.pth.
    """
    path = os.path.join(data_path, "models", f"F_inv-{margin}.pth")
    if not os.path.exists(path):
        return None
    if margin == "stat":
        net = F_Inv(hidden=64, K=16, beta=2.0).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
    else:
        net = load_F_inv_withlocs(path, device)
    net.eval()
    if margin == "stat":
        return net
    return lambda u: net(u, coords_test.unsqueeze(0).expand(u.shape[0], -1, -1))


def range_draws(loc, log_scale, R, generator=None):
    """
    R draws of the range from the third coordinate of q(theta) if the range is
    variational (Train_BNFC --vb_range), else None.
    """
    if loc.shape[0] < 3:
        return None
    eps = torch.randn(R, generator=generator, device=loc.device)
    return torch.exp(loc[2] + torch.exp(log_scale[2]) * eps)


# =========================
# Kriging operators (Sigma does not depend on theta, see build_covariance)
# =========================
def kriging_operators(coords_train, coords_test, jitter=1e-6, cov_params=None):
    n = coords_train.shape[0]
    n_test = coords_test.shape[0]

    all_coords = torch.cat((coords_train, coords_test), dim=0).unsqueeze(0)
    K = build_covariance(torch.ones(1, 2, device=device), all_coords, cov_params)[0]

    C_train = K[:n, :n]
    C_test_train = K[n:, :n]
    C_test = K[n:, n:]

    L_train = torch.linalg.cholesky(C_train)

    # A = C_test_train C_train^{-1}
    A = torch.cholesky_solve(C_test_train.T, L_train).T            # (n_test, n)

    cond_cov = C_test - A @ C_test_train.T
    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    min_eig = torch.linalg.eigvalsh(cond_cov).min()
    if min_eig < jitter:
        cond_cov = cond_cov + (jitter - min_eig) * torch.eye(n_test, device=device)
    L_cond = torch.linalg.cholesky(cond_cov)

    return L_train, A, L_cond


# =========================
# Posterior predictive sampler
# =========================
@torch.no_grad()
def posterior_predictive(
    u_obs,            # (B, n)
    coords_train,     # (n, 2)
    coords_test,      # (n_test, 2)
    loc,
    log_scale,
    neural_sampler,
    F_inv,
    factor,
    kriging,
    R,
    L,
    num_factor=1,
):
    """
    kriging: the operators (L_train, A, L_cond), or a list of R of them, one per draw r
    (variational range).
    Returns latent, copula and data-scale draws, each (B, R, n_test), and the
    posterior draws of the centred factor contribution g(v*, s_0) - E g(V, s_0)
    at the test sites, (B, R, n_test), with E g(V, s_0) the mean over the L draws v.
    """
    krigs = kriging if isinstance(kriging, list) else [kriging] * R
    B, n = u_obs.shape
    n_test = coords_test.shape[0]

    ct = coords_train.unsqueeze(0).expand(B, -1, -1)
    cs = coords_test.unsqueeze(0).expand(B, -1, -1)

    q_theta = dist.LogNormal(loc, torch.exp(log_scale) + 1e-6)
    std_normal = dist.Normal(0., 1.)

    w_draws = torch.empty(B, R, n_test, device=device)
    f_draws = torch.empty(B, R, n_test, device=device)
    u_draws = torch.empty(B, R, n_test, device=device)
    y_draws = torch.empty(B, R, n_test, device=device)

    for r in range(R):
        L_train, A, L_cond = krigs[r]
        theta = q_theta.sample((1, B))                                   # (1,B,p)
        mc_u = torch.rand(1, L, B, num_factor, device=device)            # (1,L,B,1)

        # ---- latent data w = F1^{-1}(u_obs) under theta (same draws v) ----
        w_obs = F1_inv(u_obs.clamp(1e-6, 1 - 1e-6).unsqueeze(0), ct, theta,
                       neural_sampler, factor, mc_u)[0]                    # (B,n)

        mu_train = sampler_means(ct, theta, neural_sampler, factor, mc_u)[0]  # (L,B,n)
        mu_test = sampler_means(cs, theta, neural_sampler, factor, mc_u)[0]   # (L,B,n_test)

        # ---- p(v | w): self-normalised weights phi_Sigma(w - G(v)) ----
        resid = (w_obs.unsqueeze(0) - mu_train)                           # (L,B,n)
        white = torch.linalg.solve_triangular(
            L_train, resid.reshape(-1, n).T, upper=False
        ).T.reshape(L, B, n)
        log_w = -0.5 * (white ** 2).sum(-1)                               # (L,B)
        idx = torch.distributions.Categorical(logits=log_w.T).sample()    # (B,)

        b_idx = torch.arange(B, device=device)
        mu_tr_star = mu_train[idx, b_idx]                                 # (B,n)
        mu_te_star = mu_test[idx, b_idx]                                  # (B,n_test)

        # ---- kriging draw of W_0 given w and v* ----
        cond_mean = mu_te_star + (w_obs - mu_tr_star) @ A.T               # (B,n_test)
        eps = torch.randn(B, n_test, device=device)
        w_test = cond_mean + eps @ L_cond.T

        # ---- copula scale: F1_hat at test sites with the same draws ----
        u_test = std_normal.cdf(w_test.unsqueeze(0) - mu_test).mean(0)    # (B,n_test)

        # ---- data scale: learned marginal quantile function ----
        y_test = F_inv(u_test)

        w_draws[:, r] = w_test
        u_draws[:, r] = u_test
        y_draws[:, r] = y_test
        f_draws[:, r] = mu_te_star - mu_test.mean(0)

    return w_draws, u_draws, y_draws, f_draws


# =========================
# Metrics
# =========================
def crps_ensemble(samples, truth):
    """
    samples: (R, n), truth: (n,). Returns mean CRPS over sites.
    CRPS = E|X - y| - 0.5 E|X - X'|.
    """
    R = samples.shape[0]
    term1 = np.abs(samples - truth[None]).mean(0)
    xs = np.sort(samples, axis=0)
    w = (2 * np.arange(1, R + 1) - R - 1)[:, None]
    term2 = (2.0 / R ** 2) * (w * xs).sum(0)        # = E|X - X'|
    return float(np.mean(term1 - 0.5 * term2))


def energy_score(samples, truth):
    """Multivariate energy score of the (R, n) field draws against the true field."""
    t1 = cdist(samples, truth[None]).mean()
    t2 = cdist(samples, samples).mean()
    return float(t1 - 0.5 * t2)


def replicate_metrics(samples, truth, level=0.95):
    mean = samples.mean(0)
    sd = samples.std(0)
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return {
        "mae": float(np.mean(np.abs(mean - truth))),
        "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "crps": crps_ensemble(samples, truth),
        "energy_score": energy_score(samples, truth),
        "coverage95": float(np.mean((truth >= lo) & (truth <= hi))),
        "mean_sd": float(np.mean(sd)),
        # distributional shape of the predicted vs true field (pooled over sites)
        "skewness_pred": float(skew(samples.ravel())),
        "skewness_true": float(skew(truth)),
        "kurtosis_pred": float(kurtosis(samples.ravel(), fisher=False)),
        "kurtosis_true": float(kurtosis(truth, fisher=False)),
    }


def summarise(per_rep):
    keys = per_rep[0].keys()
    out = {}
    for k in keys:
        v = np.array([m[k] for m in per_rep])
        out[k] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v)))
                  if len(v) > 1 else 0.0}
    return out


# =========================
# One scenario / factor
# =========================
def run_scenario(data_path, factor, args):
    name = os.path.basename(os.path.normpath(data_path))
    paths = model_paths(data_path, factor)
    missing = [p for k, p in paths.items() if k != "F_inv" and not os.path.exists(p)]
    if missing:
        print(f"[skip] {name}/{factor}: missing {missing}")
        return None

    test_indices = list(range(args.test_start, args.test_start + args.n_test_reps))
    u_obs, coords_train, u_true, y_true, coords_test = load_scenario(data_path, test_indices)
    u_obs, coords_train = subset_conditioning(u_obs, coords_train, args.n_cond)
    neural_sampler, _, loc, log_scale, cov_params = load_models(data_path, factor, args.num_factor)
    nets = load_weight_samples(data_path, factor, neural_sampler, args.weights)
    if args.R % len(nets):
        raise SystemExit(f"R={args.R} is not divisible by {len(nets)} weight samples")

    print(f"\n=== {name} / {factor} ({len(nets)} weight sample(s)) ===")
    print(f"q(theta): loc={loc.cpu().numpy()}, log_scale={log_scale.cpu().numpy()}")
    print(f"test replicates: {len(test_indices)}, n_obs={u_obs.shape[1]}, n_test={u_true.shape[1]}")

    ct = torch.tensor(coords_train, device=device)
    cs = torch.tensor(coords_test, device=device)
    margin = factor if args.margin == "own" else args.margin
    F_inv = load_margin(data_path, margin, cs)
    if F_inv is None:
        print(f"[skip] {name}/{factor}: missing F_inv-{margin}.pth")
        return None
    R_net = args.R // len(nets)
    ranges = range_draws(loc, log_scale, args.R)
    if ranges is None:
        kriging = kriging_operators(ct, cs, cov_params=cov_params)
        krig_net = [kriging] * len(nets)
    else:
        ops = [kriging_operators(ct, cs, cov_params=(a, cov_params[1], cov_params[2]))
               for a in ranges]
        krig_net = [ops[k * R_net:(k + 1) * R_net] for k in range(len(nets))]
        print(f"variational range: median {ranges.median().item():.4f}, "
              f"sd {ranges.std().item():.2e} over {args.R} draws")
    if cov_params is not None:
        print(f"estimated Gaussian component ({cov_params[2]}): "
              f"lengthscale={cov_params[0].item():.4f}, nugget={cov_params[1].item():.4f}")

    W, U, Y, G = [], [], [], []
    t0 = time.time()
    for start in tqdm(range(0, len(test_indices), args.batch_size)):
        u_b = torch.tensor(u_obs[start:start + args.batch_size], device=device)
        # R / S draws for each weight sample
        draws = [posterior_predictive(
            u_b, ct, cs, loc, log_scale, net, F_inv, factor,
            kr, R=R_net, L=args.L, num_factor=args.num_factor,
        ) for net, kr in zip(nets, krig_net)]
        W.append(torch.cat([d[0] for d in draws], 1).cpu().numpy())
        U.append(torch.cat([d[1] for d in draws], 1).cpu().numpy())
        Y.append(torch.cat([d[2] for d in draws], 1).cpu().numpy())
        G.append(torch.cat([d[3] for d in draws], 1).cpu().numpy())
    elapsed = time.time() - t0

    W = np.concatenate(W).astype(np.float32)     # (n_rep, R, n_test)
    U = np.concatenate(U).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32)
    G = np.concatenate(G).astype(np.float32)
    if args.y_min is not None:                   # e.g. 0 for precipitation
        Y = np.maximum(Y, args.y_min)

    per_rep_copula = [replicate_metrics(U[k], u_true[k]) for k in range(len(test_indices))]
    per_rep_data = [replicate_metrics(Y[k], y_true[k]) for k in range(len(test_indices))]

    label = args.label or factor
    metrics = {
        "scenario": name,
        "factor": label,
        "config": vars(args),
        "test_indices": test_indices,
        "margin": margin,
        "n_cond": int(coords_train.shape[0]),
        "weight_samples": len(nets),
        "time_seconds": elapsed,
        "q_theta": {"loc": loc.cpu().tolist(), "log_scale": log_scale.cpu().tolist()},
        "gaussian_component": (None if cov_params is None else
                               {"lengthscale": cov_params[0].item(),
                                "nugget": cov_params[1].item(),
                                "kernel": cov_params[2]}),
        "summary": {"copula": summarise(per_rep_copula), "data": summarise(per_rep_data)},
        "per_replicate": {"copula": per_rep_copula, "data": per_rep_data},
    }

    out = os.path.join(args.out_dir, name, label)
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
    parser = argparse.ArgumentParser(description="BNFC posterior predictive simulation")
    parser.add_argument("--scenarios", nargs="+",
                        default=["simulations_gamma/", "simulations_BNFC/"],
                        help="data directories (each with models/ subfolder)")
    parser.add_argument("--factors", nargs="+", choices=["stat", "nonstat"],
                        default=["stat", "nonstat"])
    parser.add_argument("--out_dir", type=str, default="results/")
    parser.add_argument("--test_start", type=int, default=2000,
                        help="index of the first test replicate file")
    parser.add_argument("--n_test_reps", type=int, default=100)
    parser.add_argument("--R", type=int, default=100,
                        help="posterior predictive draws per replicate")
    parser.add_argument("--L", type=int, default=200,
                        help="factor draws v per theta draw")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_factor", type=int, default=1)
    parser.add_argument("--margin", choices=["nonstat", "stat", "own"], default="nonstat",
                        help="data margin: the location-dependent or stationary marginal network "
                             "for every model, or 'own' (each BNFC variant uses its own)")
    parser.add_argument("--n_cond", type=int, default=None,
                        help="condition the prediction on a fixed subset of this many observed "
                             "sites (default: all); the models are not refitted")
    parser.add_argument("--y_min", type=float, default=None,
                        help="lower bound of the data scale (predictions are floored at it)")
    parser.add_argument("--weights", choices=["sgmcmc", "map"], default="sgmcmc",
                        help="average over the pSGLD weight samples, or use the fitted network")
    parser.add_argument("--label", type=str, default=None,
                        help="name of the output folder and of the model in the summary "
                             "(default: the factor)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.label and len(args.factors) > 1:
        raise SystemExit("--label needs a single --factors entry")
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = []
    for data_path in args.scenarios:
        for factor in args.factors:
            m = run_scenario(data_path, factor, args)
            if m is None:
                continue
            row = {"scenario": m["scenario"], "factor": m["factor"], "time_seconds": m["time_seconds"]}
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
