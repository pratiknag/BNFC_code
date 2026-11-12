#!/usr/bin/env python
# coding: utf-8

"""
Nested Monte-Carlo training for VaNFC
"""

# =========================
# Imports
# =========================
import argparse
import copy
import json
import math
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from tqdm import tqdm
from utils import *   # Encoder, NNFactor, NNFactorWithLocations, rbf_kernel

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# GP covariance builder
# =========================
def build_covariance(theta, coords, cov_params=None):
    """
    theta: (B, 3)
    coords: (B, n, 2)
    cov_params: None for the fixed SE correlation (range 0.1, nugget 1e-4),
        or (lengthscale, nugget, kernel) for the estimated correlation
        (1 - nugget) K + nugget I, which keeps unit variance, with K the
        squared exponential ("se") or exponential ("exponential") kernel.
    """
    if cov_params is not None:
        lengthscale, nugget, kernel = cov_params
        B, n, _ = coords.shape
        if kernel == "exponential":
            K = torch.exp(-torch.cdist(coords, coords) / lengthscale)
        else:
            K = rbf_kernel(coords, coords, lengthscale, torch.ones_like(theta[:, 0]))
        return (1 - nugget) * K + nugget * torch.eye(n, device=coords.device)

    B, n, _ = coords.shape

    variance = torch.ones_like(theta[:, 0])

    lengthscale = torch.tensor(0.1, device=coords.device)

    K = rbf_kernel(coords, coords, lengthscale, variance)

    I = torch.eye(n, device=coords.device)
    cov = K + 1e-4 * I
    # cov = K
    return cov

def range_covariance(coords, lengthscale, nugget, kernel="se"):
    """
    (1 - nugget) K + nugget I for each range in lengthscale (R,): returns (R, n, n).
    coords: (n, 2).
    """
    d = torch.cdist(coords, coords)[None]
    ls = lengthscale.view(-1, 1, 1)
    K = torch.exp(-d / ls) if kernel == "exponential" else torch.exp(-0.5 * d ** 2 / ls ** 2)
    return (1 - nugget) * K + nugget * torch.eye(coords.shape[0], device=coords.device)


# =========================
# Nested Monte Carlo loss
# =========================
def F1_w(
    w,
    coords,
    theta,
    neural_sampler,
    factor,
    mc_u,
    num_mc_samples=100
):
    """
    Deterministic Monte Carlo approximation using common random numbers.

    mc_u:
        Fixed uniform samples with shape (M2, num_mc_samples, B, 1)

    Returns:
        u_hat : (M2, B, n)
    """

    M2, B, n = w.shape
    device = w.device

    # ---------------------------------------------------------
    # Use FIXED Monte Carlo samples
    # ---------------------------------------------------------
    u_mc = mc_u
    num_mc_samples = mc_u.shape[1]

    a = theta[:, None, :, 0:1]
    b = theta[:, None, :, 1:2]

    # V ~ Uniform(a, b)
    v = a + (b - a).abs() * u_mc

    # ---------------------------------------------------------
    # Flatten for neural net
    # ---------------------------------------------------------
    v_flat = v.reshape(M2 * num_mc_samples * B, 1)

    # ---------------------------------------------------------
    # Conditional mean
    # ---------------------------------------------------------
    if factor == "stat":

        mean_scalar = neural_sampler(v_flat)
        mean_test = mean_scalar.repeat(1, n)

    else:

        coords_flat = (
            coords.unsqueeze(0)
            .unsqueeze(0)
            .expand(M2, num_mc_samples, -1, -1, -1)
            .reshape(M2 * num_mc_samples * B, n, 2)
        )

        mean_test = neural_sampler(
            v_flat,
            coords_flat
        )

    # ---------------------------------------------------------
    # Reshape
    # ---------------------------------------------------------
    mean_test = mean_test.reshape(
        M2,
        num_mc_samples,
        B,
        n
    )

    # ---------------------------------------------------------
    # Residuals
    # ---------------------------------------------------------
    residuals = w.unsqueeze(1) - mean_test

    # ---------------------------------------------------------
    # Gaussian CDF
    # ---------------------------------------------------------
    std = torch.ones_like(theta[:, :, 0])

    normal = dist.Normal(
        loc=torch.zeros_like(residuals),
        scale=std[:, None, :, None]
    )

    cdf_vals = normal.cdf(residuals)

    # ---------------------------------------------------------
    # Monte Carlo integration
    # ---------------------------------------------------------
    u_hat = cdf_vals.mean(dim=1)

    return u_hat

def sampler_means(coords, theta, neural_sampler, factor, mc_u):
    """
    Conditional means mu = g(v) for v = a + |b - a| * mc_u.

    coords: (B, n, 2), theta: (M2, B, p), mc_u: (M2, M1, B, 1)
    Returns: (M2, M1, B, n)
    """
    M2, M1, B, _ = mc_u.shape
    n = coords.shape[1]

    a = theta[:, None, :, 0:1]
    b = theta[:, None, :, 1:2]

    v = a + torch.abs(b - a) * mc_u
    v_flat = v.reshape(M2 * M1 * B, 1)

    if factor == "stat":
        mu = neural_sampler(v_flat).repeat(1, n)
    else:
        coords_flat = (
            coords.unsqueeze(0)
            .unsqueeze(0)
            .expand(M2, M1, -1, -1, -1)
            .reshape(M2 * M1 * B, n, 2)
        )
        mu = neural_sampler(v_flat, coords_flat)

    return mu.reshape(M2, M1, B, n)


@torch.no_grad()
def F1_inv(
    u,
    coords,
    theta,
    neural_sampler,
    factor,
    mc_u,
    tol=1e-4,
    max_iter=60
):
    """
    Bisection for w with F1(w) = u, where F1(w) = mean_k Phi(w - mu_k).
    mu does not depend on w, so the sampler is evaluated once. The search
    interval [min mu - 10, max mu + 10] always brackets the root, so the
    result no longer saturates at fixed bounds when mu drifts.
    """

    u = torch.as_tensor(
        u,
        dtype=coords.dtype,
        device=coords.device
    )

    mu = sampler_means(coords, theta, neural_sampler, factor, mc_u)   # (M2,M1,B,n)

    low = mu.amin(dim=1) - 10.
    high = mu.amax(dim=1) + 10.

    normal = dist.Normal(0., 1.)

    for _ in range(max_iter):

        mid = (low + high) / 2

        val = normal.cdf(mid.unsqueeze(1) - mu).mean(dim=1)

        low = torch.where(
            val < u,
            mid,
            low
        )

        high = torch.where(
            val >= u,
            mid,
            high
        )

        if (high - low).max() < tol:
            break

    return (low + high) / 2

def f1_w(
    w,   # (B, n_sample, n_test)
    coords,
    theta,
    neural_sampler,
    factor,
    num_mc_samples=100
):
    """
    Monte Carlo approximation of conditional copula samples.

    Returns:
        copula_samples : (B, n_sample, n_test)
    """

    # ---------------------------------------------------------
    # Sample ONE theta per batch
    # ---------------------------------------------------------
    M2, B, n = w.shape
    device = w.device
    # ---------------------------------------------------------
    # Monte Carlo only over epsilon
    # ---------------------------------------------------------
    u = torch.rand(
            M2, 
            num_mc_samples,
            B,
            1,
            device=device,
        )
    a = theta[:, None, :, 0:1]
    b = theta[:, None, :, 1:2]

    v = a + torch.abs(b - a) * u
    # ---------------------------------------------------------
    # Flatten for neural net
    # ---------------------------------------------------------
    v_flat = v.reshape(B * num_mc_samples * M2, 1)

    # ---------------------------------------------------------
    # Conditional mean
    # ---------------------------------------------------------
    if factor == "stat":

        mean_scalar = neural_sampler(v_flat)
        mean_test = mean_scalar.repeat(1, n)

    else:

        coords_flat = (
            coords.unsqueeze(0)
            .unsqueeze(0)
            .expand(M2, num_mc_samples, -1, -1, -1)
            .reshape(M2 * num_mc_samples * B, n, 2)
        )

        mean_test = neural_sampler(
            v_flat,
            coords_flat
        )

    # ---------------------------------------------------------
    # reshape back
    # ---------------------------------------------------------
    mean_test = mean_test.reshape(M2, 
                                  num_mc_samples, 
                                    B,
                                    n
                                )
    # ---------------------------------------------------------
    # residuals
    # ---------------------------------------------------------
    residuals = w.unsqueeze(1) - mean_test
    # (M2, M1, B, n)
    
    # ---------------------------------------------------------
    # Gaussian CDF
    # ---------------------------------------------------------
    std = torch.sqrt(torch.ones_like(theta[:, :, 0]))
    normal = dist.Normal(
        loc=torch.zeros_like(residuals),
        scale=std[:, None, :, None]
    )

    log_pdf = normal.log_prob(residuals)

    # ---------------------------------------------------------
    # Monte Carlo over epsilon only
    # ---------------------------------------------------------
    log_marginal = (
            torch.logsumexp(log_pdf, dim=1)
            - math.log(num_mc_samples)
        )
    
    return log_marginal



def nested_mc_loss(
    x,
    coords1,
    loc,
    log_scale,
    neural_sampler,
    prior_sigma,
    factor,
    M1,
    M2,
    num_factor,
    N,
    cov_params=None,
    vecchia=None,
    vb_range=False,
):
    """
    Negative nested Monte Carlo ELBO (eq. 3.8 of the manuscript), divided by N:

        -(1/M2) sum_r [ (N/|B|) sum_{i in B} { log f_n(w_i) - sum_j log f_1j(w_ij) }
                        + log p(theta_r) - log q(theta_r) ] / N

    theta_r ~ q(theta) is shared by all replicates of the mini-batch, and the
    same v draws are used for F_1, f_1 and f_n (common random numbers).

    vecchia: None for the exact Gaussian density of the residuals w - G(v), or a
    utils.VecchiaFactor of the locations, whose sparse inverse-Cholesky factor
    replaces the n x n Cholesky factor (cost O(n m) per residual instead of O(n^2)).

    vb_range: the range alpha of the Gaussian component is the third coordinate of
    theta, with the log-normal variational factor and half-normal prior of theta_1,
    theta_2, so Sigma is built for each draw theta_r (cov_params then supplies the
    nugget and the kernel only).
    """

    B, n = x.shape
    coords1 = coords1.expand(B, -1, -1)

    # --------------------------------------------------
    # q(theta): one draw per r, shared across the batch
    # --------------------------------------------------
    q_theta = dist.LogNormal(loc, torch.exp(log_scale) + 1e-6)
    theta_r = q_theta.rsample((M2,))                              # (M2,p)
    theta = theta_r.unsqueeze(1).expand(M2, B, -1)                 # (M2,B,p)

    mc_u = torch.rand(
        M2,
        M1,
        B,
        num_factor,
        device=coords1.device
    )
    eps = 1e-6
    u_obs = x.clamp(eps, 1 - eps).unsqueeze(0).expand(M2, B, n)

    # --------------------------------------------------
    # neural sampler means (shared by F1, f1 and the joint)
    # --------------------------------------------------
    mu = sampler_means(coords1, theta, neural_sampler, factor, mc_u)   # (M2,M1,B,n)

    # --------------------------------------------------
    # w = F1^{-1}(u): bisection without gradients, then attach the
    # implicit-function gradient dw = -dF1 / f1(w) so that w moves with
    # the sampler / theta. Without this, w is a fixed target and the
    # objective log f(w) - sum log f1(w) is unbounded in mu.
    # --------------------------------------------------
    x0 = F1_inv(u_obs,
            coords1,
            theta,
            neural_sampler,
            factor,
            mc_u
        )                                                  # (M2,B,n)

    std_normal = dist.Normal(0., 1.)
    r0 = x0.unsqueeze(1) - mu                              # (M2,M1,B,n)
    F1_x0 = std_normal.cdf(r0).mean(dim=1)
    f1_x0 = std_normal.log_prob(r0).exp().mean(dim=1)
    w = x0 - (F1_x0 - F1_x0.detach()) / f1_x0.detach().clamp_min(1e-12)

    # --------------------------------------------------
    # prior and variational terms (once per theta draw)
    # --------------------------------------------------
    log_p_theta = dist.HalfNormal(prior_sigma).log_prob(theta_r).sum(-1)   # (M2,)
    log_q_theta = q_theta.log_prob(theta_r).sum(-1)                        # (M2,)

    # --------------------------------------------------
    # Gaussian component: Sigma does not depend on theta
    # --------------------------------------------------
    residual = w.unsqueeze(1) - mu                         # (M2,M1,B,n)

    if vb_range:
        if vecchia is not None:
            raise ValueError("--vb_range is not implemented with the Vecchia approximation")
        cov = range_covariance(coords1[0], theta_r[:, 2], cov_params[1], cov_params[2])
        L = torch.linalg.cholesky(cov)                     # (M2,n,n)
        mvn = dist.MultivariateNormal(
            loc=torch.zeros(n, device=x.device),
            scale_tril=L[:, None, None]
        )
        log_phi = mvn.log_prob(residual)                   # (M2,M1,B)
    elif vecchia is None:
        cov = build_covariance(torch.ones(1, 2, device=x.device), coords1[:1], cov_params)[0]
        mvn = dist.MultivariateNormal(
            loc=torch.zeros(n, device=x.device),
            scale_tril=torch.linalg.cholesky(cov)
        )
        log_phi = mvn.log_prob(residual)                   # (M2,M1,B)
    else:
        U = (vecchia.factor("fixed", dtype=residual.dtype) if cov_params is None else
             vecchia.factor(cov_params[2], cov_params[0], cov_params[1], dtype=residual.dtype))
        log_phi = vecchia.log_prob(residual, U)

    # joint density f_n(w_i), inner MC over v
    log_joint = (
        torch.logsumexp(log_phi, dim=1)
        - math.log(M1)
    )                                                      # (M2,B)

    # marginal densities f_1j(w_ij), same MC draws as the joint
    log_marginal = (
        torch.logsumexp(std_normal.log_prob(residual), dim=1)
        - math.log(M1)
    ).sum(-1)                                              # (M2,B)

    log_copula = (log_joint - log_marginal).sum(-1)        # (M2,)

    elbo = (N / B) * log_copula + log_p_theta - log_q_theta
    return -elbo.mean() / N


# =========================
# F_inv diagnostics
# =========================
@torch.no_grad()
def evaluate_F_inv(F_inv, tensor_data_Finv, factor, plot_path, n_show=10):
    """
    tensor_data_Finv columns: 0 = copula value u, 1:3 = coords, 3 = observed y
    Prints fit metrics + a few sample predictions and saves a diagnostic plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F_inv.eval()

    u = tensor_data_Finv[:, 0:1]
    c = tensor_data_Finv[:, 1:3]
    y = tensor_data_Finv[:, 3:4]

    y_hat = F_inv(u) if factor == "stat" else F_inv(u, c)

    err = y_hat - y
    mse = (err ** 2).mean().item()
    mae = err.abs().mean().item()
    r2 = 1 - (err ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()

    print("\n===== F_inv diagnostics =====")
    print(f"MSE  = {mse:.5f}")
    print(f"RMSE = {math.sqrt(mse):.5f}")
    print(f"MAE  = {mae:.5f}")
    print(f"R^2  = {r2:.5f}")
    print(f"y     range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    print(f"y_hat range: [{y_hat.min().item():.3f}, {y_hat.max().item():.3f}]")

    idx = torch.randperm(u.shape[0], device=u.device)[:n_show]
    print(f"\n{'u':>10} {'x1':>8} {'x2':>8} {'y':>10} {'y_hat':>10}")
    for i in idx.tolist():
        print(
            f"{u[i, 0].item():10.4f} {c[i, 0].item():8.3f} {c[i, 1].item():8.3f} "
            f"{y[i, 0].item():10.4f} {y_hat[i, 0].item():10.4f}"
        )

    # ---- plot ----
    u_np, y_np, yh_np = (t.squeeze(-1).cpu().numpy() for t in (u, y, y_hat))
    sub = np.random.default_rng(0).choice(len(u_np), min(20000, len(u_np)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    lo, hi = min(y_np.min(), yh_np.min()), max(y_np.max(), yh_np.max())
    axes[0].scatter(y_np[sub], yh_np[sub], s=2, alpha=0.3)
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[0].set(xlabel="y (true)", ylabel="y_hat", title=f"Predicted vs true (R²={r2:.3f})")

    order = np.argsort(u_np[sub])
    axes[1].scatter(u_np[sub], y_np[sub], s=2, alpha=0.2, label="data")
    axes[1].plot(u_np[sub][order], yh_np[sub][order], "r", lw=1.5, label="F_inv(u)")
    axes[1].set(xlabel="u", ylabel="y", title="Learned mapping u → y")
    axes[1].legend()

    axes[2].hist((yh_np - y_np), bins=100)
    axes[2].set(xlabel="y_hat - y", ylabel="count", title=f"Residuals (RMSE={math.sqrt(mse):.3f})")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved F_inv diagnostic plot to {plot_path}\n")

    F_inv.train()
    return {"mse": mse, "mae": mae, "r2": r2}


# =========================
# TRAINING LOOP
# =========================
def sgmcmc_sample(neural_sampler, loss_fn, train_loader, N, args, out_path):
    """
    Preconditioned stochastic gradient Langevin dynamics (pSGLD; Li et al., 2016)
    over the network weights phi, started at the fitted network (the posterior
    mode). With q(eta) fixed at its estimate, the target is

        p(phi | U) propto exp{ELBO(phi)} N(phi; 0, (N wd)^{-1} I),

    whose mode is the fitted network: the Gaussian prior is the weight decay wd
    of the fit, since the fit minimises -ELBO/N + (wd / 2) |phi|^2. With the
    mini-batch gradient g of U(phi) = -ELBO(phi) + (N wd / 2) |phi|^2,

        V <- a V + (1 - a) g^2,   G = 1 / (N wd + sqrt(V)),
        phi <- phi - (step / 2) G g + sqrt(step G) xi,   xi ~ N(0, I).

    The preconditioner is floored at the prior precision N wd: weights that the
    data do not inform (e.g. of inactive ReLU units) then move on the scale of
    their prior, whereas a small constant floor would inject noise of order
    1 / sqrt(floor) into them and destroy the network.

    The first epoch only estimates the preconditioner V; after `burnin` further
    epochs, a sample is kept every `thin` steps. The samples are
    saved to out_path as a list of state dicts.
    """
    params = [p for p in neural_sampler.parameters()]
    V = [torch.zeros_like(p) for p in params]
    prec = N * args.weight_decay
    a = args.sgmcmc_alpha
    # one epoch to estimate the preconditioner, then `burnin` epochs of Langevin dynamics
    steps_burnin = (1 + args.sgmcmc_burnin) * len(train_loader)
    total = steps_burnin + args.sgmcmc_samples * args.sgmcmc_thin
    samples, losses, step = [], [], 0
    t0 = time.time()
    while step < total:
        for batch in train_loader:
            if step >= total:
                break
            loss = loss_fn(batch[:, :, 0])
            neural_sampler.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p, v in zip(params, V):
                    g = N * p.grad + prec * p
                    v.mul_(a).addcmul_(g, g, value=1 - a)
                    G = 1.0 / (prec + v.sqrt())
                    if step < len(train_loader):
                        continue            # first epoch: estimate the preconditioner only
                    p.add_(-0.5 * args.sgmcmc_step * G * g
                           + torch.sqrt(args.sgmcmc_step * G) * torch.randn_like(p))
            losses.append(loss.item())
            step += 1
            if step == len(train_loader):
                print(f"  fitted network: -ELBO/N (epoch mean) = {np.mean(losses):.4f}")
            if step > steps_burnin and (step - steps_burnin) % args.sgmcmc_thin == 0:
                samples.append(copy.deepcopy(neural_sampler.state_dict()))
                print(f"  sample {len(samples)}/{args.sgmcmc_samples}: -ELBO/N (batch mean of "
                      f"last {args.sgmcmc_thin}) = {np.mean(losses[-args.sgmcmc_thin:]):.4f}")
    torch.save(samples, out_path)
    print(f"saved {len(samples)} pSGLD weight samples to {out_path} ({time.time() - t0:.0f}s)")
    return time.time() - t0


def train(args):
    torch.manual_seed(args.seed)

    # -------------------------
    # Spatial coordinates
    # -------------------------
    # x = np.linspace(0, 1, 5)
    # y = np.linspace(0, 1, 5)
    # xv, yv = np.meshgrid(x, y)
    # coords = np.hstack((xv.reshape(args.n, 1),
    #                     yv.reshape(args.n, 1)))
    # coords = torch.tensor(coords, dtype=torch.float, device=device)

    # -------------------------
    # Dataset
    # -------------------------
    data_list, x1_list, x2_list = [], [], []

    for i in range(args.N):
        df = pd.read_csv(f"{args.data_path}sample_copula-{i}.csv")
        data_list.append(df["y"].values)
        x1_list.append(df["x1"].values)
        x2_list.append(df["x2"].values)
    x1 = x1_list[0]
    x2 = x2_list[0]

    coords = np.hstack((
        x1.reshape(-1, 1),
        x2.reshape(-1, 1)
    ))
    
    coords = torch.tensor(
        coords,
        dtype=torch.float,
        device=device
    )

    tensor_data = torch.cat(
        (
            torch.tensor(np.array(data_list)).reshape(args.N, args.n, 1),
            torch.tensor(np.array(x1_list)).reshape(args.N, args.n, 1),
            torch.tensor(np.array(x2_list)).reshape(args.N, args.n, 1),
        ),
        dim=-1
    ).float().to(device)

    train_loader = torch.utils.data.DataLoader(
        tensor_data,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # -------------------------
    # load data for F_inv: (u, s, y) pairs of the TRAINING replicates
    # (copula data and data-scale values at the observed sites)
    # -------------------------
    copula_data_list, data_list1, x1_list1, x2_list1 = [], [], [], []
    for i in range(args.N):
        df = pd.read_csv(f"{args.data_path}sample_copula-{i}.csv")
        copula_data_list.append(df["y"].values)
        x1_list1.append(df["x1"].values)
        x2_list1.append(df["x2"].values)
        df1 = pd.read_csv(f"{args.data_path}sample-{i}.csv")
        assert np.allclose(df1[["x1", "x2"]].values, df[["x1", "x2"]].values)
        data_list1.append(df1["y"].values)

    tensor_data_Finv = torch.cat(
        (
            torch.tensor(np.array(copula_data_list)).reshape(-1, 1),
            torch.tensor(np.array(x1_list1)).reshape(-1, 1),
            torch.tensor(np.array(x2_list1)).reshape(-1, 1),
            torch.tensor(np.array(data_list1)).reshape(-1, 1)
        ),
        dim=-1
    ).float().to(device)

    train_loader_Finv = torch.utils.data.DataLoader(
        tensor_data_Finv,
        batch_size=args.batch_size_Finv,
        shuffle=True
    )
    # -------------------------
    # Models (unchanged)
    # -------------------------
    # encoder = Encoder(args.n, args.out).to(device)

    if args.factor == "stat":
        neural_sampler = NNFactor(args.num_factor, 32, 32).to(device)
        F_inv = F_Inv(hidden=64,K=16,beta=2.0).to(device)
        model_name = f"{args.data_path}models/VaNFC-stat.pth"
        # encoder_name = "models/encoder-stat.pth"
        f_inv_name = f"{args.data_path}models/F_inv-stat.pth"
        params_name = f"{args.data_path}models/posterior_params_stat.pth"
    else:
        arch = {"hidden2": args.hidden, "depth": args.depth, "out_act": args.out_act,
                "n_fourier": args.n_fourier, "fourier_scale": args.fourier_scale}
        neural_sampler = NNFactorWithLocations(**arch).to(device)
        # two-sided basis: convex and concave terms in logit(u), see F_Inv_WithLocs
        F_inv = F_Inv_WithLocs(coord_dim=2,hidden=128,K=32,two_sided=True).to(device)
        model_name = f"{args.data_path}models/VaNFC-nonstat.pth"
        # encoder_name = "models/encoder-nonstat.pth"
        f_inv_name = f"{args.data_path}models/F_inv-nonstat.pth"
        params_name = f"{args.data_path}models/posterior_params_nonstat.pth"

    sgmcmc_name = f"{args.data_path}models/sgmcmc-{args.factor}.pth"
    if args.sgmcmc_only:
        # sample the weights of an existing fit: no marginal network, no ELBO fit
        args.epochs_Finv = 0

    os.makedirs(f"{args.data_path}models", exist_ok=True)

    # -------------------------
    # Optimizer
    # -------------------------
    # q(theta) starts at theta ~ (1, 2): by Remark 2.1 only the width of the
    # input interval matters, and a unit width matches the network's scale.
    # With --vb_range, the range of the Gaussian component is a third coordinate of
    # theta with a log-normal variational factor, started at --init_lengthscale.
    init = [0., math.log(2.)] + ([math.log(args.init_lengthscale)] if args.vb_range else [])
    loc = nn.Parameter(torch.tensor(init, dtype=torch.float, device=device))
    log_scale = nn.Parameter(torch.ones(len(init), dtype=torch.float, device=device) * -5.0)
    
    # Gaussian component: fixed SE correlation, or range and nugget estimated
    # (point estimates) jointly with the ELBO when --learn_cov is set.
    log_ls = nn.Parameter(torch.tensor(math.log(args.init_lengthscale), device=device))
    logit_nug = nn.Parameter(torch.logit(torch.tensor(args.init_nugget, device=device)))

    def cov_params():
        if not (args.learn_cov or args.fix_cov):
            return None
        return torch.exp(log_ls), torch.sigmoid(logit_nug), args.kernel

    optimizer = torch.optim.Adam(
        list(neural_sampler.parameters())+
        [loc, log_scale] + ([log_ls, logit_nug] if (args.learn_cov and not args.fix_cov) else []),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    prior_sigma = torch.tensor(
        args.prior_sigma_sq,
        dtype=torch.float,
        device=device
    )

    # -------------------------------
    # Training F_inv 
    # -------------------------------

    if args.factor != "stat":
        # start at the location and scale of the data: T = a + s logit(u), with
        # s matching the sd of the logistic distribution, pi / sqrt(3)
        y_all = tensor_data_Finv[:, 3]
        with torch.no_grad():
            F_inv.shift.bias.fill_(y_all.mean().item())
            F_inv.scale.bias.fill_(math.log(y_all.std().item() * math.sqrt(3) / math.pi))

    optimizer_Finv = torch.optim.Adam(
        F_inv.parameters(),
        lr=args.lr_Finv
    )

    num_epochs = args.epochs_Finv
    scheduler_Finv = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_Finv, max(1, num_epochs * len(train_loader_Finv)))

    for epoch in range(num_epochs):

        F_inv.train()

        total_loss = 0.0

        for batch in train_loader_Finv:

            # ------------------------------------------
            # batch columns:
            # 0 = copula value u
            # 1 = x1
            # 2 = x2
            # 3 = observed data y
            # ------------------------------------------

            u = batch[:, 0:1]
            c = batch[:, 1:3]
            y = batch[:, 3:4]

            optimizer_Finv.zero_grad()
            # u -> y | location
            if args.factor == "stat":
                y_hat = F_inv(u)
            else:
                y_hat = F_inv(u,c)

            loss = F.mse_loss(y_hat,y)
            loss.backward()
            optimizer_Finv.step()
            scheduler_Finv.step()

            total_loss += loss.item() * u.shape[0]

        total_loss /= len(train_loader_Finv.dataset)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {total_loss}"
            )
        torch.save(F_inv.state_dict(), f_inv_name)

    if args.finv_only:
        print(f"saved {f_inv_name} (--finv_only: copula not fitted)")
        return

    # evaluate_F_inv(
    #     F_inv,
    #     tensor_data_Finv,
    #     args.factor,
    #     f"{args.data_path}plots/F_inv_diagnostics-{args.factor}.png"
    # )
    # exit(0);
    # -------------------------
    # Nested MC ELBO training
    # -------------------------
    N_train = len(train_loader.dataset)
    # Vecchia approximation of the Gaussian component (--vecchia m > 0)
    vecchia = VecchiaFactor(coords.cpu().numpy(), args.vecchia, device) if args.vecchia > 0 else None

    def loss_fn(y_obs):
        return nested_mc_loss(y_obs, coords, loc, log_scale, neural_sampler, prior_sigma,
                              args.factor, M1=args.M1, M2=args.M2,
                              num_factor=args.num_factor, N=N_train, cov_params=cov_params(),
                              vecchia=vecchia, vb_range=args.vb_range)

    fit_time_name = f"{args.data_path}models/fit_time-{args.factor}.json"
    if args.sgmcmc_only:
        neural_sampler.load_state_dict(torch.load(model_name, map_location=device))
        ck = torch.load(params_name, map_location=device)
        with torch.no_grad():
            loc.copy_(ck["loc"]), log_scale.copy_(ck["log_scale"])
        loc.requires_grad_(False), log_scale.requires_grad_(False)
        t_s = sgmcmc_sample(neural_sampler, loss_fn, train_loader, N_train, args, sgmcmc_name)
        with open(fit_time_name) as f:
            info = json.load(f)
        info.update({"sgmcmc_seconds": t_s, "sgmcmc_samples": args.sgmcmc_samples})
        with open(fit_time_name, "w") as f:
            json.dump(info, f)
        return

    # early stopping: stop when -ELBO/N has not improved by min_delta for
    # `patience` epochs and keep the parameters of the best epoch
    best_loss, best_epoch, best_state = float("inf"), 0, None
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for batch in tqdm(train_loader):
            # x_full = batch
            y_obs = batch[:, :, 0]

            optimizer.zero_grad()
            loss = nested_mc_loss(
                y_obs,
                coords,
                loc,
                log_scale,
                neural_sampler,
                prior_sigma,
                args.factor,
                M1=args.M1,
                M2=args.M2,
                num_factor=args.num_factor,
                N=N_train,
                cov_params=cov_params(),
                vecchia=vecchia,
                vb_range=args.vb_range,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / len(train_loader)
        scheduler.step()
        if args.vb_range:
            print(f"  q(range): median={torch.exp(loc[2]).item():.4f}, "
                  f"log-sd={torch.exp(log_scale[2]).item():.2e}, "
                  f"nugget={torch.sigmoid(logit_nug).item():.4f}")
        elif args.learn_cov:
            print(f"  lengthscale={torch.exp(log_ls).item():.4f}, "
                  f"nugget={torch.sigmoid(logit_nug).item():.4f}")
        print(f"[Epoch {epoch+1}] -ELBO/N = {epoch_loss:.4f}  "
              f"q(theta): loc={loc.detach().cpu().numpy().round(3)}, "
              f"log_scale={log_scale.detach().cpu().numpy().round(3)}")

        if epoch_loss < best_loss - args.min_delta:
            best_loss, best_epoch = epoch_loss, epoch + 1
            best_state = {
                "sampler": copy.deepcopy(neural_sampler.state_dict()),
                "params": {
                    "loc": loc.detach().cpu().clone(),
                    "log_scale": log_scale.detach().cpu().clone(),
                    **({"lengthscale": (torch.exp(loc[2]) if args.vb_range
                                        else torch.exp(log_ls)).detach().cpu(),
                        "nugget": torch.sigmoid(logit_nug).detach().cpu(),
                        "kernel": args.kernel}
                       if (args.learn_cov or args.fix_cov) else {}),
                    "vb_range": args.vb_range,
                    **({"arch": arch} if args.factor != "stat" else {}),
                },
            }
            torch.save(best_state["sampler"], model_name)
            torch.save(best_state["params"], params_name)
        elif args.patience > 0 and epoch + 1 - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch+1}: no improvement of -ELBO/N by "
                  f"{args.min_delta} since epoch {best_epoch} ({best_loss:.4f})")
            break

    fit_seconds = time.time() - t0
    print(f"best epoch {best_epoch}, -ELBO/N = {best_loss:.4f}, saved {model_name}")
    info = {"time_seconds": fit_seconds, "estimated": False, "epochs_run": epoch + 1,
            "best_epoch": best_epoch, "best_loss": best_loss, "seed": args.seed,
            "n_sites": args.n, "vecchia_m": args.vecchia, "batch_size": args.batch_size,
            "M1": args.M1, "M2": args.M2, "seconds_per_epoch": fit_seconds / (epoch + 1),
            "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                   if torch.cuda.is_available() else None)}

    # SG-MCMC over the weights, started at the best epoch (q(theta) stays fixed)
    if args.sgmcmc_samples > 0:
        neural_sampler.load_state_dict(best_state["sampler"])
        with torch.no_grad():
            loc.copy_(best_state["params"]["loc"]), log_scale.copy_(best_state["params"]["log_scale"])
        loc.requires_grad_(False), log_scale.requires_grad_(False)
        t_s = sgmcmc_sample(neural_sampler, loss_fn, train_loader, N_train, args, sgmcmc_name)
        info.update({"sgmcmc_seconds": t_s, "sgmcmc_samples": args.sgmcmc_samples})
    with open(fit_time_name, "w") as f:
        json.dump(info, f)

    
# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epochs_Finv", type=int, default=100)
    parser.add_argument("--finv_only", action="store_true",
                        help="fit only the marginal quantile network T_psi and exit "
                             "(it is fitted independently of the copula)")
    parser.add_argument("--patience", type=int, default=25,
                        help="early stopping: epochs without improvement (0 = off)")
    parser.add_argument("--min_delta", type=float, default=0.01,
                        help="early stopping: minimum decrease of -ELBO/N")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 penalty of the fit; N * weight_decay is the prior precision "
                             "of the weights in the SG-MCMC")
    parser.add_argument("--sgmcmc_samples", type=int, default=20,
                        help="pSGLD weight samples after the fit (0 = none, point estimate only)")
    parser.add_argument("--sgmcmc_burnin", type=int, default=1, help="pSGLD burn-in epochs")
    parser.add_argument("--sgmcmc_thin", type=int, default=50, help="pSGLD steps between samples")
    parser.add_argument("--sgmcmc_step", type=float, default=1e-3, help="pSGLD step size")
    parser.add_argument("--sgmcmc_alpha", type=float, default=0.99,
                        help="pSGLD decay of the squared-gradient average")
    parser.add_argument("--sgmcmc_only", action="store_true",
                        help="run only the pSGLD sampler, started at an existing fit")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=int, default=3)
    parser.add_argument("--num_factor", type=int, default=1)
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--batch_size_Finv", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--lr_Finv", type=float, default=1e-3) 
    parser.add_argument("--prior_sigma_sq", type=float, default=4.0)
    parser.add_argument('--step_size', type=int, default=20, help='Step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma value.')
    parser.add_argument("--data_path", type=str,
                        default="simulations_gamma/")
    parser.add_argument("--factor",
                        choices=["stat", "nonstat"],
                        default="stat")

    parser.add_argument("--fix_cov", action="store_true",
                        help="use the Gaussian component with the range and nugget of "
                             "--init_lengthscale/--init_nugget and --kernel without estimating them "
                             "(e.g. to refit with the Gaussian component shared with other models)")
    parser.add_argument("--vb_range", action="store_true",
                        help="with --learn_cov: the range is part of the variational family "
                             "(log-normal q, half-normal prior, as theta); the nugget stays a "
                             "point estimate")
    parser.add_argument("--hidden", type=int, default=64, help="nonstat: width of the factor network")
    parser.add_argument("--depth", type=int, default=3, help="nonstat: hidden layers of the factor network")
    parser.add_argument("--out_act", choices=["softplus", "linear"], default="softplus",
                        help="nonstat: output activation of the factor network")
    parser.add_argument("--n_fourier", type=int, default=0,
                        help="nonstat: random Fourier features of the coordinates (0 = none)")
    parser.add_argument("--fourier_scale", type=float, default=3.0)
    parser.add_argument("--learn_cov", action="store_true",
                        help="estimate the SE range and the nugget of the Gaussian component")
    parser.add_argument("--kernel", choices=["se", "exponential"], default="se",
                        help="correlation kernel of the estimated Gaussian component")
    parser.add_argument("--init_lengthscale", type=float, default=0.1)
    parser.add_argument("--init_nugget", type=float, default=0.01)
    parser.add_argument("--vecchia", type=int, default=0,
                        help="Vecchia approximation of the Gaussian component with this many "
                             "nearest neighbours (0 = exact)")

    # Monte Carlo parameters
    parser.add_argument("--M1", type=int, default=100,
                        help="inner MC samples (v)")
    parser.add_argument("--M2", type=int, default=50,
                        help="outer MC samples (theta)")

    args = parser.parse_args()
    print(args)
    train(args)

if __name__ == "__main__":
    main()