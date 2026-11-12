#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
Plots and tables for the ERA5 precipitation application, from the results
written by Copula_BNFC.py and Copula_single_factor.py. Expects
    <results_dir>/<scenario>/<model>/{predictions.npz, metrics.json}
and the grid and dates of the data folder (dataGen_precipitation.py).
Nothing here re-runs a model.

Figures (maps of 15E-25E, 45N-55N, colours distinct from the simulation plots):
  map_comparison_<date>.png : (--comparison) true field, observed data, and
                              predictive mean and SD of BNFC and the comparator
  map_bnfc_<variant>_<date>.png : true field, observed data, and the BNFC
                              predictive mean, 2.5% and 97.5% quantiles and SD
  metrics_<scale>.png       : CRPS, energy score and MAE for all models
Tables:
  summary_table.csv / summary_table.tex   (as for the simulations: held-out copula
                                           log score and copula-scale CRPS, energy
                                           score and MAE, with --scale copula)
"""

# =========================
# Imports
# =========================
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import plot_BNFC_results as pbr
from application_topography import ETOPO, add_relief
from plot_BNFC_results import (MODEL_LABELS, COLORBAR_TICKSIZE, find_runs, plot_metric_summary,
                               summary_table, load_logscore)

# Sequential maps: multi-hue, light -> dark. Lightness decreases monotonically
# (CIE L* 99 -> 14), so both read in grayscale. Precipitation: yellow-green-blue-violet,
# as in rain radar maps; SD: the same colormap as the simulation figures.
CMAP_VALUE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "precip_radar", ["#ffffd9", "#d4eeb3", "#97d6b9", "#4fbcc5",
                     "#1f93c1", "#2563a8", "#2b3591", "#2a1458"])
CMAP_SD = pbr.CMAP_SD
LAND_EDGE = "#3d3c39"
UNITS = "mm/day"
# precipitation is right-skewed: a square-root colour scale keeps light rain visible
GAMMA_NORM = 0.5

PROJ = ccrs.PlateCarree()


# =========================
# Grid and dates
# =========================
def load_geometry(data_path):
    grid = pd.read_csv(os.path.join(data_path, "grid.csv"))
    dates = pd.read_csv(os.path.join(data_path, "dates.csv")).set_index("index")["date"]
    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)
    return grid, dates, meta


def to_lonlat_grid(values, grid):
    """values on the full grid (x1 fastest) -> (lat, lon) array and cell edges."""
    lon = np.unique(grid["lon"].values)
    lat = np.unique(grid["lat"].values)
    field = values.reshape(len(lat), len(lon))
    dlon, dlat = np.diff(lon)[0], np.diff(lat)[0]
    lon_e = np.concatenate((lon - dlon / 2, [lon[-1] + dlon / 2]))
    lat_e = np.concatenate((lat - dlat / 2, [lat[-1] + dlat / 2]))
    return field, lon_e, lat_e


def obs_lonlat(coords_train, grid):
    lookup = {(round(float(a), 5), round(float(b), 5)): k
              for k, (a, b) in enumerate(grid[["x1", "x2"]].values)}
    idx = [lookup[(round(float(a), 5), round(float(b), 5))] for a, b in coords_train]
    return grid["lon"].values[idx], grid["lat"].values[idx], np.array(idx)


# =========================
# Map panels
# =========================
def base_map(ax, meta, left, bottom):
    lon0, lon1 = meta["lon_range"]
    lat0, lat1 = meta["lat_range"]
    ax.set_extent([lon0, lon1, lat0, lat1], crs=PROJ)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.7, edgecolor=LAND_EDGE)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.8, edgecolor=LAND_EDGE)
    gl = ax.gridlines(crs=PROJ, draw_labels=True, lw=0.4, color="#9b9a96", alpha=0.6,
                      xlocs=np.arange(lon0 + 1, lon1, 2), ylocs=np.arange(lat0 + 1, lat1, 2))
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = left, bottom
    # same size as the tick labels of the other figures (plot_BNFC_results rcParams)
    gl.xlabel_style = gl.ylabel_style = {"size": plt.rcParams["ytick.labelsize"],
                                         "color": "#52514e"}


def map_field(ax, values, grid, meta, cmap, norm, title, left, bottom):
    field, lon_e, lat_e = to_lonlat_grid(values, grid)
    im = ax.pcolormesh(lon_e, lat_e, field, cmap=cmap, norm=norm, transform=PROJ,
                       shading="flat", rasterized=True)
    base_map(ax, meta, left, bottom)
    ax.set_title(title)
    return im


def map_obs(ax, y_obs, lon, lat, meta, cmap, norm, title, left, bottom):
    ax.scatter(lon, lat, c=y_obs, cmap=cmap, norm=norm, s=46, transform=PROJ,
               edgecolors=LAND_EDGE, linewidths=0.6, zorder=3)
    base_map(ax, meta, left, bottom)
    ax.set_title(title)


def value_norm(*fields):
    vmax = float(max(np.nanmax(f) for f in fields))
    return PowerNorm(GAMMA_NORM, vmin=0.0, vmax=max(vmax, 1e-3))


def sqrt_ticks(norm, n=5):
    """Ticks evenly spaced on the square-root colour scale, rounded to 2 significant digits."""
    t = np.linspace(0.0, np.sqrt(norm.vmax), n) ** 2
    return [0.0] + [float(f"{v:.2g}") for v in t[1:]]


# =========================
# Figures
# =========================
# --margin_path: data-scale draws recomputed from the stored copula draws with
# this marginal network instead of the stored data_samples
MARGIN_PATH = None


def data_draws(p, rep, y_min=0.0):
    """Data-scale draws (R, n_test) of test day rep."""
    if MARGIN_PATH is None:
        return p["data_samples"][rep]
    import torch
    from Copula_BNFC import device
    from utils import load_F_inv_withlocs
    net = load_F_inv_withlocs(MARGIN_PATH, device).eval()
    u = torch.tensor(p["copula_samples"][rep], device=device)
    c = torch.tensor(p["coords_test"], device=device)
    with torch.no_grad():
        y = net(u, c.unsqueeze(0).expand(u.shape[0], -1, -1)).cpu().numpy()
    return np.maximum(y, y_min)
def plot_comparison(runs, scenario, bnfc, comparator, rep, geom, out_dir):
    """
    2 x 3 maps on the data scale.
      row 1: true field | BNFC predictive mean | comparator predictive mean
      row 2: observed data | BNFC predictive SD | comparator predictive SD
    """
    grid, dates, meta = geom
    pb = np.load(os.path.join(runs[(scenario, bnfc)]["dir"], "predictions.npz"))
    pc = np.load(os.path.join(runs[(scenario, comparator)]["dir"], "predictions.npz"))
    truth = pb["y_true"][rep]
    lon_o, lat_o, idx_o = obs_lonlat(pb["coords_train"], grid)

    draws_b, draws_c = data_draws(pb, rep), data_draws(pc, rep)
    mean_b, sd_b = draws_b.mean(0), draws_b.std(0)
    mean_c, sd_c = draws_c.mean(0), draws_c.std(0)
    norm = value_norm(truth, mean_b, mean_c)
    sd_norm = PowerNorm(GAMMA_NORM, vmin=0.0, vmax=float(max(sd_b.max(), sd_c.max())))

    fig, axes = plt.subplots(2, 3, figsize=(16, 11.2), constrained_layout=True,
                             subplot_kw={"projection": PROJ})
    map_field(axes[0, 0], truth, grid, meta, CMAP_VALUE, norm, "True field", True, False)
    map_field(axes[0, 1], mean_b, grid, meta, CMAP_VALUE, norm,
              f"{MODEL_LABELS[bnfc]}: mean", False, False)
    im = map_field(axes[0, 2], mean_c, grid, meta, CMAP_VALUE, norm,
                   f"{MODEL_LABELS[comparator]}: mean", False, False)
    cb = fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.02, label=UNITS, format="%g",
                      ticks=sqrt_ticks(norm))
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    map_obs(axes[1, 0], truth[idx_o], lon_o, lat_o, meta, CMAP_VALUE, norm,
            "Observed data", True, True)
    map_field(axes[1, 1], sd_b, grid, meta, CMAP_SD, sd_norm,
              f"{MODEL_LABELS[bnfc]}: SD", False, True)
    im_sd = map_field(axes[1, 2], sd_c, grid, meta, CMAP_SD, sd_norm,
                      f"{MODEL_LABELS[comparator]}: SD", False, True)
    cb = fig.colorbar(im_sd, ax=axes[1, 1:], shrink=0.85, pad=0.02, label=UNITS,
                      format="%g", ticks=sqrt_ticks(sd_norm))
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    date = dates[int(pb["test_indices"][rep])]
    fig.suptitle(f"{date}: predictive mean and SD", fontweight="bold")
    path = os.path.join(out_dir, f"map_comparison_{date}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_bnfc_summary(runs, scenario, bnfc, rep, geom, out_dir):
    """
    2 x 3 maps of one BNFC variant on the data scale.
      row 1: true field | observed data | predictive mean
      row 2: 2.5% quantile | 97.5% quantile | predictive SD
    The first five panels share one colour scale; the SD has its own.
    """
    grid, dates, meta = geom
    p = np.load(os.path.join(runs[(scenario, bnfc)]["dir"], "predictions.npz"))
    truth = p["y_true"][rep]
    draws = data_draws(p, rep)
    lon_o, lat_o, idx_o = obs_lonlat(p["coords_train"], grid)

    mean, sd = draws.mean(0), draws.std(0)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    norm = value_norm(truth, mean, lo, hi)
    coverage = float(np.mean((truth >= lo) & (truth <= hi)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 11.2), constrained_layout=True,
                             subplot_kw={"projection": PROJ})
    map_field(axes[0, 0], truth, grid, meta, CMAP_VALUE, norm, "True field", True, False)
    map_obs(axes[0, 1], truth[idx_o], lon_o, lat_o, meta, CMAP_VALUE, norm,
            "Observed data", False, False)
    map_field(axes[0, 2], mean, grid, meta, CMAP_VALUE, norm, "Predictive mean", False, False)
    map_field(axes[1, 0], lo, grid, meta, CMAP_VALUE, norm, "2.5% quantile", True, True)
    im = map_field(axes[1, 1], hi, grid, meta, CMAP_VALUE, norm, "97.5% quantile", False, True)
    im_sd = map_field(axes[1, 2], sd, grid, meta, CMAP_SD,
                      PowerNorm(GAMMA_NORM, vmin=0.0, vmax=float(sd.max())),
                      "Predictive SD", False, True)
    # horizontal bars under the bottom row: values (first five panels) | SD
    cb = fig.colorbar(im, ax=axes[1, :2], location="bottom", shrink=0.7, pad=0.02, format="%g",
                      ticks=sqrt_ticks(norm), label=f"Precipitation ({UNITS}), panels 1-5")
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)
    cb = fig.colorbar(im_sd, ax=axes[1, 2], location="bottom", shrink=0.9, pad=0.02,
                      format="%g", ticks=sqrt_ticks(im_sd.norm, 4), label=f"SD ({UNITS})")
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    date = dates[int(p["test_indices"][rep])]
    print(f"95% interval coverage on {date}: {coverage:.2f}")
    fig.suptitle(f"{MODEL_LABELS[bnfc]}, {date}", fontweight="bold")
    path = os.path.join(out_dir, f"map_bnfc_{bnfc}_{date}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# =========================
# Inference on the fitted models (--inference)
# =========================
INF_MODELS = ["stat", "nonstat", "exponential", "gamma", "pareto", "mixgamma", "tcopula"]


def haversine_km(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pair_measures(u, qs):
    """
    Pairwise Spearman correlation and upper-tail coefficients lambda_U(q) of copula
    data u (M, n). Each margin is replaced by its ranks / (M + 1), so that exactly
    the same number of values exceeds 1 - q at every location and lambda_U(q) <= 1.
    """
    from scipy.stats import rankdata
    u = rankdata(u, axis=0) / (len(u) + 1)
    out = {"spearman": np.corrcoef(u.T)}
    for q in qs:
        E = (u > 1 - q).astype(np.float64)
        out[f"lambdaU_{q:g}"] = (E.T @ E) / (len(u) * q)
    return out


def simulate_copula(model, data_path, coords, n_sim, K, seed):
    """
    n_sim draws of the copula field at coords from the fitted model (posterior mean
    or median of the copula parameters; for BNFC, an equal share of the draws per
    weight sample), with its own Gaussian component.
    """
    import torch
    from scipy.stats import t as sp_t
    import gsm_t_copula as T
    import train_model_single_factor_HMC_NUTS as H
    from Copula_BNFC import load_models, load_weight_samples, device
    from Train_BNFC import build_covariance, sampler_means
    from heldout_loglik import hmc_cov_params

    rng = torch.Generator(device=device).manual_seed(seed)
    c = torch.tensor(coords, dtype=torch.float32, device=device)
    normal = torch.distributions.Normal(0.0, 1.0)

    def chol(cov_params):
        S = build_covariance(torch.ones(1, 2, device=device), c[None], cov_params)[0].double()
        return torch.linalg.cholesky(S).float()

    def from_nodes(G, logp, L, m):
        """W = G[k] + L z, U = F_1(W) for the discretised factor G (K, n) with log weights logp."""
        k = torch.distributions.Categorical(logits=logp).sample((m,))
        W = G[k] + (L @ torch.randn(L.shape[0], m, generator=rng, device=device)).T
        p = logp.exp()
        U = torch.empty_like(W)
        for a in range(0, m, 500):
            U[a:a + 500] = (normal.cdf(W[a:a + 500, :, None] - G.T[None]) * p).sum(-1)
        return U

    if model in ("stat", "nonstat"):
        net, _, loc, _, cov_params = load_models(data_path, model)
        nets = load_weight_samples(data_path, model, net)
        theta = torch.exp(loc).view(1, 1, -1)
        mc_u = ((torch.arange(K, device=device) + 0.5) / K).view(1, K, 1, 1)
        L = chol(cov_params)
        logp = torch.full((K,), -np.log(K), device=device)
        out = []
        with torch.no_grad():
            for nk in nets:
                G = sampler_means(c[None], theta, nk, model, mc_u)[0, :, 0, :]
                out.append(from_nodes(G, logp, L, n_sim // len(nets)))
        return torch.cat(out).cpu().numpy()
    if model == "tcopula":
        post = T.load_posterior(data_path)
        if post is None:
            raise FileNotFoundError(f"{data_path}models/posterior-tcopula.pth")
        nu, beta, gc = post
        nu, beta = float(np.mean(nu)), float(np.mean(beta))
        Lb = np.linalg.cholesky(T.correlation(coords.astype(np.float64), beta, gc))
        r = np.random.default_rng(seed)
        X = (Lb @ r.standard_normal((len(coords), n_sim))).T / np.sqrt(r.chisquare(nu, n_sim) / nu)[:, None]
        return sp_t.cdf(X, nu)
    path = os.path.join(data_path, "models", f"hmc_posterior-{model}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device)
    params = {k: ck[k].float().mean(0).to(device) for k in ("vartheta", "pi") if k in ck}
    v, logp = H.factor_nodes(model, H.base_quantiles(model, K).float().to(device), params)
    L = chol(hmc_cov_params(ck.get("gaussian_component", {"kernel": "fixed"})))
    with torch.no_grad():
        return from_nodes(v[:, None].expand(-1, len(coords)), logp, L, n_sim).cpu().numpy()


def factor_maps(data_path, grid_coords, K):
    """
    BNFC (nonstat), averaged over the weight samples: share of the latent variance
    due to the factor, var g(V, s) / {1 + var g(V, s)}, skewness of the factor
    contribution, and its covariance matrix across the grid (for correlations).
    """
    import torch
    from Copula_BNFC import load_models, load_weight_samples, device
    from Train_BNFC import sampler_means
    net, _, loc, _, cov_params = load_models(data_path, "nonstat")
    nets = load_weight_samples(data_path, "nonstat", net)
    theta = torch.exp(loc).view(1, 1, -1)
    mc_u = ((torch.arange(K, device=device) + 0.5) / K).view(1, K, 1, 1)
    c = torch.tensor(grid_coords, dtype=torch.float32, device=device)[None]
    share, skew, cov = [], [], 0.0
    with torch.no_grad():
        for nk in nets:
            G = sampler_means(c, theta, nk, "nonstat", mc_u)[0, :, 0, :].double()   # (K, n)
            Gc = G - G.mean(0)
            var = (Gc ** 2).mean(0)
            share.append((var / (1 + var)).cpu().numpy())
            skew.append(((Gc ** 3).mean(0) / var.clamp_min(1e-12) ** 1.5).cpu().numpy())
            cov = cov + (Gc.T @ Gc / K).cpu().numpy() / len(nets)
    gc = None if cov_params is None else \
        {"kernel": cov_params[2], "lengthscale": float(cov_params[0]), "nugget": float(cov_params[1])}
    return np.array(share), np.array(skew), cov, gc


def dependence(args, grid, coords_obs, u_train, out_dir, models=None):
    """
    Dependence of the observed locations, models against the empirical copula of the
    training days. Writes the RMSE of the pairwise Spearman correlations and lambda_U(q)
    of every model (inference_dependence_rmse.csv) and the figure of chi(q) against the
    level in four distance classes for BNFC (nonstat), with the 95% band over the
    weight samples (inference_dependence.png, Figure S8 of the online supplement).
    """
    from Copula_BNFC import load_models, load_weight_samples
    lon_o, lat_o, _ = obs_lonlat(coords_obs, grid)
    qs = (0.05, 0.01)
    keys = ["spearman"] + [f"lambdaU_{q:g}" for q in qs]
    dist = haversine_km(lon_o, lat_o)
    iu = np.triu_indices(len(lon_o), 1)
    emp = pair_measures(u_train, qs)
    sims, draws = {}, {}
    for m in models or INF_MODELS:
        try:
            draws[m] = simulate_copula(m, args.data_path, coords_obs, args.n_sim, args.K, args.seed)
            sims[m] = pair_measures(draws[m], qs)
            print(f"simulated {m}")
        except FileNotFoundError as e:
            print(f"[skip] {m}: {e}")
    rows = [{"model": m, **{k: float(np.sqrt(np.mean((sims[m][k][iu] - emp[k][iu]) ** 2)))
                            for k in keys}} for m in sims]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "inference_dependence_rmse.csv"), index=False)

    # chi(q) = P(U_k > 1 - q | U_j > 1 - q) against the tail level, averaged over the
    # pairs in each distance class: empirical against BNFC (nonstat), with the 95% band
    # over the weight samples (simulate_copula draws an equal share from each, in order)
    levels = np.array([0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005])
    classes = [(0, 150), (150, 300), (300, 600), (600, np.inf)]
    net, *_ = load_models(args.data_path, "nonstat")
    S = len(load_weight_samples(args.data_path, "nonstat", net))

    def chi_by_class(u):
        m = pair_measures(u, levels)
        return np.array([[m[f"lambdaU_{q:g}"][iu][(dist[iu] >= lo) & (dist[iu] < hi)].mean()
                          for q in levels] for lo, hi in classes])

    chi_emp = chi_by_class(u_train)
    chi_mod = chi_by_class(draws["nonstat"])
    band = np.array([chi_by_class(d) for d in np.split(draws["nonstat"], S)])
    lo_b, hi_b = np.quantile(band, [0.025, 0.975], axis=0)
    color = pbr.MODEL_COLORS["nonstat"]
    x = 1 - levels
    fig, axes = plt.subplots(1, len(classes), figsize=(19, 5.6), sharey=True,
                             constrained_layout=True)
    for k, (ax, (lo, hi)) in enumerate(zip(axes, classes)):
        ax.fill_between(x, lo_b[k], hi_b[k], color=color, alpha=0.3, lw=0,
                        label=f"{MODEL_LABELS['nonstat']}, 95% band over weight samples")
        ax.plot(x, chi_mod[k], "-", lw=2.6, color=color, label=MODEL_LABELS["nonstat"])
        ax.plot(x, chi_emp[k], "o", color="black", ms=8, label="Empirical (training days)")
        n_pairs = int(((dist[iu] >= lo) & (dist[iu] < hi)).sum())
        ax.set_title(f"{lo}-{hi:g} km" if np.isfinite(hi) else f"> {lo} km")
        ax.text(0.04, 0.04, f"{n_pairs} pairs", transform=ax.transAxes, color=pbr.TEXT_MUTED)
        ax.set_xscale("function", functions=(lambda v: -np.log(1 - v), lambda t: 1 - np.exp(-t)))
        ax.set_xticks([0.8, 0.9, 0.95, 0.99])
        ax.set_xticklabels(["0.8", "0.9", "0.95", "0.99"])
        ax.set_xlabel("Level $1-q$")
        ax.set_ylim(0, 1)
        ax.grid(color=pbr.GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"$\chi(q)$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncols=3, frameon=False, fontsize=15)
    path = os.path.join(out_dir, "inference_dependence.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)
    return rows


def inference(args):
    import gsm_t_copula as T
    from Copula_BNFC import load_csv_fields
    grid, dates, meta = load_geometry(args.data_path)
    scenario = os.path.basename(os.path.normpath(args.data_path))
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    N = meta["N_train"]
    u_train, coords_obs = load_csv_fields(os.path.join(args.data_path, "sample_copula-{}.csv"),
                                          range(N))
    lon_o, lat_o, idx_o = obs_lonlat(coords_obs, grid)
    grid_coords = grid[["x1", "x2"]].values.astype(np.float32)
    qs = (0.05, 0.01)

    # ---- (1) factor maps and the correlation with a reference location ----
    share, skew, cov, gc = factor_maps(args.data_path, grid_coords, args.K)
    ref = int(np.argmin((lon_o - args.ref_lon) ** 2 + (lat_o - args.ref_lat) ** 2))
    ref_grid = idx_o[ref]
    R = T.base_correlation(grid_coords.astype(np.float64), gc or T.FIXED)
    cor_model = (R[ref_grid] + cov[ref_grid]) / np.sqrt((1 + cov[ref_grid, ref_grid]) * (1 + np.diag(cov)))
    z = np.clip(u_train, 1e-6, 1 - 1e-6)
    from scipy.stats import norm as sp_norm
    zs = sp_norm.ppf(z)
    cor_emp = np.corrcoef(zs.T)[ref]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), constrained_layout=True,
                             subplot_kw={"projection": PROJ})
    panels = [(share.mean(0), "Factor share of latent variance", "Blues", (0, None)),
              (skew.mean(0), "Skewness of factor contribution", "RdBu_r", "sym"),
              (cor_model, "Correlation with reference location", "RdBu_r", (-1, 1))]
    for k, (ax, (vals, title, cmap, lim)) in enumerate(zip(axes, panels)):
        if lim == "sym":
            a = float(np.abs(vals).max())
            norm = matplotlib.colors.Normalize(-a, a)
        else:
            norm = matplotlib.colors.Normalize(lim[0], lim[1] if lim[1] is not None else float(vals.max()))
        im = map_field(ax, vals, grid, meta, cmap, norm, title, k == 0, True)
        if os.path.exists(ETOPO):
            # relief: 500 and 1000 m contours, names of the relief units on the first two panels
            add_relief(ax, labels=k < 2)
        if k == 2:
            ax.scatter(lon_o, lat_o, c=cor_emp, cmap=cmap, norm=norm, s=60, transform=PROJ,
                       edgecolors=LAND_EDGE, linewidths=0.7, zorder=3)
            ax.scatter([lon_o[ref]], [lat_o[ref]], marker="*", s=420, c="gold",
                       edgecolors="black", transform=PROJ, zorder=4)
        cb = fig.colorbar(im, ax=ax, location="bottom", shrink=0.85, pad=0.03)
        cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)
    path = os.path.join(out_dir, "inference_factor_maps.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)
    if args.maps_only:
        return

    # ---- (2) pairwise dependence: models against the empirical copula ----
    rows = dependence(args, grid, coords_obs, u_train, out_dir)

    # ---- (3) posterior of the factor over the test period ----
    pred = os.path.join(args.results_dir, scenario, "nonstat", "predictions.npz")
    ts = None
    if os.path.exists(pred) and "factor_samples" in np.load(pred).files:
        p = np.load(pred)
        f = p["factor_samples"].mean(-1)                                # (n_days, R): spatial mean
        prcp = p["y_true"].mean(1)
        days = pd.to_datetime([dates[int(i)] for i in p["test_indices"]])
        lo, med, hi = np.quantile(f, [0.025, 0.5, 0.975], axis=1)
        ts = pd.DataFrame({"date": days, "precip_mean": prcp, "factor_median": med,
                           "factor_lo": lo, "factor_hi": hi})
        ts.to_csv(os.path.join(out_dir, "inference_factor_timeseries.csv"), index=False)
        fig, ax = plt.subplots(2, 1, figsize=(16, 8.4), sharex=True, constrained_layout=True,
                               gridspec_kw={"height_ratios": [1, 1.3]})
        ax[0].bar(days, prcp, width=1.0, color="#2a78d6")
        ax[0].set_ylabel(f"Domain mean ({UNITS})")
        ax[1].fill_between(days, lo, hi, color="#f6c4ad", label="95% posterior interval")
        ax[1].plot(days, med, color=pbr.MODEL_COLORS["nonstat"], lw=2, label="Posterior median")
        ax[1].axhline(0, color=pbr.TEXT_MUTED, lw=1)
        ax[1].set_ylabel("Centred factor\n(domain mean)")
        ax[1].legend(frameon=False, loc="upper left")
        k = int(np.argmax(prcp))
        for a in ax:
            a.axvline(days[k], color=pbr.TEXT_MUTED, ls="--", lw=1.2)
            a.grid(color=pbr.GRID, lw=0.8)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        path = os.path.join(out_dir, "inference_factor_timeseries.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("wrote", path)

    # ---- summaries for the text ----
    summ = {"reference_location": [float(lon_o[ref]), float(lat_o[ref])],
            "gaussian_component": gc,
            "range_km_east_west": None if gc is None else gc["lengthscale"] * 10 * 111.2 * np.cos(np.radians(50)),
            "range_km_north_south": None if gc is None else gc["lengthscale"] * 10 * 111.2,
            "factor_share_mean": float(share.mean()), "factor_share_min": float(share.mean(0).min()),
            "factor_share_max": float(share.mean(0).max()),
            "factor_share_sd_over_weight_samples": float(share.std(0).mean()),
            "skewness_min": float(skew.mean(0).min()), "skewness_max": float(skew.mean(0).max()),
            "corr_ref_model_vs_empirical_at_obs": float(np.corrcoef(cor_model[idx_o], cor_emp)[0, 1]),
            "rmse": rows}
    if ts is not None:
        summ["factor_vs_precip_correlation"] = float(np.corrcoef(ts.factor_median, ts.precip_mean)[0, 1])
        summ["wettest_day"] = str(ts.date.iloc[int(np.argmax(ts.precip_mean))].date())
    with open(os.path.join(out_dir, "inference_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(json.dumps(summ, indent=2, default=float))


# =========================
# Complete data with the Vecchia approximation (--vecchia_table)
# =========================
def vecchia_table(args):
    """
    BNFC (nonstat) fitted to the n_obs observed locations with the exact Gaussian
    component against BNFC (nonstat) fitted to the complete data (all grid
    locations) with the Vecchia approximation: cost of the fit, estimated
    Gaussian component, and held-out and predictive performance on the same test
    days, conditioning on the same observed locations.
    """
    import torch
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    cols = []
    for label, data_path, res in (("Observed locations, exact", args.data_path, args.results_dir),
                                  ("Complete data, Vecchia", args.full_data_path, args.full_results_dir)):
        scen = os.path.basename(os.path.normpath(data_path))
        with open(os.path.join(data_path, "models", "fit_time-nonstat.json")) as f:
            ft = json.load(f)
        ck = torch.load(os.path.join(data_path, "models", "posterior_params_nonstat.pth"))
        with open(os.path.join(res, scen, "nonstat", "metrics.json")) as f:
            met = json.load(f)["summary"]["copula"]
        ls = pd.read_csv(os.path.join(res, "heldout_loglik.csv"))
        ls = ls[(ls.scenario == scen) & (ls.model == "nonstat")].groupby("set").log_score.mean()
        cols.append({"label": label,
                     "n_sites": ft.get("n_sites"), "vecchia_m": ft.get("vecchia_m", 0),
                     "lengthscale": float(ck["lengthscale"]), "nugget": float(ck["nugget"]),
                     "seconds_per_epoch": ft.get("seconds_per_epoch"),
                     "fit_hours": (ft["time_seconds"] + ft.get("sgmcmc_seconds", 0)) / 3600,
                     "peak_gpu_memory_gb": ft.get("peak_gpu_memory_gb"),
                     "logscore_obs": ls.get("observed"), "logscore_full": ls.get("full"),
                     **{k: met[k]["mean"] for k in ("crps", "energy_score", "mae", "coverage95")}})
    df = pd.DataFrame(cols)
    df.to_csv(os.path.join(out_dir, "vecchia_complete_data.csv"), index=False)
    spec = [("Locations used for fitting", "n_sites", "{:.0f}"),
            ("Vecchia neighbours $m$", "vecchia_m", "{:.0f}"),
            ("Range $\\hat\\ell$", "lengthscale", "{:.3f}"),
            ("Nugget $\\hat\\tau$", "nugget", "{:.4f}"),
            ("Seconds per epoch", "seconds_per_epoch", "{:.0f}"),
            ("Fit time (hours)", "fit_hours", "{:.2f}"),
            ("Peak GPU memory (GB)", "peak_gpu_memory_gb", "{:.1f}"),
            ("Log score (observed)", "logscore_obs", "{:.1f}"),
            ("Log score (all locations)", "logscore_full", "{:.0f}"),
            ("CRPS", "crps", "{:.4f}"), ("Energy score", "energy_score", "{:.3f}"),
            ("MAE", "mae", "{:.4f}"), ("Coverage (95\\%)", "coverage95", "{:.3f}")]
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             "& " + " & ".join(c["label"] for c in cols) + r" \\", r"\midrule"]
    for lab, key, fmt in spec:
        cells = ["--" if (c[key] is None or (isinstance(c[key], float) and np.isnan(c[key])))
                 else "$" + fmt.format(c[key]) + "$" for c in cols]
        if key == "vecchia_m":
            cells = ["exact" if c[key] == 0 else f"${c[key]}$" for c in cols]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "vecchia_complete_data.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(df.round(4).to_string(index=False))


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Plot the ERA5 precipitation application")
    parser.add_argument("--results_dir", type=str, default="results_application/")
    parser.add_argument("--data_path", type=str, default="application_precipitation_v2/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="defaults to <results_dir>/figures")
    parser.add_argument("--replicates", type=int, nargs="*", default=None,
                        help="positions of test days to map (default: the wettest test day)")
    parser.add_argument("--bnfc", choices=["stat", "nonstat"], default="nonstat")
    parser.add_argument("--comparison", action="store_true",
                        help="also map BNFC against --comparator (mean and SD)")
    parser.add_argument("--comparator", default="mixgamma",
                        choices=["stat", "exponential", "gamma", "pareto", "mixgamma", "tcopula"])
    parser.add_argument("--scale", choices=["data", "copula"], default="copula",
                        help="scale of the table: copula (held-out log score and copula-scale "
                             "CRPS, energy score and MAE, as for the simulations) or data")
    parser.add_argument("--exclude_from_plots", nargs="*", default=[],
                        help="models left out of the metric figures (still in the table)")
    parser.add_argument("--data_root", type=str, default=".",
                        help="directory containing the data folder (fit times)")
    parser.add_argument("--inference", action="store_true",
                        help="inference on the fitted models (factor maps, dependence against "
                             "the empirical copula, factor posterior over the test period)")
    parser.add_argument("--ref_lon", type=float, default=16.37, help="reference location (Vienna)")
    parser.add_argument("--ref_lat", type=float, default=48.21)
    parser.add_argument("--n_sim", type=int, default=20000,
                        help="inference: replicates simulated from each model")
    parser.add_argument("--K", type=int, default=500, help="inference: factor nodes")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vecchia_table", action="store_true",
                        help="table of BNFC (nonstat) fitted to the observed locations against the "
                             "complete data with the Vecchia approximation")
    parser.add_argument("--full_data_path", type=str, default="application_precipitation_full/")
    parser.add_argument("--full_results_dir", type=str, default="results_application_full/")
    parser.add_argument("--margin_path", type=str, default=None,
                        help="maps: marginal network (F_inv-nonstat.pth) applied to the stored "
                             "copula draws instead of the stored data-scale draws")
    parser.add_argument("--maps_only", action="store_true",
                        help="inference: only the factor maps (Figure 3 of the manuscript)")
    parser.add_argument("--dependence_only", action="store_true",
                        help="inference: only the dependence-across-distance figure and table")
    args = parser.parse_args()
    global MARGIN_PATH
    MARGIN_PATH = args.margin_path
    if args.vecchia_table:
        vecchia_table(args)
        return
    if args.dependence_only:
        from Copula_BNFC import load_csv_fields
        grid, _, meta = load_geometry(args.data_path)
        u_train, coords_obs = load_csv_fields(
            os.path.join(args.data_path, "sample_copula-{}.csv"), range(meta["N_train"]))
        out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
        dependence(args, grid, coords_obs, u_train, out_dir)
        return
    if args.inference:
        inference(args)
        return

    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # results of the current data folder only (other folders have other data scales)
    scenario = os.path.basename(os.path.normpath(args.data_path))
    runs = {k: v for k, v in find_runs(args.results_dir).items() if k[0] == scenario}
    if not runs:
        raise SystemExit(f"no results for {scenario} under {args.results_dir}")
    pbr.SCENARIO_LABELS[scenario] = "ERA5 precipitation"
    geom = load_geometry(args.data_path)

    if (scenario, args.bnfc) in runs:
        p = np.load(os.path.join(runs[(scenario, args.bnfc)]["dir"], "predictions.npz"))
        reps = args.replicates
        if not reps:
            reps = [int(np.argmax(p["y_true"].mean(1)))]          # wettest test day
        for rep in reps:
            print("wrote", plot_bnfc_summary(runs, scenario, args.bnfc, rep, geom, out_dir))
            if args.comparison and (scenario, args.comparator) in runs:
                print("wrote", plot_comparison(runs, scenario, args.bnfc, args.comparator,
                                               rep, geom, out_dir))
    else:
        print(f"[skip maps] {scenario}: no BNFC ({args.bnfc}) run")

    plot_runs = {k: v for k, v in runs.items() if k[1] not in args.exclude_from_plots}
    for scale in ("data", "copula"):
        print("wrote", plot_metric_summary(plot_runs, scale, out_dir))
    logscore = load_logscore(args.results_dir)
    if logscore is not None:
        logscore = logscore[logscore.scenario == scenario]
    for path in summary_table(runs, out_dir, args.data_root, scale=args.scale,
                              logscore=logscore if args.scale == "copula" else None):
        print("wrote", path)


if __name__ == "__main__":
    main()
