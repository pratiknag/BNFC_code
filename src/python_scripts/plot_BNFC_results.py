#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
Plots and tables comparing BNFC with the parametric single-factor copula
models, from the results written by Copula_BNFC.py and
Copula_single_factor.py. Expects
    <results_dir>/<scenario>/<model>/{predictions.npz, metrics.json}
with <model> in {stat, nonstat, exponential, gamma, pareto, mixgamma, tcopula}
(stat and nonstat average over the pSGLD weight samples). Nothing here re-runs a model.

Figures (sized for the manuscript, large headings):
  heatmaps_<scenario>_rep<k>.png : true field vs BNFC vs a comparator predictive
                                   mean and SD on the data scale
  metrics_<scale>.png            : CRPS, energy score and MAE
                                   for all models and scenarios
Tables:
  summary_table.csv / summary_table.tex
"""

# =========================
# Imports
# =========================
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Models in a fixed order; categorical colours assigned in that order, never cycled.
MODEL_ORDER = ["stat", "nonstat", "exponential", "gamma", "pareto", "mixgamma", "tcopula"]
MODEL_LABELS = {
    "stat": "BNFC (stat)",
    "nonstat": "BNFC (nonstat)",
    "exponential": "Exponential",
    "gamma": "Gamma",
    "pareto": "Pareto",
    "mixgamma": "Gamma mixture",
    "tcopula": "Student-t",
    "m1": "Gaussian loading",
    "m2": "Gamma loading",
}
MODEL_COLORS = {
    "stat": "#2a78d6",
    "nonstat": "#eb6834",
    "exponential": "#1baf7a",
    "gamma": "#eda100",
    "pareto": "#e87ba4",
    "mixgamma": "#7a5cc7",
    "tcopula": "#8a6a3d",
    "m1": "#5b6770",
    "m2": "#b5452f",
}
SCENARIO_LABELS = {
    "simulations_gamma": "Gamma-factor data",
    "simulations_signchange": "Sign-change data",
    "simulations_mixgamma": "Regional mixture data",
}
# two-line model names for crowded x axes (five or more models per panel)
TICK_LABELS = {
    "stat": "BNFC\n(stat)",
    "nonstat": "BNFC\n(nonstat)",
    "exponential": "Expo-\nnential",
    "gamma": "Gamma",
    "pareto": "Pareto",
    "mixgamma": "Gamma\nmix.",
    "tcopula": "Student-\n$t$",
    "m1": "Gauss.\nloading",
    "m2": "Gamma\nloading",
}
# loading copulas (M1_gaussian_rank1.py, M2_gamma_varying_loading.py), only in the log-score
# plot and the heatmaps
LOADING_MODELS = ["m1", "m2"]
COMMON_FACTORS = ["exponential", "gamma", "pareto", "mixgamma"]
TEXT_MUTED = "#52514e"
GRID = "#e6e5e1"

# Sequential maps: single hue, light -> dark.
CMAP_VALUE = "Blues"
CMAP_SD = "Purples"
# heatmaps with all models of Table 1 (--loading_dir): copula values on a scale centred at the
# median 0.5 (red above, blue below), SDs on a neutral white -> charcoal scale (small = light),
# capped at the 99th percentile over all models so that a few large values do not wash it out
CMAP_FIELD = "Spectral_r"
CMAP_FIELD_SD = matplotlib.colors.LinearSegmentedColormap.from_list("ink", ["#ffffff", "#26262b"])
FIELD_TITLESIZE, FIELD_LABELSIZE, FIELD_TICKSIZE = 30, 28, 25
FIELD_TITLES = {"exponential": "Exponential", "gamma": "Gamma", "mixgamma": "Gamma mixture",
                "pareto": "Pareto", "tcopula": "Student-$t$", "m1": "Gauss. loading",
                "m2": "Gamma loading"}
COLORBAR_TICKSIZE = 21

METRICS = [
    ("crps", "CRPS"),
    ("energy_score", "Energy score"),
    ("mae", "MAE"),
]

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 17,
    "legend.fontsize": 20,
    "figure.titlesize": 26,
})


# =========================
# Loading
# =========================
def find_runs(results_dir, models=MODEL_ORDER):
    runs = {}
    for mpath in sorted(glob.glob(os.path.join(results_dir, "*", "*", "metrics.json"))):
        with open(mpath) as f:
            metrics = json.load(f)
        if models is not None and metrics["factor"] not in models:
            continue
        runs[(metrics["scenario"], metrics["factor"])] = {
            "dir": os.path.dirname(mpath), "metrics": metrics,
            "scenario": metrics["scenario"], "factor": metrics["factor"],
        }
    return runs


def scenario_label(name):
    return SCENARIO_LABELS.get(name, name.replace("simulations_", ""))


# =========================
# Spatial helpers
# =========================
def to_grid(values, coords):
    xs = np.unique(np.round(coords[:, 0], 6))
    ys = np.unique(np.round(coords[:, 1], 6))
    grid = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, np.round(coords[:, 0], 6))
    iy = np.searchsorted(ys, np.round(coords[:, 1], 6))
    grid[iy, ix] = values
    return grid, [xs.min(), xs.max(), ys.min(), ys.max()]


def show_field(ax, values, coords, cmap, vmin, vmax, title):
    grid, extent = to_grid(values, coords)
    im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def observed_values(y_true_rep, coords_test, coords_train):
    """True data at the observed sites (they are a subset of the grid)."""
    lookup = {(round(float(x), 5), round(float(y), 5)): k for k, (x, y) in enumerate(coords_test)}
    idx = [lookup[(round(float(x), 5), round(float(y), 5))] for x, y in coords_train]
    return y_true_rep[idx]


# =========================
# Heatmap comparison: BNFC vs Gamma
# =========================
def choose_bnfc(runs, scenario, preference, scale="data"):
    if preference in ("stat", "nonstat"):
        return preference if (scenario, preference) in runs else None
    cands = [f for f in ("stat", "nonstat") if (scenario, f) in runs]
    if not cands:
        return None
    return min(cands, key=lambda f: runs[(scenario, f)]["metrics"]["summary"][scale]["crps"]["mean"])


def second_best(runs, scenario, exclude, scale="data"):
    """Model with the lowest CRPS on `scale` among those other than `exclude`."""
    cands = [m for (scen, m) in runs if scen == scenario and m != exclude]
    if not cands:
        return None
    return min(cands, key=lambda m: runs[(scenario, m)]["metrics"]["summary"][scale]["crps"]["mean"])


def plot_heatmaps(runs, scenario, bnfc, comparator, rep, out_dir, scale="data"):
    """
    2 x 3 panel on the data scale.
      row 1: true field | BNFC predictive mean | comparator predictive mean
      row 2: observed data | BNFC predictive SD | comparator predictive SD
    Means share one colour scale, and the two SDs share another, so that the
    predictive uncertainty of the two models can be compared directly.
    """
    pb = np.load(os.path.join(runs[(scenario, bnfc)]["dir"], "predictions.npz"))
    pc = np.load(os.path.join(runs[(scenario, comparator)]["dir"], "predictions.npz"))
    coords = pb["coords_test"]
    obs = pb["coords_train"]
    samples, truth_key = (("copula_samples", "u_true") if scale == "copula"
                          else ("data_samples", "y_true"))
    truth = pb[truth_key][rep]

    mean_b, sd_b = pb[samples][rep].mean(0), pb[samples][rep].std(0)
    mean_c, sd_c = pc[samples][rep].mean(0), pc[samples][rep].std(0)

    vmin = float(min(truth.min(), mean_b.min(), mean_c.min()))
    vmax = float(max(truth.max(), mean_b.max(), mean_c.max()))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2), constrained_layout=True)
    show_field(axes[0, 0], truth, coords, CMAP_VALUE, vmin, vmax, "True field")
    show_field(axes[0, 1], mean_b, coords, CMAP_VALUE, vmin, vmax,
               f"{MODEL_LABELS[bnfc]}: mean")
    im = show_field(axes[0, 2], mean_c, coords, CMAP_VALUE, vmin, vmax,
                    f"{MODEL_LABELS[comparator]}: mean")
    cb = fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    # observed data at the 100 conditioning sites, on the mean colour scale
    ax = axes[1, 0]
    y_obs = observed_values(truth, coords, obs)
    ax.scatter(obs[:, 0], obs[:, 1], c=y_obs, cmap=CMAP_VALUE, vmin=vmin, vmax=vmax,
               s=45, edgecolors=TEXT_MUTED, linewidths=0.6)
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_aspect("equal")
    ax.set_title("Observed data")
    ax.set_xticks([])
    ax.set_yticks([])

    sd_max = float(max(sd_b.max(), sd_c.max()))
    for ax, model, sd in ((axes[1, 1], bnfc, sd_b), (axes[1, 2], comparator, sd_c)):
        im_sd = show_field(ax, sd, coords, CMAP_SD, 0.0, sd_max, f"{MODEL_LABELS[model]}: SD")
    cb = fig.colorbar(im_sd, ax=axes[1, 1:], shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=COLORBAR_TICKSIZE)

    idx = int(pb["test_indices"][rep])
    fig.suptitle(f"{scenario_label(scenario)}: predictive mean and SD"
                 + (" (copula scale)" if scale == "copula" else ""), fontweight="bold")
    path = os.path.join(out_dir, f"heatmaps_{scenario}_rep{idx}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_heatmaps_models(runs, scenario, models, rep, out_dir, scale="data"):
    """
    2 x (1 + len(models)) panel, the models in the order of Table 1.
      row 1: true field with the observed locations marked | predictive mean of each model
      row 2: (empty)                                       | predictive SD of each model
    Means and true field share one colour scale, centred at 0.5; the SDs share another.
    """
    preds = {m: np.load(os.path.join(runs[(scenario, m)]["dir"], "predictions.npz")) for m in models}
    p0 = preds[models[0]]
    coords, obs = p0["coords_test"], p0["coords_train"]
    samples, truth_key = (("copula_samples", "u_true") if scale == "copula"
                          else ("data_samples", "y_true"))
    truth = p0[truth_key][rep]
    mean = {m: p[samples][rep].mean(0) for m, p in preds.items()}
    sd = {m: p[samples][rep].std(0) for m, p in preds.items()}
    half = float(max(np.abs(truth - 0.5).max(), *(np.abs(v - 0.5).max() for v in mean.values())))
    vmin, vmax = 0.5 - half, 0.5 + half
    sd_max = float(np.quantile(np.concatenate(list(sd.values())), 0.99))

    nc = 1 + len(models)
    fig, axes = plt.subplots(2, nc, figsize=(4.5 * nc, 9.6), constrained_layout=True)
    show_field(axes[0, 0], truth, coords, CMAP_FIELD, vmin, vmax, "True field")
    axes[0, 0].scatter(obs[:, 0], obs[:, 1], s=22, facecolors="none", edgecolors="black",
                       linewidths=1.0)
    axes[1, 0].axis("off")
    for j, m in enumerate(models, start=1):
        title = FIELD_TITLES.get(m, MODEL_LABELS[m])
        im = show_field(axes[0, j], mean[m], coords, CMAP_FIELD, vmin, vmax, title)
        im_sd = show_field(axes[1, j], sd[m], coords, CMAP_FIELD_SD, 0.0, sd_max, "")
    for ax in axes[0]:
        ax.title.set_fontsize(FIELD_TITLESIZE)
    axes[0, 1].set_ylabel("Predictive mean", fontsize=FIELD_LABELSIZE)
    axes[1, 1].set_ylabel("Predictive SD", fontsize=FIELD_LABELSIZE)
    for row, mappable in ((0, im), (1, im_sd)):
        cb = fig.colorbar(mappable, ax=axes[row, :], shrink=0.9, pad=0.01,
                          extend="max" if row == 1 else "neither")
        cb.ax.tick_params(labelsize=FIELD_TICKSIZE)

    idx = int(p0["test_indices"][rep])
    path = os.path.join(out_dir, f"heatmaps_{scenario}_rep{idx}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# =========================
# Metrics across models and scenarios
# =========================
def plot_metric_summary(runs, scale, out_dir):
    scenarios = [s for s in SCENARIO_LABELS if any(k[0] == s for k in runs)]
    scenarios += sorted({k[0] for k in runs} - set(scenarios))
    models = [m for m in MODEL_ORDER if any(k[1] == m for k in runs)]
    width = 0.72
    offsets = {m: (i - (len(models) - 1) / 2) * width / len(models) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 6.4), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), METRICS):
        for (scen, model), run in runs.items():
            if model not in offsets or scen not in scenarios:
                continue
            s = run["metrics"]["summary"][scale][key]
            x = scenarios.index(scen) + offsets[model]
            ax.errorbar(x, s["mean"], yerr=2 * s["se"], fmt="o", ms=10, lw=2.5, capsize=0,
                        color=MODEL_COLORS[model])
        ax.set_yscale("log")
        for b in range(len(scenarios) - 1):          # separate the scenarios
            ax.axvline(b + 0.5, color=TEXT_MUTED, lw=1, ls="--")
        ax.set_title(label)
        ax.set_xticks(range(len(scenarios)))
        # one word per line, so that three scenario names fit under each panel
        ax.set_xticklabels([scenario_label(s).replace(" ", "\n") for s in scenarios],
                           fontsize=15)
        ax.set_xlim(-0.6, len(scenarios) - 0.4)
        ax.grid(axis="y", color=GRID, lw=0.8, which="both")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=MODEL_COLORS[m],
                          label=MODEL_LABELS[m]) for m in models]
    fig.legend(handles=handles, loc="outside lower center", frameon=False,
               ncols=(len(models) + 1) // 2)
    path = os.path.join(out_dir, f"metrics_{scale}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# =========================
# Held-out copula log score (heldout_loglik.py)
# =========================
def load_logscore(results_dir):
    """Per-replicate held-out log scores, or None if heldout_loglik.py has not been run."""
    path = os.path.join(results_dir, "heldout_loglik.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def logscore_summary(ls):
    """Mean log score per replicate and its standard error, by scenario, set and model."""
    g = ls.groupby(["scenario", "set", "model"]).log_score
    return pd.DataFrame({"mean": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.count())})


def best_common_factor(ls, scenario):
    """Common factor copula with the highest mean held-out log score over the full grid."""
    d = ls[(ls.scenario == scenario) & (ls.set == "full") & ls.model.isin(COMMON_FACTORS)]
    return d.groupby("model").log_score.mean().idxmax() if len(d) else None


# reference BNFC variant of each scenario in the log-score plot: BNFC (stat) for the
# gamma-factor data, whose factor has the same effect at every location, else BNFC (nonstat)
LOGSCORE_REFERENCE = {"simulations_gamma": "stat"}


def plot_logscore(ls, out_dir, reference=None, exclude=("pareto",), best_only=True,
                  sets=("full",)):
    """
    Paired difference in held-out log score per location, model minus the reference
    BNFC variant of each scenario (`reference`: dict scenario -> model, default
    LOGSCORE_REFERENCE, else "nonstat"), one panel per scenario. By default only the
    log score over all grid locations ("full"); with sets=("observed", "full") also at
    the observed locations (filled markers; full grid open). The other BNFC variant is
    shown only when the reference is BNFC (stat). Bars are +-2 standard errors of the
    paired differences; values below zero mean that the reference assigns a higher
    density to the test replicates. With best_only, the common factor copulas are
    represented by the best of them in each scenario (best_common_factor), as in Table 1.
    """
    reference = LOGSCORE_REFERENCE if reference is None else reference
    ref_of = (lambda s: reference.get(s, "nonstat")) if isinstance(reference, dict) \
        else (lambda s: reference)
    scenarios = [s for s in SCENARIO_LABELS if s in set(ls.scenario)]
    present = set(ls.model)

    def models_for(s):
        ref = ref_of(s)
        out = []
        for m in MODEL_ORDER + LOADING_MODELS:
            if m not in present or m == ref or m in exclude:
                continue
            if m == "stat" and ref == "nonstat":
                continue
            if best_only and m in COMMON_FACTORS and m != best_common_factor(ls, s):
                continue
            out.append(m)
        return out

    models_by = {s: models_for(s) for s in scenarios}
    n_max = max(len(m) for m in models_by.values())
    all_sets = {"observed": ("o", "Observed locations"), "full": ("D", "All grid locations")}
    sets = [(name, *all_sets[name]) for name in sets]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(max(18, 4.8 * n_max), 6.6),
                             constrained_layout=True, sharey=False)
    for ax, scen in zip(np.atleast_1d(axes), scenarios):
        ref, models = ref_of(scen), models_by[scen]
        for k, (set_name, marker, _) in enumerate(sets):
            d = ls[(ls.scenario == scen) & (ls.set == set_name)]
            wide = d.pivot(index="replicate", columns="model", values="log_score")
            n_sites = d.n_sites.iloc[0]
            for i, m in enumerate(models):
                diff = (wide[m] - wide[ref]) / n_sites
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                x = i + (k - (len(sets) - 1) / 2) * 0.3
                filled = set_name == "observed" or len(sets) == 1
                ax.errorbar(x, diff.mean(), yerr=2 * se, fmt=marker, ms=11, lw=2.5, capsize=0,
                            color=MODEL_COLORS[m], mfc=MODEL_COLORS[m] if filled else "white",
                            mew=2.5)
        ax.axhline(0, color=MODEL_COLORS[ref], lw=2)
        ax.set_title(scenario_label(scen), fontsize=28)
        ax.tick_params(axis="y", labelsize=21)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([f"{TICK_LABELS[m]}\n(best)"
                            if best_only and m in COMMON_FACTORS
                            else TICK_LABELS.get(m, MODEL_LABELS[m]) for m in models], fontsize=19)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    refs = {ref_of(s) for s in scenarios}
    ref_label = MODEL_LABELS[refs.pop()] if len(refs) == 1 else "BNFC"
    np.atleast_1d(axes)[0].set_ylabel(f"Log score per location\nminus {ref_label}", fontsize=24)
    if len(sets) > 1:
        handles = [plt.Line2D([], [], marker=mk, ls="", ms=10, color=TEXT_MUTED,
                              mfc=TEXT_MUTED if s == "observed" else "white", mew=2.5, label=lab)
                   for s, mk, lab in sets]
        fig.legend(handles=handles, loc="outside lower center", frameon=False, ncols=2,
                   fontsize=24)
    path = os.path.join(out_dir, "logscore_differences.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# =========================
# Tables
# =========================
def fit_time(data_root, scen, model):
    """Fitting time from <data_root>/<scen>/models: fit_time-<model>.json (BNFC:
    the ELBO fit plus the pSGLD sampling of the weights), hmc_posterior-<model>.json
    (common factor) or posterior-tcopula.json. Returns (seconds, estimated)."""
    models_dir = os.path.join(data_root, scen, "models")
    name = {"stat": "fit_time-stat.json", "nonstat": "fit_time-nonstat.json",
            "tcopula": "posterior-tcopula.json"}.get(model, f"hmc_posterior-{model}.json")
    path = os.path.join(models_dir, name)
    if not os.path.exists(path):
        return None, False
    with open(path) as f:
        d = json.load(f)
    return d["time_seconds"] + d.get("sgmcmc_seconds", 0.0), bool(d.get("estimated", False))


def summary_table(runs, out_dir, data_root=".", fit_time_row=True, scale="data", logscore=None):
    rows = []
    for (scen, model), run in runs.items():
        m = run["metrics"]
        d = m["summary"][scale]
        rows.append({
            "scenario": scen, "model": model,
            **{f"{k}": d[k]["mean"] for k in d},
            **{f"{k}_se": d[k]["se"] for k in d},
            "copula_crps": m["summary"]["copula"]["crps"]["mean"],
            "copula_crps_se": m["summary"]["copula"]["crps"]["se"],
            "prediction_time_seconds": m["time_seconds"],
        })
    df = pd.DataFrame(rows)
    if logscore is not None:
        # held-out copula log score per replicate: observed locations and full grid
        ls = logscore_summary(logscore).reset_index()
        for set_name, key in (("observed", "logscore_obs"), ("full", "logscore_full")):
            d = ls[ls.set == set_name].rename(columns={"mean": key, "se": key + "_se"})
            df = df.merge(d[["scenario", "model", key, key + "_se"]], on=["scenario", "model"], how="left")
    csv_path = os.path.join(out_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)

    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    scenarios = [s for s in SCENARIO_LABELS if s in set(df["scenario"])]
    scenarios += sorted(set(df["scenario"]) - set(scenarios))

    # ---- LaTeX table (booktabs): mean with standard error, best value in bold ----
    bnfc = [m for m in models if m in ("stat", "nonstat")]
    para = [m for m in models if m not in ("stat", "nonstat", "tcopula")]
    gsm = [m for m in models if m == "tcopula"]
    models = bnfc + para + gsm
    short = {"stat": "stat", "nonstat": "nonstat", "exponential": "Exponential",
             "gamma": "Gamma", "pareto": "Pareto", "mixgamma": "Gamma mix.",
             "tcopula": "Student-$t$"}
    roman = ["i", "ii", "iii", "iv", "v"]
    # (label, key, decimals, best = "min" or "truth")
    spec = [("CRPS", "crps", 3, "min"),
            ("Energy score", "energy_score", 2, "min"),
            ("MAE", "mae", 3, "min"),
            ("Skewness", "skewness_pred", 2, "truth"),
            ("Kurtosis", "kurtosis_pred", 2, "truth"),
            ("CRPS (copula)", "copula_crps", 4, "min")]
    if scale == "copula":
        # on the copula scale the margins are uniform: no shape rows
        spec = [("CRPS", "crps", 4, "min"),
                ("Energy score", "energy_score", 3, "min"),
                ("MAE", "mae", 4, "min")]
        if logscore is None:
            spec.append(("Coverage (95\\%)", "coverage95", 3, "nominal"))
        else:
            spec += [("Log score (observed)", "logscore_obs", 1, "max"),
                     ("Log score (all locations)", "logscore_full", 0, "max")]
    truth_key = {"skewness_pred": "skewness_true", "kurtosis_pred": "kurtosis_true"}

    def value(scen, model, key):
        r = df[(df.scenario == scen) & (df.model == model)]
        return None if r.empty else r.iloc[0]

    truth_col = scale == "data"          # the true field has no shape rows on the copula scale
    ncol = len(models) + 1 + truth_col
    lines = [r"\begin{tabular}{l" + "c" * (len(models) + truth_col) + "}", r"\toprule"]
    head = [""]
    rules = []
    col = 2
    if bnfc:
        head.append(r"\multicolumn{%d}{c}{BNFC}" % len(bnfc))
        rules.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(bnfc) - 1))
        col += len(bnfc)
    if para:
        head.append(r"\multicolumn{%d}{c}{Common factor copula}" % len(para))
        rules.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(para) - 1))
        col += len(para)
    if gsm:
        head.append(r"\multicolumn{%d}{c}{Scale mixture}" % len(gsm))
        rules.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(gsm) - 1))
    if truth_col:
        head.append("")
    lines.append(" & ".join(head) + r" \\")
    lines.append("".join(rules))
    lines.append("Measure & " + " & ".join(short[m] for m in models)
                 + (r" & True field" if truth_col else "") + r" \\")

    for i, scen in enumerate(scenarios):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textbf{(%s) %s}} \\[2pt]"
                     % (ncol, roman[i], scenario_label(scen)))
        for label, key, dec, best in spec:
            rows_m = {m: value(scen, m, key) for m in models}
            means = {m: r[key] for m, r in rows_m.items() if r is not None}
            truth = None
            if best == "truth":
                truth = df[df.scenario == scen][truth_key[key]].iloc[0]
                target = min(means, key=lambda m: abs(means[m] - truth))
            elif best == "nominal":
                target = min(means, key=lambda m: abs(means[m] - 0.95))
            elif best == "max":
                target = max(means, key=means.get)
            else:
                target = min(means, key=means.get)
            cells = []
            for m in models:
                r = rows_m[m]
                if r is None:
                    cells.append("--")
                    continue
                if pd.isna(r[key]):
                    cells.append("--")
                    continue
                v = f"{r[key]:.{dec}f}"
                v = r"$\mathbf{" + v + "}$" if m == target else f"${v}$"
                cells.append(v + r"\,{\scriptsize$(" + f"{r[key + '_se']:.{dec}f}" + ")$}")
            t = "--" if truth is None else f"${truth:.{dec}f}$"
            lines.append(r"\quad " + label + " & " + " & ".join(cells)
                         + (f" & {t}" if truth_col else "") + r" \\")
        if not fit_time_row:
            continue
        cells = []
        for m in models:
            sec, est = fit_time(data_root, scen, m)
            cells.append("--" if sec is None else f"${sec:.0f}$" + (r"$^{*}$" if est else ""))
        lines.append(r"\quad Fit time (s) & " + " & ".join(cells)
                     + (r" & --" if truth_col else "") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(out_dir, "summary_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    if scale == "data":
        truth = df.groupby("scenario")[["skewness_true", "kurtosis_true"]].first()
        truth.to_csv(os.path.join(out_dir, "true_shape.csv"))
    return csv_path, tex_path


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Plot BNFC vs single-factor copula results")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="defaults to <results_dir>/figures")
    parser.add_argument("--replicates", type=int, nargs="+", default=[0],
                        help="positions (not file indices) of test replicates to map")
    parser.add_argument("--bnfc", choices=["best", "stat", "nonstat"], default="best",
                        help="BNFC variant in the heatmaps (best = lowest data-scale CRPS)")
    parser.add_argument("--bnfc_for", nargs="*",
                        default=["simulations_gamma=stat", "simulations_signchange=nonstat",
                                 "simulations_mixgamma=nonstat"],
                        help="per-scenario BNFC variant overriding --bnfc, as scenario=variant")
    parser.add_argument("--data_root", type=str, default=".",
                        help="directory containing the simulations_* folders (fit times)")
    parser.add_argument("--comparator", default="second",
                        choices=["second", "stat", "nonstat", "exponential", "gamma", "pareto",
                                 "mixgamma", "tcopula"],
                        help="model next to BNFC in the heatmaps (second = second-best model "
                             "by data-scale CRPS, i.e. the best model other than the BNFC shown)")
    parser.add_argument("--comparator_for", nargs="*",
                        default=["simulations_gamma=gamma", "simulations_signchange=tcopula",
                                 "simulations_mixgamma=mixgamma"],
                        help="per-scenario comparator overriding --comparator, as scenario=model")
    parser.add_argument("--scale", choices=["data", "copula"], default="data",
                        help="scale of the heatmaps, the table and the model ranking")
    parser.add_argument("--no_fit_time", action="store_true",
                        help="omit the fit-time rows (e.g. when only the prediction differs)")
    parser.add_argument("--extra_logscore", nargs="*", default=[],
                        help="further heldout_loglik.csv files (e.g. of the loading copulas) "
                             "added to the log-score plot only")
    parser.add_argument("--loading_dir", default=None,
                        help="results of the loading copulas (e.g. results_M1_M2/simulations/); "
                             "if given, the heatmaps show the models of Table 1: BNFC, the best "
                             "common factor, the Student-t copula and the two loading copulas")
    parser.add_argument("--best_from", default="results/",
                        help="directory with heldout_loglik.csv that picks the best common factor")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    runs = find_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no results found under {args.results_dir}")

    bnfc_for = dict(item.split("=", 1) for item in args.bnfc_for)
    comparator_for = dict(item.split("=", 1) for item in args.comparator_for)

    logscore = load_logscore(args.results_dir)
    loading_runs = (find_runs(args.loading_dir, models=LOADING_MODELS) if args.loading_dir else {})

    for scenario in sorted({k[0] for k in runs}):
        bnfc = choose_bnfc(runs, scenario, bnfc_for.get(scenario, args.bnfc), args.scale)
        if args.loading_dir:
            ls_best = load_logscore(args.best_from)
            best = best_common_factor(ls_best, scenario) if ls_best is not None else None
            models = [m for m in (bnfc, best, "tcopula", "m1", "m2")
                      if m is not None and ((scenario, m) in runs or (scenario, m) in loading_runs)]
            for rep in args.replicates:
                print("wrote", plot_heatmaps_models({**runs, **loading_runs}, scenario, models,
                                                    rep, out_dir, args.scale), f"({models})")
            continue
        comparator = comparator_for.get(scenario, args.comparator)
        if comparator == "second" and bnfc is not None:
            comparator = second_best(runs, scenario, bnfc, args.scale)
        if bnfc is None or (scenario, comparator) not in runs:
            print(f"[skip heatmaps] {scenario}: need a BNFC run and a {comparator} run")
            continue
        n_rep = np.load(os.path.join(runs[(scenario, bnfc)]["dir"], "predictions.npz"))["y_true"].shape[0]
        for rep in args.replicates:
            if rep < n_rep:
                print(f"wrote {plot_heatmaps(runs, scenario, bnfc, comparator, rep, out_dir, args.scale)}"
                      f"  (BNFC variant: {bnfc}, comparator: {comparator})")

    for scale in ("data", "copula"):
        print("wrote", plot_metric_summary(runs, scale, out_dir))
    if logscore is not None and {"stat", "nonstat"} <= set(logscore.model):
        extra = [pd.read_csv(p) for p in args.extra_logscore if os.path.exists(p)]
        print("wrote", plot_logscore(pd.concat([logscore] + extra, ignore_index=True), out_dir))
    for p in summary_table(runs, out_dir, args.data_root, fit_time_row=not args.no_fit_time,
                           scale=args.scale, logscore=logscore if args.scale == "copula" else None):
        print("wrote", p)


if __name__ == "__main__":
    main()
