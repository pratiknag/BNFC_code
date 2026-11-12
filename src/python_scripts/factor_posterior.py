#!/usr/bin/env python
# coding: utf-8

"""
Posterior of the factor given the observations of a test replicate, under each
fitted model and under the true data-generating model of the simulations.

By Remark 1 of the manuscript, the network g is not identifiable, but the
centred factor contribution

    g~(V, s) = g(V, s) - E g(V, s)

is: the Gaussian component has unit variance in every model, so its law at
every location, and its conditional law given the data, are determined by the
process. The same holds for the common factor models (V0 - E V0, the same at
every location) and for the truth T~(U, s) = T(U, s) - E T(U, s).

True posterior. The data scale of the simulations is the latent field itself,
so the observations w at the conditioning sites give the exact posterior of U on
a midpoint grid u_k, k = 1..K, with weights proportional to
phi_Sigma{w - T(u_k, s_obs)}, and hence that of T~(U, s) at every grid location.
The realised factor of each test replicate is recovered by replaying the random
number generator of the data generator (verified by rebuilding a saved field).

Model posteriors. Copula_BNFC.py and Copula_single_factor.py save the draws of
the centred factor contribution, g~(v*, s) with v* ~ p(v | w), as factor_samples
in predictions.npz (R draws per test replicate).

Measures, at all grid locations and averaged over the test replicates:
    coverage of the realised factor contribution T~(U_i, s) by the central
    (1 - a) posterior intervals, and ACE (as in plot_calibration.py);
    relative W1: the Wasserstein-1 distance between the model posterior and
    the true posterior, averaged over locations and divided by the average prior
    standard deviation of T~(U, s), so that 0 is exact and 1 is of the order of
    the prior spread.

Output in <results_dir>/figures/: factor_posterior.png (posterior along a
transect for one test replicate), factor_posterior_table.csv / .tex.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gamma as sp_gamma

import dataGen_MixGamma
import dataGen_SignChange
from dataGen_SignChange import se_correlation
from plot_BNFC_results import (SCENARIO_LABELS, MODEL_LABELS, MODEL_COLORS, TEXT_MUTED, GRID,
                               scenario_label)

FACTOR_MODELS = ["stat", "nonstat", "exponential", "gamma", "pareto", "mixgamma"]
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
N_TRAIN, N_TEST, N_OBS = 2000, 100, 100


# =========================
# True data-generating models: T(u, coords) -> (..., n), seed, factor draw
# =========================
def gamma_T(u, coords):
    q = sp_gamma.ppf(np.asarray(u), a=6.0, scale=3.0)[..., None]
    return np.broadcast_to(q, q.shape[:-1] + (len(coords),))


TRUTH = {
    # factor draw -> u in (0, 1), in the order of the generator's RNG calls
    "simulations_gamma": dict(T=gamma_T, seed=2025,
                              draw=lambda rng: sp_gamma.cdf(rng.gamma(6.0, 3.0), a=6.0, scale=3.0)),
    "simulations_signchange": dict(T=dataGen_SignChange.target_function, seed=2026,
                                   draw=lambda rng: rng.uniform()),
    "simulations_mixgamma": dict(T=dataGen_MixGamma.target_function, seed=2027,
                                 draw=lambda rng: rng.uniform()),
}


def grid_coords(grid=30):
    x = np.linspace(0, 1, grid)
    xv, yv = np.meshgrid(x, x)
    return np.column_stack((xv.ravel(), yv.ravel()))


def realised_factors(data_path, truth):
    """u_i of the test replicates, by replaying the generator; verified on one replicate."""
    coords = grid_coords()
    rng = np.random.default_rng(truth["seed"])
    rng.choice(len(coords), N_OBS, replace=False)
    u, z_first = [], None
    for i in range(N_TRAIN + N_TEST):
        ui = truth["draw"](rng)
        z = rng.standard_normal(len(coords))
        if i >= N_TRAIN:
            u.append(ui)
            if z_first is None:
                z_first = z
    L = np.linalg.cholesky(se_correlation(coords))
    W = truth["T"](u[0], coords) + L @ z_first
    saved = pd.read_csv(os.path.join(data_path, f"sample_full-{N_TRAIN}.csv"))
    assert np.allclose(saved[["x1", "x2"]].values, coords)
    err = np.abs(W - saved["y"].values).max()
    if err > 1e-6:
        raise SystemExit(f"{data_path}: replayed field differs from the saved one (max error {err:.2e})")
    return np.array(u)


# =========================
# True posterior of the centred factor contribution
# =========================
def true_posterior(data_path, truth, coords_cond, coords_test, test_indices, K):
    """
    Weights (n_rep, K) over the nodes u_k, and the centred factor contribution
    at the nodes, Tc (K, n_test), with its prior standard deviation (n_test,).
    """
    u_nodes = (np.arange(K) + 0.5) / K
    T_test = truth["T"](u_nodes, coords_test)                       # (K, n_test)
    Tc = T_test - T_test.mean(0)
    T_cond = truth["T"](u_nodes, coords_cond)                       # (K, n)
    L = np.linalg.cholesky(se_correlation(coords_cond))
    logw = []
    for i in test_indices:
        df = pd.read_csv(os.path.join(data_path, f"sample-{i}.csv"))
        lookup = {(round(x, 4), round(y, 4)): v for x, y, v in zip(df.x1, df.x2, df.y)}
        w = np.array([lookup[(round(float(x), 4), round(float(y), 4))] for x, y in coords_cond])
        white = np.linalg.solve(L, (w[None, :] - T_cond).T)        # (n, K)
        logw.append(-0.5 * (white ** 2).sum(0))
    logw = np.array(logw)
    wts = np.exp(logw - logw.max(1, keepdims=True))
    return wts / wts.sum(1, keepdims=True), Tc, Tc.std(0)


def weighted_quantiles(values, order, weights, probs):
    """values (K, m) with per-column ascending order (K, m), weights (K,): (len(probs), m)."""
    v = np.take_along_axis(values, order, 0)
    cw = np.cumsum(weights[order], 0)                                # (K, m)
    out = np.empty((len(probs), values.shape[1]))
    for j in range(values.shape[1]):
        k = np.searchsorted(cw[:, j], probs * cw[-1, j]).clip(0, len(v) - 1)
        out[:, j] = v[k, j]
    return out


# =========================
# Evaluation of one scenario
# =========================
def evaluate(results_dir, data_path, K, rep_plot, loading_dir=None):
    scen = os.path.basename(os.path.normpath(data_path))
    truth = TRUTH[scen]
    runs = {m: os.path.join(results_dir, scen, m, "predictions.npz") for m in FACTOR_MODELS}
    runs = {m: p for m, p in runs.items() if os.path.exists(p)}
    if not runs:
        return [], None
    p0 = np.load(next(iter(runs.values())))
    coords_cond, coords_test = p0["coords_train"], p0["coords_test"]
    test_indices = p0["test_indices"]
    u_real = realised_factors(data_path, truth)[test_indices - N_TRAIN]
    wts, Tc, prior_sd = true_posterior(data_path, truth, coords_cond, coords_test, test_indices, K)
    order = np.argsort(Tc, 0)
    realised = truth["T"](u_real, coords_test) - truth["T"]((np.arange(K) + 0.5) / K,
                                                             coords_test).mean(0)  # (n_rep, n_test)
    probs = (np.arange(100) + 0.5) / 100

    rows, plot = [], {"coords": coords_test, "realised": realised[rep_plot], "models": {}}
    q_true_plot = weighted_quantiles(Tc, order, wts[rep_plot], np.array([0.025, 0.5, 0.975]))
    plot["true"] = q_true_plot
    for m, path in runs.items():
        F = np.load(path)["factor_samples"]                           # (n_rep, R, n_test or 1)
        F = np.broadcast_to(F, F.shape[:2] + (len(coords_test),))
        cov = []
        for lev in LEVELS:
            lo, hi = np.quantile(F, [(1 - lev) / 2, (1 + lev) / 2], axis=1)
            cov.append(np.mean((realised >= lo) & (realised <= hi)))
        cov = np.array(cov)
        w1 = []
        for r in range(F.shape[0]):
            qm = np.quantile(F[r], probs, axis=0)                     # (100, n_test)
            qt = weighted_quantiles(Tc, order, wts[r], probs)
            w1.append(np.abs(qm - qt).mean(0).mean())
        rows.append({"scenario": scen, "model": m,
                     **{f"cov{int(round(l * 100))}": c for l, c in zip(LEVELS, cov)},
                     "ace": float(np.mean(np.abs(cov - LEVELS))),
                     "rel_w1": float(np.mean(w1) / prior_sd.mean()),
                     "rel_w1_se": float(np.std(w1, ddof=1) / np.sqrt(len(w1)) / prior_sd.mean())})
        plot["models"][m] = np.quantile(F[rep_plot], [0.025, 0.5, 0.975], axis=0)
    # models shown in the figure only (not in the table), e.g. the gamma loading copula
    for m in set(PANEL_MODELS[scen]) - set(runs):
        path = os.path.join(loading_dir or results_dir, scen, m, "predictions.npz")
        if loading_dir is None or not os.path.exists(path):
            continue
        P = np.load(path)
        assert np.array_equal(P["test_indices"], test_indices)
        F = np.broadcast_to(P["factor_samples"], P["factor_samples"].shape[:2] + (len(coords_test),))
        plot["models"][m] = np.quantile(F[rep_plot], [0.025, 0.5, 0.975], axis=0)
    plot["test_index"] = int(test_indices[rep_plot])
    return rows, plot


# =========================
# Figure: posterior along a transect
# =========================
TRANSECT = {  # (coordinate that varies, fixed coordinate value)
    "simulations_gamma": ("s1", 0.5),
    "simulations_signchange": ("s2", 0.5),
    "simulations_mixgamma": ("s1", 0.5),
}


# models shown in each panel: BNFC (stat) alone for the gamma-factor data,
# BNFC (nonstat) and the gamma loading copula otherwise
PANEL_MODELS = {"simulations_gamma": ("stat",),
                "simulations_signchange": ("nonstat", "m2"),
                "simulations_mixgamma": ("nonstat", "m2")}
# both BNFC models in one colour: they never share a panel
COLORS = {**MODEL_COLORS, "nonstat": MODEL_COLORS["stat"]}


def plot_transects(plots, out_dir):
    scen = [s for s in SCENARIO_LABELS if s in plots]
    fig, axes = plt.subplots(1, len(scen), figsize=(6.4 * len(scen), 6.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scen):
        pl = plots[sc]
        c = pl["coords"]
        var, fixed = TRANSECT[sc]
        k_var, k_fix = (0, 1) if var == "s1" else (1, 0)
        level = np.unique(c[:, k_fix])[np.argmin(np.abs(np.unique(c[:, k_fix]) - fixed))]
        sel = np.where(np.isclose(c[:, k_fix], level))[0]
        sel = sel[np.argsort(c[sel, k_var])]
        x = c[sel, k_var]
        qt = pl["true"][:, sel]
        ax.fill_between(x, qt[0], qt[2], color="#a9a7a1", alpha=0.55, lw=0)
        ax.plot(x, qt[1], color="#3d3c39", lw=2.6)
        for m in PANEL_MODELS[sc]:
            if m not in pl["models"]:
                continue
            q = pl["models"][m][:, sel]
            ax.fill_between(x, q[0], q[2], color=COLORS[m], alpha=0.22, lw=0)
            ax.plot(x, q[0], color=COLORS[m], lw=1.6, ls="--")
            ax.plot(x, q[2], color=COLORS[m], lw=1.6, ls="--")
            ax.plot(x, q[1], color=COLORS[m], lw=3.6)
        ax.set_title(scenario_label(sc))
        ax.set_xlabel(f"${var[0]}_{var[1]}$ (with ${'s_2' if var == 's1' else 's_1'}={level:.2f}$)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Centred factor contribution")
    from matplotlib.patches import Patch
    handles = [Patch(color="#a9a7a1", alpha=0.55, label="True posterior (95%)"),
               plt.Line2D([], [], color="#3d3c39", lw=2.6, label="True posterior median")]
    labels = {**MODEL_LABELS, "stat": "BNFC", "nonstat": "BNFC"}
    for m in ("stat", "nonstat", "gamma", "mixgamma", "m2"):
        if any(m in pl["models"] and m in PANEL_MODELS[sc] for sc, pl in plots.items()) \
                and labels[m] not in [h.get_label().split(" (")[0] for h in handles]:
            handles.append(Patch(color=COLORS[m], alpha=0.5,
                                 label=f"{labels[m]} (median, 95%)"))
    fig.legend(handles=handles, loc="outside lower center", frameon=False, ncols=3, fontsize=15)
    path = os.path.join(out_dir, "factor_posterior.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_table(df, out_dir):
    df.to_csv(os.path.join(out_dir, "factor_posterior_table.csv"), index=False)
    models = [m for m in FACTOR_MODELS if m in set(df.model)]
    short = {"stat": "stat", "nonstat": "nonstat", "exponential": "Exp.", "gamma": "Gamma",
             "pareto": "Pareto", "mixgamma": "Gamma mix.", "m1": "Gauss. load.", "m2": "Gamma load."}
    spec = [("Coverage 50\\%", "cov50", 3, "nominal", 0.5),
            ("Coverage 80\\%", "cov80", 3, "nominal", 0.8),
            ("Coverage 95\\%", "cov95", 3, "nominal", 0.95),
            ("ACE", "ace", 3, "min", None),
            ("Relative $W_1$", "rel_w1", 3, "min", None)]
    nb = sum(m in ("stat", "nonstat") for m in models)
    lines = [r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
             " & " + (r"\multicolumn{%d}{c}{BNFC} & " % nb if nb else "")
             + r"\multicolumn{%d}{c}{%s} \\" % (len(models) - nb, "Varying loading" if set(models) <= {"m1", "m2"}
                                                  else "Common factor copula"),
             (r"\cmidrule(lr){2-%d}" % (1 + nb) if nb else "")
             + r"\cmidrule(lr){%d-%d}" % (2 + nb, 1 + len(models)),
             "Measure & " + " & ".join(short[m] for m in models) + r" \\"]
    roman = ["i", "ii", "iii"]
    for i, sc in enumerate([s for s in SCENARIO_LABELS if s in set(df.scenario)]):
        lines += [r"\midrule", r"\multicolumn{%d}{l}{\textbf{(%s) %s}} \\[2pt]"
                  % (len(models) + 1, roman[i], scenario_label(sc))]
        d = df[df.scenario == sc].set_index("model")
        for label, key, dec, best, nominal in spec:
            vals = {m: d.loc[m, key] for m in models if m in d.index}
            target = (min(vals, key=lambda m: abs(vals[m] - nominal)) if best == "nominal"
                      else min(vals, key=vals.get))
            cells = [("--" if m not in vals else
                      (r"$\mathbf{%.*f}$" % (dec, vals[m]) if m == target else "$%.*f$" % (dec, vals[m])))
                     for m in models]
            lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "factor_posterior_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Posterior of the factor against the true posterior")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--scenarios", nargs="+",
                        default=["simulations_gamma/", "simulations_signchange/", "simulations_mixgamma/"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--K", type=int, default=2000, help="quadrature nodes for the true posterior")
    parser.add_argument("--loading_dir", type=str, default=None,
                        help="results of the loading copulas (m2 is shown in the figure only)")
    parser.add_argument("--rep", type=int, default=0, help="test replicate (position) in the figure")
    parser.add_argument("--models", nargs="+", default=FACTOR_MODELS,
                        help="models under <results_dir>/<scenario>/ (m1, m2: the loading copulas)")
    args = parser.parse_args()
    FACTOR_MODELS[:] = args.models
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17})

    rows, plots = [], {}
    for d in args.scenarios:
        r, pl = evaluate(args.results_dir, d, args.K, args.rep, args.loading_dir)
        rows += r
        if pl is not None:
            plots[os.path.basename(os.path.normpath(d))] = pl
    df = pd.DataFrame(rows)
    write_table(df, out_dir)
    print("wrote", plot_transects(plots, out_dir))
    pd.set_option("display.width", 200)
    print(df[["scenario", "model", "cov50", "cov80", "cov95", "ace", "rel_w1", "rel_w1_se"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
