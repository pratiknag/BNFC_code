#!/usr/bin/env python
# coding: utf-8

"""
Calibration of the posterior predictive distributions on the copula scale,
from the predictions written by Copula_BNFC.py (BNFC with pSGLD weight samples, stat and
nonstat), Copula_single_factor.py and gsm_t_copula.py. Nothing is re-run.

Calibration is assessed at the unobserved grid locations only: at the
conditioning locations the prediction is almost exact for every model.
For each model and scenario:

    coverage of the central (1 - a) prediction intervals, a = 0.9, ..., 0.05
    ACE    : mean absolute difference between empirical and nominal coverage
    width  : mean width of the 95% intervals (sharpness)
    PIT    : probability integral transform of the truth under the predictive
             draws (randomised ranks), whose histogram is flat under calibration

Output in <results_dir>/figures/: calibration.png, pit_histograms.png,
calibration_table.csv and calibration_table.tex. With --logscore_dir, the
held-out log score at all grid locations is added to the table. With --loading_dir, the
Gaussian and gamma loading copulas are added, and the figures show the models of Table 1:
the common factor copulas are represented by the best of them in each scenario (highest
held-out log score over the full grid in --best_from).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_BNFC_results import (find_runs, scenario_label, SCENARIO_LABELS, MODEL_LABELS,
                               MODEL_COLORS, TEXT_MUTED, GRID, load_logscore,
                               best_common_factor, COMMON_FACTORS, LOADING_MODELS)

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
ORDER = ["stat", "nonstat", "exponential", "gamma", "pareto", "mixgamma", "tcopula", "m1", "m2"]
LABELS = MODEL_LABELS
COLORS = MODEL_COLORS
# in the figures of the paper (with `best`), one BNFC model per scenario, labelled "BNFC":
# BNFC (stat) for the gamma-factor data, BNFC (nonstat) otherwise
BNFC_MODEL = {"simulations_gamma": "stat"}
BNFC_COLOR = MODEL_COLORS["stat"]


def bnfc_model(sc):
    return BNFC_MODEL.get(sc, "nonstat")


def color_of(m, best):
    return BNFC_COLOR if best and m in ("stat", "nonstat") else COLORS[m]


def unobserved_mask(coords_test, coords_train):
    obs = {(round(float(x), 5), round(float(y), 5)) for x, y in coords_train}
    return np.array([(round(float(x), 5), round(float(y), 5)) not in obs for x, y in coords_test])


def calibration(path, seed=0):
    """Coverage per level, 95% width and PIT values at the unobserved locations."""
    p = np.load(path)
    keep = unobserved_mask(p["coords_test"], p["coords_train"])
    S = p["copula_samples"][:, :, keep]                     # (n_rep, R, m)
    y = p["u_true"][:, keep]                                 # (n_rep, m)
    cov = []
    for lev in LEVELS:
        lo, hi = np.quantile(S, [(1 - lev) / 2, (1 + lev) / 2], axis=1)
        cov.append(np.mean((y >= lo) & (y <= hi)))
    lo, hi = np.quantile(S, [0.025, 0.975], axis=1)
    rng = np.random.default_rng(seed)
    below = (S < y[:, None]).sum(1)
    ties = (S == y[:, None]).sum(1)
    pit = (below + rng.uniform(size=below.shape) * (ties + 1)) / (S.shape[1] + 1)
    return {"coverage": np.array(cov), "width95": float(np.mean(hi - lo)), "pit": pit.ravel()}


def collect(results_dir, loading_dir=None):
    runs = find_runs(results_dir, models=ORDER)
    if loading_dir:
        runs.update(find_runs(loading_dir, models=LOADING_MODELS))
    out = {}
    for (scen, model), run in runs.items():
        out[(scen, model)] = calibration(os.path.join(run["dir"], "predictions.npz"))
    return out


def scenarios_of(cal):
    s = [x for x in SCENARIO_LABELS if any(k[0] == x for k in cal)]
    return s + sorted({k[0] for k in cal} - set(s))


def shown(cal, sc, m, best):
    """Whether model m is drawn for scenario sc: with `best`, only the best common factor."""
    if best is not None and m in ("stat", "nonstat"):
        return (sc, m) in cal and m == bnfc_model(sc)
    return (sc, m) in cal and m != "pareto" and (best is None or m not in COMMON_FACTORS
                                                 or m == best.get(sc))


def plot_coverage(cal, out_dir, title, best=None):
    scen = scenarios_of(cal)
    fig, axes = plt.subplots(1, len(scen), figsize=(5.6 * len(scen), 5.8),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    models = [m for m in ORDER if any(k[1] == m for k in cal)]
    for ax, sc in zip(axes, scen):
        ax.axhline(0, color=TEXT_MUTED, lw=1.2)
        for m in models:
            if not shown(cal, sc, m, best):
                continue
            c = cal[(sc, m)]["coverage"]
            ax.plot(LEVELS, c - LEVELS, marker="o", ms=6, lw=2.2, color=color_of(m, best))
        ax.set_title(scenario_label(sc))
        ax.set_xlabel("Nominal coverage")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Empirical minus nominal coverage")
    drawn = [m for m in models if any(shown(cal, sc, m, best) for sc in scen)]
    if best:   # one legend entry for the BNFC model of each scenario
        drawn = ["stat"] + [m for m in drawn if m not in ("stat", "nonstat")]
    label = {m: (f"{LABELS[m]} (best common factor)" if best and m in COMMON_FACTORS else LABELS[m])
             for m in drawn}
    if best:
        label["stat"] = "BNFC"
    handles = [plt.Line2D([], [], color=color_of(m, best), lw=2.2, marker="o", label=label[m])
               for m in drawn]
    fig.legend(handles=handles, loc="outside lower center", frameon=False,
               ncols=3 if best else 4, fontsize=17 if best else 15)
    fig.suptitle(title, fontweight="bold")
    path = os.path.join(out_dir, "calibration.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_pit(cal, out_dir, models=("stat", "nonstat", "mixgamma", "tcopula"), best=None):
    """PIT histograms, scenarios in rows. With `best`, the columns are BNFC (stat for the
    gamma-factor data, nonstat otherwise), the best common factor of each scenario and the
    Gaussian and gamma loading copulas."""
    scen = scenarios_of(cal)
    if best:
        models = ["bnfc", "best", "m1", "m2"]
    models = [m for m in models if m in ("best", "bnfc") or any(k[1] == m for k in cal)]
    fig, axes = plt.subplots(len(scen), len(models), figsize=(4.2 * len(models), 3.4 * len(scen)),
                             constrained_layout=True, squeeze=False)
    bins = np.linspace(0, 1, 11)
    for i, sc in enumerate(scen):
        for j, col in enumerate(models):
            ax = axes[i, j]
            m = best.get(sc) if col == "best" else bnfc_model(sc) if col == "bnfc" else col
            if (sc, m) in cal:
                ax.hist(cal[(sc, m)]["pit"], bins=bins, density=True, color=color_of(m, best),
                        alpha=0.9, edgecolor="white")
                if col == "best":
                    ax.text(0.5, 1.88, LABELS[m], ha="center", va="top", fontsize=16,
                            color=TEXT_MUTED)
            ax.axhline(1, color=TEXT_MUTED, lw=1.2, ls="--")
            ax.set_ylim(0, 2)
            ax.set_xticks([0, 0.5, 1])
            if i == 0:
                ax.set_title("Best common factor" if col == "best" else "BNFC" if col == "bnfc"
                             else LABELS[m],
                             fontsize=20 if best else 16)
            if j == 0:
                ax.set_ylabel(scenario_label(sc).replace(" data", ""), fontsize=20 if best else 16)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    path = os.path.join(out_dir, "pit_histograms.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def table(cal, out_dir, logscore=None):
    rows = []
    for (sc, m), c in cal.items():
        r = {"scenario": sc, "model": m, "width95": c["width95"],
             "ace": float(np.mean(np.abs(c["coverage"] - LEVELS)))}
        r.update({f"cov{int(round(l * 100))}": v for l, v in zip(LEVELS, c["coverage"])})
        rows.append(r)
    df = pd.DataFrame(rows)
    if logscore is not None:
        ls = logscore[logscore.set == "full"].groupby(["scenario", "model"]).log_score.mean()
        df["logscore_full"] = [ls.get((s, m), np.nan) for s, m in zip(df.scenario, df.model)]
    df.to_csv(os.path.join(out_dir, "calibration_table.csv"), index=False)

    models = [m for m in ORDER if m in set(df.model)]
    short = {"stat": "stat", "nonstat": "nonstat", "exponential": "Exp.", "gamma": "Gamma", "pareto": "Pareto",
             "mixgamma": "Gamma mix.", "tcopula": "Student-$t$", "m1": "Gaussian", "m2": "Gamma"}
    spec = [("Coverage 50\\%", "cov50", 3, "nominal", 0.5),
            ("Coverage 80\\%", "cov80", 3, "nominal", 0.8),
            ("Coverage 95\\%", "cov95", 3, "nominal", 0.95),
            ("ACE", "ace", 3, "min", None),
            ("Width 95\\%", "width95", 3, None, None)]
    if logscore is not None:
        spec.append(("Log score (all)", "logscore_full", 0, "max", None))
    lines = [r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule"]
    groups = [("BNFC", [m for m in models if m in ("stat", "nonstat")]),
              ("Common factor copula", [m for m in models if m in
                                        ("exponential", "gamma", "pareto", "mixgamma")]),
              ("Scale mixture", [m for m in models if m == "tcopula"]),
              ("Varying loading", [m for m in models if m in LOADING_MODELS])]
    head, rules, col = [""], [], 2
    for g, ms in groups:
        if ms:
            head.append(r"\multicolumn{%d}{c}{%s}" % (len(ms), g))
            rules.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(ms) - 1))
            col += len(ms)
    lines += [" & ".join(head) + r" \\", "".join(rules),
              "Measure & " + " & ".join(short[m] for m in models) + r" \\"]
    roman = ["i", "ii", "iii", "iv"]
    for i, sc in enumerate(scenarios_of(cal)):
        lines += [r"\midrule", r"\multicolumn{%d}{l}{\textbf{(%s) %s}} \\[2pt]"
                  % (len(models) + 1, roman[i], scenario_label(sc))]
        d = df[df.scenario == sc].set_index("model")
        for label, key, dec, best, nominal in spec:
            vals = {m: d.loc[m, key] for m in models if m in d.index and not pd.isna(d.loc[m, key])}
            target = None
            if not vals:
                best = None
            if best == "nominal":
                target = min(vals, key=lambda m: abs(vals[m] - nominal))
            elif best == "min":
                target = min(vals, key=vals.get)
            elif best == "max":
                target = max(vals, key=vals.get)
            cells = []
            for m in models:
                if m not in vals:
                    cells.append("--")
                    continue
                v = f"{vals[m]:.{dec}f}"
                cells.append(r"$\mathbf{" + v + "}$" if m == target else f"${v}$")
            lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(out_dir, "calibration_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return df


def main():
    parser = argparse.ArgumentParser(description="Calibration of the predictive distributions")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--logscore_dir", type=str, default=None,
                        help="directory with heldout_loglik.csv (default: results_dir)")
    parser.add_argument("--title", type=str, default="Calibration at the unobserved locations")
    parser.add_argument("--loading_dir", type=str, default=None,
                        help="results of the loading copulas (e.g. results_M1_M2/simulations/); "
                             "if given, they are added and the figures show the models of Table 1")
    parser.add_argument("--best_from", type=str, default="results/",
                        help="directory with heldout_loglik.csv that picks the best common factor")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17})

    cal = collect(args.results_dir, args.loading_dir)
    best = None
    if args.loading_dir:
        ls = load_logscore(args.best_from)
        best = {sc: best_common_factor(ls, sc) for sc in scenarios_of(cal)}
    print("wrote", plot_coverage(cal, out_dir, args.title, best))
    print("wrote", plot_pit(cal, out_dir, best=best))
    ls = load_logscore(args.logscore_dir or args.results_dir)
    if ls is not None and args.loading_dir:
        extra = sorted(glob.glob(os.path.join(args.loading_dir, "*", "m[12]", "heldout_loglik.csv")))
        ls = pd.concat([ls] + [pd.read_csv(f) for f in extra], ignore_index=True)
    df = table(cal, out_dir, ls)
    pd.set_option("display.width", 220)
    print(df[["scenario", "model", "cov50", "cov80", "cov95", "ace", "width95"]
             + (["logscore_full"] if "logscore_full" in df else [])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
