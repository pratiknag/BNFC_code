"""
Rows of the two comparison tables of the main text, from the saved outputs:

  Table 1 (simulation study):  results/simulation_table_rows.tex
  Table 2 (ERA5 application):  results_application/figures/application_table_rows.tex

Columns: BNFC (stat), BNFC (nonstat), best common factor copula (highest mean held-out log
score over the full grid), Student-t copula, Gaussian loading copula (m1), gamma loading
copula (m2). Sources:

  CRPS, energy score, MAE, log scores: <comparison dir>/comparison.csv (compare_M1_M2.py);
      means over the test replicates with standard errors sd / sqrt(n)
  PICP and MPIW of the 90% intervals: predictions.npz of each model, at the grid locations
      that are not conditioned on (800 in the simulations, 924 in the application)
  fit times: <data dir>/models/*.json; for BNFC the fit plus the pSGLD weight samples.
      Fit times marked "estimated" in fit_time-*.json are marked with an asterisk.

The best value of each measure is in bold (PICP: closest to 0.90). Run from the repository
root after run_simulations.sh and run_application.sh:

  python src/python_scripts/make_main_tables.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

COMMON_FACTORS = ["exponential", "gamma", "pareto", "mixgamma"]
COLUMNS = ["stat", "nonstat", "best", "tcopula", "m1", "m2"]
FACTOR_NAMES = {"exponential": "exponential", "gamma": "gamma", "pareto": "Pareto",
                "mixgamma": "gamma mixture"}
SIMULATIONS = [("simulations_gamma", "(i) Gamma-factor data"),
               ("simulations_signchange", "(ii) Sign-change data"),
               ("simulations_mixgamma", "(iii) Regional mixture data")]
# label, comparison.csv measure, decimals, rule
ROWS = [("CRPS", "crps", 4, "min"), ("Energy score", "energy_score", 3, "min"),
        ("MAE", "mae", 4, "min"), ("Log score (observed)", "logscore_observed", 1, "max"),
        ("Log score (all locations)", "logscore_full", 0, "max"),
        ("PICP $90\\%$", "picp", 3, "p90"), ("MPIW $90\\%$", "mpiw", 3, "min")]


def se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x))


def interval_scores(path, level=0.90):
    """Per-replicate PICP and MPIW of the central `level` interval at the unobserved locations."""
    p = np.load(path)
    observed = {(round(float(a), 5), round(float(b), 5)) for a, b in p["coords_train"]}
    keep = np.array([(round(float(a), 5), round(float(b), 5)) not in observed
                     for a, b in p["coords_test"]])
    samples, u = p["copula_samples"][:, :, keep], p["u_true"][:, keep]
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
    return ((u >= lo) & (u <= hi)).mean(1), (hi - lo).mean(1)


def fit_time(models_dir, m):
    """Wall-clock fit time in seconds and whether it is an estimate."""
    if m in ("stat", "nonstat"):
        j = json.load(open(os.path.join(models_dir, f"fit_time-{m}.json")))
        return j["time_seconds"] + j.get("sgmcmc_seconds", 0), bool(j.get("estimated"))
    name = f"hmc_posterior-{m}.json" if m in COMMON_FACTORS else f"posterior-{m}.json"
    return json.load(open(os.path.join(models_dir, name)))["time_seconds"], False


def cell(value, err, decimals, bold):
    a, b = f"{value:.{decimals}f}", f"{err:.{decimals}f}"
    a = f"\\mathbf{{{a}}}" if bold else a
    return f"${a}$\\,{{\\scriptsize$({b})$}}"


def block(comparison_csv, prediction_path, models_dir):
    """Table rows of one data set. prediction_path(model) -> predictions.npz."""
    c = pd.read_csv(comparison_csv)
    full = c[(c.measure == "logscore_full") & c.model.isin(COMMON_FACTORS)]
    best = full.sort_values("mean").model.iloc[-1]
    models = [best if m == "best" else m for m in COLUMNS]
    val = {}
    for m in models:
        for _, key, _, _ in ROWS[:5]:
            r = c[(c.measure == key) & (c.model == m)].iloc[0]
            val[m, key] = (r["mean"], r["se"])
        picp, mpiw = interval_scores(prediction_path(m))
        val[m, "picp"], val[m, "mpiw"] = (picp.mean(), se(picp)), (mpiw.mean(), se(mpiw))
    lines = []
    for label, key, d, rule in ROWS:
        v = [val[m, key][0] for m in models]
        if rule == "p90":
            b = int(np.argmin([abs(x - 0.9) for x in v]))
        else:
            b = int(np.argmin(v) if rule == "min" else np.argmax(v))
        cells = [cell(*val[m, key], d, i == b) for i, m in enumerate(models)]
        lines.append(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
    times = []
    for m in models:
        t, estimated = fit_time(models_dir, m)
        times.append(f"${t:.0f}$" + ("$^{*}$" if estimated else ""))
    lines.append("\\quad Fit time (s) & " + " & ".join(times) + " \\\\")
    return best, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip_simulations", action="store_true")
    parser.add_argument("--skip_application", action="store_true")
    args = parser.parse_args()

    if not args.skip_simulations:
        out = ["% Table 1: rows written by make_main_tables.py"]
        for i, (sc, title) in enumerate(SIMULATIONS):
            def pred(m, sc=sc):
                if m in ("m1", "m2"):
                    return f"results_M1_M2/simulations/{sc}/{m}/predictions.npz"
                return f"results/{sc}/{m}/predictions.npz"
            best, lines = block(f"results_M1_M2/simulations/{sc}/comparison.csv", pred,
                                f"{sc}/models")
            if i:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{title}}} \\quad "
                       f"(best common factor: {FACTOR_NAMES[best]})}} \\\\[2pt]")
            out += lines
        path = "results/simulation_table_rows.tex"
        open(path, "w").write("\n".join(out) + "\n")
        print("\n".join(out))
        print("wrote", path)

    if not args.skip_application:
        D = "application_precipitation_v2"

        def pred(m):
            if m in ("m1", "m2"):
                return f"results_M1_M2/application/{D}/{m}/predictions.npz"
            return f"results_application/{D}/{m}/predictions.npz"
        best, lines = block(f"results_M1_M2/application/{D}/comparison.csv", pred, f"{D}/models")
        out = [f"% Table 2: rows written by make_main_tables.py (best common factor: "
               f"{FACTOR_NAMES[best]})"] + lines
        os.makedirs("results_application/figures", exist_ok=True)
        path = "results_application/figures/application_table_rows.tex"
        open(path, "w").write("\n".join(out) + "\n")
        print("\n".join(out))
        print("wrote", path)


if __name__ == "__main__":
    main()
