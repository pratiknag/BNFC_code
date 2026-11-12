#!/usr/bin/env python
# coding: utf-8

"""
Paired comparison of the baselines M1 (M1_gaussian_rank1.py) and M2 with the models
of the main comparison, per test replicate (Tables 1 and 2 of the manuscript).

Reads the held-out log scores of the main models from <main_dir>/heldout_loglik.csv
(heldout_loglik.py) and their per-replicate copula metrics from
<main_dir>/<scenario>/<model>/metrics.json (Copula_BNFC.py and the competitors), and
the same files of the baselines from <base_dir>/<scenario>/<model>/. Nothing is refitted.

Output: <base_dir>/<scenario>/comparison.csv with, for each model and measure, the
mean, its standard error and the mean paired difference to --reference with its
standard error over the test replicates.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

MEASURES = ["crps", "energy_score", "mae", "coverage95"]


def load(main_dir, base_dir, scen, main_models, base_models):
    ls = pd.read_csv(os.path.join(main_dir, "heldout_loglik.csv"))
    ls = ls[(ls.scenario == scen) & ls.model.isin(main_models)]
    rows = [ls]
    for m in base_models:
        p = os.path.join(base_dir, scen, m, "heldout_loglik.csv")
        if os.path.exists(p):
            rows.append(pd.read_csv(p))
    ls = pd.concat(rows)
    per = {}
    for m in main_models + base_models:
        d = base_dir if m in base_models else main_dir
        p = os.path.join(d, scen, m, "metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            mt = json.load(f)
        per[m] = pd.DataFrame(mt["per_replicate"]["copula"], index=mt["test_indices"])
    return ls, per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="application_precipitation_v2")
    parser.add_argument("--main_dir", default="results_application/")
    parser.add_argument("--base_dir", default="results_M1_M2/application/")
    parser.add_argument("--main_models", nargs="+",
                        default=["stat", "nonstat", "exponential", "gamma", "pareto", "tcopula"])
    parser.add_argument("--base_models", nargs="+", default=["m1", "m1_K49", "m2"])
    parser.add_argument("--reference", default="nonstat")
    args = parser.parse_args()

    ls, per = load(args.main_dir, args.base_dir, args.scenario, args.main_models, args.base_models)
    out = []
    for set_name, d in ls.groupby("set"):
        wide = d.pivot(index="replicate", columns="model", values="log_score")
        for m in wide.columns:
            diff = wide[m] - wide[args.reference]
            out.append({"measure": f"logscore_{set_name}", "model": m, "mean": wide[m].mean(),
                        "se": wide[m].std(ddof=1) / math.sqrt(len(wide)),
                        "diff_to_ref": diff.mean(),
                        "diff_se": diff.std(ddof=1) / math.sqrt(len(wide))})
    for k in MEASURES:
        if args.reference not in per:
            break
        ref = per[args.reference][k]
        for m, df in per.items():
            diff = df[k] - ref.loc[df.index]
            out.append({"measure": k, "model": m, "mean": df[k].mean(),
                        "se": df[k].std(ddof=1) / math.sqrt(len(df)),
                        "diff_to_ref": diff.mean(),
                        "diff_se": diff.std(ddof=1) / math.sqrt(len(df))})
    out = pd.DataFrame(out)
    path = os.path.join(args.base_dir, args.scenario, "comparison.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(f"paired differences: model - {args.reference}")
    print(out.round(4).to_string(index=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
