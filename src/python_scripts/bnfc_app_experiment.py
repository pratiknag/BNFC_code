#!/usr/bin/env python
# coding: utf-8
# Created by Pratik Nag, October 2026 (approximate)

"""
Experiment folders for tuning BNFC (nonstat) on the ERA5 application without
touching application_precipitation_v2/models/.

    setup:     python bnfc_app_experiment.py setup --tag <tag>
    summarise: python bnfc_app_experiment.py summarise

An experiment folder experiments/bnfc_app/<tag>/ holds symbolic links to the copula and
data files of application_precipitation_v2/ (all training and test days) and, as
sample_copula_full-<i>.csv for the validation days, the 1,024-location copula data of
application_precipitation_full/ (identical at the 100 observed locations), so that
Train_BNFC.py, heldout_loglik.py and Copula_BNFC.py run on it unchanged and write
their models to experiments/bnfc_app/<tag>/models/.

Tuning protocol: fit on the training days before --val_start (1994-2021), score
the validation days --val_start..N_train-1 (2022) at the observed locations and on
the full grid, and select on the full-grid validation log score. The test days
(2024) are used only once, for the selected configuration refitted on all training days.
"""

import argparse
import glob
import json
import os
import shutil

import pandas as pd

SRC = "application_precipitation_v2"
FULL = "application_precipitation_full"
ROOT = "experiments/bnfc_app"


def setup(args):
    meta = json.load(open(os.path.join(SRC, "meta.json")))
    n_train, n_test = meta["N_train"], meta["N_test"]
    d = os.path.join(ROOT, args.tag)
    os.makedirs(os.path.join(d, "models"), exist_ok=True)

    def link(src, name):
        t = os.path.join(d, name)
        if not os.path.lexists(t):
            os.symlink(os.path.abspath(src), t)

    for i in range(n_train + n_test):
        link(os.path.join(SRC, f"sample_copula-{i}.csv"), f"sample_copula-{i}.csv")
        link(os.path.join(SRC, f"sample-{i}.csv"), f"sample-{i}.csv")
    for i in range(args.val_start, n_train):
        link(os.path.join(FULL, f"sample_copula-{i}.csv"), f"sample_copula_full-{i}.csv")
    for i in range(n_train, n_train + n_test):
        for f in ("sample_copula_full", "sample_full"):
            link(os.path.join(SRC, f"{f}-{i}.csv"), f"{f}-{i}.csv")
    for f in ("meta.json", "dates.csv", "grid.csv", "obs_index.txt"):
        link(os.path.join(SRC, f), f)
    shutil.copy(os.path.join(SRC, "models", "F_inv-nonstat.pth"), os.path.join(d, "models"))
    for f in args.link_models:                   # e.g. competitors' posteriors for reference scores
        link(os.path.join(SRC, "models", f), os.path.join("models", f))
    print(d)


def summarise(args):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "val", "heldout_loglik_summary.csv"))):
        tag = p.split(os.sep)[-3]
        s = pd.read_csv(p)
        row = {"tag": tag}
        for _, r in s.iterrows():
            row[f"{r.model}_{r.set}"] = r.mean_log_score
        cfg = os.path.join(ROOT, tag, "config.txt")
        if os.path.exists(cfg):
            row["config"] = open(cfg).read().strip()
        ft = os.path.join(ROOT, tag, "models", "fit_time-nonstat.json")
        if os.path.exists(ft):
            info = json.load(open(ft))
            row.update({"best_loss": info["best_loss"], "epochs": info["epochs_run"],
                        "fit_s": round(info["time_seconds"])})
        for m in glob.glob(os.path.join(ROOT, tag, "val", "m1*", "heldout_loglik.csv")):
            d = pd.read_csv(m)
            for set_name, g in d.groupby("set"):
                row[f"{d.model.iloc[0]}_{set_name}"] = g.log_score.mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(df.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--tag", required=True)
    s.add_argument("--val_start", type=int, default=10226, help="first validation day (2022-01-01)")
    s.add_argument("--link_models", nargs="*", default=[])
    sub.add_parser("summarise")
    args = parser.parse_args()
    setup(args) if args.cmd == "setup" else summarise(args)


if __name__ == "__main__":
    main()
