#!/usr/bin/env python
# coding: utf-8

"""
Figure 2 of the manuscript: predictions of BNFC (nonstat) for one test day of the
ERA5 application, drawn with a larger nugget of the Gaussian component.

The fitted nugget of the application (about 1e-5) makes the predictive distribution
collapse onto the observed value at each conditioning location, so that the quantile
and SD maps show isolated points there. For the map only, the nugget is replaced by
--nugget; the network weights (pSGLD samples), the factor and the range are those of
the fit. All other results use the fitted nugget. Writes
    <results_dir>/<scenario>/nonstat_nugget<t>/predictions.npz   draws for that day
    <results_dir>/<scenario>/nonstat_nugget<t>/day_metrics.json  CRPS and 95% coverage on
                                                                 the data scale, with the
                                                                 nugget t and the fitted one
    <out_dir>/map_bnfc_nonstat_<date>_nugget<t>.png              2 x 3 maps (plot_application.py)

Run from the repository root after run_application.sh (or finalize_after_runs.sh):
    env/bin/python src/python_scripts/plot_application_nugget_map.py --nugget 0.1 --R 1000
"""

# =========================
# Imports
# =========================
import argparse
import json
import os

import numpy as np
import torch

from Copula_BNFC import (load_scenario, load_models, load_weight_samples, load_margin,
                         kriging_operators, posterior_predictive, replicate_metrics, device)
import plot_application as pa


def draw_day(u_obs, ct, cs, loc, log_scale, nets, F_inv, cov, R, L):
    """Copula and data-scale draws (1, R, n_test) of one day, R split over the weight samples."""
    kr = kriging_operators(ct, cs, cov_params=cov)
    u_b = torch.tensor(u_obs, device=device)
    draws = [posterior_predictive(u_b, ct, cs, loc, log_scale, net, F_inv, "nonstat", kr,
                                  R=R // len(nets), L=L) for net in nets]
    U = torch.cat([d[1] for d in draws], 1).cpu().numpy()
    Y = np.maximum(torch.cat([d[2] for d in draws], 1).cpu().numpy(), 0.0)
    return U, Y


def main():
    parser = argparse.ArgumentParser(description="Figure 2: BNFC (nonstat) map with a larger nugget")
    parser.add_argument("--data_path", default="application_precipitation_v2/",
                        help="data folder with models/ of the fitted BNFC (nonstat)")
    parser.add_argument("--margin_path", default="application_precipitation_full/",
                        help="folder whose models/F_inv-nonstat.pth is the marginal network")
    parser.add_argument("--results_dir", default="results_application/")
    parser.add_argument("--out_dir", default="results_application/figures/")
    parser.add_argument("--nugget", type=float, default=0.1)
    parser.add_argument("--R", type=int, default=1000, help="predictive draws")
    parser.add_argument("--L", type=int, default=200, help="factor draws per predictive draw")
    parser.add_argument("--day", type=int, default=None,
                        help="test index (default: wettest test day by domain mean)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    scen = os.path.basename(os.path.normpath(args.data_path))
    if args.day is None:
        p0 = np.load(os.path.join(args.results_dir, scen, "nonstat", "predictions.npz"))
        args.day = int(p0["test_indices"][int(np.argmax(p0["y_true"].mean(1)))])
    u_obs, coords_train, u_true, y_true, coords_test = load_scenario(args.data_path, [args.day])
    sampler, _, loc, log_scale, cov = load_models(args.data_path, "nonstat")
    nets = load_weight_samples(args.data_path, "nonstat", sampler)
    ct = torch.tensor(coords_train, device=device)
    cs = torch.tensor(coords_test, device=device)
    F_inv = load_margin(args.margin_path, "nonstat", cs)
    geom = pa.load_geometry(args.data_path)
    date = geom[1][args.day]
    print(f"day {args.day} ({date}), fitted range {cov[0].item():.3f}, nugget {cov[1].item():.2e}, "
          f"kernel {cov[2]}, {len(nets)} weight samples")

    tag = f"nugget{args.nugget:g}"
    d = os.path.join(args.results_dir, scen, f"nonstat_{tag}")
    os.makedirs(d, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {"day": args.day, "date": date, "R": args.R}
    for name, t in ((tag, torch.tensor(args.nugget, device=device)), ("fitted", cov[1])):
        U, Y = draw_day(u_obs, ct, cs, loc, log_scale, nets, F_inv, (cov[0], t, cov[2]),
                        args.R, args.L)
        m = replicate_metrics(Y[0], y_true[0])
        metrics[name] = {"nugget": float(t), "crps_data": float(m["crps"]),
                         "coverage95": float(m["coverage95"]), "mean_sd": float(m["mean_sd"])}
        print(f"{name}: nugget {float(t):.2e}, CRPS {m['crps']:.3f} mm/day, "
              f"coverage {m['coverage95']:.3f}")
        if name != tag:
            continue
        np.savez_compressed(os.path.join(d, "predictions.npz"), copula_samples=U, data_samples=Y,
                            u_obs=u_obs, u_true=u_true, y_true=y_true, coords_train=coords_train,
                            coords_test=coords_test, test_indices=np.array([args.day]))
        # plot_bnfc_summary names the file after the model and date; written next to the
        # draws, then moved, so that the map with the fitted nugget is not overwritten
        path = pa.plot_bnfc_summary({(scen, "nonstat"): {"dir": d}}, scen, "nonstat", 0, geom, d)
        out = os.path.join(args.out_dir, f"map_bnfc_nonstat_{date}_{tag}.png")
        os.replace(path, out)
        print("wrote", out)
    with open(os.path.join(d, "day_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
