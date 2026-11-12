"""
Relief of the ERA5 domain for the factor maps of the application (Figure 3 of the manuscript)
and its relation to the factor share and the skewness of the factor contribution.

Elevation: ETOPO1 (NOAA, 1 arc-minute global relief), every second node over 15-25 E,
45-55 N, downloaded from the NOAA CoastWatch ERDDAP server (dataset etopo180) to
application_precipitation_v2/topography/etopo1_15-25E_45-55N.nc.

Run from the repository root: writes results_application/figures/topography_summary.json
(rank correlations of the factor share and skewness with the mean elevation of each grid
cell, and their averages by elevation class).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ETOPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     "application_precipitation_v2/topography/etopo1_15-25E_45-55N.nc")

# main relief units of the domain, (lon, lat) of the label
REGIONS = {
    "Eastern\nAlps": (15.75, 47.35),
    "Sudetes": (16.2, 50.95),
    "Western\nCarpathians": (19.6, 49.45),
    "Eastern\nCarpathians": (24.45, 47.9),
    "Southern\nCarpathians": (23.6, 45.3),
    "Apuseni": (22.75, 46.55),
    "Pannonian\nBasin": (19.6, 46.75),
    "North European Plain": (19.5, 52.6),
}


def load_etopo(path=ETOPO):
    """lon (m,), lat (k,), elevation (k, m) in metres."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        return (np.asarray(d["longitude"][:], float), np.asarray(d["latitude"][:], float),
                np.asarray(d["altitude"][:], float))


def cell_elevation(grid, path=ETOPO):
    """Mean elevation of the ETOPO nodes nearest to each grid location (its grid cell)."""
    lon, lat, z = load_etopo(path)
    glon, glat = np.unique(grid["lon"].values), np.unique(grid["lat"].values)
    ix = np.abs(lon[:, None] - glon[None]).argmin(1)          # ETOPO lon -> grid column
    iy = np.abs(lat[:, None] - glat[None]).argmin(1)
    out = np.zeros((len(glat), len(glon)))
    cnt = np.zeros_like(out)
    np.add.at(out, (iy[:, None], ix[None, :]), z)
    np.add.at(cnt, (iy[:, None], ix[None, :]), 1)
    out /= cnt
    col = np.searchsorted(glon, grid["lon"].values)
    row = np.searchsorted(glat, grid["lat"].values)
    return out[row, col]


def add_relief(ax, levels=(500, 1000), labels=True, fontsize=10.5):
    """Elevation contours (thin: 500 m, thick: 1000 m) and the names of the relief units."""
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe
    lon, lat, z = load_etopo()
    from scipy.ndimage import gaussian_filter
    zs = gaussian_filter(z, 1.5)
    ax.contour(lon, lat, zs, levels=list(levels), colors="#6b3f12",
               linewidths=[0.9, 1.8][:len(levels)], alpha=0.9, transform=ccrs.PlateCarree(),
               zorder=2)
    if labels:
        for name, (x, y) in REGIONS.items():
            ax.text(x, y, name, transform=ccrs.PlateCarree(), ha="center", va="center",
                    fontsize=fontsize, style="italic", color="#1f1e1c", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def summary(share, skew, grid):
    elev = cell_elevation(grid)
    s, k = share.mean(0), skew.mean(0)
    cls = pd.cut(elev, [-np.inf, 200, 500, 1000, np.inf], labels=["<200 m", "200-500 m", "500-1000 m", ">1000 m"])
    df = pd.DataFrame({"elev": elev, "share": s, "skew": k, "cls": cls})
    by = df.groupby("cls", observed=True).agg(n=("share", "size"), share=("share", "mean"),
                                              skew=("skew", "mean"))
    return {
        "spearman_share_elevation": float(spearmanr(elev, s)[0]),
        "spearman_skew_elevation": float(spearmanr(elev, k)[0]),
        "by_elevation_class": {c: {kk: float(v) for kk, v in r.items()} for c, r in by.iterrows()},
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_application import factor_maps, load_geometry
    grid, _, _ = load_geometry("application_precipitation_v2/")
    share, skew, _, _ = factor_maps("application_precipitation_v2/",
                                    grid[["x1", "x2"]].values.astype(np.float32), 2000)
    out = summary(share, skew, grid)
    print(json.dumps(out, indent=1))
    with open("results_application/figures/topography_summary.json", "w") as f:
        json.dump(out, f, indent=1)
