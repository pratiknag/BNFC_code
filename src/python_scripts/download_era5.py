#!/usr/bin/env python
# coding: utf-8

"""
Download the ERA5 hourly total precipitation over central Europe (15E-25E, 45N-55N) used in the
application, one request per year, from the Copernicus Climate Data Store (CDS):

    dataset  reanalysis-era5-single-levels, product type reanalysis
    variable total_precipitation, all hours, years 1994-2024, area [N 55, W 15, S 45, E 25]

Each year is downloaded as a zip file and extracted to

    <out_dir>/era5_precipitation_<year>/data_stream-oper_stepType-accum.nc

which is the layout read by dataGen_precipitation.py (run_application.sh, ERA5_DIR=<out_dir>).

Needs a CDS account and the cdsapi package with a key in ~/.cdsapirc, see
https://cds.climate.copernicus.eu/how-to-api. The licence of the dataset must be accepted once
on its CDS page. About 30 requests of a few minutes each, plus queueing time at the CDS.

    python src/python_scripts/download_era5.py --out_dir era5/
"""
import argparse
import os
import zipfile

YEARS = range(1994, 2025)


def request(year):
    return {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [55, 15, 45, 25],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="era5/")
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    args = parser.parse_args()

    import cdsapi
    client = cdsapi.Client()
    os.makedirs(args.out_dir, exist_ok=True)
    for year in args.years:
        folder = os.path.join(args.out_dir, f"era5_precipitation_{year}")
        if os.path.exists(os.path.join(folder, "data_stream-oper_stepType-accum.nc")):
            print(f"{year}: present, skipped")
            continue
        archive = folder + ".zip"
        print(f"{year}: requesting")
        client.retrieve("reanalysis-era5-single-levels", request(year)).download(archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(folder)
        os.remove(archive)
        print(f"{year}: extracted to {folder}")


if __name__ == "__main__":
    main()
