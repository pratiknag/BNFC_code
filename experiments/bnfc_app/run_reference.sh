#!/usr/bin/env bash
# Created by Pratik Nag, November 2026 (approximate)
# Reference scores for run_bnfc_app_tuning.sh: Student-t copula (Gaussian component of
# application_precipitation_v2/models/competitor_kernel.txt) and Gaussian loading copula (K = 25),
# refitted on 1994-2021 and scored on the 2022 validation days (columns tcopula_* and m1_* of
# summary.csv). Run from the repository root: bash experiments/bnfc_app/run_reference.sh
set -euo pipefail
P=${PYTHON:-env/bin/python}; S=src/python_scripts; D=experiments/bnfc_app/ref_competitors/
$P $S/bnfc_app_experiment.py setup --tag ref_competitors > /dev/null
echo "Student-t copula and M1 (K=25), fitted on 1994-2021" > $D/config.txt
$P $S/gsm_t_copula.py fit --data_path $D --N 10226 --kernel se --lengthscale 0.13131488859653473 --nugget 0.017772505059838295
$P $S/heldout_loglik.py --scenarios $D --models tcopula --test_start 10226 --n_test_reps 365 --out_dir $D/val/
$P $S/M1_gaussian_rank1.py fit --data_path $D --N 10226 --basis_grid 5
$P $S/M1_gaussian_rank1.py score --data_path $D --basis_grid 5 --test_start 10226 --n_test_reps 365 --out_dir $D/val_m1/
mkdir -p $D/val/m1 && cp $D/val_m1/ref_competitors/m1/heldout_loglik.csv $D/val/m1/
echo "[$(date '+%F %T')] reference done"
