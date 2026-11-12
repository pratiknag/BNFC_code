#!/usr/bin/env bash
# Created by Pratik Nag, October 2026 (approximate)
# ERA5 precipitation application: Sections 5.2-5.3 of the manuscript and Sections S7-S8 of the
# supplement. Run from the repository root:
#
#   setsid nohup bash run_application.sh > logs/run_application.log 2>&1 < /dev/null &
#
#   PYTHON        Python interpreter (default env/bin/python if present, else python)
#   BATRAM_PYTHON interpreter of the transport map environment (default env_batram/bin/python)
#   ERA5_DIR      folder with era5_precipitation_<year>/
#                 (default ../FNO-DSTM-EnKF/datasets/precip_reanalysis_datasets)
#   HMC_FACTORS   common factor copulas (default "exponential gamma pareto")
#   FORCE=1       regenerate the data and refit every model. By default a data or fitting stage
#                 is skipped when its output exists, so an interrupted run can be resumed.
#                 Predictions, scores, tables and figures are always recomputed.
#   DRY_RUN=1     print the commands without running them
#
# Stages (per-stage logs in logs/app_*.log):
#   1. data: 2-day mean daily precipitation on a 32 x 32 grid, randomized empirical PIT with one
#      tie-breaking draw per day; 100 observed locations (application_precipitation_v2/) and the
#      same days with all 1,024 locations observed (application_precipitation_full/)
#   2. marginal quantile network T_psi, trained on the training days at all 1,024 locations and
#      shared by every model
#   3. preliminary BNFC (stat) fit; its range and nugget (models/competitor_kernel.txt) define the
#      Gaussian component of the common factor copulas and the Student-t copula
#   4. common factor copulas (HMC/NUTS) and the Student-t copula
#   5. BNFC (stat), then BNFC (nonstat) with configs/bnfc_application.conf (chosen on the
#      validation days of 2022 by run_bnfc_app_tuning.sh, which need not be rerun), each followed
#      by 20 pSGLD weight samples; Gaussian loading copula (K = 25, 49) and gamma loading copula
#   6. predictions given the 100 observed locations, held-out copula log score, paired comparison
#      with BNFC (nonstat): results_M1_M2/application/application_precipitation_v2/comparison.csv
#      (Tables 2 and S8)
#   7. results_application/figures/: rows of Table 2 (application_table_rows.tex), calibration,
#      Figure 2 (nugget 0.1 for the map only), Figure 3, Figures S8-S9, posterior summaries of
#      Section S7 (bayes_insights.json) and the relation to the relief (topography_summary.json;
#      ETOPO1 is downloaded if missing)
#   8. complete data: BNFC (nonstat) with the Vecchia approximation (m = 30) on all 1,024
#      locations, predictions on an 80 x 80 grid (results_application_full/dense/, Figure S10),
#      and the timing benchmark of Table S9 (results_application_full/benchmark/)
#   9. Bayesian transport map at the 100 observed locations (CPU, env_batram/; ERA5 column of
#      Table S6): results_M1_M2/transport_map_application/comparison.csv
#
# Runtime on one RTX 4090: about 30 hours, most of it in the HMC fits (the gamma loading copula
# alone takes about 14 hours) and BNFC (nonstat); stage 9 adds about 35 minutes on 24 CPU threads.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${PYTHON:-}" ]]; then P=$PYTHON
elif [[ -x env/bin/python ]]; then P=env/bin/python
else P=python; fi
ERA5_DIR=${ERA5_DIR:-era5/}
BATRAM_PYTHON=${BATRAM_PYTHON:-env_batram/bin/python}
HMC_FACTORS=${HMC_FACTORS-exponential gamma pareto}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
# the Pareto fit reaches the maximum NUTS tree depth on these data
PARETO_OPTS="--max_tree_depth 6 --warmup 100 --num_samples 200"
S=src/python_scripts
D=application_precipitation_v2/
DF=application_precipitation_full/
RES=results_application/
RESF=results_application_full/
RESL=results_M1_M2/application/
TOPO=${D}topography/etopo1_15-25E_45-55N.nc
source configs/bnfc_application.conf
mkdir -p logs

step() { echo "[$(date '+%F %T')] $*"; }
# run LOG CMD...: run CMD with its output in LOG (runa: append to LOG)
run()  { local log=$1; shift; if [[ "$DRY_RUN" == "1" ]]; then echo "  $* > $log"; else "$@" > "$log" 2>&1; fi; }
runa() { local log=$1; shift; if [[ "$DRY_RUN" == "1" ]]; then echo "  $* >> $log"; else "$@" >> "$log" 2>&1; fi; }
# todo FILE: true if FILE must be (re)made
todo() { [[ "$FORCE" == "1" || ! -e $1 ]]; }
params() {  # range and nugget of a BNFC fit
  $P -c "import torch; p = torch.load('${D}models/posterior_params_$1.pth'); print(float(p['lengthscale']), float(p['nugget']))"
}

# 1. data
if todo ${D}meta.json || todo ${DF}meta.json; then
  step "data"
  run logs/app_data.log $P $S/dataGen_precipitation.py --era5_dir "$ERA5_DIR" --out_dir $D \
    --smooth_days 2 --tie_noise shared
  run logs/app_data_full.log $P $S/dataGen_precipitation.py --era5_dir "$ERA5_DIR" --out_dir $DF \
    --smooth_days 2 --tie_noise shared --train_all_sites
fi
[[ "$DRY_RUN" == "1" ]] || mkdir -p ${D}models ${DF}models
if [[ "$DRY_RUN" == "1" && ! -f ${D}meta.json ]]; then
  N_TRAIN=10591; N_TEST=200   # values of the data of the paper, for printing the commands only
else
  N_TRAIN=$($P -c "import json; print(json.load(open('${D}meta.json'))['N_train'])")
  N_TEST=$($P -c "import json; print(json.load(open('${D}meta.json'))['N_test'])")
fi
step "training days: $N_TRAIN, test days: $N_TEST"

# 2. marginal network on all grid locations (same number of gradient steps as with 100
#    locations and mini-batches of 256), shared by both data folders
if todo ${DF}models/F_inv-nonstat.pth; then
  step "marginal network"
  run logs/app_margin.log $P $S/Train_BNFC.py --data_path=$DF --factor=nonstat --N "$N_TRAIN" \
    --n 1024 --finv_only --epochs_Finv 100 --batch_size_Finv 2560 --lr_Finv 1e-3
fi
[[ "$DRY_RUN" == "1" ]] || cp ${DF}models/F_inv-nonstat.pth ${D}models/F_inv-nonstat.pth

# 3. preliminary BNFC (stat): Gaussian component of the competitors. The final BNFC (stat) fit
#    overwrites its model files, so its range and nugget are kept in competitor_kernel.txt.
if todo ${D}models/competitor_kernel.txt; then
  step "preliminary BNFC (stat)"
  run logs/app_bnfc_stat_preliminary.log $P $S/Train_BNFC.py --data_path=$D --factor=stat \
    --N "$N_TRAIN" --epochs 20 --lr 1e-4 --step_size 10 --learn_cov --kernel se \
    --init_lengthscale 0.2 --init_nugget 0.01 --epochs_Finv 0
  [[ "$DRY_RUN" == "1" ]] || params stat > ${D}models/competitor_kernel.txt
fi
if [[ -f ${D}models/competitor_kernel.txt ]]; then read LS0 NUG0 < ${D}models/competitor_kernel.txt
else LS0=LS0; NUG0=NUG0; fi
step "competitors' Gaussian component: SE, range $LS0, nugget $NUG0"

# 4. common factor copulas and the Student-t copula (a failed HMC fit is reported and skipped)
for F in $HMC_FACTORS; do
  if todo ${D}models/hmc_posterior-$F.pth; then
    step "HMC $F"
    OPTS=""; [[ "$F" == "pareto" ]] && OPTS=$PARETO_OPTS
    run logs/app_hmc_$F.log $P $S/train_model_single_factor_HMC_NUTS.py --data_path=$D --factor=$F \
      --N "$N_TRAIN" --kernel se --lengthscale "$LS0" --nugget "$NUG0" $OPTS \
      || step "[warn] HMC $F failed, see logs/app_hmc_$F.log"
  fi
done
if todo ${D}models/posterior-tcopula.pth; then
  step "Student-t copula"
  run logs/app_tcopula.log $P $S/gsm_t_copula.py fit --data_path $D --N "$N_TRAIN" --kernel se \
    --lengthscale "$LS0" --nugget "$NUG0"
fi

# 5. BNFC with early stopping and pSGLD weight samples, then the loading copulas
if todo ${D}models/sgmcmc-stat.pth; then
  step "BNFC (stat)"
  run logs/app_bnfc_stat.log $P $S/Train_BNFC.py --data_path=$D --factor=stat --N "$N_TRAIN" \
    --epochs 100 --lr 1e-3 --step_size 20 --learn_cov --kernel se --init_lengthscale "$LS0" \
    --init_nugget "$NUG0" --epochs_Finv 0
fi
if todo ${D}models/sgmcmc-nonstat.pth; then
  step "BNFC (nonstat), configs/bnfc_application.conf"
  run logs/app_bnfc_nonstat.log $P $S/Train_BNFC.py --data_path=$D --factor=nonstat --N "$N_TRAIN" \
    $BNFC_APP_NONSTAT --epochs_Finv 0
fi
# Gaussian loading copula with K = 25 and, as a check, K = 49 loading bumps; gamma loading copula;
# range and nugget estimated jointly with the loading
for G in 5 7; do
  K=$((G * G)); f=${D}models/posterior-m1.pth; [[ $K == 49 ]] && f=${D}models/posterior-m1_K49.pth
  if todo $f; then
    step "Gaussian loading copula, K = $K"
    run logs/app_m1_fit_K$K.log $P $S/M1_gaussian_rank1.py fit --data_path $D --N "$N_TRAIN" --basis_grid $G
  fi
done
if todo ${D}models/posterior-m2.pth; then
  step "gamma loading copula"
  run logs/app_m2_fit.log $P $S/M2_gamma_varying_loading.py fit --data_path $D --N "$N_TRAIN"
fi

# 6. predictions given the 100 observed locations (precipitation floored at 0), held-out copula
#    log score at the observed locations and on the full grid, paired comparison
step "predictions and held-out log score"
COMMON="--scenarios $D --out_dir $RES --test_start $N_TRAIN --n_test_reps $N_TEST --y_min 0"
run logs/app_predict.log $P $S/Copula_BNFC.py --factors stat nonstat $COMMON
if [[ -n "$HMC_FACTORS" ]]; then
  runa logs/app_predict.log $P $S/Copula_single_factor.py --factors $HMC_FACTORS $COMMON
fi
runa logs/app_predict.log $P $S/gsm_t_copula.py predict $COMMON
run logs/app_logscore.log $P $S/heldout_loglik.py --scenarios $D --out_dir $RES \
  --test_start "$N_TRAIN" --n_test_reps "$N_TEST"
LTEST="--test_start $N_TRAIN --n_test_reps $N_TEST --out_dir $RESL"
for G in 5 7; do
  K=$((G * G))
  run logs/app_m1_score_K$K.log $P $S/M1_gaussian_rank1.py score --data_path $D --basis_grid $G $LTEST
  run logs/app_m1_predict_K$K.log $P $S/M1_gaussian_rank1.py predict --data_path $D --basis_grid $G \
    $LTEST --y_min 0
done
run logs/app_m2_score.log $P $S/M2_gamma_varying_loading.py score --data_path $D $LTEST
run logs/app_m2_predict.log $P $S/M2_gamma_varying_loading.py predict --data_path $D $LTEST --y_min 0
run logs/app_compare.log $P $S/compare_M1_M2.py --main_dir $RES --base_dir $RESL \
  --base_models m1 m1_K49 m2 --reference nonstat

# 7. tables and figures
step "tables and figures"
run logs/app_main_tables.log $P $S/make_main_tables.py --skip_simulations
run logs/app_plots.log $P $S/plot_application.py --results_dir $RES --data_path $D --scale copula \
  --margin_path ${DF}models/F_inv-nonstat.pth
runa logs/app_plots.log $P $S/plot_calibration.py --results_dir $RES \
  --title "Calibration at the 924 unobserved grid locations (100 conditioning locations)"
runa logs/app_plots.log $P $S/plot_application_nugget_map.py --data_path $D --margin_path $DF \
  --results_dir $RES --out_dir ${RES}figures/ --nugget 0.1 --R 1000
runa logs/app_plots.log $P $S/plot_application.py --results_dir $RES --data_path $D --inference \
  --n_sim 40000
runa logs/app_plots.log $P $S/application_bayes_insights.py
if [[ ! -f $TOPO ]]; then
  step "download ETOPO1 relief"
  [[ "$DRY_RUN" == "1" ]] || mkdir -p $(dirname $TOPO)
  run logs/app_topography_download.log curl -fsSL -o $TOPO \
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude%5B(45.0):2:(55.0)%5D%5B(15.0):2:(25.0)%5D"
fi
runa logs/app_plots.log $P $S/application_topography.py

# 8. complete data with the Vecchia approximation, dense-grid predictions, timing benchmark
if todo ${DF}models/sgmcmc-nonstat.pth; then
  step "BNFC (nonstat) on the complete data (Vecchia, m = 30)"
  if [[ -f ${D}models/posterior_params_stat.pth ]]; then read LS NUG < <(params stat); else LS=LS; NUG=NUG; fi
  run logs/app_bnfc_full_vecchia.log $P $S/Train_BNFC.py --data_path=$DF --factor=nonstat \
    --N "$N_TRAIN" --n 1024 --epochs 20 --lr 1e-3 --step_size 10 --learn_cov --kernel se \
    --init_lengthscale "$LS" --init_nugget "$NUG" --vecchia 30 --M2 5 --batch_size 4 --epochs_Finv 0
fi
step "predictions on the 80 x 80 grid"
run logs/app_dense.log $P $S/predict_dense_grid.py --data_path $DF --out_dir ${RESF}dense/ \
  --dates 2024-07-02 2024-09-14 --G 80
if todo ${RESF}benchmark/vecchia_benchmark.csv; then
  step "Vecchia timing benchmark"
  run logs/app_vecchia_benchmark.log $P $S/heldout_loglik.py --benchmark_vecchia --scenarios $D \
    --out_dir ${RESF}benchmark/ --test_start "$N_TRAIN" --n_test_reps "$N_TEST" --K 500
fi

# 9. Bayesian transport map (Katzfuss and Schafer) at the 100 observed locations, on CPU in its own
#    environment. The map needs memory of order n x N^2; with all training days (or 4,000) it
#    exceeds 126 GB, so it is fitted to the 2,000 most recent training days, by 100 Adam steps.
if [[ ! -x $BATRAM_PYTHON ]]; then
  step "[skip] transport map: $BATRAM_PYTHON not found (see README)"
elif todo results_M1_M2/transport_map_application/comparison.csv; then
  step "Bayesian transport map"
  run logs/app_transport_map.log $BATRAM_PYTHON $S/transport_map_baseline.py --scenarios $D \
    --results_dir $RES --out_dir results_M1_M2/transport_map_application/ --N 2000 \
    --train_start $((N_TRAIN - 2000)) --test_start "$N_TRAIN" --n_test_reps "$N_TEST" \
    --max_nn 30 --nsteps 100 --lr 0.01 --threads 24
fi
step "done"
