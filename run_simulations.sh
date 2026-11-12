#!/usr/bin/env bash
# Created by Pratik Nag, October 2026 (approximate)
# Simulation study: Section 5.1 of the manuscript and Section S6 of the supplement. Generates
# the data, fits every model, and writes every simulation table and figure. Run from the
# repository root:
#
#   setsid nohup bash run_simulations.sh > logs/run_simulations.log 2>&1 < /dev/null &
#
#   PYTHON         Python interpreter (default env/bin/python if present, else python)
#   BATRAM_PYTHON  interpreter of the transport map environment (default env_batram/bin/python)
#   FORCE=1        regenerate the data and refit every model. By default a data or fitting stage
#                  is skipped when its output exists, so an interrupted run can be resumed.
#                  Predictions, scores, tables and figures are always recomputed.
#   DRY_RUN=1      print the commands without running them
#
# Scenarios: (i) gamma-factor data, (ii) sign-change data, (iii) regional mixture data.
# Models:
#   BNFC (stat, nonstat): settings per scenario in configs/bnfc_simulations.conf (tuned with
#     run_bnfc_sim_tuning.sh, which need not be rerun); early stopping, then pSGLD weight samples
#   common factor copulas with exponential, gamma, Pareto and gamma-mixture factors (HMC/NUTS)
#   Student-t copula (Gaussian scale mixture, exact grid posterior)
#   Gaussian loading copula (m1) and gamma loading copula (m2) (HMC/NUTS)
#   Bayesian transport map (observed locations only; CPU, env_batram/)
# Every model uses the Gaussian component of the data-generating process (SE, range 0.1).
# A single BNFC (nonstat) fit fills the GPU, so every stage runs in sequence.
#
# Outputs used in the manuscript and supplement (see README.md for the full list):
#   results/simulation_table_rows.tex     Table 1 (make_main_tables.py)
#   results/figures/                      Figure 1; Tables S2-S3; Figures S1-S5, S7
#   results_n25/figures/                  sparse conditioning: Tables S4-S5; Figures S4-S6
#   results_M1_M2/simulations/<scenario>/comparison.csv
#                                         paired comparisons, all models (Table 1)
#   results_M1_M2/transport_map/comparison.csv: Table S6 (simulation columns)
# Per-stage logs are written to logs/.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${PYTHON:-}" ]]; then P=$PYTHON
elif [[ -x env/bin/python ]]; then P=env/bin/python
else P=python; fi
BATRAM_PYTHON=${BATRAM_PYTHON:-env_batram/bin/python}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
S=src/python_scripts
SCENARIOS="simulations_gamma/ simulations_signchange/ simulations_mixgamma/"
source configs/bnfc_simulations.conf
# loading copulas: Gaussian component of the data-generating process; 100 test replicates
LOADING_COV="--fix_cov --init_lengthscale 0.1 --init_nugget 1e-4"
TEST="--test_start 2000 --n_test_reps 100"
mkdir -p logs

step() { echo "[$(date '+%F %T')] $*"; }
# run LOG CMD...: run CMD with its output in LOG (runa: append to LOG)
run()  { local log=$1; shift; if [[ "$DRY_RUN" == "1" ]]; then echo "  $* > $log"; else "$@" > "$log" 2>&1; fi; }
runa() { local log=$1; shift; if [[ "$DRY_RUN" == "1" ]]; then echo "  $* >> $log"; else "$@" >> "$log" 2>&1; fi; }
# todo FILE: true if FILE must be (re)made
todo() { [[ "$FORCE" == "1" || ! -e $1 ]]; }

# ---------------------------------------------------------------------------
# 1. data: training replicates 0-1999, test replicates 2000-2099
# ---------------------------------------------------------------------------
declare -A DATAGEN=([simulations_gamma/]=dataGen_Gamma.py
                    [simulations_signchange/]=dataGen_SignChange.py
                    [simulations_mixgamma/]=dataGen_MixGamma.py)
for D in $SCENARIOS; do
  if todo ${D}sample_copula_full-2099.csv; then
    step "data $D"
    run logs/data_$(basename $D).log $P $S/${DATAGEN[$D]} --out_dir $D
  fi
done

# ---------------------------------------------------------------------------
# 2. model fits
# ---------------------------------------------------------------------------
bnfc_fit() {  # scenario factor
  local D=$1 F=$2 sc=$(basename $1)
  local args="$BNFC_SIM_COMMON ${BNFC_SIM[$sc/$F]}" log=logs/fit_${sc}_$F.log
  if [[ "$FORCE" != "1" && -f ${D}models/sgmcmc-$F.pth ]]; then return; fi
  if [[ "$FORCE" != "1" && -f ${D}models/fit_time-$F.json ]] && grep -q epochs_run ${D}models/fit_time-$F.json; then
    # fitted with early stopping, weights not yet sampled
    step "BNFC $sc $F: pSGLD weight samples"
    runa $log $P $S/Train_BNFC.py --data_path=$D --factor=$F $args --sgmcmc_only
  else
    step "BNFC $sc $F: fit and pSGLD weight samples"
    run $log $P $S/Train_BNFC.py --data_path=$D --factor=$F $args
  fi
}

for D in $SCENARIOS; do
  sc=$(basename $D)
  # BNFC; the fit also fits the marginal quantile network T_psi
  for F in stat nonstat; do bnfc_fit $D $F; done
  # common factor copulas (HMC/NUTS)
  for F in exponential gamma pareto mixgamma; do
    if todo ${D}models/hmc_posterior-$F.pth; then
      step "HMC $sc $F"
      run logs/hmc_${sc}_$F.log $P $S/train_model_single_factor_HMC_NUTS.py --data_path=$D --factor=$F
    fi
  done
  # Student-t copula (Gaussian scale mixture)
  if todo ${D}models/posterior-tcopula.pth; then
    step "Student-t copula $sc"
    run logs/tcopula_$sc.log $P $S/gsm_t_copula.py fit --data_path $D
  fi
  # Gaussian (m1) and gamma (m2) loading copulas (HMC/NUTS)
  if todo ${D}models/posterior-m1.pth; then
    step "Gaussian loading copula $sc"
    run logs/m1_fit_$sc.log $P $S/M1_gaussian_rank1.py fit --data_path $D --N 2000 $LOADING_COV
  fi
  if todo ${D}models/posterior-m2.pth; then
    step "gamma loading copula $sc"
    run logs/m2_fit_$sc.log $P $S/M2_gamma_varying_loading.py fit --data_path $D --N 2000 $LOADING_COV
  fi
done

# ---------------------------------------------------------------------------
# 3. posterior predictive simulation of the test replicates, given all 100 observed
#    locations (results/) and a fixed subset of 25 of them (results_n25/, same fits)
# ---------------------------------------------------------------------------
for OUT in results/ results_n25/; do
  COND=""; [[ "$OUT" == "results_n25/" ]] && COND="--n_cond 25"
  tag=$(basename $OUT)
  step "predictions -> $OUT"
  [[ "$DRY_RUN" == "1" ]] || rm -rf ${OUT}simulations_* ${OUT}summary.csv
  run logs/predict_bnfc_$tag.log $P $S/Copula_BNFC.py --scenarios $SCENARIOS --factors stat nonstat \
    $COND --out_dir $OUT
  run logs/predict_factor_$tag.log $P $S/Copula_single_factor.py --scenarios $SCENARIOS \
    --factors exponential gamma pareto mixgamma $COND --out_dir $OUT
  run logs/predict_tcopula_$tag.log $P $S/gsm_t_copula.py predict --scenarios $SCENARIOS $COND --out_dir $OUT
done
for D in $SCENARIOS; do
  sc=$(basename $D)
  step "loading copulas: held-out log score and predictions, $sc"
  for M in m1:M1_gaussian_rank1 m2:M2_gamma_varying_loading; do
    m=${M%%:*}; script=${M#*:}.py
    run logs/${m}_score_$sc.log $P $S/$script score --data_path $D --out_dir results_M1_M2/simulations/ $TEST
    run logs/${m}_predict_$sc.log $P $S/$script predict --data_path $D --out_dir results_M1_M2/simulations/ $TEST
    run logs/${m}_predict_n25_$sc.log $P $S/$script predict --data_path $D \
      --out_dir results_M1_M2/simulations_n25/ $TEST --n_cond 25
  done
done

# ---------------------------------------------------------------------------
# 4. held-out copula log score of the test replicates (observed locations, full grid)
# ---------------------------------------------------------------------------
step "held-out copula log score"
run logs/heldout_loglik.log $P $S/heldout_loglik.py --scenarios $SCENARIOS --out_dir results/

# ---------------------------------------------------------------------------
# 5. tables and figures
#    results/figures/: Table S2 (summary_table.tex), Figure 1 (logscore_differences.png),
#    Figure S3 (heatmaps_*_rep2000.png); results_n25/figures/: Table S5, Figure S6;
#    calibration: Tables S3-S4, Figures S4-S5; factor posterior against the true posterior:
#    Figure S7 (factor_posterior.png); paired comparison of all models with BNFC (nonstat):
#    results_M1_M2/simulations/<scenario>/comparison.csv; rows of Table 1
# ---------------------------------------------------------------------------
step "tables and figures"
run logs/plots.log $P $S/plot_BNFC_results.py --scale copula --loading_dir results_M1_M2/simulations/ \
  --extra_logscore results_M1_M2/simulations/simulations_*/m[12]/heldout_loglik.csv
runa logs/plots.log $P $S/plot_BNFC_results.py --results_dir results_n25/ --no_fit_time --scale copula \
  --loading_dir results_M1_M2/simulations_n25/
run logs/calibration.log $P $S/plot_calibration.py --results_dir results/ \
  --loading_dir results_M1_M2/simulations/ \
  --title "Calibration at the 800 unobserved locations (100 conditioning locations)"
runa logs/calibration.log $P $S/plot_calibration.py --results_dir results_n25/ \
  --loading_dir results_M1_M2/simulations_n25/ \
  --title "Calibration at the 875 unobserved locations (25 conditioning locations)"
run logs/factor_posterior.log $P $S/factor_posterior.py --results_dir results/ \
  --loading_dir results_M1_M2/simulations/
runa logs/factor_posterior.log $P $S/factor_posterior.py --results_dir results_n25/ \
  --loading_dir results_M1_M2/simulations_n25/
for R in results_M1_M2/simulations/ results_M1_M2/simulations_n25/; do
  runa logs/factor_posterior.log $P $S/factor_posterior.py --results_dir $R --models m1 m2 --out_dir ${R}figures/
done
for D in $SCENARIOS; do
  sc=$(basename $D)
  run logs/compare_$sc.log $P $S/compare_M1_M2.py --scenario $sc --main_dir results/ \
    --base_dir results_M1_M2/simulations/ --base_models m1 m2 --reference nonstat \
    --main_models stat nonstat exponential gamma pareto mixgamma tcopula
done
# rows of Table 1 of the manuscript: results/simulation_table_rows.tex
run logs/main_tables.log $P $S/make_main_tables.py --skip_application

# ---------------------------------------------------------------------------
# 6. Bayesian transport map (Katzfuss and Schafer) at the observed locations, on CPU in its
#    own environment (batram needs Python >= 3.10): Section S6.6 and Table S6,
#    results_M1_M2/transport_map/{heldout_loglik,comparison}.csv; about 5 hours
# ---------------------------------------------------------------------------
if [[ ! -x $BATRAM_PYTHON ]]; then
  step "[skip] transport map: $BATRAM_PYTHON not found (see README)"
elif todo results_M1_M2/transport_map/comparison.csv; then
  step "Bayesian transport map"
  run logs/transport_map_simulations.log $BATRAM_PYTHON $S/transport_map_baseline.py \
    --scenarios $SCENARIOS --threads 24 --nsteps 300
fi

# ---------------------------------------------------------------------------
# 7. tail flexibility figures of Section S3 (Figures S1-S2: zeta_plot.png, correlation_matrices.png)
# ---------------------------------------------------------------------------
if todo results/figures/correlation_matrices.png; then
  step "tail flexibility figures"
  run logs/tail_dependence.log $P $S/Copula_tail_dependence.py --fig_dir results/figures/
fi
step "done"
