#!/usr/bin/env bash
# Selection of the BNFC (nonstat) configuration of the ERA5 application, with the range of the
# Gaussian component in the variational family (--vb_range). The result is already in
# configs/bnfc_application.conf, so this script need not be rerun to reproduce the results.
# Nothing in application_precipitation_v2/models/ is changed. Run from the repository root,
# once per round of experiments:
#
#   setsid nohup bash run_bnfc_app_tuning.sh experiments/bnfc_app/configs_r1.txt \
#     > logs/bnfc_app_tuning_r1.log 2>&1 < /dev/null &
#   (then configs_r2.txt; experiments/bnfc_app/run_reference.sh scores the Student-t and
#   Gaussian loading copulas on the same validation days)
#
# <configs file>: one experiment per line, "<tag> <Train_BNFC.py options>".
# Each experiment fits on 1994-2021 (days 0-10225) and is scored on the 365 validation
# days of 2022, at the 100 observed locations and on the full grid of 1,024 locations
# (fitted network, no weight samples). Everything goes to experiments/bnfc_app/<tag>/:
#   train.log, score.log, models/, val/heldout_loglik_summary.csv
# and the table of all experiments to experiments/bnfc_app/summary.csv; the configuration with
# the highest full-grid validation score (r1_h128_lin_exp) is refitted to all training days by
# run_application.sh.
#   PYTHON     Python interpreter (default env/bin/python if present, else python)
#   PARALLEL   experiments run at the same time on the GPU (default 3)
set -euo pipefail
export PYTHONUNBUFFERED=1
if [[ -n "${PYTHON:-}" ]]; then P=$PYTHON
elif [[ -x env/bin/python ]]; then P=env/bin/python
else P=python; fi
S=src/python_scripts
ROOT=experiments/bnfc_app
CONFIGS=${1:?configs file}
PARALLEL=${PARALLEL:-3}
VAL_START=10226
N_VAL=365
COMMON="--factor=nonstat --N $VAL_START --learn_cov --vb_range --epochs 100 --patience 15 \
  --lr 1e-3 --step_size 20 --M2 5 --sgmcmc_samples 0 --epochs_Finv 0 \
  --init_lengthscale 0.135 --init_nugget 0.057"
export P S ROOT COMMON VAL_START N_VAL
step() { echo "[$(date '+%F %T')] $*"; }
export -f step

run_one() {
  tag=$1; shift
  D=$ROOT/$tag/
  if [[ -f $D/val/heldout_loglik_summary.csv ]]; then step "$tag: already done"; return; fi
  $P $S/bnfc_app_experiment.py setup --tag "$tag" > /dev/null
  echo "$*" > $D/config.txt
  step "$tag: fit ($*)"
  $P $S/Train_BNFC.py --data_path=$D $COMMON "$@" > $D/train.log 2>&1 \
    || { step "$tag: FIT FAILED, see $D/train.log"; return; }
  step "$tag: validation score"
  $P $S/heldout_loglik.py --scenarios $D --models nonstat --weights map \
    --test_start $VAL_START --n_test_reps $N_VAL --out_dir $D/val/ > $D/score.log 2>&1 \
    || { step "$tag: SCORE FAILED, see $D/score.log"; return; }
  step "$tag: done, $(grep -E 'full +nonstat' $D/score.log | sed 's/  */ /g')"
}
export -f run_one

step "configs: $CONFIGS, $PARALLEL at a time"
grep -v '^\s*#' "$CONFIGS" | grep -v '^\s*$' | \
  xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _
$P $S/bnfc_app_experiment.py summarise
step "all done"
