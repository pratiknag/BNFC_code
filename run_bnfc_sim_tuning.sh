#!/usr/bin/env bash
# Created by Pratik Nag, November 2026 (approximate)
# Selection of the BNFC settings of the simulation study (Section S6.8 and Table S7 of the
# supplement). The result is already in configs/bnfc_simulations.conf, so this script need not
# be rerun to reproduce the results. It needs the data of run_simulations.sh (stage 1) and, for
# the reference score, its HMC fit of the gamma factor model. Run from the repository root:
#
#   setsid nohup bash run_bnfc_sim_tuning.sh > logs/bnfc_sim_tuning.log 2>&1 < /dev/null &
#
#   PYTHON   Python interpreter (default env/bin/python if present, else python)
#
# Every configuration is fitted to replicates 0-1799 and scored by the held-out copula log score
# of the validation replicates 1800-1999 (observed locations and full grid); the test replicates
# 2000-2099 are not used. BNFC (stat) is tuned on the gamma-factor data, BNFC (nonstat) on the
# sign-change and regional mixture data. The configuration with the highest full-grid validation
# score is written to experiments/bnfc_sim/selected.csv; copy it into configs/bnfc_simulations.conf
# and run run_simulations.sh, which refits it to all 2,000 training replicates.
# Outputs in experiments/bnfc_sim/: <scenario>/<config>/ (fits, fit.log, val.log, val/),
# reference/ (gamma factor model), validation_summary.csv, selected.csv. About 28 hours on one
# RTX 4090; finished configurations are skipped.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -n "${PYTHON:-}" ]]; then P=$PYTHON
elif [[ -x env/bin/python ]]; then P=env/bin/python
else P=python; fi
S=src/python_scripts
T=experiments/bnfc_sim
mkdir -p $T
step() { echo "[$(date '+%F %T')] $*"; }

COMMON_TUNE="--N 1800 --batch_size 20 --sgmcmc_samples 20 --sgmcmc_thin 50 --seed 123"
VAL="--test_start 1800 --n_test_reps 200"

# configurations: name -> options added to COMMON_TUNE (later options override earlier ones).
# The 128 x 4 networks, M1 = 200 and the Fourier features use smaller mini-batches to fit in
# 24 GB of GPU memory (the Monte Carlo likelihood per replicate is unchanged).
declare -A STAT_CFG=(
  [s0_base]="--epochs 100 --patience 25 --min_delta 0.01 --lr 1e-3 --step_size 20 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 1e-3"
  [s1_long]="--epochs 200 --patience 60 --min_delta 0.001 --lr 1e-3 --step_size 40 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 1e-3"
  [s2_lr3e-3]="--epochs 200 --patience 60 --min_delta 0.001 --lr 3e-3 --step_size 40 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 1e-3"
  [s3_M1_300]="--epochs 200 --patience 60 --min_delta 0.001 --lr 1e-3 --step_size 40 --gamma 0.5 --M1 300 --M2 50 --sgmcmc_step 1e-3"
  [s4_sgld3e-4]="--epochs 200 --patience 60 --min_delta 0.001 --lr 1e-3 --step_size 40 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 3e-4"
  [s5_sgld3e-3]="--epochs 200 --patience 60 --min_delta 0.001 --lr 1e-3 --step_size 40 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 3e-3"
  [s6_wd1e-5]="--epochs 200 --patience 60 --min_delta 0.001 --lr 1e-3 --step_size 40 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 1e-3 --weight_decay 1e-5"
  [s7_batch50]="--epochs 200 --patience 60 --min_delta 0.001 --lr 1e-3 --step_size 40 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 1e-3 --batch_size 50"
  [s8_M1_300_lr3e-3]="--epochs 200 --patience 60 --min_delta 0.001 --lr 3e-3 --step_size 40 --gamma 0.5 --M1 300 --M2 50 --sgmcmc_step 1e-3"
)
NS_BASE="--patience 40 --min_delta 0.001 --gamma 0.5 --M2 50 --sgmcmc_step 1e-3"
declare -A NONSTAT_CFG=(
  [n0_base]="--epochs 100 --patience 25 --min_delta 0.01 --lr 1e-3 --step_size 20 --gamma 0.5 --M1 100 --M2 50 --sgmcmc_step 1e-3 --hidden 64 --depth 3 --out_act softplus"
  [n1_long]="$NS_BASE --epochs 200 --lr 1e-3 --step_size 40 --M1 100 --hidden 64 --depth 3 --out_act softplus"
  [n2_linear]="$NS_BASE --epochs 200 --lr 1e-3 --step_size 40 --M1 100 --hidden 64 --depth 3 --out_act linear"
  [n3_wide]="$NS_BASE --epochs 200 --lr 1e-3 --step_size 40 --M1 100 --hidden 128 --depth 4 --out_act softplus --batch_size 5"
  [n4_fourier]="$NS_BASE --epochs 200 --lr 1e-3 --step_size 40 --M1 100 --hidden 64 --depth 3 --out_act softplus --n_fourier 16 --fourier_scale 2 --batch_size 10"
  [n5_lr3e-3]="$NS_BASE --epochs 200 --lr 3e-3 --step_size 40 --M1 100 --hidden 64 --depth 3 --out_act softplus"
  [n6_M1_200]="$NS_BASE --epochs 200 --lr 1e-3 --step_size 40 --M1 200 --hidden 64 --depth 3 --out_act softplus --batch_size 10"
  [n7_wide_linear]="$NS_BASE --epochs 200 --lr 1e-3 --step_size 40 --M1 100 --hidden 128 --depth 4 --out_act linear --batch_size 5"
)

# full-grid fields of the validation replicates: the scenario data hold them only for the test
# replicates, so they are replayed from the seeded generator with replicates 0-1799 as training
# data (same seed, so the draws are identical to those of the scenario data)
declare -A DATAGEN=([gamma]=dataGen_Gamma.py [signchange]=dataGen_SignChange.py [mixgamma]=dataGen_MixGamma.py)
for s in gamma signchange mixgamma; do
  if [[ ! -f $T/data_$s/sample_copula_full-1999.csv ]]; then
    step "validation fields $s"
    $P $S/${DATAGEN[$s]} --out_dir $T/data_$s/ --N_train 1800 --N_test 300 > $T/data_$s.log 2>&1
  fi
done

# folder of one configuration: symbolic links to the data, its own models/
make_dir() {  # scenario cfg
  local d=$T/$1/$2
  if [[ ! -d $d/models ]]; then
    mkdir -p $d/models
    for f in $1/*.csv $1/*.txt; do ln -s "$(realpath $f)" $d/; done
  fi
  for i in $(seq 1800 1999); do
    ln -sf "$(realpath $T/data_${1#simulations_}/sample_copula_full-$i.csv)" $d/
  done
  echo $d/
}

tune() {  # scenario factor cfg args
  local sc=$1 F=$2 cfg=$3 args=$4 d
  d=$(make_dir $sc $cfg)
  if [[ -f ${d}val/heldout_loglik_summary.csv ]]; then return; fi
  echo "$args" > ${d}args.txt
  if [[ ! -f ${d}models/sgmcmc-$F.pth ]]; then
    step "tune $sc $F $cfg: fit"
    $P $S/Train_BNFC.py --data_path=$d --factor=$F $COMMON_TUNE $args > ${d}fit.log 2>&1 \
      || { step "[fail] $sc $cfg, see ${d}fit.log"; return; }
  fi
  step "tune $sc $F $cfg: validation log score"
  $P $S/heldout_loglik.py --scenarios $d --models $F $VAL --out_dir ${d}val/ > ${d}val.log 2>&1 \
    || step "[fail score] $sc $cfg, see ${d}val.log"
  grep "mean log score" ${d}val.log || true
}

# reference: validation score of the data-generating gamma factor model (HMC fit of
# run_simulations.sh to all 2,000 replicates, so the validation replicates are in its training
# data: optimistic)
if [[ ! -f $T/reference/heldout_loglik_summary.csv ]]; then
  step "reference score"
  d=$(make_dir simulations_gamma reference_hmc)
  ln -sf "$(realpath simulations_gamma/models/hmc_posterior-gamma.pth)" ${d}models/
  $P $S/heldout_loglik.py --scenarios $d --models gamma $VAL --out_dir $T/reference/ \
    > $T/reference.log 2>&1 || step "[fail] reference, see $T/reference.log"
fi
for cfg in $(printf '%s\n' "${!STAT_CFG[@]}" | sort); do
  tune simulations_gamma stat $cfg "${STAT_CFG[$cfg]}"
done
for sc in simulations_signchange simulations_mixgamma; do
  for cfg in $(printf '%s\n' "${!NONSTAT_CFG[@]}" | sort); do
    tune $sc nonstat $cfg "${NONSTAT_CFG[$cfg]}"
  done
done

step "validation summary"
$P - <<'EOF'
import glob, pandas as pd
rows = []
for f in glob.glob("experiments/bnfc_sim/simulations_*/*/val/heldout_loglik_summary.csv"):
    sc, cfg = f.split("/")[2:4]
    for _, r in pd.read_csv(f).iterrows():
        rows.append({"scenario": sc, "cfg": cfg, "set": r["set"], "model": r["model"],
                     "val_log_score": r["mean_log_score"], "se": r["se"]})
df = pd.DataFrame(rows).sort_values(["scenario", "set", "val_log_score"], ascending=[True, True, False])
df.to_csv("experiments/bnfc_sim/validation_summary.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
best = df[df.set == "full"].groupby("scenario").head(1)
best[["scenario", "cfg", "model"]].to_csv("experiments/bnfc_sim/selected.csv", index=False)
print("\nselected (copy into configs/bnfc_simulations.conf):\n", best.to_string(index=False))
EOF
step "done"
