#!/usr/bin/env bash
# Run every setting with every frozen sequence model.
#   original split    -> reduced_original (DPFunc's own split, filtered proteins) and full (DPFunc's whole benchmark)
#   reshuffled split  -> reduced_resplit
# Tables come from make_split_tables.sh on the data branch (copy benchmark_tables/ into this folder first).
# Finished steps are skipped, so the script can be restarted. DPFunc predictions, when you have them, are scored next to the rest:
#   DPFUNC_original=runs/dpfunc_reduced_original.csv bash benchmark/run_all.sh   (file format: benchmark/README.md, section 4)
set -euo pipefail
PLMS=${PLMS:-"facebook/esm2_t12_35M_UR50D facebook/esm2_t33_650M_UR50D"}
SETTINGS=${SETTINGS:-"reduced_resplit reduced_original full"}

for setting in $SETTINGS; do
  for plm in $PLMS; do
    tag=$(basename "$plm")
    extra=()
    var="DPFUNC_${setting}"
    [ -n "${!var:-}" ] && extra+=(--pred "DPFunc=${!var}")
    python benchmark/run.py --table "benchmark_tables/${setting}.csv" --out "runs/${setting}/${tag}" --plm "$plm" "${extra[@]}"
  done
done
