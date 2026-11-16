#!/usr/bin/env bash
# Build the two filtered datasets (original split, reshuffled split) with the same filters, check their structures,
# and write the benchmark tables. Needs: mf_manifest_from_data_dpfunc.tsv in this folder, the MMseqs2 binary on PATH,
# pip install -r requirements.txt, and internet access (UniProt, RCSB, AlphaFold DB; cached in dpfunc_filter_cache/).
#
#   bash make_split_tables.sh                       # uses the settings below
#   MONOMER_MODE=strict MIN_PLDDT=80 bash make_split_tables.sh
#
# original    DPFunc's own train/valid/test assignment is kept for the proteins that survive the filters
# reshuffled  the survivors are re-split 80/10/10, GO vocabulary and weights from the new training set, ML-RUS undersampling
set -euo pipefail

MANIFEST=${MANIFEST:-mf_manifest_from_data_dpfunc.tsv}
MONOMER_MODE=${MONOMER_MODE:-supported}   # strict | lenient | supported (see docs/DATASET_CURATION.md)
MIN_PLDDT=${MIN_PLDDT:-70}                # AlphaFold-only proteins below this mean pLDDT are dropped
MIN_TERM=${MIN_TERM:-10}
COMMON=(--manifest "$MANIFEST" --ontology mf --monomer-mode "$MONOMER_MODE" --pdb-oligomer-check
        --check-alphafold --min-plddt "$MIN_PLDDT" --download-structures --min-term-count "$MIN_TERM")

echo "== original split =="
python dpfunc_filter.py "${COMMON[@]}" --outdir filtered_mf_original

echo "== reshuffled split =="
python dpfunc_filter.py "${COMMON[@]}" --outdir filtered_mf_reshuffled --resplit 0.8 0.1 0.1 --undersample-frac 0.1

echo "== structure quality =="
for d in filtered_mf_original filtered_mf_reshuffled; do
  python structure_qc.py --manifest "$d/filtered_manifest.tsv" --root . --sequences "$d/annotations.tsv" --rcsb \
      --out "$d/structure_qc.tsv" --drop "$d/filtered_manifest.qc.tsv"
done
echo "Review structure_qc.tsv in both folders. The tables below use the unfiltered manifests; to apply the QC result, move"
echo "filtered_manifest.qc.tsv over filtered_manifest.tsv in the folder first."

echo "== benchmark tables =="
python build_benchmark_tables.py --manifest "$MANIFEST" --cache-dir dpfunc_filter_cache \
    --filtered-original filtered_mf_original --filtered-resplit filtered_mf_reshuffled --outdir benchmark_tables
ls benchmark_tables
