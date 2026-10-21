# Benchmark tables (molecular function)

`build_benchmark_tables.py` writes the three tables used to compare a sequence model, a sequence+structure model and DPFunc
(the models and the scoring live on the `plm-1d-prediction` branch, folder `benchmark/`).

| table | proteins | split (train / valid / test) | GO terms |
|---|---|---|---|
| `full` | all DPFunc MF benchmark proteins that UniProt still maps and whose sequence has at most 1000 residues (30,371 of 34,164) | 29,383 / 369 / 619 (DPFunc's own) | 1,615 |
| `reduced_original` | the 690 filtered monomers (same filters as the curated set, no re-split, no undersampling) | 672 / 4 / 14 (DPFunc's own) | 124 |
| `reduced_resplit` | the 635 curated monomers | 497 / 69 / 69 (re-split) | 106 |

GO roots are removed from the labels in every table. The label set is chosen from the training proteins of each table (terms seen in
at least 10 of them), so the three tables have different vocabularies and their scores are not on the same label set.
The 4 validation and 14 test proteins of `reduced_original` make its scores noisy (85 of its 124 terms have no positive test
protein); it is included because it keeps DPFunc's split unchanged.

## Build

```bash
pip install -r requirements.txt

# 0. the DPFunc manifest (not stored in this repository): copy
#    DPFunc_demo_pipeline/templates/mf_manifest_from_data_dpfunc.tsv from https://github.com/CSUBioGroup/DPFunc
#    into the repository root.

# 1. the filtered monomers with DPFunc's original split
python dpfunc_filter.py --manifest mf_manifest_from_data_dpfunc.tsv --ontology mf --outdir filtered_mf_original \
    --pdb-oligomer-check --check-alphafold --min-term-count 10

# 2. the tables
python build_benchmark_tables.py --manifest mf_manifest_from_data_dpfunc.tsv --cache-dir dpfunc_filter_cache \
    --filtered-original filtered_mf_original --filtered-resplit data/curated_mf \
    --structure-dir /path/to/DPFunc/DataProcess/AF2DB \
    --outdir benchmark_tables
```

Step 1 queries UniProt, RCSB and AlphaFold DB for every protein of the manifest the first time it runs (internet needed, expect a
long run) and stores the answers in `dpfunc_filter_cache/`; step 2 reads the UniProt sequences from `dpfunc_filter_cache/uniprot.tsv`,
so run step 1 first. The curated re-split table needs no filter run: its proteins and labels are `data/curated_mf/filtered_manifest.tsv`.
`--structure-dir` is optional and can be repeated: a protein without a structure file is looked up by its UniProt accession when the model runs.

Each table comes with `<name>_dpfunc_manifest.tsv`, the same proteins in DPFunc's manifest format, so DPFunc is trained and
tested on exactly the rows the other models see. Next: `benchmark/README.md` on the `plm-1d-prediction` branch.
