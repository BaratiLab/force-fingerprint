# Benchmark tables (molecular function)

`build_benchmark_tables.py` writes the three tables used to compare a sequence model, a sequence+structure model and DPFunc
(the models and the scoring live on the `plm-1d-prediction` branch, folder `benchmark/`).

| table | proteins | split |
|---|---|---|
| `full` | all DPFunc MF benchmark proteins that UniProt still maps and whose sequence has at most 1000 residues (30,371 of 34,164) | DPFunc's own: 29,383 / 369 / 619 |
| `reduced_original` | the 690 filtered monomers (same filters as the curated set, no re-split, no undersampling) | DPFunc's own: 672 / 4 / 14 |
| `reduced_resplit` | the 635 curated monomers | re-split: 497 / 69 / 69 |

GO roots are removed from the labels in every table. The 4 validation and 14 test proteins of `reduced_original` make its
scores noisy (85 of its 124 terms have no positive test protein); it is included because it keeps DPFunc's split unchanged.

## Build

```bash
# 1. the filtered monomers with DPFunc's original split (reuses dpfunc_filter_cache/, so it is fast once the curated run exists)
python dpfunc_filter.py --manifest mf_manifest_from_data_dpfunc.tsv --ontology mf --outdir filtered_mf_original \
    --pdb-oligomer-check --check-alphafold --min-term-count 10

# 2. the tables
python build_benchmark_tables.py --manifest mf_manifest_from_data_dpfunc.tsv --cache-dir dpfunc_filter_cache \
    --filtered-original filtered_mf_original --filtered-resplit filtered_mf \
    --structure-dir filtered_mf/structures --structure-dir /path/to/DPFunc/DataProcess/AF2DB \
    --outdir benchmark_tables
```

Step 1 needs internet for the first run only; step 2 reads the UniProt sequences from `dpfunc_filter_cache/uniprot.tsv`.
`--structure-dir` is optional: a protein without a file there is looked up by its UniProt accession when the model runs.

Each table comes with `<name>_dpfunc_manifest.tsv`, the same proteins in DPFunc's manifest format, so DPFunc is trained and
tested on exactly the rows the other models see.
