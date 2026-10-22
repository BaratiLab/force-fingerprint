# Benchmark: sequence model, sequence+structure model and DPFunc

Molecular-function (MF) prediction on three tables that are built on the `data` branch
(`build_benchmark_tables.py`). Every method is trained on the same training proteins, tested on the same test
proteins and scored by the same code (`score.py`).

| setting | proteins | split | what it answers |
|---|---|---|---|
| `full` | the whole DPFunc MF benchmark (30,371 proteins with a UniProt sequence of at most 1000 residues) | DPFunc's own | how do the models compare with DPFunc on its own data? |
| `reduced_original` | the 690 filtered soluble monomers | DPFunc's own: 672 / 4 / 14 | same filtered proteins, split untouched |
| `reduced_resplit` | the 635 curated monomers | re-split 497 / 69 / 69 | the curated benchmark used in the slides |

Each table keeps the GO terms seen in at least 10 of its own training proteins (1,615 terms for `full`, 124 for `reduced_original`, 106 for `reduced_resplit`), so the label sets differ between settings and scores are not comparable across them.

The `reduced_original` split has 4 validation and 14 test proteins, and 85 of its 124 terms have no positive test protein.
Treat its scores, and early stopping on 4 validation proteins, as very noisy; the bootstrap intervals in the results table show how much.

Compared in each setting:

1. sequence only (`--structure off`)
2. sequence + structure, the default configuration: a frozen structure teacher guides the sequence model in training; the prediction uses the sequence alone
3. the same trained model with the structure looked up and used at inference
4. DPFunc, trained on the same manifest (below)
5. a reference that predicts each term's training frequency for every protein

## 1. Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Build the tables on the `data` branch (`git clone --branch data https://github.com/BaratiLab/force-fingerprint.git force-fingerprint-data`, then follow
`docs/BENCHMARK.md` there) and copy the resulting `benchmark_tables/` folder into the root of this repository (the commands below use `benchmark_tables/<setting>.csv`).
Labels are the GO terms seen in at least 10 training proteins, GO roots removed (`--min-count`).

## 2. Resolve structures and embed sequences once (recommended, needs internet)

```bash
python mp.py prepare --csv benchmark_tables/full.csv --targets go_terms:multilabel --set data.min_label_count=10
```

Structures come from a local file when the table has one, otherwise from RCSB or AlphaFold DB through `uniprot_id`.
Models with mean pLDDT below 70 are ignored (that protein is then predicted from its sequence alone). For `full` this makes
tens of thousands of requests; the result is cached in `cache/`, so later runs are offline. The ESM-2 embeddings are cached too.

## 3. Run a setting

```bash
python benchmark/run.py --table benchmark_tables/full.csv --out runs/full
python benchmark/run.py --table benchmark_tables/reduced_original.csv --out runs/reduced_original
python benchmark/run.py --table benchmark_tables/reduced_resplit.csv --out runs/reduced_resplit
```

Add `--dry-run` to print the commands first, `--set train.batch_size=64` (or any other config key) to change a setting.
Steps whose output exists are skipped, so an interrupted run can be restarted. Results: `runs/<setting>/scores.csv` and `scores.md`.

## 4. DPFunc on the same proteins

`build_benchmark_tables.py` also writes `<setting>_dpfunc_manifest.tsv`, with exactly the rows of the table above. Follow the
DPFunc repository's pipeline with that manifest (`DPFunc_demo_pipeline/scripts/build_data_demo.py`, then `run_dpfunc.py train`
and `run_dpfunc.py predict`, MF ontology) and export its test-set scores as a table with the columns

```
protein_id   go_term      score
3HAO_HUMAN   GO:0016491   0.93
```

Then add it to the comparison:

```bash
python benchmark/score.py --table benchmark_tables/full.csv --pred dpfunc=dpfunc_test_scores.tsv \
    --pred "sequence only=runs/full/pred_seq_off_go_terms_probs.npz" --out runs/full/with_dpfunc
```

or pass `--pred dpfunc=dpfunc_test_scores.tsv` to `run.py` directly. Terms outside the label set above are ignored, terms DPFunc does not
score count as 0, and the comparison covers only the test proteins that every method predicted.

## Reading the results

`scores.csv` has one row per method: Fmax (protein-centric, CAFA style), AUPR over all protein-term pairs (micro) and averaged over
terms (macro), and a 95% bootstrap interval of Fmax over test proteins. Differences smaller than the interval are not evidence of a better method.
Published DPFunc numbers (Wang et al., Nat. Commun. 2025) come from other test sets and are not comparable with these scores.

## Results (`reduced_resplit`)

Evaluated on the 635 curated monomer benchmark (`benchmark_tables/reduced_resplit.csv`: 497 train / 69 valid / 69 test) across 106 Molecular Function GO terms (see `benchmark/results/`):

| method | n_proteins | n_terms | Fmax | 95% CI low | 95% CI high | AUPR (micro) | AUPR (macro) |
|---|---|---|---|---|---|---|---|
| sequence only | 69 | 106 | 0.684 | 0.621 | 0.758 | 0.616 | 0.621 |
| sequence + structure (teacher; sequence only at inference) | 69 | 106 | 0.616 | 0.563 | 0.679 | 0.582 | 0.586 |
| sequence + structure (structure used at inference) | 69 | 106 | 0.614 | 0.560 | 0.675 | 0.591 | 0.593 |
| train frequency (reference) | 69 | 106 | 0.380 | 0.347 | 0.421 | 0.286 | 0.074 |

> **Note on sequence vs. sequence+structure:** Sequence+structure does not necessarily improve over sequence alone here because the validation and test sets have very few proteins (69 each) and too much noise for structure guidance to show a statistically significant edge over strong pretrained ESM-2 sequence embeddings.

## Not covered

Only MF is set up. Proteins longer than 1000 residues are left out of every setting (about 10% of the full benchmark), so the `full` test set
is smaller than DPFunc's own.
