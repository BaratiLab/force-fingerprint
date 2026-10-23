# MultiPeptide

Predict properties of proteins from their **sequence**, using 3D **structure** as a teacher while training.

You give it a table with a sequence column and one or more columns you want to predict (a number, a yes/no, a class, a set of labels). It looks up a 3D structure for each protein, trains a model on
sequence *and* structure, and ships a model that **needs only a sequence at prediction time**. Structure at prediction time is optional.

```
 training                                                     prediction (default)
 sequence  --> frozen ESM-2 --> sequence model --\            new sequence --> frozen ESM-2 --> sequence model --> your targets
 structure --> geometry-only graph network -------> contrastive space + per-modality experts
 (teacher, frozen after its own training)          (structure knowledge is distilled into the sequence model)
```

## Where this fits

This code is a proof of concept in two stages. **Stage 1 (this branch):** take the DPFunc benchmark, replace DPFunc's method with a protein language model and a sequence+structure model, and benchmark them against DPFunc on the same proteins (`benchmark/`). **Stage 2 (proposal):** if that works, replace the DPFunc benchmark inputs with the one-dimensional force fingerprints from UIUC and repeat the same comparison. The force fingerprints replace the DPFunc data; they are not an extra input on top of it. The code can already read per-residue arrays (`--feature-files`, section 5) and has been tested on per-residue values derived from structures, but it has not been run on real force descriptors, which are not available yet.

> **About the bundled demo:** `examples/demo_only_data/` holds a small demo table (section 2). It is for checking that the code runs, **not** a realistic use case - its targets (AlphaFold pLDDT, GO-derived labels) are not properties you would actually predict.

## The default behaviour, at a glance

| | default | how to change it |
|---|---|---|
| Training uses | sequence **and** structure (structure is the teacher) | `--structure off` for a pure sequence model |
| Where structures come from | your file, your PDB id, a PDB search on your sequence, then AlphaFold DB (needs a `uniprot_id` column) | `--structure generate` also folds proteins with none using ESMFold |
| Low-confidence AlphaFold/ESMFold models (mean pLDDT < 70) | ignored; that protein uses its sequence only | `--set structure.min_plddt=80` |
| Models | **one per target column** (you are asked; Enter = this) | `--mode joint` = one model for all targets |
| Prediction uses | **sequence only** (no structure lookup, fast) | `predict ... --structure on` also uses structures |

## 1. Install (a reproducible environment)

Tested with **Python 3.9.6 on macOS (Apple silicon)**; Linux, Windows and other Python versions are untested. A GPU is not required: it uses CUDA if present, else Apple's MPS, else the CPU
(CPU and CUDA are untested; all timings below are from an Apple M5).

```bash
git clone --branch plm-1d-prediction https://github.com/BaratiLab/force-fingerprint.git
cd force-fingerprint
python3 -m venv .venv                  # a private environment, so nothing else on your machine is touched
source .venv/bin/activate              # (Windows: .venv\Scripts\activate - untested)
python -m pip install --upgrade pip
pip install -r requirements.txt        # exact versions: torch 2.2.2, numpy 1.26.4, pandas 2.3.3, scipy, scikit-learn, PyYAML, requests, transformers 4.40.2
```
The install was re-checked from scratch in a clean environment (about 20 s with a warm pip cache; a cold download of PyTorch is larger). Leave the environment with `deactivate`; re-enter it with the `source` line.

**First run downloads:** the ESM-2 weights (about 130 MB, from Hugging Face) and the structures you ask for (RCSB / AlphaFold DB). Everything is cached under `cache/` (about 0.3 MB per protein for embeddings, plus the structures),
so later runs are fast. You need internet access for the first run only if something is not cached yet.

## 2. Quick start with your own data

**Step 1 - make a table (CSV or TSV).** One row per protein. Required: a sequence column. Everything else is optional. Column names do not have to match; they are detected.

| column | meaning |
|---|---|
| id | any unique id (row numbers are used if there is none) |
| sequence | amino-acid sequence |
| `structure_file` / `pdb_id` / `uniprot_id` | hints for finding the structure: a local `.pdb`/`.cif`, e.g. `1UBQ` or `1UBQ.A`, and a UniProt accession for AlphaFold DB |
| `split` | `train` / `valid` / `test`; if absent a group-aware 80/10/10 split is made |
| `interpro` | optional InterPro ids separated by `;` (e.g. `IPR010329;IPR011051`); with structure on, they guide the structure readout (DPFunc-style domain-guided attention). Vocabulary comes from the training split only |
| `group` | homology cluster (e.g. from MMseqs2) so relatives stay in one split |
| *every other column* | a **target** to predict (or an input, see section 5) |

**Step 2 - check what it understood** (offline, instant, no downloads):
```bash
python mp.py inspect --csv my.csv
```
It prints which column is the sequence, which is the id, the split, and for each target its **type** (regression, binary, multiclass or multilabel) and how many values are filled. Fix anything wrong with
`--columns id=name sequence=seq split=fold` or `--targets activity solubility:regression`. Empty cells are treated as missing labels (never filled in).

**Step 3 - train.**
```bash
python mp.py train --csv my.csv
```
With several targets it explains the one-model-per-target / one-joint-model choice with a time estimate and asks; press Enter for the recommended one model per target. Results are written to `runs/default/` (change with `--out`).

**Step 4 - predict on new proteins** (they only need an id and a sequence):
```bash
python mp.py predict --run runs/default --csv new_proteins.csv --out predictions.csv
```

Optional, before step 3: `python mp.py prepare --csv my.csv` fetches structures and embeds sequences without training, so you can read `cache/structure_report.tsv` and see which proteins got a structure and why others did not.

**Optional: run the demo to check your install.**

> **The demo is NOT a realistic use case.** It exists only to exercise the code. Its targets - the AlphaFold pLDDT of each protein's model, a catalytic-activity flag derived from GO annotations, and 106 GO terms -
> are by-products of structure prediction and database curation, not measurements you would want to predict, and its 634 proteins are soluble monomers only. Do not read its scores as how the tool will perform on real properties of your own proteins.

It lives in `examples/demo_only_data/` (157 real proteins with real structures, about 5 MB; see the `README.txt` there for its provenance). Run it with:
`python mp.py train --csv examples/demo_only_data/demo_only_curated_mf.csv`

## 3. The choices you make

| Flag | Values | What it does |
|---|---|---|
| `--mode` | `per_target`, `joint` | One model per target column (recommended: each target gets its own encoders, early stopping and metric, so a hard target cannot drag down an easy one; costs one training per target), or one shared model for all targets (about T times cheaper for T targets, but one early-stopping point is a compromise) |
| `--structure` | `off`, `on`, `generate` | `off`: pure sequence model, no structure lookup. `on` (training default): use structures. `generate`: also fold proteins with none using ESMFold (up to 400 residues; the sequence is sent to api.esmatlas.com). At **prediction** the default is `off` |
| `--targets` | column names, optionally `name:kind` | Choose targets or force a type: `regression`, `binary`, `multiclass`, `multilabel` (labels separated by `;`) |
| `--columns` | `id=… sequence=… split=…` | Say which column is what when detection guesses wrong |
| `--features`, `--feature-files` | see section 5 | Extra input columns / files |
| `--out`, `--run` | paths | Training output folder; trained run to predict with |
| `--set` | `section.key=value` … | Any other setting in `config.yaml`, e.g. `--set train.epochs=30 structure.min_plddt=80`. Mistyped keys are rejected |

Commands: `inspect` (offline check), `prepare` (fetch structures + embed), `train`, `predict`.

## 4. How structures are found, and the confidence filter

For each protein the sources are tried in a fixed order and the first that covers at least 80% of your sequence wins: your `structure_file` -> your `pdb_id` -> an RCSB search by your sequence -> AlphaFold DB for your
`uniprot_id` -> (only with `--structure generate`) ESMFold. Experimental structures are preferred over predicted ones. AlphaFold DB is a **lookup** of existing predictions, so it works only when the table has a `uniprot_id`.

AlphaFold and ESMFold store their confidence (pLDDT) in the file's B-factor column. A model whose **mean pLDDT is below 70** is ignored and that protein trains and predicts on its sequence alone. This also applies to
AlphaFold files you supply yourself; experimental PDB structures are never filtered. `cache/structure_report.tsv` records the source, chain, coverage, mean pLDDT and the reason for every miss.

## 5. Extra input columns (any kind of feature)

Every column that is not an id, sequence, split, group or structure hint is a *target* unless you say it is an *input*. Inputs become extra modalities that are aligned with sequence and structure:
```bash
# a group of table columns: numbers are standardised (+ missing flags), text columns are one-hot encoded
python mp.py train --csv my.csv --features chem=logp,charge,mw --targets activity
# per-protein .npz files named in a column, one value row per structure residue: X [L,D] (required), pair [L,L], prior [L] in 0-1, vec [Dv] (optional)
python mp.py train --csv my.csv --feature-files md=md_file --targets activity
```
Statistics are fitted on the training split only and saved with the model. A protein missing a group has that input masked, never filled in. Per-residue arrays use the structure's residue graph, so they need structure
(`--structure on`); summarise repeated measurements into channels yourself (e.g. mean and std).

## 6. Prediction: sequence first, structure on request

```bash
python mp.py predict --run runs/default --csv new.csv --out preds.csv                       # sequence only (default)
python mp.py predict --run runs/default --csv new.csv --out preds.csv --structure on        # also look up and use structures
python mp.py predict --run runs/default --csv new.csv --out preds.csv --structure generate  # ... and fold the ones with none
```
With `--structure on` a protein uses its structure only if one was found and passed the pLDDT filter; otherwise it silently falls back to the sequence model. The `structure_used` column says which path each protein took.
Values are on the natural scale: regression in original units, binary as a probability, multiclass as the class plus per-class probabilities, multilabel as labels with probability >= 0.5 (full probabilities in an `.npz`).

## 7. What you get after training

For one model per target (the default) each target has its own folder; with `--mode joint` the same files sit directly in the run folder.
```
runs/default/
  summary.csv                       test score per target (the headline numbers)
  config.yaml                       the exact settings used
  per_target/<target>/
      best.pt                       the trained model (used by predict)
      results.json                  details (below)
      test_predictions.csv          predictions for the test split
```
Metrics (each target uses its own): regression - Spearman (also Pearson, R2, RMSE, MAE); binary - AUROC (AUPRC, MCC, accuracy); multiclass - macro-F1 (accuracy, majority baseline); multilabel - Fmax (AUPR).
The headline scores are for the **deployed, structure-free path**, with 95% group-bootstrap intervals. `results.json` adds `_with_structure` (the same model when a protein's structure is used), `_only` (each modality alone),
`_gates` (learned weight of each modality), `_controls` (`struct_shuffled`: every protein given another protein's structure - if the model really uses structure this must hurt; `without_<group>` for extra inputs)
and `_alignment` (how well sequence and structure agree in the shared space, including retrieval among length-matched candidates, which is the honest number).

## 8. How it works (short)

* **Sequence side:** frozen ESM-2 (35M) residue embeddings -> a small transformer -> one vector per protein.
* **Structure side:** a C-alpha contact graph (CA-CA < 10 A, 24 nearest) with a geometry-only network (distances, backbone angles, sequence offsets). It never sees residue letters, so matching a sequence to its structure is a real task.
* **Teacher/student training (when structure is used):** (A) the structure network is trained alone on your targets, then **frozen**; (B) the sequence side and the heads are trained with the frozen structure space as a fixed contrastive target
  (a protein's sequence and structure vectors must match each other more than other proteins'; relatives are not used as negatives; batches group similar lengths so protein length cannot be used as a shortcut) and with the structure randomly hidden
  from the head, so the structure-free path that is deployed is trained too.
* **Prediction heads:** every modality has its own expert head, per-target gates weight the experts over the modalities a protein has, and a zero-initialised interaction term learns only the residual. This keeps a gradient flowing to every encoder
  even when one modality (the strong frozen ESM features) already fits the training set.

## 9. Demo results (a code check - not representative of real use)

> These numbers come from the demo data described in section 2, whose targets (AlphaFold pLDDT, a GO-derived flag, GO terms) are not the kind of property you would actually predict. They show that the pipeline runs and how the controls read; they are not evidence of performance on your data.

Measured on the full 634-protein version of the demo (the bundled one is a 157-protein sample of it): two seeds, 20 epochs, 69 test proteins; targets: AlphaFold pLDDT (Spearman), a catalytic-activity flag (AUROC), 106 GO terms (Fmax); test scores, mean of 2 seeds, deployed (structure-free) path.

| model | pLDDT | catalytic | GO terms | mean |
|---|---|---|---|---|
| sequence only (`--structure off`) | 0.798 | 0.862 | 0.471 | 0.710 |
| teacher/student (the default) | 0.812 | 0.859 | 0.504 | 0.725 |

* On the demo, **structure adds no measurable information beyond the ESM sequence embedding**: removing or shuffling it changes scores by about +-0.02, and teacher/student's edge over sequence-only (+0.015 mean) is within the noise of 2 seeds x 69 proteins.
  Whether structure helps on **your** targets is exactly what `_controls`, `_gates` and `_alignment` in `results.json` will tell you; do not assume it does.
* One model per target vs one joint model: a single-seed comparison of sequence-only models on the demo favoured one model per target (0.771 vs 0.710 mean, mostly on GO terms). One seed is suggestive, not proof.

## 10. Time and resources (Apple M5; proteins of median ~300 residues; measured on the full 634-protein version of the demo - the bundled one is a 157-protein sample of it - at 386 and 634 rows, larger sizes extrapolated)

| model | per epoch | per 1,000 rows |
|---|---|---|
| sequence + structure, student phase | 7.5 s at 386 rows, 15 s at 634 rows | ~24 s |
| structure teacher phase | 11 s at 634 rows | ~17 s |
| sequence only (`--structure off`) | 3-4 s at 634 rows | ~6 s |

One model takes about epochs x 0.024 s x rows - for example 20 epochs on 634 rows is about 5 minutes, 100 epochs on 1,000 rows about 40 minutes (the teacher phase adds a smaller share on top). One model per target multiplies this by the number of targets.
One-off preparation: embeddings ~0.1 s per sequence; first structure lookup 0.1-4 s per protein (the PDB search is the slowest), cached afterwards.

## 11. Troubleshooting

| You see | Do this |
|---|---|
| `Could not find a sequence column` | `--columns sequence=<your column>` |
| a column is skipped as "free text" | it has too many distinct text values to be a class; force a type with `--targets name:multiclass`, or ignore it |
| `no usable target columns` | every non-id column was skipped or is an input; name the targets with `--targets` |
| `unknown setting 'train.epoch'` | a mistyped `--set` key; the available keys are in `config.yaml` |
| many proteins use "none (sequence-only)" | read `cache/structure_report.tsv`: usually no `uniprot_id`/`pdb_id`, the structure did not cover 80% of your sequence, or mean pLDDT < 70. Add hints, lower `--set structure.min_plddt=…`, or use `--structure generate` |
| out of memory | `--set train.batch_size=16`, or lower `data.max_len` (default 1000 residues) |
| download errors on first run | the first run needs internet for the ESM-2 weights and any structures not cached yet |
| a warning about few rows | under ~100 proteins results are noisy; rely on the confidence intervals |

## 12. What is tested, and the limits

* `python tests/smoke.py` (a few minutes, real proteins from the bundled demo data, no network) trains one epoch of two scenarios that cover every option, plus a pure sequence model:
  **A, sequence at inference** = the default configuration with the demo's `interpro` column (domain guide on; teacher/student training, one model per target, sequence-only prediction), and
  **B, sequence + structure at inference** = one joint model with a tabular feature group, per-residue array features (all keys, on real values, some proteins without a file) and a user `group` column, predicted on the sequence alone and with `--structure on`.
  It also checks the pLDDT filter on real AlphaFold and experimental files, input features leaving the target list, and the loud failure on a mistyped setting. It passes.
* Structure lookup was tested against the live services: a local file, an RCSB download by `pdb_id` (1UBQ), AlphaFold DB by `uniprot_id` (P62987), the RCSB sequence search, and ESMFold.
* Small datasets (hundreds of proteins) overfit easily. Splits group by shared PDB entry and 4-mer similarity (roughly >50-60% identity); remote homologs are not caught - supply a `group` column from MMseqs2 for a strict split.
* Not modelled: free-text columns, sequences longer than `data.max_len`, non-standard residues (mapped to X). Not verified: Linux/Windows, CUDA, CPU, Python versions other than 3.9.

## Benchmark against DPFunc

`benchmark/` compares the sequence model, the sequence+structure model and DPFunc on the DPFunc MF benchmark, on a filtered monomer subset with DPFunc's original split, and on the curated re-split. See `benchmark/README.md`.

## Project layout

```
mp.py                  command line (inspect / prepare / train / predict)
config.yaml            every setting, with comments
requirements.txt       pinned versions
multipeptide/          columns (detect roles) - data (table, splits, batches) - targets (types, losses, metrics) - features (extra inputs)
                       fetch + structure (find structures, pLDDT filter) - plm (ESM-2 embeddings) - model - losses - pipeline (train, predict) - diagnostics
tests/smoke.py         the end-to-end test
benchmark/             run.py (train + score a benchmark table), score.py (one scorer for every method), README.md
examples/demo_only_data/   small demo table + structures (DEMO ONLY, see section 2)
```
