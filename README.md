# force-fingerprint

Curation and filtering pipeline for the [DPFunc](https://github.com/CSUBioGroup/DPFunc) protein function prediction benchmark.

> **Molecular Function (MF)** is the Gene Ontology (GO) domain describing elemental biochemical activities (e.g., catalytic activity, ligand binding) exerted by an individual gene product at the molecular level.

Role in the project: this dataset is the benchmark on which a protein language model and a multimodal model are compared with DPFunc (stage 1). The proposal is then to replace the DPFunc data with the 1D force fingerprints from UIUC and run the same pipeline and comparison on them (stage 2).

Detailed biological criteria, mathematical formulations, and benchmark statistics are documented in [docs/DATASET_CURATION.md](docs/DATASET_CURATION.md).

Benchmark tables for comparing models with DPFunc (full benchmark, filtered set with the original split, curated re-split): [docs/BENCHMARK.md](docs/BENCHMARK.md).

---

## Quickstart

```bash
git clone https://github.com/BaratiLab/force-fingerprint.git
cd force-fingerprint
pip install -r requirements.txt
```

The full pipeline (Option B below) also needs:

- **metapredict** (disorder prediction). Requires Python development headers (`Python.h`). If the build fails with `Python.h: No such file or directory`, use a Python that ships its headers, such as a conda/micromamba environment or a uv-managed Python (`uv python install 3.11`); no root access is needed for either.
- **MMseqs2** (sequence-redundancy filtering), an external binary that must be on `PATH` or passed with `--mmseqs /path/to/mmseqs`:
  ```bash
  conda install -c bioconda mmseqs2
  ```

### Option A: Use Pre-Curated Dataset
The curated manifests, train/val/test splits, and loss weights from release v0.1.0 are tracked in `data/curated_mf/`. They were produced by an earlier version of the pipeline (strict monomers, length < 600, no disorder or redundancy filtering); see [Benchmark Curation Statistics](#benchmark-curation-statistics-v010-release).

To download and unpack the matching 635 3D structures:
```bash
mkdir -p filtered_mf

# Via curl:
curl -sL https://github.com/BaratiLab/force-fingerprint/releases/download/v0.1.0/structures_mf.tar.gz -o filtered_mf/structures_mf.tar.gz
tar -xzf filtered_mf/structures_mf.tar.gz -C filtered_mf

# Or via GitHub CLI:
gh release download v0.1.0 -p "structures_mf.tar.gz" -D filtered_mf
tar -xzf filtered_mf/structures_mf.tar.gz -C filtered_mf
```

### Option B: Run the Full Filtering Pipeline from Scratch
```bash
python dpfunc_filter.py \
  --manifest mf_manifest_from_data_dpfunc.tsv \
  --ontology mf \
  --outdir filtered_mf \
  --pdb-oligomer-check \
  --check-alphafold \
  --resplit 0.8 0.1 0.1 \
  --min-term-count 10 \
  --undersample-frac 0.1 \
  --download-structures
```

With current defaults this also applies the disorder filter and MMseqs2 redundancy filtering, and keeps sequences shorter than 1,200 residues, so its output differs from the v0.1.0 dataset. Commonly adjusted options:

| Option | Default | Effect |
| :--- | :--- | :--- |
| `--max-len` | `600` | Keep sequences with length < max-len. |
| `--monomer-mode` | `strict` | `strict`, `lenient`, or `supported` (see below). All three are evaluated and recorded in `annotations.tsv` on every run. |
| `--pdb-hetero-as-complex` | off | In `supported` mode, treat PDB assemblies with a real protein partner as complex evidence. |
| `--max-disordered` | `10` | Drop proteins with an internal disordered run longer than this (`-1` disables). |
| `--max-terminal-disordered` | `25` | Also cap the length of N-/C-terminal disordered tails (`-1` disables). |
| `--disorder-threshold` | `0.5` | metapredict score above which a residue counts as disordered. |
| `--min-seq-id` | `0.4` | MMseqs2 identity threshold for redundancy filtering (`0` disables). |
| `--cluster-coverage` / `--cluster-cov-mode` | `0.8` / `0` | MMseqs2 alignment coverage requirement (`0` = both sequences). |
| `--dedup-name-prefix` | off | Also keep at most one protein per UniProt entry-name prefix (e.g. `CATA_HUMAN` / `CATA_MOUSE`). |
| `--skip-identity-verify` | off | Skip the all-against-all check that no two kept proteins exceed the identity threshold. |

All external UniProt, PDB, AlphaFold, metapredict, and MMseqs2 results are cached in `dpfunc_filter_cache/` for fast re-runs. Changing the UniProt fields the script requests triggers a one-time re-download of the UniProt cache.

---

## Dataset: DPFunc Filter Criteria

Applied in this order; every dropped protein is logged with a reason in `annotations.tsv`.

- **UniProt mapping**: DPFunc IDs are mapped to current UniProtKB entries; obsolete or unmappable IDs are dropped.
- **Length**: $30 \le \text{length} < 600$ (configurable with `--min-len` / `--max-len`).
- **No membrane proteins**: Drops transmembrane, intramembrane, and lipid-anchored proteins.
- **Monomers**: Excludes oligomers and stable complex members, in one of three modes:
  - `strict`: UniProt's subunit annotation says monomer and contains no oligomer or complex language.
  - `lenient`: also accepts "monomer and homodimer"-style annotations.
  - `supported`: drops proteins with positive evidence of a stable complex (UniProt text about the protein itself, Complex Portal membership, homo-oligomeric PDB assemblies), then keeps only proteins with monomer support, ranked in `monomer_tier` (1 = UniProt and PDB agree, 2 = UniProt only, 3 = PDB only).
- **Disorder**: Drops proteins with an internal disordered region longer than 10 residues (metapredict per-residue scores > 0.5). Terminal tails are limited to 25 residues by default.
- **Sequence redundancy**: Clusters with MMseqs2 (≥ 40% identity, 80% coverage) and keeps one protein per cluster, preferring human, mouse, yeast, then *E. coli*, reviewed entries, proteins with PDB structures, and longer sequences. A verification search then guarantees no two kept proteins exceed the threshold.
- **Structures**: Audits experimental PDB structures vs. AlphaFold DB models and downloads coordinates.
- **Manage class imbalance**: Prunes root GO terms, defines vocabulary strictly on train split ($\ge 10$ occurrences), applies ML-RUS, and generates `pos_weight` and `sample_weight`.

---

## Repository Layout & Data Locations

```text
force-fingerprint/
├── data/
│   └── curated_mf/             # Tracked in Git: v0.1.0 curated manifest, loss weights, stats (~280 KB)
│       ├── filtered_manifest.tsv  # Curated manifest (635 proteins) with 80/10/10 split assignments
│       ├── pos_weight.npy         # Precomputed 106-term BCE loss weights for PyTorch
│       ├── term_stats.tsv         # Label counts, frequencies, and imbalance ratios per GO term
│       ├── structures_to_fetch.tsv# Direct download URLs for all verified PDB & AlphaFold models
│       ├── report.json            # Machine-readable attrition funnel and drop counts
│       └── report.txt             # Human-readable filtering report
├── filtered_mf/                # Local run output directory (structures unpack here)
│   ├── structures/             # 635 downloaded 3D structures (.pdb and .cif files)
│   └── structures_mf.tar.gz    # Compressed 82 MB archive (from release)
└── dpfunc_filter_cache/        # Cached API responses (UniProt, RCSB, AlphaFold, metapredict, MMseqs2)
```

---

## Benchmark Curation Statistics (v0.1.0 Release; MF Split, Re-split 80/10/10))

### Filtering Funnel

| Stage / Filter | Train | Valid | Test | Total | Dropped |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Manifest** | 33,046 | 419 | 699 | **34,164** | — |
| UniProt mapping | 32,751 | 410 | 687 | 33,848 | 316 |
| Length ($30 \le \text{len} < 600$) | 23,084 | 289 | 490 | 23,863 | 9,985 |
| Non-membrane | 15,320 | 203 | 292 | 15,815 | 8,048 |
| Monomers (strict) | 676 | 4 | 14 | 694 | 15,121 |
| **Re-split (80 / 10 / 10)** | **555** | **69** | **70** | **694** | — |
| In-vocab GO terms ($\ge 10$ train) | 552 | 69 | 69 | 690 | 4 |
| After ML-RUS | **497** | **69** | **69** | **635** | 55 |

### Structure Availability & Downloads

| Split | AlphaFold Only | PDB + AlphaFold | Total Kept | Downloaded Coordinates |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 236 | 261 | 497 | 497 (100%) |
| **Validation** | 32 | 37 | 69 | 69 (100%) |
| **Test** | 31 | 38 | 69 | 69 (100%) |
| **Total** | **299** | **336** | **635** | **635 / 635 (100%)** |

*(3,643 unique experimental PDB entry IDs mapped across the 336 PDB-backed proteins. The curated dataset follows a PDB-first strategy: 336 experimental structures are downloaded directly from the RCSB PDB, falling back to AlphaFold DB models for the 299 proteins lacking experimental data).*

---

## Key Design Decisions

### 1. Why Filter First, Then Re-split?
The raw DPFunc manifest came with pre-assigned splits (33,046 train / 419 valid / 699 test). Because the biological filters eliminate the large majority of candidate proteins (~98% in the v0.1.0 run), preserving the original split leaves only a handful of validation and test proteins (4 and 14 in v0.1.0), making model evaluation statistically meaningless.

The pipeline filters 34,164 all proteins across the whole dataset first, then re-partitions the survivors with `--resplit 0.8 0.1 0.1` into a balanced trail / validation / test split. The GO vocabulary and class weights are derived strictly from the new train split.

### 2. Monomer Evidence: Strict vs. Supported
Most UniProt entries never state an oligomeric state; many have no subunit comment or only list binding partners. `strict` mode therefore requires explicit proof of monomericity and removes the vast majority of proteins, including many monomers. `supported` mode first drops proteins with positive evidence of a stable complex, then retains only those with monomer support, recording how strong that support is in `monomer_tier` so the strongest evidence can be preferred when selecting the final set.

### 3. Disorder Before Redundancy
The disorder filter runs before redundancy filtering so that each cluster's representative is chosen only among proteins that pass it; otherwise a cluster could lose its only usable member.

### 4. Redundancy Verification
MMseqs2 clustering only guarantees that each member resembles its own cluster's MMseqs2 representative. Because the pipeline chooses its own representatives (by species and annotation quality), two kept proteins from different clusters could still be similar. An all-against-all MMseqs2 search of the kept set, followed by a greedy pass in representative-priority order, removes any remaining pairs above the threshold.

### 5. PDB vs. AlphaFold Structure Strategy
AlphaFold DB generated predictions for virtually the entire UniProt proteome, including proteins that already have experimental crystal structures. In our audit:
- **PDB + AlphaFold**: Both an experimental RCSB PDB structure and an AlphaFold DB prediction exist.
- **AlphaFold Only**: No experimental structure exists in the PDB; AlphaFold is the only 3D model available.

The pipeline operates in **PDB-first mode** (`--structure-source-pref pdb`, the default):
- For proteins with experimental structures, experimental coordinates are downloaded directly from the RCSB PDB (preferring native `.pdb`, falling back to mmCIF `.cif` if only CIF was deposited).
- For proteins lacking experimental structures, pre-computed predictions are fetched from AlphaFold DB (`.pdb`).
- Users can optionally pass `--structure-source-pref alphafold` if they prefer AlphaFold predictions for all proteins, for uniform backbone modeling.

---

## License

[MIT](LICENSE)

## Structure quality report

`structure_qc.py` checks the structures of a filtered manifest (PDB and AlphaFold): coverage of the UniProt sequence, sequence match, chain breaks,
C-alpha clashes, mean pLDDT for AlphaFold models, resolution for PDB entries (`--rcsb`) and AlphaFold-versus-PDB RMSD (`--af-root`). Proteins without a
structure are reported as `no_structure` and kept by `--drop`, since the sequence model does not need one.

```bash
python structure_qc.py --manifest filtered_mf/filtered_manifest.tsv --root . --sequences filtered_mf/annotations.tsv --rcsb --out structure_qc.tsv
```
