# force-fingerprint

Curation and filtering pipeline for the [DPFunc](https://github.com/CSUBioGroup/DPFunc) protein function prediction benchmark.

> **Molecular Function (MF)** is the Gene Ontology (GO) domain describing elemental biochemical activities (e.g., catalytic activity, ligand binding) exerted by an individual gene product at the molecular level.

Detailed biological criteria, mathematical formulations, and benchmark statistics are documented in [docs/DATASET_CURATION.md](docs/DATASET_CURATION.md).

Benchmark tables for comparing models with DPFunc (full benchmark, filtered set with the original split, curated re-split): [docs/BENCHMARK.md](docs/BENCHMARK.md).

---

## Quickstart

```bash
git clone https://github.com/BaratiLab/force-fingerprint.git
cd force-fingerprint
pip install -r requirements.txt
```

### Option A: Use Pre-Curated Dataset
The curated manifests, train/val/test splits, and loss weights are already tracked in `data/curated_mf/`. 

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

All external UniProt and PDB queries are cached in `dpfunc_filter_cache/` for instant re-runs.

---

## Dataset: DPFunc Filter Criteria

- **Monomers**: Single-chain monomers only (UniProt subunit annotations + RCSB PDB crystal symmetry). Drops oligomers and complexes.
- **No membrane proteins**: Drops transmembrane, intramembrane, and lipid-anchored proteins.
- **<600 residues**: Restricts sequence length to $30 \le \text{length} < 600$.
- **Manage class imbalance**: Prunes root GO terms, defines vocabulary strictly on train split ($\ge 10$ occurrences), applies ML-RUS, and generates `pos_weight` and `sample_weight`.
- **Check structures available or not - PDB vs AlphaFold**: Audits experimental PDB structures vs. AlphaFold DB models and downloads coordinates.

---

## Repository Layout & Data Locations

```text
force-fingerprint/
├── data/
│   └── curated_mf/             # Tracked in Git: curated manifest, loss weights, stats (~280 KB)
│       ├── filtered_manifest.tsv  # Curated manifest (635 proteins) with 80/10/10 split assignments
│       ├── pos_weight.npy         # Precomputed 106-term BCE loss weights for PyTorch
│       ├── term_stats.tsv         # Label counts, frequencies, and imbalance ratios per GO term
│       ├── structures_to_fetch.tsv# Direct download URLs for all verified PDB & AlphaFold models
│       ├── report.json            # Machine-readable attrition funnel and drop counts
│       └── report.txt             # Human-readable filtering report
├── filtered_mf/                # Local run output directory (structures unpack here)
│   ├── structures/             # 635 downloaded 3D structures (.pdb and .cif files)
│   └── structures_mf.tar.gz    # Compressed 82 MB archive (from release)
└── dpfunc_filter_cache/        # Cached API responses (uniprot.tsv, rcsb, alphafold)
```

---

## Benchmark Curation Statistics (MF Split, Re-split 80/10/10)

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
The raw DPFunc manifest came with pre-assigned splits (33,046 train / 419 valid / 699 test). Because biological monomer and non-membrane filters eliminate ~98% of candidate proteins, preserving the original split leaves only **4 validation** and **14 test** proteins—making model evaluation statistically meaningless.

The pipeline filters all 34,164 proteins across the whole dataset first, then re-partitions the surviving clean monomers with `--resplit 0.8 0.1 0.1` into a balanced **497 train / 69 validation / 69 test** split. The GO vocabulary and class weights are derived strictly from the new train split to prevent data leakage.

### 2. PDB vs. AlphaFold Structure Strategy
AlphaFold DB generated predictions for virtually the entire UniProt proteome, including proteins that already have experimental crystal structures. In our audit:
- **PDB + AlphaFold**: Both an experimental RCSB PDB structure and an AlphaFold DB prediction exist.
- **AlphaFold Only**: No experimental structure exists in the PDB; AlphaFold is the only 3D model available.

The pipeline operates in **PDB-first mode** (`--structure-source-pref pdb`, the default):
- For all **336 proteins** with experimental structures, experimental coordinates are downloaded directly from the RCSB PDB (preferring native `.pdb`, falling back to mmCIF `.cif` if only CIF was deposited).
- For the **299 proteins** lacking experimental structures, pre-computed predictions are fetched from AlphaFold DB (`.pdb`).
- Users can optionally pass `--structure-source-pref alphafold` if they prefer AlphaFold predictions across all 635 proteins for uniform backbone modeling.

---

## License

[MIT](LICENSE)
