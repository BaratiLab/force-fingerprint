# DPFunc Dataset Curation & Filtering Pipeline: Technical Specification

This document provides a comprehensive technical, biological, and mathematical description of the curation pipeline implemented in `dpfunc_filter.py`. The pipeline curates protein datasets from the [DPFunc benchmark](https://github.com/CSUBioGroup/DPFunc) for structure-based protein function prediction under rigorous biological and machine-learning constraints.

---

## Table of Contents
1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Biological and Structural Filtering Criteria](#2-biological-and-structural-filtering-criteria)
   - [2.1 Monomer Selection and Validation](#21-monomer-selection-and-validation)
   - [2.2 Membrane Protein Exclusion](#22-membrane-protein-exclusion)
   - [2.3 Sequence Length Constraints](#23-sequence-length-constraints)
3. [Multi-Label Class Imbalance Curation](#3-multi-label-class-imbalance-curation)
   - [3.1 Information Leakage Prevention](#31-information-leakage-prevention)
   - [3.2 Root Term Removal](#32-root-term-removal)
   - [3.3 Multi-Label Random Undersampling (ML-RUS)](#33-multi-label-random-undersampling-ml-rus)
   - [3.4 Loss and Sampler Formulations](#34-loss-and-sampler-formulations)
   - [3.5 Multi-Label Imbalance Metrics](#35-multi-label-imbalance-metrics)
4. [3D Coordinate Retrieval and Structure Auditing](#4-3d-coordinate-retrieval-and-structure-auditing)
   - [4.1 Experimental PDB vs. AlphaFold DB](#41-experimental-pdb-vs-alphafold-db)
   - [4.2 AlphaFold DB v6 API Resolution](#42-alphafold-db-v6-api-resolution)
   - [4.3 Resilient HTTP Download Protocol](#43-resilient-http-download-protocol)
5. [Empirical Curation Funnel (Molecular Function Benchmark)](#5-empirical-curation-funnel-molecular-function-benchmark)
6. [Generated Artifacts and File Schemas](#6-generated-artifacts-and-file-schemas)
7. [Downstream PyTorch & DPFunc Integration](#7-downstream-pytorch--dpfunc-integration)

---

## 1. Introduction and Motivation

Structure-based deep learning models such as **DPFunc** predict Gene Ontology (GO) terms by constructing residue-contact graphs and combining them with sequence language embeddings (e.g., ESM-2) and InterPro domain features.

Standard benchmark datasets (such as the 34,164-protein Molecular Function benchmark) often contain confounding biological factors that bias graph representations:
- **Oligomers and Multi-Protein Complexes**: Tertiary contact graphs of isolated monomers from multi-chain complexes miss interfacial binding surfaces and quaternary contacts, leading to misleading contact maps.
- **Membrane-Bound Proteins**: Transmembrane helices and lipid-associated domains undergo unique hydrophobic packing within lipid bilayers that differ fundamentally from soluble globular proteins, skewing spatial graph convolutions.
- **Extreme Sequence Lengths**: Very long macromolecules (> 600 residues) introduce quadratic memory scaling in graph attention mechanisms and GCNs, while short peptides (< 30 residues) lack stable tertiary folds.
- **Severe Class Imbalance**: Multi-label GO distributions have extreme power-law (long-tail) characteristics, where a handful of generic terms dominate thousands of rare specific terms, leading unweighted networks to collapse into majority-class predictions.

The `dpfunc_filter.py` pipeline resolves all of these issues systematically.

---

## 2. Biological and Structural Filtering Criteria

Every candidate protein is evaluated against curated databases (UniProtKB and RCSB PDB) through cached API calls stored in `--cache-dir` (default: `dpfunc_filter_cache/`).

```
Candidate Protein
  │
  ├── 1. Sequence Length [30 <= len < 600]
  │
  ├── 2. Membrane Exclusion [Features + Keywords + Subcellular text]
  │
  ├── 3. Monomer Verification [UniProt SUBUNIT regex + RCSB PDB crystal symmetry]
  │
  └── 4. Label Filtering [GO root pruning + train vocab thresholding + ML-RUS]
```

### 2.1 Monomer Selection and Validation

UniProt's `SUBUNIT` annotation field (`cc_subunit`) free-text contains detailed experimental descriptions of quaternary structure. The pipeline uses a two-tier verification mechanism:

#### A. Free-Text Regex Pattern Parsing
Annotations are classified into discrete quaternary states:
- **Strict Monomer**: Matches explicit monomeric affirmations (e.g., `^monomer(?:\b|[.;])`, `exists as a monomer`, `predominantly monomeric`, `functional as a monomer`).
- **Negative Pattern Filtering**: Explicitly detects and rejects false positives such as `fails to form a monomer`, `inactive as a monomer`, `does not exist as a monomer`, or `monomer in equilibrium with dimer`.
- **Oligomers**: Matches terms indicating quaternary assembly (`homodimer`, `heterotetramer`, `hexamer`, `octamer`, `pentamer`, `forms a complex with`, `interacts with`).

Under `--monomer-mode strict` (default), only unambiguous, exclusively monomeric proteins are retained. Under `--monomer-mode lenient`, proteins that function both as monomers and dimers under differing conditions are permitted.

#### B. RCSB PDB Experimental Assembly Symmetry
When `--pdb-oligomer-check` is enabled, the pipeline cross-checks the experimental biological assembly of all PDB structures mapped to the protein via the RCSB PDB GraphQL API (`rcsb_struct_symmetry`):
- PDB entries with **$C_1$ point group symmetry** and `stoichiometry == "A"` (single chain in the asymmetric unit / biological assembly) confirm a true monomer.
- Entries with $C_2$, $C_3$, $D_2$, or higher cyclic/dihedral point group symmetry with homo-oligomeric stoichiometry ($A_2, A_3, A_4, \dots$) vote against monomer status.
- Proteins with conflicting or missing UniProt comments are discarded if experimental assemblies demonstrate oligomerization.

### 2.2 Membrane Protein Exclusion

To ensure that only soluble globular proteins are retained, the pipeline applies a multi-layered exclusion filter:

1. **Feature Annotations (`ft_transmem` and `ft_intramem`)**:
   UniProt sequence features marking alpha-helical transmembrane regions or intramembrane loops.
2. **Lipidation Features (`ft_lipid`)**:
   Excludes proteins post-translationally modified with lipid anchors that embed them into the bilayer (e.g., `GPI-anchor`, `S-diacylglycerol cysteine`, `N-myristoyl glycine`).
3. **Controlled UniProt Keyword IDs (`keywordid`)**:
   Flags any of the following controlled ontology keywords:
   - `KW-0472`: Membrane
   - `KW-0812`: Transmembrane
   - `KW-1133`: Transmembrane helix
   - `KW-1134`: Transmembrane beta strand
   - `KW-0336`: GPI-anchor
   - `KW-1003`: Cell membrane
   - `KW-0999`: Integral membrane protein
   - `KW-1000`: Peripheral membrane protein
4. **Subcellular Location Text (`cc_subcellular_location`)**:
   Scans free-text location comments for `membrane`, `microsome`, `endoplasmic reticulum membrane`, or `mitochondrial inner membrane` (can be disabled via `--no-subcell-text`).

### 2.3 Sequence Length Constraints

- **Minimum Length**: `--min-len 30` (drops peptide fragments lacking stable folded domains).
- **Maximum Length**: `--max-len 600` (strictly requires $\text{length} < 600$ residues to eliminate high-dimensional graph memory scaling).

---

## 3. Multi-Label Class Imbalance Curation

Gene Ontology annotations follow a directed acyclic graph (DAG) structure with severe power-law frequency distributions.

### 3.1 Information Leakage Prevention
To enforce strict machine-learning best practices, the vocabulary of active GO terms is derived **solely from the training split** using `--min-term-count` (default: 10). Validation and test sets are never inspected to determine the target vocabulary, preventing data leakage. Any term appearing exclusively in validation or test splits is excluded.

### 3.2 Root Term Removal
Ontology root terms encompass every protein in their respective namespace by definition and provide zero discriminative signal. The following terms are pruned before vocabulary construction:
- `GO:0003674` (Molecular Function root)
- `GO:0008150` (Biological Process root)
- `GO:0005575` (Cellular Component root)

### 3.3 Multi-Label Random Undersampling (ML-RUS)

Standard random undersampling collapses multi-label datasets because dropping a sample annotated with a majority label may unintentionally remove a rare minority label.

The pipeline implements **ML-RUS** (*Charte et al., 2015*):
1. Compute the Imbalance Ratio per label ($IRLbl_c$):
   \[
   IRLbl_c = \frac{\max_{c'} n_{c'}}{n_c}
   \]
   where $n_c$ is the positive count for class $c$.
2. Compute the Mean Imbalance Ratio ($MeanIR$):
   \[
   MeanIR = \frac{1}{|C|} \sum_{c \in C} IRLbl_c
   \]
3. Identify majority labels ($IRLbl_c < MeanIR$).
4. For training samples containing only majority labels, iteratively undersample by fraction `--undersample-frac` (default: 0.1) until the imbalance ratio reaches equilibrium or the target retention threshold is reached, protecting all samples possessing at least one minority label.

### 3.4 Loss and Sampler Formulations

To train deep models on the curated dataset, the pipeline exports precomputed class loss weights and sample weights:

#### A. Positive Class Loss Weights (`pos_weight.npy`)
For binary cross-entropy with logits (`torch.nn.BCEWithLogitsLoss`), the weight for positive instances of class $c$ in a training split of $N$ samples is:
\[
\text{pos\_weight}_c = \min\left( \frac{N - n_c}{n_c}, \text{max\_pos\_weight} \right)
\]
This balances the gradient contribution between rare positive annotations and abundant negative background.

#### B. Class-Balanced Effective Number of Samples (`term_stats.tsv`)
Following *Cui et al. (CVPR 2019)*, the effective volume of samples accounting for data overlap is:
\[
E_{n_c} = \frac{1 - \beta^{n_c}}{1 - \beta}, \quad \text{with } \beta = 0.999
\]
The normalized class weight is:
\[
CB_c = \frac{1}{E_{n_c}} \Bigg/ \left( \frac{1}{|C|} \sum_{c'=1}^{|C|} \frac{1}{E_{n_{c'}}} \right)
\]

#### C. Per-Protein Sample Weights (`sample_weight` column)
For PyTorch's `WeightedRandomSampler`, each protein $i$ with active labels $L_i \subseteq C$ receives a sample weight:
\[
w_i = \frac{1}{|L_i|} \sum_{c \in L_i} \text{pos\_weight}_c
\]

### 3.5 Multi-Label Imbalance Metrics

The pipeline outputs quantitative multi-label metrics in `filter_report.json`:
- **$MeanIR$**: Average imbalance ratio across all classes.
- **$MaxIR$**: Worst-case class imbalance ratio.
- **$CVIR$ (Coefficient of Variation of IR)**: Degree of variance among class frequencies:
  \[
  CVIR = \frac{1}{MeanIR} \sqrt{\frac{1}{|C| - 1} \sum_{c=1}^{|C|} (IRLbl_c - MeanIR)^2}
  \]
- **$SCUMBLE$ (Score of ConcUrrence of rajority and Minority LaBEls)**: Measures the co-occurrence of rare and frequent labels in identical proteins (*Charte et al., 2014*). A lower SCUMBLE score indicates cleaner label separability.

---

## 4. 3D Coordinate Retrieval and Structure Auditing

### 4.1 Experimental PDB vs. AlphaFold DB
DPFunc requires 3D atomic coordinates (`.pdb` or `.cif`) to compute spatial distance matrices and $k$-nearest-neighbor residue contact graphs. The pipeline audits both:
- **Experimental Structures**: High-resolution X-ray or cryo-EM coordinates from RCSB PDB (`https://files.rcsb.org/download/{pdb_id}.pdb` or `.cif`).
- **Predicted Models**: AlphaFold DB tertiary structure predictions (`https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb`).

Under the default **PDB-first strategy** (`--structure-source-pref pdb`), the pipeline downloads genuine experimental coordinates from RCSB PDB for all 336 proteins where experimental structures exist (334 as native `.pdb`, 2 as `.cif`), and falls back to AlphaFold DB models for the remaining 299 proteins. Furthermore, the pipeline automatically aligns 3D structure chains against UniProt sequences to resolve the exact matching polypeptide chain in the manifest.

### 4.2 AlphaFold DB v6 API Resolution
AlphaFold DB migrated from model release `v4` to `v6`. Static URLs querying `_v4.pdb` fail with HTTP 404 errors. 

`dpfunc_filter.py` resolves active models dynamically:
1. Queries `https://alphafold.ebi.ac.uk/api/prediction/{acc}` for each surviving protein.
2. Extracts the official `pdbUrl` (e.g. `AF-{acc}-F1-model_v6.pdb`) and records the global mean `pLDDT` confidence score (`globalMetricValue`).
3. Saves verified URLs into `filtered_manifest.tsv` and `structures_to_fetch.tsv`.

### 4.3 Resilient HTTP Download Protocol
When `--download-structures` is specified, coordinate files are downloaded into `<outdir>/structures/`:
- **Browser-Compliant Headers**: Includes `User-Agent`, `Referer`, and `sec-ch-ua` headers to avoid CDN throttling.
- **Exponential Backoff**: Intercepts HTTP 429 ("Too Many Requests") and retries with progressive backoff delays.
- **Threadpool Throttling**: Restricts concurrent download threads (`dl_workers = min(workers, 4)`) to respect EBI host limits.
- **Atomic Writes**: Writes downloads to `.part` temporary files before atomic rename, preventing partial or corrupted coordinate files.

---

## 5. Empirical Curation Funnel (Molecular Function Benchmark)

Below is the execution trace of the pipeline running on the full **34,164-protein** DPFunc Molecular Function benchmark (`mf_manifest_from_data_dpfunc.tsv`).

### 5.1 Why Re-splitting is Critical
When preserving the original DPFunc splits, the biological filters drop ~98% of candidate proteins, leaving:
- Train: 605
- Valid: **4**
- Test: **14**

Because testing on 14 proteins cannot produce statistically meaningful evaluations, passing `--resplit 0.8 0.1 0.1` filters the whole dataset first, then re-partitions the 694 surviving clean monomers into a balanced 80/10/10 split before deriving vocabulary and class weights.

### 5.2 Attrition Funnel Table (with `--resplit 0.8 0.1 0.1`)
| Stage / Filter | Train | Valid | Test | Total Retained | Drop Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Manifest** | 33,046 | 419 | 699 | **34,164** | — |
| **UniProt ID Mapping** | 32,751 | 410 | 687 | **33,848** | 316 (unmapped / obsolete) |
| **Sequence Length ($30 \le \text{len} < 600$)** | 23,084 | 289 | 490 | **23,863** | 9,971 ($\ge 600$) + 14 ($< 30$) |
| **Non-Membrane Filter** | 15,320 | 203 | 292 | **15,815** | 8,048 (membrane / TM / lipid) |
| **Monomer (Strict Verification)** | 676 | 4 | 14 | **694** | 15,121 (oligomers / complexes) |
| **Structure Availability** | 676 | 4 | 14 | **694** | 0 (all 694 have 3D models) |
| **Re-split (80 / 10 / 10)** | **555** | **69** | **70** | **694** | — |
| **In-Vocabulary GO Labels ($\ge 10$)** | 552 | 69 | 69 | **690** | 4 (zero remaining GO terms) |
| **After ML-RUS Undersampling** | **497** | **69** | **69** | **635** | 55 (majority-class train samples) |

### 5.3 Structure Availability of Retained Proteins
| Split | AlphaFold Model Only | PDB + AlphaFold | Total Retained | Successfully Downloaded |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 236 | 261 | 497 | 497 (100%) |
| **Valid** | 32 | 37 | 69 | 69 (100%) |
| **Test** | 31 | 38 | 69 | 69 (100%) |
| **Total** | **299** | **336** | **635** | **635 / 635 (100%)** |

*(Note: Across the 336 proteins with experimental structures, 3,643 unique experimental PDB entry IDs were mapped.)*

### 5.4 Imbalance Mitigation Impact
| Metric | Before Vocab Filter (892 terms) | After Vocab Filter (106 terms) | After ML-RUS Undersampling |
| :--- | :---: | :---: | :---: |
| **MeanIR** | 265.51 | 21.34 | **19.99** |
| **MaxIR** | 420.0 | 42.0 | **38.20** |
| **CVIR** | 0.608 | 0.524 | **0.494** |
| **SCUMBLE** | 0.621 | 0.355 | **0.356** |
| **Label Cardinality** | 9.85 | 6.77 | **7.01** |

---

## 6. Generated Artifacts and File Schemas

Execution with `--outdir filtered_mf` generates the following production files:

```text
filtered_mf/
├── filtered_manifest.tsv       # Main curated manifest ready for DPFunc (635 rows)
├── pos_weight.npy              # 1D float32 NumPy array of BCE loss weights (length 106)
├── term_stats.tsv              # Per-term positive counts, frequencies, and weights (106 GO terms)
├── structures_to_fetch.tsv     # Direct URLs for PDB and AlphaFold downloads (1,412 entries)
├── report.json                 # Machine-readable attrition counts, drop reasons, metrics
├── report.txt                  # Human-readable filtering and class-imbalance summary
└── structures/                 # Downloaded 3D coordinates (633 .pdb + 2 .cif files)
```

### Manifest Schema (`filtered_manifest.tsv`)
| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `protein_id` | str | `3HAO_HUMAN` | UniProt entry name or DPFunc identifier |
| `split` | str | `train` | Dataset partition (`train`, `valid`, `test`) |
| `pdb_file` | str | `filtered_mf/structures/3HAO_HUMAN.pdb` | Local path to downloaded structure |
| `chain_id` | str | `A` | Target polypeptide chain (auto-resolved to matching protein chain) |
| `go_terms` | str | `GO:0016491;GO:0003824` | Semicolon-delimited curated GO annotations |
| `interpro_terms` | str | `IPR010329;IPR011051` | Semicolon-delimited InterPro domain IDs |
| `accession` | str | `P46952` | Primary UniProt accession |
| `length` | int | `286` | Residue sequence length |
| `structure_source` | str | `PDB+AlphaFold` | Coordinate provenance (`AlphaFold` or `PDB+AlphaFold`) |
| `pdb_ids` | str | `2QNK;5TK5;5TKQ` | Experimental RCSB PDB entry IDs |
| `alphafold_id` | str | `P46952` | AlphaFold DB entry accession |
| `af_plddt` | float | `95.81` | Mean AlphaFold model confidence score |
| `sample_weight` | float | `0.94277` | PyTorch `WeightedRandomSampler` sampling weight |

---

## 7. Downstream PyTorch & DPFunc Integration

### Loading Class Loss Weights in PyTorch
```python
import numpy as np
import torch
import torch.nn as nn

# Load precomputed positive weights matching the 106-term vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = torch.from_numpy(np.load("data/curated_mf/pos_weight.npy")).to(device)

# Supply pos_weight to BCE loss to handle rare-class imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Loading Sample Weights for Data Loading
```python
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

manifest = pd.read_csv("filtered_mf/filtered_manifest.tsv", sep="\t")
train_df = manifest[manifest["split"] == "train"]

weights = torch.tensor(train_df["sample_weight"].values, dtype=torch.float32)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Running DPFunc Feature Construction
```bash
python DPFunc_demo_pipeline/scripts/build_data_demo.py \
  --manifest filtered_mf/filtered_manifest.tsv \
  --workspace workspace_mf_monomer \
  --ontology mf \
  --pdb-dir filtered_mf/structures \
  --inter-idx ./data/inter_idx.pkl \
  --go-obo ./data/go.obo
```
