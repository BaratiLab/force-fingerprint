# DPFunc Dataset Curation & Filtering Pipeline: Technical Specification

This document provides a comprehensive technical, biological, and mathematical description of the curation pipeline implemented in `dpfunc_filter.py`. The pipeline curates protein datasets from the [DPFunc benchmark](https://github.com/CSUBioGroup/DPFunc) for structure-based protein function prediction under rigorous biological and machine-learning constraints.

---

## Table of Contents
1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Biological and Structural Filtering Criteria](#2-biological-and-structural-filtering-criteria)
   - [2.1 UniProt Mapping and Sequence Length](#21-uniprot-mapping-and-sequence-length)
   - [2.2 Membrane Protein Exclusion](#22-membrane-protein-exclusion)
   - [2.3 Monomer Selection and Validation](#23-monomer-selection-and-validation)
   - [2.4 Intrinsic Disorder Filtering](#24-intrinsic-disorder-filtering)
   - [2.5 Sequence Redundancy Filtering](#25-sequence-redundancy-filtering)
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
5. [Empirical Curation Funnel (v0.1.0 Release)](#5-empirical-curation-funnel-v010-release)
6. [Generated Artifacts and File Schemas](#6-generated-artifacts-and-file-schemas)
7. [Downstream PyTorch & DPFunc Integration](#7-downstream-pytorch--dpfunc-integration)

---

## 1. Introduction and Motivation

Structure-based deep learning models such as **DPFunc** predict Gene Ontology (GO) terms by constructing residue-contact graphs and combining them with sequence language embeddings (e.g., ESM-2) and InterPro domain features.

Standard benchmark datasets (such as the 34,164-protein Molecular Function benchmark) often contain confounding biological factors that bias graph representations:
- **Oligomers and Multi-Protein Complexes**: Tertiary contact graphs of isolated monomers from multi-chain complexes miss interfacial binding surfaces and quaternary contacts, leading to misleading contact maps.
- **Membrane-Bound Proteins**: Transmembrane helices and lipid-associated domains undergo unique hydrophobic packing within lipid bilayers that differ fundamentally from soluble globular proteins, skewing spatial graph convolutions.
- **Long Intrinsically Disordered Regions**: Disordered segments have no stable conformation, so their coordinates (low-confidence regions of AlphaFold models, or regions missing from experimental structures) produce contact graphs that do not reflect a real fold.
- **Sequence Redundancy**: Close homologs (often the same protein from several species) inflate the dataset with near-duplicate examples and, when they fall in different splits, leak information from train to test.
- **Extreme Sequence Lengths**: Very long macromolecules (> 600 residues by default) introduce quadratic memory scaling in graph attention mechanisms and GCNs, while short peptides (< 30 residues) lack stable tertiary folds.
- **Severe Class Imbalance**: Multi-label GO distributions have extreme power-law (long-tail) characteristics, where a handful of generic terms dominate thousands of rare specific terms, leading unweighted networks to collapse into majority-class predictions.

The `dpfunc_filter.py` pipeline addresses each of these systematically.

---

## 2. Biological and Structural Filtering Criteria

Every candidate protein is evaluated against curated databases (UniProtKB, RCSB PDB, Complex Portal cross-references) and local predictors (metapredict, MMseqs2). Remote lookups and local predictions are cached in `--cache-dir` (default: `dpfunc_filter_cache/`). Filters are applied in the order below, and every dropped protein is recorded with a reason in `annotations.tsv`.

```
Candidate Protein
  │
  ├── 1. UniProt Mapping       [current UniProtKB entry; optional --reviewed-only]
  │
  ├── 2. Sequence Length       [30 <= len < 1200]
  │
  ├── 3. Membrane Exclusion    [Features + Keywords + Subcellular text]
  │
  ├── 4. Monomer Verification  [strict / lenient / supported]
  │
  ├── 5. Disorder              [metapredict: no internal disordered run > 14 aa]
  │
  ├── 6. Sequence Redundancy   [MMseqs2 clustering at 40% identity + verification pass]
  │
  ├── 7. Structure Availability [PDB and/or AlphaFold DB] → optional re-split
  │
  └── 8. Label Filtering       [GO root pruning + train vocab thresholding + ML-RUS]
```

### 2.1 UniProt Mapping and Sequence Length

DPFunc identifiers (UniProt entry names, e.g. `1433B_HUMAN`) are mapped to current UniProtKB entries through the UniProt ID-mapping service. Obsolete or unmappable identifiers are dropped. With `--reviewed-only`, only Swiss-Prot entries are kept.

- **Minimum Length**: `--min-len 30` (drops peptide fragments lacking stable folded domains).
- **Maximum Length**: `--max-len 1200` (requires $\text{length} < 1200$ residues to limit graph memory scaling).

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

### 2.3 Monomer Selection and Validation

The monomer step combines UniProt's free-text `SUBUNIT` annotation (`cc_subunit`), Complex Portal cross-references (`xref_complexportal`), and, with `--pdb-oligomer-check`, the biological assemblies of the protein's PDB entries. Three modes are available through `--monomer-mode`. All three are evaluated on every run and recorded in `annotations.tsv` (`monomer_keep_strict`, `monomer_keep_lenient`, `monomer_keep_supported`); only the selected mode drops proteins.

#### A. PDB Assembly Evidence (`--pdb-oligomer-check`)
For each protein, the first `--max-pdb-per-protein` (default 5) PDB entries are taken in the order UniProt lists them. For each entry, the RCSB PDB GraphQL API supplies the oligomeric state of the global symmetry (`rcsb_struct_symmetry`) of **assembly 1**, and that entry casts one vote:
- `Monomer` → monomer vote.
- `Homo n-mer` → homo-oligomer vote.
- `Hetero ...` → hetero vote. For `supported` mode this is further split: **hetero-complex** if the entry contains another polypeptide entity of at least `--min-partner-len` residues (default 50) mapped to a *different* UniProt accession (a real protein partner), otherwise **hetero-other**. Antibody fragments, nanobodies, designed binders and peptides usually lack a UniProt mapping or are short, so they fall into hetero-other.

Entries without assembly data do not vote. The verdict is *monomer* if monomer votes are at least as many as homo-oligomer votes (and, in `supported` mode with `--pdb-hetero-as-complex`, as hetero-complex votes) and at least one monomer vote exists; *homo-oligomer* if homo votes are more numerous; *hetero-complex* if hetero-complex votes dominate and `--pdb-hetero-as-complex` is set; otherwise *hetero* (neutral) or *none*. Hetero votes are neutral by default because the PDB over-represents crystallized transient complexes.

#### B. `strict` and `lenient` Modes
The subunit text (evidence tags and `SUBUNIT:` prefixes removed, lowercased) is classified with three patterns:
- **Monomer**: the words `monomer`, `monomeric`, `monomers`.
- **Negation**: phrases such as `not (a) monomer`, `never monomeric`, `rarely a monomer`, `fails to form a` / `unable to form a` / `does not form a monomer`, `monomer is inactive`, `monomeric unstable`. A negation anywhere cancels all positive monomer mentions.
- **Oligomer/complex**: `dimer` … `dodecamer`, `oligomer`, `multimer` (with optional homo-/hetero- prefixes and -ic/-s/-izes/-ization suffixes), `homo-n-mer`, `heteromer`, `component of`, `subunit of`, `part of the … complex`, `forms … complex`, `assembles into`, `filament`.

This yields the classes *monomer* (monomer only), *mixed* (both), *oligomer*, *interacts* (text present but neither pattern matches, typically "Interacts with …"), and *unknown* (no annotation).
- `strict` keeps *monomer* proteins, unless `--pdb-oligomer-check` finds mostly homo-oligomeric assemblies.
- `lenient` also keeps *mixed* proteins, and does not apply the PDB homo-oligomer veto.
- With `--allow-unknown-subunit`, *unknown* and *interacts* proteins are also kept (when the PDB check is on, only if PDB assemblies are monomeric).

Because the patterns scan the whole text, mentions of partners ("Interacts with the catalytic subunit of PKA", "Binds to the Ku heterodimer") can classify a monomeric protein as oligomeric or mixed. Together with sparse annotation coverage, this is why `strict` mode removes the large majority of proteins.

#### C. `supported` Mode
`supported` mode only counts statements whose subject is the protein itself. The subunit text is split into clauses at `.` and `;`, and each clause is tested **at its start**:
- **self_oligomer**: a clause opens with an oligomer or complex-membership statement, optionally preceded by routine curation qualifiers (`probable`, `predominantly`, `mainly`, …) and verbs (`is`, `exists as`, `forms`, `the active form is`, …): e.g. "Homodimer", "Probable homotetramer", "Forms homodimers", "Component of the 26S proteasome". Hedged statements ("May form homodimers", "Can dimerize") do not count.
- **self_monomer**: a clause opens with a monomer statement ("Monomer", "Exists as a monomer").
- **mixed**: both, or a monomer clause that also names an oligomer ("Monomer and homodimer", "Monomer; homodimer upon ligand binding").
- **partner_only**: text present but no statement about the protein's own state.
- **unknown**: no annotation.

A protein is **dropped** if any of the following holds (complex evidence):
1. its subunit class is *self_oligomer*;
2. it has a Complex Portal cross-reference (a curated stable complex);
3. its PDB verdict is *homo-oligomer*;
4. its PDB verdict is *hetero-complex* (only with `--pdb-hetero-as-complex`).

Otherwise it is **kept** only with monomer support, recorded as `monomer_tier`:

| Tier | Evidence |
| :---: | :--- |
| 1 | UniProt *self_monomer* and PDB verdict *monomer* |
| 2 | UniProt *self_monomer*, PDB verdict neutral or absent |
| 3 | PDB verdict *monomer*, with UniProt *partner_only*, *unknown*, or *mixed* |

Proteins with neither (including *mixed* without PDB monomer support) are dropped. Tiers 1 and 3 require `--pdb-oligomer-check`. Tier 3 is the weakest evidence: a monomeric PDB assembly can come from a crystallized domain or truncated construct. The tier is computed in every mode and exported in `filtered_manifest.tsv`, so stronger evidence can be preferred when selecting a final set.

### 2.4 Intrinsic Disorder Filtering

Per-residue disorder scores $s_1, \dots, s_L \in [0, 1]$ are predicted with [metapredict](https://github.com/idptools/metapredict) (non-standard residues `U, O, B, Z, J, X` are mapped to close standard analogues first). A residue is disordered if $s_i > \tau$, with $\tau$ = `--disorder-threshold` (default 0.5). Maximal runs of consecutive disordered residues are classified as:
- **N-terminal**: the run starts at residue 1;
- **C-terminal**: the run ends at residue $L$;
- **internal**: all other runs. A run covering the entire sequence counts as internal, so fully disordered proteins are not excused as terminal.

A protein is dropped if its longest internal run exceeds `--max-disordered` (default 14, i.e. internal runs must be shorter than 15 residues):
\[
\max_{r \in \text{internal runs}} |r| > 14
\]
Terminal tails are allowed by default; `--max-terminal-disordered N` additionally drops proteins whose longest terminal tail exceeds $N$ residues. `--max-disordered -1` disables the filter. Scores are cached by sequence hash in a file named for the installed metapredict version, because different versions give different scores.

This step runs **before** redundancy filtering so that cluster representatives are chosen only among proteins that pass it.

### 2.5 Sequence Redundancy Filtering

Surviving sequences are clustered with [MMseqs2](https://github.com/soedinglab/MMseqs2) `easy-cluster` using `--min-seq-id` (default 0.4), `--cluster-coverage` (`-c`, default 0.8), and `--cluster-cov-mode` (default 0, coverage required of both sequences). Coverage matters: without it, a short domain matching part of a long protein can merge unrelated proteins, but a strict bidirectional requirement can also keep orthologs of different lengths apart.

**Representative selection.** One protein is kept per cluster, chosen by this priority:
1. species, in the order of `PREFERRED_SPECIES` (human, mouse, yeast, *E. coli*; read from the UniProt entry-name suffix);
2. reviewed (Swiss-Prot) entries;
3. proteins with at least one PDB entry;
4. longer sequences;
5. `protein_id`, for determinism.

**Verification pass.** MMseqs2's greedy clustering only guarantees that each member resembles its own cluster's MMseqs2 representative. Because the pipeline picks its own representatives, two kept proteins from different clusters can still exceed the thresholds. The kept set is therefore searched against itself (`easy-search`, same identity and coverage thresholds, sensitivity `-s 7.5` by default via `--verify-sensitivity`). A pair counts as redundant if either search direction passes. Proteins are then visited in the priority order above, and a protein is kept only if it has no redundant pair with an already-kept protein. This single greedy pass guarantees that no two surviving proteins share a passing hit. It can be disabled with `--skip-identity-verify`.

**Optional entry-name prefix deduplication.** With `--dedup-name-prefix`, proteins sharing a UniProt entry-name prefix (e.g. `CATA_HUMAN`, `CATA_MOUSE`) are added as extra redundant pairs in the same greedy pass, so at most one protein per prefix survives. Matching is exact (paralogs such as `1433B` and `1433E` are not merged), orthologs with unrelated mnemonics are not caught, and TrEMBL entries (accession-based names) are unaffected.

Dropped proteins record the kept protein they conflicted with in `identity_rep`, with distinct drop reasons for clustering, the verification pass, and prefix deduplication. Because this step runs before the optional re-split, no two proteins in different splits exceed the identity threshold.

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

The pipeline outputs quantitative multi-label metrics in `report.json`:
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

Under the default **PDB-first strategy** (`--structure-source-pref pdb`), the pipeline downloads experimental coordinates from RCSB PDB for proteins where experimental structures exist, and falls back to AlphaFold DB models for the rest (in the v0.1.0 release: 336 experimental structures, 334 as native `.pdb` and 2 as `.cif`, and 299 AlphaFold models). The pipeline also aligns structure chains against UniProt sequences to resolve the exact matching polypeptide chain in the manifest.

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

## 5. Empirical Curation Funnel (v0.1.0 Release)

> The figures in this section describe the **v0.1.0** dataset (tracked in `data/curated_mf/`), produced by an earlier version of the pipeline: strict monomer filtering, $30 \le \text{length} < 600$, and no disorder or sequence-redundancy filtering. Runs with the current pipeline and defaults will produce different numbers.

Below is the execution trace of that run on the full **34,164-protein** DPFunc Molecular Function benchmark (`mf_manifest_from_data_dpfunc.tsv`).

### 5.1 Why Re-splitting is Critical
When preserving the original DPFunc splits, the biological filters dropped ~98% of candidate proteins, leaving:
- Train: 605
- Valid: **4**
- Test: **14**

Because testing on 14 proteins cannot produce statistically meaningful evaluations, passing `--resplit 0.8 0.1 0.1` filters the whole dataset first, then re-partitions the surviving proteins into a balanced 80/10/10 split before deriving vocabulary and class weights.

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

Execution with `--outdir filtered_mf` generates the following files (counts in comments refer to the v0.1.0 run):

```text
filtered_mf/
├── filtered_manifest.tsv       # Main curated manifest ready for DPFunc (635 rows)
├── annotations.tsv             # Every input protein with its drop reason and per-filter evidence
├── pos_weight.npy              # 1D float32 NumPy array of BCE loss weights (length 106)
├── term_stats.tsv              # Per-term positive counts, frequencies, and weights (106 GO terms)
├── structures_to_fetch.tsv     # Direct URLs for PDB and AlphaFold downloads (1,412 entries)
├── report.json                 # Machine-readable attrition counts, drop reasons, metrics
├── report.txt                  # Human-readable filtering and class-imbalance summary
└── structures/                 # Downloaded 3D coordinates (633 .pdb + 2 .cif files)
```

`report.json` also includes the distribution of subunit classes after the membrane filter (`subunit_class_after_membrane_filter`, `subunit_self_class_after_membrane_filter`) and `monomer_mode_comparison`: how many proteins each monomer mode would keep, plus per-tier counts for `supported`.

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
| `monomer_tier` | int | `1` | `supported`-mode monomer tier (1 = strongest; see §2.3C), computed in every mode; blank if the protein does not meet `supported` criteria |
| `structure_source` | str | `PDB+AlphaFold` | Coordinate provenance (`AlphaFold` or `PDB+AlphaFold`) |
| `pdb_ids` | str | `2QNK;5TK5;5TKQ` | Experimental RCSB PDB entry IDs |
| `alphafold_id` | str | `P46952` | AlphaFold DB entry accession |
| `af_plddt` | float | `95.81` | Mean AlphaFold model confidence score |
| `sample_weight` | float | `0.94277` | PyTorch `WeightedRandomSampler` sampling weight |

### Per-Filter Evidence (`annotations.tsv`)
In addition to identifiers, split, length, and `drop_reason`, `annotations.tsv` records the evidence behind each filtering decision:

| Columns | Filter | Content |
| :--- | :--- | :--- |
| `membrane_evidence` | Membrane | Which feature, keyword, or text triggered exclusion |
| `subunit_class`, `subunit_text` | Monomer (strict/lenient) | Whole-text class and cleaned subunit text |
| `subunit_self_class`, `subunit_self_clause` | Monomer (supported) | Clause-level class and the clause that triggered it |
| `complex_portal` | Monomer (supported) | Complex Portal IDs, if any |
| `pdb_oligomer`, `pdb_oligomer_votes` | Monomer (strict/lenient) | PDB verdict and vote counts |
| `pdb_complex`, `pdb_complex_votes` | Monomer (supported) | PDB verdict with hetero-complex / hetero-other split |
| `monomer_keep_strict`, `monomer_keep_lenient`, `monomer_keep_supported` | Monomer | Keep/drop decision under each mode |
| `monomer_tier`, `monomer_evidence` | Monomer | Tier (supported mode) and the reason for the selected mode's decision |
| `disorder_internal_max`, `disorder_nterm`, `disorder_cterm` | Disorder | Longest internal disordered run and terminal tail lengths |
| `identity_cluster`, `identity_rep`, `name_prefix` | Redundancy | MMseqs2 cluster, the kept protein a dropped protein conflicted with, entry-name prefix |
| `structure_source`, `pdb_ids`, `alphafold_id`, `af_plddt`, `local_structure` | Structure | Structure availability and AlphaFold confidence |

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
