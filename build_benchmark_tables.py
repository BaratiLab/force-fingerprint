#!/usr/bin/env python3
"""
Build the three benchmark tables (molecular function only) used to compare a sequence model,
a sequence+structure model and DPFunc on the same proteins.

  full               every DPFunc benchmark protein that UniProt still knows, DPFunc's own split
  reduced_original   the filtered monomer set (dpfunc_filter.py without --resplit), DPFunc's own split
                     -> only 4 validation and 14 test proteins; expect very noisy scores
  reduced_resplit    the filtered monomer set after --resplit 0.8 0.1 0.1 and ML-RUS (filtered_mf/)

For every setting two files are written to --outdir:
  <name>.csv                 input for the sequence / structure models (see the plm-1d-prediction branch)
  <name>_dpfunc_manifest.tsv the same proteins in DPFunc's manifest format, for DPFunc_demo_pipeline/scripts/build_data_demo.py

Both files hold exactly the same rows, so every method sees the same proteins and the same split.

Inputs:
  --manifest          the DPFunc manifest (mf_manifest_from_data_dpfunc.tsv)
  --cache-dir         dpfunc_filter cache; its uniprot.tsv supplies the sequences
  --filtered-original output folder of   dpfunc_filter.py ... (no --resplit, no --undersample-frac)
  --filtered-resplit  output folder of the curated, re-split run (filtered_mf)
  --structure-dir     optional folder(s) with structure files, e.g. DPFunc's DataProcess/AF2DB (<protein_id>.pdb);
                      the structures/ folder next to each filtered manifest is searched too. Proteins without a
                      file get an empty structure_file and are looked up by uniprot_id when the model runs

Usage:
  python build_benchmark_tables.py \
      --manifest mf_manifest_from_data_dpfunc.tsv --cache-dir dpfunc_filter_cache \
      --filtered-original filtered_mf_original --filtered-resplit filtered_mf \
      --structure-dir /path/to/DPFunc/DataProcess/AF2DB --outdir benchmark_tables
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("build_benchmark_tables")

GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}


def load_sequences(cache_dir):
    uni = pd.read_csv(cache_dir / "uniprot.tsv", sep="\t", dtype=str).fillna("")
    uni = uni[uni.sequence != ""].drop_duplicates("query_id")
    return uni.set_index("query_id")[["accession", "sequence"]]


def drop_roots(terms):
    return ";".join(t for t in str(terms).split(";") if t and t not in GO_ROOTS)


def structure_paths(df, dirs):
    """First existing structure file per protein, searching the given folders for <name of pdb_file> and <protein_id>.pdb."""
    out = []
    for pid, given in zip(df.protein_id, df.get("pdb_file", [""] * len(df))):
        names = [n for n in (Path(str(given)).name, f"{pid}.pdb") if n and n != "."]
        hits = [d / n for d in dirs for n in names if (d / n).exists()]
        out.append(str(hits[0].resolve()) if hits else "")
    return out


def finish(df, seqs, max_len, dirs):
    df = df.join(seqs[["sequence"]], on="protein_id", how="inner")
    df["go_terms"] = df.go_terms.map(drop_roots)
    n0 = len(df)
    df = df[(df.go_terms != "") & (df.sequence.str.len() <= max_len)].copy()
    log.info("  %d -> %d proteins (labels left after root removal, length <= %d)", n0, len(df), max_len)
    df["structure_file"] = structure_paths(df, dirs)
    return df.reset_index(drop=True)


def write(df, name, outdir):
    csv = pd.DataFrame({
        "protein_id": df.protein_id, "sequence": df.sequence, "uniprot_id": df.accession,
        "structure_file": df.structure_file, "chain_id": df.chain_id.where(df.chain_id != "", "A"),
        "split": df.split, "interpro": df.interpro_terms, "go_terms": df.go_terms,
    })
    csv.to_csv(outdir / f"{name}.csv", index=False)
    manifest = pd.DataFrame({
        "protein_id": df.protein_id, "split": df.split,
        "pdb_file": df.structure_file.where(df.structure_file != "", df.get("pdb_file", "")),
        "chain_id": csv.chain_id, "go_terms": df.go_terms, "interpro_terms": df.interpro_terms,
    })
    manifest.to_csv(outdir / f"{name}_dpfunc_manifest.tsv", sep="\t", index=False)
    counts = df.split.value_counts().reindex(["train", "valid", "test"]).fillna(0).astype(int)
    log.info("%-17s train %d / valid %d / test %d   with a local structure file: %d", name, *counts, (df.structure_file != "").sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("dpfunc_filter_cache"))
    ap.add_argument("--filtered-original", type=Path, required=True)
    ap.add_argument("--filtered-resplit", type=Path, default=Path("filtered_mf"))
    ap.add_argument("--structure-dir", type=Path, action="append", default=[],
                    help="folder with structure files (<protein_id>.pdb); repeat for several folders")
    ap.add_argument("--max-len", type=int, default=1000, help="longest sequence kept in every setting (ESM-2 reads at most 1022 residues)")
    ap.add_argument("--outdir", type=Path, default=Path("benchmark_tables"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)
    seqs = load_sequences(args.cache_dir)

    log.info("full")
    full = pd.read_csv(args.manifest, sep="\t", dtype=str).fillna("")
    full["accession"] = full.protein_id.map(seqs.accession).fillna("")
    write(finish(full, seqs, args.max_len, args.structure_dir), "full", args.outdir)

    for name, folder in (("reduced_original", args.filtered_original), ("reduced_resplit", args.filtered_resplit)):
        log.info(name)
        kept = pd.read_csv(folder / "filtered_manifest.tsv", sep="\t", dtype=str).fillna("")
        write(finish(kept, seqs, args.max_len, [folder / "structures", *args.structure_dir]), name, args.outdir)


if __name__ == "__main__":
    main()
