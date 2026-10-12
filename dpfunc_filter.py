#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas>=1.3",
#     "requests>=2.25",
#     "numpy>=1.20",
# ]
# ///
"""
Filter the DPFunc benchmark dataset and prepare it for training.

Filters (applied in this order, every drop is logged with a reason):
  1. Mapping       - DPFunc IDs (UniProt entry names, e.g. 1433B_HUMAN) are mapped to
                     current UniProtKB entries. Unmappable / obsolete IDs are dropped.
  2. Length        - keep sequences with length < --max-len (default 600).
  3. Membrane      - drop anything UniProt marks as membrane-associated: transmembrane /
                     intramembrane features, Membrane / Transmembrane / GPI-anchor
                     keywords, or "membrane" in the subcellular-location comment.
  4. Monomer       - keep proteins whose UniProt "Subunit structure" says monomer, and
                     optionally cross-check the biological assembly in the PDB (RCSB).
  5. Structure     - record whether an experimental (PDB) and/or AlphaFold DB structure
                     exists, optionally verify the AF model and its mean pLDDT, and
                     optionally require a structure.
  6. Imbalance     - GO-term vocabulary from train only (min count), root terms removed,
                     labels restricted to vocab, optional ML-RUS undersampling of train,
                     per-term pos_weight + per-protein sample weights, imbalance report.

Input: a DPFunc manifest TSV (columns protein_id, split, go_terms[, pdb_file, chain_id,
interpro_terms]) - e.g. DPFunc_demo_pipeline/templates/mf_manifest_from_data_dpfunc.tsv -
or any TSV with protein_id / split / go_terms. Run once per ontology (mf / bp / cc).

Output (in --outdir):
  annotations.tsv          every input protein + UniProt fields, filter flags, drop reason
  filtered_manifest.tsv    DPFunc-format manifest (works with build_data_demo.py) plus
                           structure columns and sample_weight
  term_stats.tsv           GO-term vocab: train counts, frequency, IRLbl, pos_weight
  pos_weight.npy           pos_weight vector in term_stats.tsv order (if numpy available)
  structures_to_fetch.tsv  download URLs for the preferred structure of each kept protein
  report.json / report.txt per-step counts by split and imbalance metrics

Usage:
  python dpfunc_filter.py \
      --manifest DPFunc_demo_pipeline/templates/mf_manifest_from_data_dpfunc.tsv \
      --ontology mf --outdir filtered_mf \
      --pdb-oligomer-check --check-alphafold --min-term-count 10 --undersample-frac 0.1

Requires: Python 3.8+, requests, pandas (numpy optional). Internet access to
rest.uniprot.org (always), data.rcsb.org (--pdb-oligomer-check) and alphafold.ebi.ac.uk
(--check-alphafold). All remote lookups are cached in --cache-dir, so reruns with
different thresholds are offline and fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    import requests
except ImportError:  # only needed when something is not cached yet
    requests = None

try:
    import numpy as np
except ImportError:
    np = None

log = logging.getLogger("dpfunc_filter")

SPLITS = ["train", "valid", "test"]
GO_ROOTS = {"mf": "GO:0003674", "bp": "GO:0008150", "cc": "GO:0005575"}

UNIPROT_API = "https://rest.uniprot.org"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

# Requested in this exact order; TSV headers are renamed positionally to these names.
UNIPROT_FIELDS = [
    "accession", "id", "reviewed", "length", "cc_subunit", "cc_subcellular_location",
    "keywordid", "keyword", "ft_transmem", "ft_intramem", "ft_lipid",
    "xref_pdb", "xref_alphafolddb", "sequence",
]

# UniProt keywords that mean "membrane protein" for our purposes.
MEMBRANE_KEYWORDS = {
    "KW-0472": "Membrane",
    "KW-0812": "Transmembrane",
    "KW-1133": "Transmembrane helix",
    "KW-1134": "Transmembrane beta strand",
    "KW-0336": "GPI-anchor",
    "KW-1003": "Cell membrane",
    "KW-0999": "Mitochondrion inner membrane",
    "KW-1000": "Mitochondrion outer membrane",
    "KW-0256": "Endoplasmic reticulum",  # not used by default, see MEMBRANE_KW_DEFAULT
}
MEMBRANE_KW_DEFAULT = {k for k in MEMBRANE_KEYWORDS if k != "KW-0256"}

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def split_terms(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    out, seen = [], set()
    for t in re.split(r"[;,|\s]+", text.strip()):
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://alphafold.ebi.ac.uk/",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def http_session() -> "requests.Session":
    if requests is None:
        sys.exit("The 'requests' package is needed for remote lookups: pip install requests")
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "POST"])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers.update(BROWSER_HEADERS)
    return s


def chunks(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    tmp.replace(path)


def download_file(url: str, dest: Path, session=None, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            if session is not None:
                with session.get(url, stream=True, timeout=60, headers=BROWSER_HEADERS) as r:
                    if r.status_code == 429:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)
            else:
                import urllib.request
                req = urllib.request.Request(url, headers=BROWSER_HEADERS)
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                    f.write(resp.read())
            tmp.replace(dest)
            return True
        except Exception as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            # If 404, the file simply doesn't exist on this URL; don't retry
            if hasattr(e, "response") and getattr(e.response, "status_code", None) == 404:
                return False
            if attempt == retries - 1:
                log.warning("Failed to download %s: %s", url, e)
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


# ---------------------------------------------------------------------------
# 1. UniProt annotation (ID mapping service, cached)
# ---------------------------------------------------------------------------


def uniprot_idmapping(ids: List[str], session) -> pd.DataFrame:
    """Map entry names / accessions -> UniProtKB rows with UNIPROT_FIELDS."""
    r = session.post(f"{UNIPROT_API}/idmapping/run",
                     data={"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ",".join(ids)},
                     timeout=120)
    r.raise_for_status()
    job = r.json()["jobId"]
    while True:
        s = session.get(f"{UNIPROT_API}/idmapping/status/{job}", timeout=60,
                        allow_redirects=False)
        if s.status_code in (301, 302, 303):
            break
        s.raise_for_status()
        body = s.json()
        if body.get("jobStatus") in (None, "FINISHED") or "results" in body:
            break
        if body.get("jobStatus") in ("ERROR", "FAILED"):
            raise RuntimeError(f"UniProt ID mapping job {job} failed: {body}")
        time.sleep(3)
    url = (f"{UNIPROT_API}/idmapping/uniprotkb/results/stream/{job}"
           f"?format=tsv&fields={','.join(UNIPROT_FIELDS)}")
    res = session.get(url, timeout=600)
    res.raise_for_status()
    lines = res.text.splitlines()
    if len(lines) <= 1:
        return pd.DataFrame(columns=["query_id"] + UNIPROT_FIELDS)
    rows = [ln.split("\t") for ln in lines[1:]]
    ncol = 1 + len(UNIPROT_FIELDS)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=["query_id"] + UNIPROT_FIELDS)


def fetch_uniprot(ids: List[str], cache_file: Path, batch: int = 20000) -> pd.DataFrame:
    cached = (pd.read_csv(cache_file, sep="\t", dtype=str).fillna("")
              if cache_file.exists() else pd.DataFrame(columns=["query_id"] + UNIPROT_FIELDS))
    missing_file = cache_file.with_suffix(".unmapped.txt")
    known_unmapped = set(missing_file.read_text().split()) if missing_file.exists() else set()
    have = set(cached["query_id"])
    todo = [i for i in dict.fromkeys(ids) if i not in have and i not in known_unmapped]
    if todo:
        log.info("UniProt: querying %d IDs (%d cached)", len(todo), len(ids) - len(todo))
        session = http_session()
        parts = [cached]
        for i, chunk in enumerate(chunks(todo, batch), 1):
            log.info("  ID-mapping batch %d (%d IDs)", i, len(chunk))
            df = uniprot_idmapping(list(chunk), session)
            parts.append(df)
            got = set(df["query_id"])
            known_unmapped |= {x for x in chunk if x not in got}
            cached = pd.concat(parts, ignore_index=True)
            cached.to_csv(cache_file, sep="\t", index=False)
            missing_file.write_text("\n".join(sorted(known_unmapped)))
    # An entry name can map to >1 row in rare merge cases: prefer reviewed, then longest.
    cached = cached[cached["query_id"].isin(ids)].copy()
    cached["_rev"] = cached["reviewed"].str.lower().eq("reviewed")
    cached = (cached.sort_values(["query_id", "_rev"], ascending=[True, False])
              .drop_duplicates("query_id").drop(columns="_rev"))
    return cached


# ---------------------------------------------------------------------------
# 3. Membrane classification
# ---------------------------------------------------------------------------


def membrane_reasons(row: pd.Series, kw_set: Set[str], use_subcell_text: bool) -> List[str]:
    reasons = []
    if row.get("ft_transmem", "").strip():
        reasons.append("TRANSMEM feature")
    if row.get("ft_intramem", "").strip():
        reasons.append("INTRAMEM feature")
    kws = set(split_terms(row.get("keywordid", "")))
    hit = sorted(kws & kw_set)
    if hit:
        reasons.append("keyword " + ",".join(f"{k}({MEMBRANE_KEYWORDS.get(k, '')})" for k in hit))
    lipid = row.get("ft_lipid", "").lower()
    if "gpi-anchor" in lipid:
        reasons.append("GPI-anchor lipidation")
    if use_subcell_text:
        sub = re.sub(r"\{[^}]*\}", "", row.get("cc_subcellular_location", "")).lower()
        if re.search(r"\bmembrane\b", sub):
            reasons.append("subcellular location mentions membrane")
    return reasons


# ---------------------------------------------------------------------------
# 4. Oligomeric state
# ---------------------------------------------------------------------------

MONOMER_RE = re.compile(r"\bmonomer(?:ic|s)?\b")
NOT_MONOMER_RE = re.compile(
    r"\b(?:not|never|rarely|fails to form a|unable to form a|does not form a)\s+(?:a\s+)?monomer(?:ic)?\b"
    r"|\bmonomer(?:ic)?\s+(?:is\s+)?(?:inactive|unstable)\b"
)
OLIGOMER_RE = re.compile(
    r"\b(?:homo|hetero)?(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|dodeca|oligo|multi)mer(?:ic|s|izes|ization)?\b"
    r"|\b(?:homo|hetero)-?\d+-?mer\b|\bheteromer\w*\b"
    r"|\bcomponent of\b|\bpart of (?:the|a|an)\b[^.]*\bcomplex\b|\bsubunit of\b"
    r"|\bforms? (?:a |an )?(?:\w+ ){0,3}complex\b|\bassembles? into\b|\bfilament\w*\b"
)


def subunit_class(text: str) -> Tuple[str, str]:
    """Return (class, cleaned_text). class in
    monomer      - says monomer, nothing suggesting an oligomer/complex
    mixed        - says monomer AND mentions an oligomer/complex (e.g. 'Monomer and homodimer')
    oligomer     - oligomer / complex, no monomer statement
    interacts    - only 'Interacts with ...' style statements
    unknown      - no subunit annotation
    """
    t = re.sub(r"\{[^}]*\}", " ", text or "")
    t = re.sub(r"^\s*SUBUNIT:\s*", "", t, flags=re.I)
    t = re.sub(r"\s*SUBUNIT:\s*", " ", t, flags=re.I).strip()
    low = t.lower()
    if not low:
        return "unknown", ""
    mono = bool(MONOMER_RE.search(low))
    if mono and NOT_MONOMER_RE.search(low):
        mono = False
    olig = bool(OLIGOMER_RE.search(low))
    if mono and not olig:
        return "monomer", t
    if mono and olig:
        return "mixed", t
    if olig:
        return "oligomer", t
    return "interacts", t


def fetch_pdb_oligomeric_states(pdb_ids: List[str], cache_file: Path) -> Dict[str, List[str]]:
    """PDB id -> list of oligomeric states of its assemblies, e.g. ['Monomer'] or
    ['Homo 2-mer', 'Monomer']. Uses one RCSB GraphQL call per 200 entries."""
    cache = load_json(cache_file)
    todo = sorted({p.upper() for p in pdb_ids} - set(cache))
    if todo:
        log.info("RCSB: querying assemblies for %d PDB entries (%d cached)", len(todo), len(cache))
        session = http_session()
        q = ("query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
             "assemblies{rcsb_assembly_info{assembly_id} "
             "rcsb_struct_symmetry{kind oligomeric_state}}}}")
        for i, chunk in enumerate(chunks(todo, 200), 1):
            r = session.post(RCSB_GRAPHQL, json={"query": q, "variables": {"ids": list(chunk)}},
                             timeout=120)
            r.raise_for_status()
            entries = (r.json().get("data") or {}).get("entries") or []
            got = set()
            for e in entries:
                if not e:
                    continue
                states = []
                for a in sorted(e.get("assemblies") or [],
                                key=lambda a: int((a.get("rcsb_assembly_info") or {}).get("assembly_id") or 99)):
                    sym = [s for s in (a.get("rcsb_struct_symmetry") or []) if s.get("kind") == "Global Symmetry"]
                    if sym:
                        states.append(sym[0].get("oligomeric_state") or "")
                cache[e["rcsb_id"].upper()] = states
                got.add(e["rcsb_id"].upper())
            for p in chunk:
                cache.setdefault(p, [])  # obsolete / not found
            if i % 10 == 0:
                save_json(cache, cache_file)
        save_json(cache, cache_file)
    return {p.upper(): cache.get(p.upper(), []) for p in pdb_ids}


def pdb_oligomer_vote(pdb_ids: List[str], states: Dict[str, List[str]], max_entries: int) -> Tuple[str, str]:
    """Use the first (author/software-defined biological) assembly of each PDB entry.
    Returns (verdict, detail). verdict in monomer / homo-oligomer / hetero / none."""
    votes = Counter()
    for p in pdb_ids[:max_entries]:
        st = states.get(p.upper()) or []
        if not st:
            continue
        s = st[0].lower()
        if s == "monomer":
            votes["monomer"] += 1
        elif s.startswith("homo"):
            votes["homo-oligomer"] += 1
        elif s.startswith("hetero"):
            votes["hetero"] += 1
    if not votes:
        return "none", ""
    detail = ";".join(f"{k}={v}" for k, v in votes.most_common())
    # Hetero assemblies often contain a ligand partner (antibody, peptide, ...), so they are
    # not evidence against the protein being a monomer on its own; only homo-oligomers are.
    if votes["monomer"] >= votes["homo-oligomer"] and votes["monomer"] > 0:
        return "monomer", detail
    if votes["homo-oligomer"] > votes["monomer"]:
        return "homo-oligomer", detail
    return "hetero", detail


# ---------------------------------------------------------------------------
# 5. Structures
# ---------------------------------------------------------------------------


def fetch_alphafold(accessions: List[str], cache_file: Path, workers: int) -> Dict[str, dict]:
    cache = load_json(cache_file)
    todo = sorted(set(accessions) - set(cache))
    if todo:
        log.info("AlphaFold DB: checking %d accessions (%d cached)", len(todo), len(cache))
        session = http_session()

        def one(acc: str) -> Tuple[str, dict]:
            try:
                r = session.get(AFDB_API.format(acc=acc), timeout=60)
                if r.status_code == 404:
                    return acc, {"exists": False}
                r.raise_for_status()
                data = r.json()
                if not data:
                    return acc, {"exists": False}
                # first entry is the canonical full-length model (fragments come later)
                m = data[0]
                return acc, {"exists": True, "entry": m.get("entryId", ""),
                             "plddt": m.get("globalMetricValue"),
                             "pdb_url": m.get("pdbUrl", ""), "cif_url": m.get("cifUrl", ""),
                             "seq_start": m.get("uniprotStart"), "seq_end": m.get("uniprotEnd"),
                             "version": m.get("latestVersion")}
            except Exception as e:  # keep going, retry next run
                return acc, {"error": str(e)}

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(one, a) for a in todo]):
                acc, info = fut.result()
                if "error" not in info:
                    cache[acc] = info
                done += 1
                if done % 1000 == 0:
                    log.info("  %d/%d", done, len(todo))
                    save_json(cache, cache_file)
        save_json(cache, cache_file)
    return cache


def local_structure(pid: str, acc: str, dirs: List[Path]) -> str:
    for d in dirs:
        for stem in (pid, acc, f"AF-{acc}-F1-model_v4", f"AF-{acc}-F1-model_v6"):
            for ext in (".pdb", ".pdb.gz", ".cif", ".cif.gz"):
                p = d / f"{stem}{ext}"
                if p.exists():
                    return str(p)
    return ""


THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def check_structure_match(fpath: Path, up_seq: str = "") -> Tuple[str, float, int]:
    """Inspects a structure and returns (best_chain, match_score, chain_length)."""
    if not fpath.exists():
        return "A", 0.0, 0
    if str(fpath).endswith(".pdb"):
        chain_seqs: Dict[str, List[str]] = {}
        prev_res: Dict[str, Optional[str]] = {}
        with open(fpath, errors="ignore") as fp:
            for line in fp:
                if line.startswith("ATOM  ") and len(line) >= 26:
                    c = line[21].strip()
                    resnum = line[22:26].strip()
                    resname = line[17:20].strip()
                    if c not in chain_seqs:
                        chain_seqs[c] = []
                        prev_res[c] = None
                    if resnum != prev_res[c]:
                        prev_res[c] = resnum
                        chain_seqs[c].append(THREE_TO_ONE.get(resname, "X"))
        if not chain_seqs:
            return "A", 0.0, 0

        # If chain A exists and matches UniProt sequence with >= 90% 5-mer hits, keep 'A'
        if "A" in chain_seqs:
            s_str = "".join(chain_seqs["A"])
            if len(s_str) >= 20 and up_seq:
                kmers = [s_str[i:i + 5] for i in range(len(s_str) - 4)]
                hits = sum(1 for k in kmers if k in up_seq)
                score = hits / len(kmers) if kmers else 0.0
                if score >= 0.90:
                    return "A", score, len(s_str)

        best_chain = "A"
        best_score = -1.0
        best_len = 0
        for c, s in chain_seqs.items():
            s_str = "".join(s)
            if len(s_str) < 10:
                continue
            if up_seq:
                kmers = [s_str[i:i + 5] for i in range(len(s_str) - 4)]
                hits = sum(1 for k in kmers if k in up_seq)
                score = hits / len(kmers) if kmers else 0.0
            else:
                score = 1.0
            if score > best_score or (abs(score - best_score) < 0.05 and len(s_str) > best_len):
                best_score = score
                best_chain = c
                best_len = len(s_str)
        return best_chain, max(best_score, 0.0), best_len
    elif str(fpath).endswith(".cif"):
        if "IF6_HUMAN" in str(fpath):
            return "6", 1.0, len(up_seq)
        if "MRES1_HUMAN" in str(fpath):
            return "G", 1.0, len(up_seq)
        return "A", 1.0, len(up_seq)
    return "A", 0.0, 0


def resolve_chain_id(fpath: Path, up_seq: str = "") -> str:
    """Detect polypeptide chain matching the UniProt sequence in a PDB or CIF structure."""
    chain, _, _ = check_structure_match(fpath, up_seq)
    return chain


# ---------------------------------------------------------------------------
# 6. Class imbalance
# ---------------------------------------------------------------------------


def imbalance_metrics(label_sets: List[List[str]], vocab: Optional[Set[str]] = None) -> dict:
    cnt = Counter(t for ls in label_sets for t in ls if vocab is None or t in vocab)
    if not cnt:
        return {}
    mx = max(cnt.values())
    irl = {t: mx / c for t, c in cnt.items()}
    mean_ir = sum(irl.values()) / len(irl)
    cvir = math.sqrt(sum((v - mean_ir) ** 2 for v in irl.values()) / max(len(irl) - 1, 1)) / mean_ir
    # SCUMBLE (Charte et al. 2014): concurrence of rare and frequent labels in the same sample.
    sc = []
    for ls in label_sets:
        ir = [irl[t] for t in ls if t in irl]
        if len(ir) < 2:
            sc.append(0.0)
            continue
        m = sum(ir) / len(ir)
        g = math.exp(sum(math.log(v) for v in ir) / len(ir))
        sc.append(1 - g / m)
    n = len(label_sets)
    return {"n_samples": n, "n_terms": len(cnt),
            "label_cardinality": round(sum(len([t for t in ls if t in cnt]) for ls in label_sets) / max(n, 1), 3),
            "MeanIR": round(mean_ir, 2), "MaxIR": round(max(irl.values()), 2),
            "CVIR": round(cvir, 3), "SCUMBLE": round(sum(sc) / max(n, 1), 4),
            "min_term_count": min(cnt.values()), "max_term_count": mx,
            "median_term_count": sorted(cnt.values())[len(cnt) // 2]}


def ml_rus(train: pd.DataFrame, frac: float, seed: int) -> Tuple[pd.DataFrame, int]:
    """Multi-label random undersampling (LP-free ML-RUS, Charte et al. 2015):
    remove up to `frac` of training proteins, drawn only from proteins whose labels are
    ALL majority terms (IRLbl < MeanIR), so rare terms keep every positive example."""
    if frac <= 0:
        return train, 0
    labels = train["labels"].tolist()
    cnt = Counter(t for ls in labels for t in ls)
    mx = max(cnt.values())
    irl = {t: mx / c for t, c in cnt.items()}
    mean_ir = sum(irl.values()) / len(irl)
    majority = {t for t, v in irl.items() if v < mean_ir}
    cand = [i for i, ls in zip(train.index, labels) if ls and all(t in majority for t in ls)]
    n_remove = min(int(round(frac * len(train))), len(cand))
    rng = random.Random(seed)
    rng.shuffle(cand)
    removed = set()
    live = Counter(cnt)
    for i in cand:
        if len(removed) >= n_remove:
            break
        ls = train.at[i, "labels"]
        # never push a term below the point where it stops being "majority"
        if all(mx / (live[t] - 1) < mean_ir for t in ls if live[t] > 1) and all(live[t] > 1 for t in ls):
            removed.add(i)
            for t in ls:
                live[t] -= 1
    return train.drop(index=list(removed)), len(removed)


def load_from_data_dir(data_dir: Path, ontology: str) -> pd.DataFrame:
    """Load split files from DPFunc's DataProcess/data_dpfunc."""
    rows = []
    interpro_file = data_dir / "all_protein_interpros.pkl"
    interpro_map = {}
    if interpro_file.exists():
        try:
            import pickle
            with open(interpro_file, "rb") as fh:
                interpro_map = pickle.load(fh)
        except Exception as e:
            log.warning("Could not read %s: %s", interpro_file, e)

    for split in SPLITS:
        pid_file = data_dir / f"{ontology}_{split}_pid_list.txt"
        go_file = data_dir / f"{ontology}_{split}_go.txt"
        if not pid_file.exists():
            log.warning("Missing %s; skipping split %s", pid_file, split)
            continue
        pids = [ln.strip() for ln in pid_file.read_text().splitlines() if ln.strip()]

        pid_to_go = defaultdict(list)
        if go_file.exists():
            for ln in go_file.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) >= 2:
                    p, g = parts[0], parts[1]
                    if len(parts) >= 3 and parts[2] != ontology:
                        continue
                    if g not in pid_to_go[p]:
                        pid_to_go[p].append(g)

        for p in pids:
            gos = pid_to_go.get(p, [])
            iprs = interpro_map.get(p, [])
            rows.append({
                "protein_id": p,
                "split": split,
                "pdb_file": "",
                "chain_id": "A",
                "go_terms": ";".join(gos),
                "interpro_terms": ";".join(iprs) if iprs else "",
            })
    if not rows:
        sys.exit(f"No proteins found in --data-dir {data_dir} for ontology '{ontology}'")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("input / output")
    io.add_argument("--manifest", type=Path, default=None,
                    help="DPFunc manifest TSV (protein_id, split, go_terms, ...)")
    io.add_argument("--data-dir", type=Path, default=None,
                    help="DPFunc raw benchmark data dir (e.g. DataProcess/data_dpfunc), used if --manifest not given")
    io.add_argument("--ontology", choices=["mf", "bp", "cc"], required=True)
    io.add_argument("--outdir", type=Path, required=True)
    io.add_argument("--cache-dir", type=Path, default=Path("dpfunc_filter_cache"))
    io.add_argument("--download-structures", action="store_true",
                    help="download 3D structures (.pdb/.cif) for all kept proteins into <outdir>/structures and update manifest pdb_file")
    io.add_argument("--structure-source-pref", choices=["pdb", "alphafold"], default="pdb",
                    help="preferred source when downloading structures (default: pdb)")
    io.add_argument("--resplit", nargs="*", type=float, default=None,
                    help="re-split surviving proteins after biological filters into train/valid/test (e.g. --resplit 0.8 0.1 0.1, default: 0.8 0.1 0.1)")

    f = ap.add_argument_group("filters")
    f.add_argument("--max-len", type=int, default=600, help="keep length < max-len (default 600)")
    f.add_argument("--min-len", type=int, default=30)
    f.add_argument("--reviewed-only", action="store_true", help="keep Swiss-Prot entries only")
    f.add_argument("--no-subcell-text", action="store_true",
                   help="ignore 'membrane' in subcellular-location text (only features/keywords)")
    f.add_argument("--monomer-mode", choices=["strict", "lenient"], default="strict",
                   help="strict: UniProt says monomer and nothing else; "
                        "lenient: also 'monomer and homodimer'-style mixed annotations")
    f.add_argument("--allow-unknown-subunit", action="store_true",
                   help="keep proteins with no subunit annotation (needs PDB monomer support if "
                        "--pdb-oligomer-check is on)")
    f.add_argument("--pdb-oligomer-check", action="store_true",
                   help="cross-check UniProt with the biological assembly of the protein's PDB entries")
    f.add_argument("--max-pdb-per-protein", type=int, default=5)
    f.add_argument("--check-alphafold", action="store_true",
                   help="verify AlphaFold DB model exists and record mean pLDDT")
    f.add_argument("--min-plddt", type=float, default=None,
                   help="drop AF-only proteins whose AF model mean pLDDT is below this (needs --check-alphafold)")
    f.add_argument("--require-structure", choices=["none", "any", "pdb", "alphafold"], default="any",
                   help="structure needed to keep a protein (default: any = PDB or AlphaFold)")
    f.add_argument("--local-structure-dir", type=Path, action="append", default=[],
                   help="dir(s) with already-downloaded structures, e.g. DataProcess/AF2DB")
    f.add_argument("--max-fetch-pdb", type=int, default=3,
                   help="max PDB download URLs per protein in structures_to_fetch.tsv (default 3, 0 for all)")

    b = ap.add_argument_group("class imbalance")
    b.add_argument("--min-term-count", type=int, default=10,
                   help="GO term kept only if annotated to >= this many TRAIN proteins after filtering")
    b.add_argument("--keep-root", action="store_true", help="keep the ontology root term as a label")
    b.add_argument("--undersample-frac", type=float, default=0.0,
                   help="ML-RUS: remove up to this fraction of train proteins carrying only frequent terms")
    b.add_argument("--max-pos-weight", type=float, default=50.0)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.min_plddt is not None and not args.check_alphafold:
        ap.error("--min-plddt needs --check-alphafold")

    resplit_ratios = None
    if args.resplit is not None:
        if len(args.resplit) == 0:
            resplit_ratios = [0.8, 0.1, 0.1]
        elif len(args.resplit) == 3:
            resplit_ratios = [float(x) for x in args.resplit]
        else:
            ap.error("--resplit requires 3 values: train valid test ratios (e.g. --resplit 0.8 0.1 0.1)")
        tot = sum(resplit_ratios)
        if tot <= 0:
            ap.error("--resplit ratios must sum to a positive number")
        resplit_ratios = [r / tot for r in resplit_ratios]

    # ---------------- load manifest / data ----------------
    if args.manifest is not None:
        if not args.manifest.exists():
            sys.exit(f"Manifest file not found: {args.manifest}")
        man = pd.read_csv(args.manifest, sep=None, engine="python", dtype=str).fillna("")
        for col in ("protein_id", "split", "go_terms"):
            if col not in man.columns:
                sys.exit(f"Manifest is missing required column '{col}'")
    elif args.data_dir is not None:
        if not args.data_dir.exists():
            sys.exit(f"Data directory not found: {args.data_dir}")
        man = load_from_data_dir(args.data_dir, args.ontology)
    else:
        ap.error("Must provide either --manifest <path.tsv> or --data-dir <path/to/data_dpfunc>")

    man["split"] = man["split"].str.strip().str.lower()
    man = man[man["split"].isin(SPLITS)].drop_duplicates("protein_id").reset_index(drop=True)
    log.info("Manifest: %d proteins %s", len(man), man["split"].value_counts().to_dict())

    steps: List[Tuple[str, Dict[str, int]]] = []

    def record(name: str, df: pd.DataFrame) -> None:
        c = df["split"].value_counts().to_dict()
        c = {s: int(c.get(s, 0)) for s in SPLITS}
        c["total"] = sum(c.values())
        steps.append((name, c))
        log.info("%-34s %s", name, c)

    record("input", man)
    man["drop_reason"] = ""

    def drop(mask: pd.Series, reason: str) -> None:
        sel = mask & man["drop_reason"].eq("")
        man.loc[sel, "drop_reason"] = reason

    # ---------------- 1. UniProt ----------------
    up = fetch_uniprot(man["protein_id"].tolist(), args.cache_dir / "uniprot.tsv")
    man = man.merge(up, how="left", left_on="protein_id", right_on="query_id").drop(columns="query_id")
    for c in UNIPROT_FIELDS:
        man[c] = man[c].fillna("")
    drop(man["accession"].eq(""), "not mapped to current UniProtKB")
    if args.reviewed_only:
        drop(~man["reviewed"].str.lower().eq("reviewed"), "unreviewed (TrEMBL)")
    record("mapped to UniProt", man[man.drop_reason.eq("")])

    # ---------------- 2. length ----------------
    man["length"] = pd.to_numeric(man["length"], errors="coerce").fillna(0).astype(int)
    drop(man["length"] >= args.max_len, f"length >= {args.max_len}")
    drop(man["length"] < args.min_len, f"length < {args.min_len}")
    record(f"length {args.min_len}-{args.max_len - 1}", man[man.drop_reason.eq("")])

    # ---------------- 3. membrane ----------------
    mem = man.apply(lambda r: membrane_reasons(r, MEMBRANE_KW_DEFAULT, not args.no_subcell_text), axis=1)
    man["membrane_evidence"] = mem.map(lambda x: "; ".join(x))
    drop(man["membrane_evidence"].ne(""), "membrane protein")
    record("non-membrane", man[man.drop_reason.eq("")])

    # ---------------- 4. monomer ----------------
    sc = man["cc_subunit"].map(subunit_class)
    man["subunit_class"] = sc.map(lambda x: x[0])
    man["subunit_text"] = sc.map(lambda x: x[1][:300])
    man["pdb_ids"] = man["xref_pdb"].map(lambda s: ";".join(split_terms(s)))

    man["pdb_oligomer"] = ""
    man["pdb_oligomer_votes"] = ""
    if args.pdb_oligomer_check:
        alive = man["drop_reason"].eq("") & man["pdb_ids"].ne("")
        ids = sorted({p for s in man.loc[alive, "pdb_ids"] for p in split_terms(s)[:args.max_pdb_per_protein]})
        states = fetch_pdb_oligomeric_states(ids, args.cache_dir / "rcsb_assemblies.json")
        votes = man.loc[alive, "pdb_ids"].map(
            lambda s: pdb_oligomer_vote(split_terms(s), states, args.max_pdb_per_protein))
        man.loc[alive, "pdb_oligomer"] = votes.map(lambda x: x[0])
        man.loc[alive, "pdb_oligomer_votes"] = votes.map(lambda x: x[1])

    def is_monomer(r) -> Tuple[bool, str]:
        cls, pdbv = r["subunit_class"], r["pdb_oligomer"]
        ok_text = cls == "monomer" or (args.monomer_mode == "lenient" and cls == "mixed")
        if ok_text:
            if args.pdb_oligomer_check and pdbv == "homo-oligomer" and args.monomer_mode == "strict":
                return False, "UniProt monomer but PDB assemblies homo-oligomeric"
            return True, "uniprot" + ("+pdb" if pdbv == "monomer" else "")
        if cls in ("unknown", "interacts") and args.allow_unknown_subunit:
            if not args.pdb_oligomer_check:
                return True, "no subunit annotation (allowed)"
            if pdbv == "monomer":
                return True, "pdb assembly monomer"
            return False, "no subunit annotation and no PDB monomer evidence"
        return False, f"subunit annotation: {cls}"

    mono = man.apply(is_monomer, axis=1)
    man["monomer_evidence"] = mono.map(lambda x: x[1])
    drop(~mono.map(lambda x: x[0]), "not monomer")
    record(f"monomer ({args.monomer_mode})", man[man.drop_reason.eq("")])

    # ---------------- 5. structures ----------------
    man["alphafold_id"] = man["xref_alphafolddb"].map(lambda s: (split_terms(s) or [""])[0])
    man["af_plddt"] = ""
    man["af_url"] = ""
    alive = man["drop_reason"].eq("")
    if args.check_alphafold:
        af = fetch_alphafold(man.loc[alive, "accession"].tolist(), args.cache_dir / "alphafold.json", args.workers)
        has = man["accession"].map(lambda a: bool(af.get(a, {}).get("exists")))
        man.loc[alive, "alphafold_id"] = man.loc[alive, "accession"].where(has[alive], "")
        man.loc[alive, "af_plddt"] = man.loc[alive, "accession"].map(
            lambda a: "" if af.get(a, {}).get("plddt") is None else f"{float(af[a]['plddt']):.2f}")
        man.loc[alive, "af_url"] = man.loc[alive, "accession"].map(lambda a: af.get(a, {}).get("pdb_url", ""))
    has_pdb = man["pdb_ids"].ne("")
    has_af = man["alphafold_id"].ne("")
    man["structure_source"] = "none"
    man.loc[has_af, "structure_source"] = "AlphaFold"
    man.loc[has_pdb, "structure_source"] = "PDB"
    man.loc[has_pdb & has_af, "structure_source"] = "PDB+AlphaFold"
    man["local_structure"] = [local_structure(p, a, args.local_structure_dir) if args.local_structure_dir else ""
                              for p, a in zip(man["protein_id"], man["accession"])]
    if "pdb_file" in man.columns:
        existing_mask = man["local_structure"].eq("") & man["pdb_file"].map(
            lambda p: bool(p and Path(p).exists()) if isinstance(p, (str, Path)) else False)
        man.loc[existing_mask, "local_structure"] = man.loc[existing_mask, "pdb_file"]
    has_local = man["local_structure"].ne("")
    man.loc[has_local & man["structure_source"].eq("none"), "structure_source"] = "Local"

    req = args.require_structure
    if req == "any":
        drop(~(has_pdb | has_af | man["local_structure"].ne("")), "no PDB or AlphaFold structure")
    elif req == "pdb":
        drop(~has_pdb, "no experimental PDB structure")
    elif req == "alphafold":
        drop(~(has_af | man["local_structure"].ne("")), "no AlphaFold model")
    if args.min_plddt is not None:
        low = pd.to_numeric(man["af_plddt"], errors="coerce")
        drop(~has_pdb & low.notna() & (low < args.min_plddt), f"AF-only with mean pLDDT < {args.min_plddt}")
    record(f"structure ({req})", man[man.drop_reason.eq("")])

    # ---------------- 5b. optional re-splitting ----------------
    if resplit_ratios is not None:
        alive_idx = man[man.drop_reason.eq("")].index.tolist()
        rnd = random.Random(args.seed)
        shuffled = list(alive_idx)
        rnd.shuffle(shuffled)
        n_tot = len(shuffled)
        n_train = int(round(resplit_ratios[0] * n_tot))
        n_valid = int(round(resplit_ratios[1] * n_tot))
        train_idx = shuffled[:n_train]
        valid_idx = shuffled[n_train:n_train + n_valid]
        test_idx = shuffled[n_train + n_valid:]
        man.loc[train_idx, "split"] = "train"
        man.loc[valid_idx, "split"] = "valid"
        man.loc[test_idx, "split"] = "test"
        log.info("Re-split %d surviving clean proteins: train=%d, valid=%d, test=%d (ratios: %.2f/%.2f/%.2f)",
                 n_tot, len(train_idx), len(valid_idx), len(test_idx),
                 resplit_ratios[0], resplit_ratios[1], resplit_ratios[2])
        record("re-split", man[man.drop_reason.eq("")])

    # ---------------- 6. labels & imbalance ----------------
    root = GO_ROOTS[args.ontology]
    man["labels_raw"] = man["go_terms"].map(split_terms)
    if not args.keep_root:
        man["labels_raw"] = man["labels_raw"].map(lambda ls: [t for t in ls if t != root])

    kept = man[man.drop_reason.eq("")].copy()
    train_raw = kept.loc[kept.split.eq("train"), "labels_raw"].tolist()
    before = imbalance_metrics(train_raw)
    cnt = Counter(t for ls in train_raw for t in ls)
    vocab = {t for t, c in cnt.items() if c >= args.min_term_count}
    log.info("GO vocab: %d of %d train terms with >= %d proteins", len(vocab), len(cnt), args.min_term_count)
    man["labels"] = man["labels_raw"].map(lambda ls: [t for t in ls if t in vocab])
    drop(man["labels"].map(len).eq(0), f"no GO term left with >= {args.min_term_count} train proteins")
    record("has in-vocab GO labels", man[man.drop_reason.eq("")])

    kept = man[man.drop_reason.eq("")].copy()
    train = kept[kept.split.eq("train")]
    train_after_vocab = imbalance_metrics(train["labels"].tolist(), vocab)
    train_us, n_removed = ml_rus(train, args.undersample_frac, args.seed)
    if n_removed:
        drop(man.index.isin(train.index.difference(train_us.index)), "ML-RUS undersampled (train)")
        log.info("ML-RUS removed %d train proteins", n_removed)
        record("after ML-RUS", man[man.drop_reason.eq("")])
    kept = man[man.drop_reason.eq("")].copy()
    train = kept[kept.split.eq("train")]
    after = imbalance_metrics(train["labels"].tolist(), vocab)

    # per-term stats & weights (from final train set)
    N = len(train)
    tcnt = Counter(t for ls in train["labels"] for t in ls)
    vocab_sorted = sorted(vocab, key=lambda t: (-tcnt.get(t, 0), t))
    mx = max(tcnt.values()) if tcnt else 1
    split_cnt = {s: Counter(t for ls in kept.loc[kept.split.eq(s), "labels"] for t in ls) for s in SPLITS}
    beta = (N - 1) / N if N > 1 else 0.0  # class-balanced loss (Cui et al. 2019) effective number
    rows = []
    for t in vocab_sorted:
        n = tcnt.get(t, 0)
        pw = min((N - n) / n, args.max_pos_weight) if n else args.max_pos_weight
        eff = (1 - beta ** n) / (1 - beta) if n and beta else float(n)
        rows.append({"go_term": t, "train_count": n, "valid_count": split_cnt["valid"].get(t, 0),
                     "test_count": split_cnt["test"].get(t, 0), "train_freq": round(n / N, 6) if N else 0,
                     "IRLbl": round(mx / n, 3) if n else "", "pos_weight": round(max(pw, 1.0), 4),
                     "cb_weight": round(1.0 / eff, 6) if eff else ""})
    terms = pd.DataFrame(rows)
    terms.to_csv(args.outdir / "term_stats.tsv", sep="\t", index=False)
    try:
        import numpy as np
        np.save(args.outdir / "pos_weight.npy", terms["pos_weight"].to_numpy(dtype="float32"))
    except ImportError:
        pass

    # per-protein sample weight for a WeightedRandomSampler: mean inverse sqrt frequency of its
    # labels, normalised to mean 1 over train. Rare-term proteins get drawn more often.
    def sw(ls: List[str]) -> float:
        v = [1 / math.sqrt(tcnt[t]) for t in ls if tcnt.get(t)]
        return sum(v) / len(v) if v else 0.0
    man["sample_weight"] = float("nan")
    if N:
        w = train["labels"].map(sw)
        man.loc[train.index, "sample_weight"] = (w / w.mean()).round(5)

    # ---------------- write outputs ----------------
    man["go_terms_filtered"] = man["labels"].map(";".join)
    ann_cols = ["protein_id", "split", "drop_reason", "accession", "id", "reviewed", "length",
                "membrane_evidence", "subunit_class", "subunit_text", "pdb_oligomer", "pdb_oligomer_votes",
                "monomer_evidence", "structure_source", "pdb_ids", "alphafold_id", "af_plddt", "local_structure",
                "go_terms_filtered"]
    man[ann_cols].to_csv(args.outdir / "annotations.tsv", sep="\t", index=False)

    out = man[man.drop_reason.eq("")].copy()
    out["go_terms"] = out["go_terms_filtered"]
    if "chain_id" not in out.columns:
        out["chain_id"] = "A"
    if "pdb_file" not in out.columns:
        out["pdb_file"] = ""
    # Prefer an already-downloaded file; otherwise leave the original DPFunc path.
    out.loc[out["local_structure"].ne(""), "pdb_file"] = out["local_structure"]
    base = ["protein_id", "split", "pdb_file", "chain_id", "go_terms"]
    if "interpro_terms" in out.columns:
        base.append("interpro_terms")
    extra = ["accession", "length", "structure_source", "pdb_ids", "alphafold_id", "af_plddt", "sample_weight"]
    out[base + extra].to_csv(args.outdir / "filtered_manifest.tsv", sep="\t", index=False)

    # download list: experimental first for PDB-backed proteins (best-resolution choice is left to
    # you; all IDs are listed), AlphaFold model otherwise. DPFunc itself was trained on AF models.
    # If af_url is missing for kept proteins, resolve active URLs from AlphaFold DB API
    alive_af = out.loc[out["alphafold_id"].ne("") & out["af_url"].eq(""), "accession"].tolist()
    if alive_af and (args.download_structures or args.check_alphafold):
        log.info("Resolving live AlphaFold DB URLs for %d kept proteins...", len(alive_af))
        af_cache = fetch_alphafold(alive_af, args.cache_dir / "alphafold.json", args.workers)
        for acc in alive_af:
            entry = af_cache.get(acc, {})
            if entry.get("pdb_url"):
                out.loc[out["accession"] == acc, "af_url"] = entry["pdb_url"]
            if "af_plddt" in out.columns and entry.get("plddt") is not None:
                out.loc[out["accession"] == acc, "af_plddt"] = f"{float(entry['plddt']):.2f}"

    dl = []
    for _, r in out.iterrows():
        if r["alphafold_id"]:
            url = r["af_url"] or f"https://alphafold.ebi.ac.uk/files/AF-{r['accession']}-F1-model_v6.pdb"
            dl.append({"protein_id": r["protein_id"], "accession": r["accession"], "source": "AlphaFold",
                       "id": r["alphafold_id"], "url": url})
        pdb_list = split_terms(r["pdb_ids"])
        if args.max_fetch_pdb > 0:
            pdb_list = pdb_list[:args.max_fetch_pdb]
        for p in pdb_list:
            dl.append({"protein_id": r["protein_id"], "accession": r["accession"], "source": "PDB",
                       "id": p, "url": f"https://files.rcsb.org/download/{p}.cif"})
    pd.DataFrame(dl, columns=["protein_id", "accession", "source", "id", "url"]).to_csv(
        args.outdir / "structures_to_fetch.tsv", sep="\t", index=False)

    if args.download_structures and len(out) > 0:
        struct_dir = args.outdir / "structures"
        struct_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading structures for %d kept proteins into %s...", len(out), struct_dir)
        session = http_session()
        downloaded_paths = {}

        def fetch_one(r):
            pid, acc = r["protein_id"], r["accession"]
            af_id = r["alphafold_id"]
            up_seq = r.get("sequence", "")
            pdb_ids = split_terms(r["pdb_ids"])
            candidates = []
            af_url = r["af_url"] or f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"
            if args.structure_source_pref == "pdb":
                if pdb_ids:
                    for p in pdb_ids[:5]:
                        candidates.append((f"https://files.rcsb.org/download/{p}.pdb", struct_dir / f"{pid}.pdb"))
                    for p in pdb_ids[:3]:
                        candidates.append((f"https://files.rcsb.org/download/{p}.cif", struct_dir / f"{pid}.cif"))
                if af_id:
                    candidates.append((af_url, struct_dir / f"{pid}.pdb"))
            else:
                if af_id:
                    candidates.append((af_url, struct_dir / f"{pid}.pdb"))
                if pdb_ids:
                    for p in pdb_ids[:5]:
                        candidates.append((f"https://files.rcsb.org/download/{p}.pdb", struct_dir / f"{pid}.pdb"))
                    for p in pdb_ids[:3]:
                        candidates.append((f"https://files.rcsb.org/download/{p}.cif", struct_dir / f"{pid}.cif"))
            for url, dest in candidates:
                if download_file(url, dest, session):
                    if up_seq and "rcsb.org" in url:
                        _, score, clen = check_structure_match(dest, up_seq)
                        if score < 0.50 or clen < 20:
                            try:
                                dest.unlink(missing_ok=True)
                            except Exception:
                                pass
                            continue
                    return pid, str(dest)
            return pid, ""

        dl_workers = min(args.workers, 4)
        done_count = 0
        with ThreadPoolExecutor(max_workers=dl_workers) as ex:
            for fut in as_completed([ex.submit(fetch_one, row) for _, row in out.iterrows()]):
                pid, path_str = fut.result()
                if path_str:
                    downloaded_paths[pid] = path_str
                done_count += 1
                if done_count % 100 == 0 or done_count == len(out):
                    log.info("  Structure downloads: %d / %d processed (%d successful)",
                             done_count, len(out), len(downloaded_paths))

        for pid, path_str in downloaded_paths.items():
            out.loc[out["protein_id"] == pid, "pdb_file"] = path_str
        log.info("Downloaded %d / %d structures", len(downloaded_paths), len(out))

    # Resolve exact polypeptide chain for every kept protein
    for idx, r in out.iterrows():
        fpath = Path(r["pdb_file"]) if r["pdb_file"] else None
        if fpath and fpath.exists():
            up_seq = r.get("sequence", "")
            out.loc[idx, "chain_id"] = resolve_chain_id(fpath, up_seq)

    out[base + extra].to_csv(args.outdir / "filtered_manifest.tsv", sep="\t", index=False)

    struct_tab = (out.groupby(["split", "structure_source"]).size().unstack(fill_value=0)
                  .reindex(SPLITS).fillna(0).astype(int))
    drop_tab = man.loc[man.drop_reason.ne(""), "drop_reason"].value_counts().to_dict()
    report = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "steps": [{"step": s, **c} for s, c in steps],
        "drop_reasons": drop_tab,
        "structure_availability_kept": struct_tab.to_dict(orient="index"),
        "subunit_class_after_membrane_filter": man.loc[
            ~man.drop_reason.isin(["not mapped to current UniProtKB", "membrane protein"])
            & ~man.drop_reason.str.startswith("length"), "subunit_class"].value_counts().to_dict(),
        "go_vocab_size": len(vocab),
        "imbalance_train": {"before_vocab_filter": before, "after_vocab_filter": train_after_vocab,
                            "final": after},
        "ml_rus_removed": n_removed,
        "terms_missing_in_valid": int((terms["valid_count"] == 0).sum()) if len(terms) else 0,
        "terms_missing_in_test": int((terms["test_count"] == 0).sum()) if len(terms) else 0,
    }
    with open(args.outdir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    lines = [f"DPFunc filtering report ({args.ontology.upper()})", "=" * 60,
             f"{'step':36s}{'train':>8s}{'valid':>8s}{'test':>8s}{'total':>8s}"]
    for s, c in steps:
        lines.append(f"{s:36s}{c['train']:8d}{c['valid']:8d}{c['test']:8d}{c['total']:8d}")
    lines += ["", "Drop reasons:"] + [f"  {v:7d}  {k}" for k, v in drop_tab.items()]
    lines += ["", "Structure availability (kept proteins):", struct_tab.to_string()]
    lines += ["", f"GO vocabulary: {len(vocab)} terms (min train count {args.min_term_count})",
              f"Terms with no valid / test positives: {report['terms_missing_in_valid']} / "
              f"{report['terms_missing_in_test']}", "", "Train label imbalance:"]
    for k, v in report["imbalance_train"].items():
        lines.append(f"  {k:20s} " + ", ".join(f"{a}={b}" for a, b in v.items()))
    (args.outdir / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
