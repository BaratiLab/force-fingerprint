#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas>=1.3",
#     "requests>=2.25",
#     "numpy>=1.20",
#     "metapredict>=3.0",
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
  4. Monomer       - strict / lenient: keep proteins whose UniProt "Subunit structure" says
                     monomer, optionally cross-checked against PDB (RCSB) assemblies.
                     supported: drop proteins with evidence of a stable complex (UniProt text
                     with the protein as subject, Complex Portal membership, homo-oligomeric or
                     optionally hetero-complex PDB assemblies), then keep only those with
                     monomer support, recorded as monomer_tier:
                       1 = UniProt monomer and PDB assemblies monomeric
                       2 = UniProt monomer, no PDB contradiction (or no PDB data)
                       3 = PDB assemblies monomeric, no UniProt statement about the protein's
                           own state (or a monomer/oligomer equilibrium statement)
                     All three modes are evaluated and stored in annotations.tsv for comparison.
  5. Disorder      - drop proteins with a long internal disordered region (metapredict
                     per-residue scores; default: no internal run > 14 residues above 0.5).
                     Run before the identity filter so cluster representatives are chosen
                     only among proteins that pass it.
  5b. Size         - drop proteins whose estimated box size exceeds --max-box-volume
                     (default 1200 nm^3). Box size is estimated from the extents e_i of
                     non-hydrogen atoms after trimming metapredict terminal tails and
                     aligning principle axes to the cartesian axes according to C-alpha
                     SVD according to V_box = prod(e_i + 2p). Proteins without a usable
                     structure are kept.
  6. Identity      - cluster with MMseqs2 (default >= 40% identity, 80% coverage) and keep
                     one protein per cluster, preferring PREFERRED_SPECIES, reviewed entries,
                     proteins with PDB structures, then longer sequences.
  7. Structure     - record whether an experimental (PDB) and/or AlphaFold DB structure
                     exists, optionally verify the AF model and its mean pLDDT, and
                     optionally require a structure.
  8. Imbalance     - GO-term vocabulary from train only (min count), root terms removed,
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
  structures/              (--download-structures) cleaned structures, hard-linked (copied if
                           linking fails) from <cache-dir>/structures/clean; raw downloads are
                           cached in <cache-dir>/structures/raw and reused across runs. This
                           directory is cleared at the start of every run.
  report.json / report.txt per-step counts by split and imbalance metrics

Usage:
  python dpfunc_filter.py \
      --manifest DPFunc_demo_pipeline/templates/mf_manifest_from_data_dpfunc.tsv \
      --ontology mf --outdir filtered_mf \
      --pdb-oligomer-check --check-alphafold --min-term-count 10 --undersample-frac 0.1

Requires: Python 3.8+, requests, pandas, metapredict (numpy optional), and the MMseqs2
binary on PATH (conda install -c bioconda mmseqs2). Internet access to
rest.uniprot.org (always), data.rcsb.org (--pdb-oligomer-check) and alphafold.ebi.ac.uk
(--check-alphafold). All remote lookups are cached in --cache-dir, so reruns with
different thresholds are offline and fast.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
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
    "xref_pdb", "xref_alphafolddb", "xref_complexportal", "sequence",
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

PREFERRED_SPECIES = [
    "human",
    "mouse",
    "yeast",
    "ecoli",
]

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
        Path(dest).touch()
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
    stale = [c for c in UNIPROT_FIELDS if c not in cached.columns]
    if stale:
        # UNIPROT_FIELDS changed since the cache was written: refetch everything once.
        log.warning("UniProt cache lacks fields %s; re-querying all IDs", stale)
        cached = pd.DataFrame(columns=["query_id"] + UNIPROT_FIELDS)
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


# --- "supported" mode: only statements whose grammatical subject is the protein itself count.
# UniProt subunit comments normally open each sentence/clause with the protein's own state
# ("Homodimer.", "Component of the X complex."), while partners appear later in the
# sentence ("Interacts with the Ku heterodimer"), so matching at the start of each clause
# avoids most false positives. Hedged statements ("May form homodimers", "Can dimerize")
# deliberately do not count.
_OLIGO = (r"(?:homo-?|hetero-?)?(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|dodeca|oligo|multi)mer(?:ic|s)?"
          r"|(?:homo|hetero)-?\d+-?mer|(?:homo|hetero)mer(?:ic|s)?")
_HEDGE = r"(?:(?:probable|probably|likely|predominantly|mainly|mostly|primarily)\s+)?"
_LEAD = (r"(?:(?:is|exists as|forms|assembles as|occurs as|functions as|"
         r"the (?:active|functional|native|mature) (?:form|enzyme|protein) is)\s+(?:an?\s+)?)?")
SELF_OLIGO_RE = re.compile(rf"^{_HEDGE}{_LEAD}(?:{_OLIGO})\b|^{_HEDGE}(?:component|part|subunit) of\b")
SELF_MONO_RE = re.compile(rf"^{_HEDGE}{_LEAD}monomer(?:ic|s)?\b")
ANY_OLIGO_RE = re.compile(rf"\b(?:{_OLIGO})\b")


def subunit_self_class(text: str) -> Tuple[str, str]:
    """Classify UniProt subunit text by what it says about the protein itself.
    Returns (class, triggering clause). class in
    self_oligomer - a clause opens with an oligomer / complex-membership statement
    self_monomer  - a clause opens with 'Monomer' and no oligomer statement anywhere for the protein
    mixed         - both, or a monomer clause that also names an oligomer ('Monomer and homodimer')
    partner_only  - text present but only describes partners ('Interacts with ...')
    unknown       - no subunit annotation
    """
    t = re.sub(r"\{[^}]*\}", " ", text or "")
    t = re.sub(r"\s*SUBUNIT:\s*", " ", t, flags=re.I).strip()
    if not t:
        return "unknown", ""
    clauses = [c.strip().lower() for c in re.split(r"(?<=[.;])\s+", t) if c.strip()]
    olig = [c for c in clauses if SELF_OLIGO_RE.match(c)]
    mono = [c for c in clauses if SELF_MONO_RE.match(c)]
    if olig and mono:
        return "mixed", f"{mono[0]} | {olig[0]}"[:200]
    if olig:
        return "self_oligomer", olig[0][:200]
    if mono:
        eq = [c for c in mono if ANY_OLIGO_RE.search(c)]
        return ("mixed", eq[0][:200]) if eq else ("self_monomer", mono[0][:200])
    return "partner_only", ""


def fetch_pdb_entries(pdb_ids: List[str], cache_file: Path) -> Dict[str, dict]:
    """PDB id -> {"states": oligomeric state of each assembly in assembly-id order,
    e.g. ['Homo 2-mer', 'Monomer'], "entities": [{"type", "len", "uniprot"}, ...]}.
    Uses one RCSB GraphQL call per 200 entries."""
    cache = load_json(cache_file)
    todo = sorted({p.upper() for p in pdb_ids} - set(cache))
    if todo:
        log.info("RCSB: querying assemblies for %d PDB entries (%d cached)", len(todo), len(cache))
        session = http_session()
        q = ("query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
             "assemblies{rcsb_assembly_info{assembly_id} "
             "rcsb_struct_symmetry{kind oligomeric_state}} "
             "polymer_entities{entity_poly{type rcsb_sample_sequence_length} "
             "rcsb_polymer_entity_container_identifiers{uniprot_ids}}}}")
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
                ents = []
                for pe in e.get("polymer_entities") or []:
                    ep = pe.get("entity_poly") or {}
                    ids_ = (pe.get("rcsb_polymer_entity_container_identifiers") or {}).get("uniprot_ids") or []
                    ents.append({"type": ep.get("type") or "", "len": ep.get("rcsb_sample_sequence_length") or 0,
                                 "uniprot": ids_})
                cache[e["rcsb_id"].upper()] = {"states": states, "entities": ents}
                got.add(e["rcsb_id"].upper())
            for p in chunk:
                cache.setdefault(p, {"states": [], "entities": []})  # obsolete / not found
            if i % 10 == 0:
                save_json(cache, cache_file)
        save_json(cache, cache_file)
    return {p.upper(): cache.get(p.upper(), {"states": [], "entities": []}) for p in pdb_ids}


def pdb_complex_vote(pdb_ids: List[str], entries: Dict[str, dict], accession: str,
                     max_entries: int, min_partner_len: int, hetero_counts: bool) -> Tuple[str, str]:
    """Like pdb_oligomer_vote, but splits hetero assemblies into
    hetero-complex (another protein entity >= min_partner_len residues with a different UniProt
    accession, i.e. a real protein partner, not a peptide / antibody / designed binder, which
    usually lack a UniProt mapping) and hetero-other (neutral).
    Verdict in monomer / homo-oligomer / hetero-complex / hetero / none. hetero-complex is only
    returned as the verdict when hetero_counts is True."""
    votes = Counter()
    for p in pdb_ids[:max_entries]:
        info = entries.get(p.upper()) or {}
        st = info.get("states") or []
        if not st:
            continue
        s = st[0].lower()
        if s == "monomer":
            votes["monomer"] += 1
        elif s.startswith("homo"):
            votes["homo-oligomer"] += 1
        elif s.startswith("hetero"):
            partner = any(
                "polypeptide" in (en.get("type") or "") and (en.get("len") or 0) >= min_partner_len
                and en.get("uniprot") and accession not in en["uniprot"]
                for en in info.get("entities") or [])
            votes["hetero-complex" if partner else "hetero-other"] += 1
    if not votes:
        return "none", ""
    detail = ";".join(f"{k}={v}" for k, v in votes.most_common())
    mono, homo, het = votes["monomer"], votes["homo-oligomer"], votes["hetero-complex"]
    if not hetero_counts:
        het = 0
    if mono > 0 and mono >= homo and mono >= het:
        return "monomer", detail
    if homo > mono and homo >= het:
        return "homo-oligomer", detail
    if het > mono:
        return "hetero-complex", detail
    return "hetero", detail


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
# 5. Disorder
# ---------------------------------------------------------------------------

# metapredict only accepts the 20 standard residues; map the rare extras to close analogues.
_NONSTANDARD = str.maketrans({"U": "C", "O": "K", "B": "N", "Z": "Q", "J": "L", "X": "G"})


def seq_key(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:20]


def predict_disorder_scores(seqs: Iterable[str], cache_dir: Path) -> Dict[str, List[float]]:
    """Per-residue metapredict disorder scores, keyed by seq_key(sequence).
    Cached per metapredict version, since different versions give different scores."""
    try:
        import metapredict as meta
    except ImportError:
        sys.exit("The disorder filter needs metapredict: pip install metapredict")
    version = getattr(meta, "__version__", "unknown")
    cache_file = cache_dir / f"metapredict_{version}.json"
    cache = load_json(cache_file)
    todo = {seq_key(s): s for s in seqs if s and seq_key(s) not in cache}
    if todo:
        log.info("metapredict %s: predicting disorder for %d sequences (%d cached)",
                 version, len(todo), len(cache))
        for i, (k, s) in enumerate(todo.items(), 1):
            scores = meta.predict_disorder(s.upper().translate(_NONSTANDARD))
            cache[k] = [round(float(v), 3) for v in scores]
            if i % 1000 == 0:
                log.info("  %d/%d", i, len(todo))
                save_json(cache, cache_file)
        save_json(cache, cache_file)
    return cache


def disorder_runs(scores: Sequence[float], threshold: float) -> Tuple[int, int, int]:
    """Return (longest internal run, N-terminal run, C-terminal run) of consecutive residues
    with score > threshold. A run covering the whole sequence counts as internal, so fully
    disordered proteins are not excused as 'terminal'."""
    n = len(scores)
    runs, start = [], None
    for i, v in enumerate(scores):
        if v > threshold:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n))
    internal = nterm = cterm = 0
    for s, e in runs:
        length = e - s
        if s == 0 and e == n:
            internal = max(internal, length)
        elif s == 0:
            nterm = length
        elif e == n:
            cterm = length
        else:
            internal = max(internal, length)
    return internal, nterm, cterm


# ---------------------------------------------------------------------------
# 6. Identity
# ---------------------------------------------------------------------------


def species_mnemonic(entry_name: str) -> str:
    """'1433B_HUMAN' -> 'human' (UniProt entry-name species mnemonic)."""
    return entry_name.rsplit("_", 1)[-1].lower() if "_" in (entry_name or "") else ""


def _mmseqs_workdir(seqs: Dict[str, str], cache_dir: Path, tag: str, params: str) -> Path:
    """Cache directory keyed by a hash of the sequences and parameters."""
    h = hashlib.sha1()
    for k in sorted(seqs):
        h.update(f"{k}\t{seqs[k]}\n".encode())
    h.update(f"{tag}|{params}".encode())
    return cache_dir / "mmseqs" / f"{tag}_{h.hexdigest()[:16]}"


def _run_mmseqs(seqs: Dict[str, str], workdir: Path, mmseqs_bin: str, args_: List[str]) -> None:
    exe = shutil.which(mmseqs_bin)
    if exe is None:
        sys.exit(f"MMseqs2 binary '{mmseqs_bin}' not found: conda install -c bioconda mmseqs2")
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "input.fasta"
    with open(fasta, "w") as fh:
        for k, s in seqs.items():
            fh.write(f">{k}\n{s}\n")
    cmd = [exe, args_[0], str(fasta)] + [a.replace("{fasta}", str(fasta)) for a in args_[1:]]
    subprocess.run(cmd, check=True)
    shutil.rmtree(workdir / "tmp", ignore_errors=True)


def run_mmseqs_cluster(seqs: Dict[str, str], cache_dir: Path, min_seq_id: float, coverage: float,
                       cov_mode: int, threads: int, mmseqs_bin: str = "mmseqs") -> Dict[str, str]:
    """Cluster {id: sequence} with MMseqs2 easy-cluster; return {member_id: cluster_id}.
    Results are cached by a hash of the inputs and parameters."""
    workdir = _mmseqs_workdir(seqs, cache_dir, "cluster", f"{min_seq_id}|{coverage}|{cov_mode}")
    result = workdir / "clu_cluster.tsv"
    if not result.exists():
        log.info("MMseqs2: clustering %d sequences (min id %.2f, coverage %.2f, cov-mode %d)",
                 len(seqs), min_seq_id, coverage, cov_mode)
        _run_mmseqs(seqs, workdir, mmseqs_bin,
                    ["easy-cluster", str(workdir / "clu"), str(workdir / "tmp"),
                     "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode),
                     "--threads", str(threads), "-v", "1"])
    clu = pd.read_csv(result, sep="\t", header=None, names=["cluster", "member"], dtype=str)
    return dict(zip(clu["member"], clu["cluster"]))


def run_mmseqs_pairs(seqs: Dict[str, str], cache_dir: Path, min_seq_id: float, coverage: float,
                     cov_mode: int, threads: int, mmseqs_bin: str = "mmseqs",
                     sensitivity: float = 7.5, max_seqs: int = 1000) -> pd.DataFrame:
    """All-against-all MMseqs2 search of {id: sequence}, applying the same identity and
    coverage thresholds as the clustering. Returns the passing non-self hits as a DataFrame
    with columns query, target, fident, qcov, tcov (both directions may appear)."""
    workdir = _mmseqs_workdir(seqs, cache_dir, "verify",
                              f"{min_seq_id}|{coverage}|{cov_mode}|{sensitivity}|{max_seqs}")
    result = workdir / "hits.m8"
    cols = ["query", "target", "fident", "qcov", "tcov"]
    if not result.exists():
        log.info("MMseqs2: all-vs-all verification search on %d sequences", len(seqs))
        _run_mmseqs(seqs, workdir, mmseqs_bin,
                    ["easy-search", "{fasta}", str(result), str(workdir / "tmp"),
                     "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode),
                     "-s", str(sensitivity), "--max-seqs", str(max_seqs),
                     "--format-output", ",".join(cols), "--threads", str(threads), "-v", "1"])
    if result.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    hits = pd.read_csv(result, sep="\t", header=None, names=cols, dtype={"query": str, "target": str})
    return hits[hits["query"] != hits["target"]].reset_index(drop=True)


def priority_order(df: pd.DataFrame) -> List:
    """Index of df sorted by representative priority: preferred species, reviewed,
    has a PDB entry, longer sequence, then protein_id for determinism."""
    rank = {s.lower(): i for i, s in enumerate(PREFERRED_SPECIES) if isinstance(s, str)}
    key = pd.DataFrame({
        "species_rank": df["id"].map(lambda x: rank.get(species_mnemonic(x), len(rank))),
        "unreviewed": ~df["reviewed"].str.lower().eq("reviewed"),
        "no_pdb": df["pdb_ids"].eq(""),
        "neg_len": -df["length"],
        "pid": df["protein_id"],
    }, index=df.index)
    return key.sort_values(["species_rank", "unreviewed", "no_pdb", "neg_len", "pid"]).index.tolist()


def pick_cluster_representatives(df: pd.DataFrame, cluster_col: str) -> pd.Series:
    """True for the one protein kept per cluster, the first in priority_order."""
    first = df.loc[priority_order(df)].drop_duplicates(cluster_col).index
    return pd.Series(df.index.isin(first), index=df.index)


def resolve_redundant_pairs(df: pd.DataFrame, hits: pd.DataFrame) -> Dict:
    """Greedy pass in priority order: keep a protein only if none of its hits is already
    kept. Hits are treated as undirected (a pair counts if either search direction passes),
    so no two kept proteins share a passing hit. Returns {dropped_index: kept protein_id}."""
    nbrs: Dict[str, Set[str]] = defaultdict(set)
    for q, t in zip(hits["query"], hits["target"]):
        nbrs[q].add(t)
        nbrs[t].add(q)
    kept: Set[str] = set()
    dropped = {}
    for idx in priority_order(df):
        pid = df.at[idx, "protein_id"]
        clash = sorted(nbrs[pid] & kept)
        if clash:
            dropped[idx] = clash[0]
        else:
            kept.add(pid)
    return dropped


# ---------------------------------------------------------------------------
# 7. Structures
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


# --- structure parsing (PDB and mmCIF) --------------------------------------
# Residues are (one-letter code, C-alpha xyz or None, heavy-atom xyz list), in file order,
# from the first model, ATOM records only, altloc blank/A. Coordinates in Angstrom.

_CIF_TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'|"(?:[^"]|"(?=\S))*"|\S+""")


def _cif_atom_rows(path: Path) -> Iterable[dict]:
    """Rows of the _atom_site loop of an mmCIF file as {field: value} dicts."""
    fields: List[str] = []
    header = in_atoms = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not in_atoms:
                if s == "loop_":
                    header, fields = True, []
                    continue
                if not header:
                    continue
                if s.startswith("_atom_site."):
                    fields.append(s.split(".", 1)[1].split()[0])
                    continue
                if not fields or s.startswith("_"):
                    header = False
                    continue
                in_atoms = True  # first data row of the _atom_site loop
            if not s or s[0] in "#_" or s.startswith("loop_") or s.startswith("data_"):
                return
            toks = [t[1:-1] if len(t) > 1 and t[0] in "'\"" and t[-1] == t[0] else t
                    for t in _CIF_TOKEN.findall(s)]
            if len(toks) == len(fields):
                yield dict(zip(fields, toks))


def _atoms(path: Path) -> Iterable[Tuple[str, tuple, str, str, str, List[float]]]:
    """(chain, residue key, residue name, atom name, element, xyz) for first-model ATOM records."""
    if str(path).endswith(".cif"):
        model = None
        for a in _cif_atom_rows(path):
            if a.get("group_PDB") != "ATOM":
                continue
            m = a.get("pdbx_PDB_model_num", "1")
            model = m if model is None else model
            if m != model or a.get("label_alt_id", ".") not in (".", "?", "A"):
                continue
            chain = a.get("auth_asym_id") or a.get("label_asym_id", "")
            ins = a.get("pdbx_PDB_ins_code", "?")
            yield (chain, (a.get("auth_seq_id") or a.get("label_seq_id", ""), "" if ins in "?." else ins),
                   a.get("auth_comp_id") or a.get("label_comp_id", ""),
                   a.get("auth_atom_id") or a.get("label_atom_id", ""), a.get("type_symbol", ""),
                   [float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"])])
        return
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                return
            if not line.startswith("ATOM  ") or len(line) < 54 or line[16] not in (" ", "A"):
                continue
            yield (line[21].strip(), (line[22:26], line[26]), line[17:20].strip(), line[12:16].strip(),
                   line[76:78].strip() if len(line) >= 78 else "",
                   [float(line[30:38]), float(line[38:46]), float(line[46:54])])


def read_chains(path: Path) -> Dict[str, List[Tuple[str, Optional[list], list]]]:
    """{chain: residues} for a .pdb or .cif file (hydrogens skipped)."""
    chains: Dict[str, List[Tuple[str, Optional[list], list]]] = {}
    last: Dict[str, tuple] = {}
    for chain, key, resname, name, elem, xyz in _atoms(path):
        if (elem or name.lstrip("0123456789")[:1]).upper() in ("H", "D"):
            continue
        res = chains.setdefault(chain, [])
        if last.get(chain) != key:
            last[chain] = key
            res.append((THREE_TO_ONE.get(resname, "X"), None, []))
        if name == "CA":
            res[-1] = (res[-1][0], xyz, res[-1][2])
        res[-1][2].append(xyz)
    return chains


def check_structure_match(fpath: Path, up_seq: str = "") -> Tuple[str, float, int]:
    """Inspects a structure and returns (best_chain, match_score, chain_length).
    Score = fraction of the chain's 5-mers found in the UniProt sequence."""
    if not fpath.exists():
        return "A", 0.0, 0
    try:
        chain_seqs = {c: "".join(r[0] for r in res) for c, res in read_chains(fpath).items()}
    except (OSError, ValueError):
        return "A", 0.0, 0
    if not chain_seqs:
        return "A", 0.0, 0

    def score_of(s: str) -> float:
        kmers = [s[i:i + 5] for i in range(len(s) - 4)]
        return sum(1 for k in kmers if k in up_seq) / len(kmers) if kmers else 0.0

    # If chain A exists and matches UniProt sequence with >= 90% 5-mer hits, keep 'A'
    if "A" in chain_seqs and up_seq and len(chain_seqs["A"]) >= 20:
        score = score_of(chain_seqs["A"])
        if score >= 0.90:
            return "A", score, len(chain_seqs["A"])

    best_chain, best_score, best_len = "A", -1.0, 0
    for c, s in chain_seqs.items():
        if len(s) < 10:
            continue
        score = score_of(s) if up_seq else 1.0
        if score > best_score or (abs(score - best_score) < 0.05 and len(s) > best_len):
            best_chain, best_score, best_len = c, score, len(s)
    return best_chain, max(best_score, 0.0), best_len


def resolve_chain_id(fpath: Path, up_seq: str = "") -> str:
    """Detect polypeptide chain matching the UniProt sequence in a PDB or CIF structure."""
    chain, _, _ = check_structure_match(fpath, up_seq)
    return chain


# --- structure cache ---------------------------------------------------------
# <cache-dir>/structures/raw/{alphafold,pdb}/<file as served>   immutable downloads
# <cache-dir>/structures/selection.json                         chosen file + chain per protein
# <cache-dir>/structures/clean/<pid>__<raw stem>_<chain>.pdb   header (kept chain's
#                                                              per-chain records), one chain,
#                                                              first model, ATOM records only
# <outdir>/structures/<pid>.pdb|.cif                           hard link to the clean file
# Hard links rather than symlinks: they need no special rights on Windows and tar archives
# them as ordinary files. They fail across filesystems, in which case the file is copied.
# Because a hard link shares the cache file's data, edit output structures only via a copy.


def raw_structure_path(url: str, cache_dir: Path) -> Path:
    sub = "alphafold" if "alphafold" in url else "pdb"
    return cache_dir / "structures" / "raw" / sub / url.rsplit("/", 1)[-1]


def select_structures(df: pd.DataFrame, args) -> Dict:
    """Choose and download the structure for each row of df, in --structure-source-pref order:
    up to 5 PDB entries (.pdb), then up to 3 as .cif, and the AlphaFold model. A PDB entry is
    accepted only if one of its chains matches the UniProt sequence (>= 50% of 5-mers, >= 20
    residues). Returns {df index: (raw file Path, chain)}; successful choices are cached."""
    af = fetch_alphafold(df["accession"].tolist(), args.cache_dir / "alphafold.json", args.workers)
    sel_file = args.cache_dir / "structures" / "selection.json"
    sel_file.parent.mkdir(parents=True, exist_ok=True)
    cache = load_json(sel_file)
    chosen, todo = {}, {}
    for idx, r in df.iterrows():
        info = af.get(r["accession"], {})
        af_url = ((info.get("pdb_url") or f"https://alphafold.ebi.ac.uk/files/AF-{r['accession']}-F1-model_v6.pdb")
                  if info.get("exists") else "")
        pdb_ids = split_terms(r["pdb_ids"])
        key = "|".join([args.structure_source_pref, ",".join(pdb_ids[:5]), af_url, seq_key(r["sequence"])])
        hit = cache.get(key)
        if hit and raw_structure_path(hit[0], args.cache_dir).exists():
            chosen[idx] = (raw_structure_path(hit[0], args.cache_dir), hit[1])
            continue
        pdb_urls = ([f"https://files.rcsb.org/download/{p}.pdb" for p in pdb_ids[:5]]
                    + [f"https://files.rcsb.org/download/{p}.cif" for p in pdb_ids[:3]])
        af_urls = [af_url] if af_url else []
        todo[idx] = (key, pdb_urls + af_urls if args.structure_source_pref == "pdb" else af_urls + pdb_urls,
                     r["sequence"])
    if todo:
        log.info("Structures: selecting and downloading for %d proteins (%d cached)", len(todo), len(chosen))
        session = http_session()

        def one(idx):
            key, urls, up_seq = todo[idx]
            for url in urls:
                raw = raw_structure_path(url, args.cache_dir)
                if not download_file(url, raw, session):
                    continue
                chain, score, clen = check_structure_match(raw, up_seq)
                if "rcsb.org" in url and up_seq and (score < 0.50 or clen < 20):
                    continue  # stays cached; it just isn't this protein's chain
                return idx, key, url, chain
            return idx, key, None, None

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as ex:
            for fut in as_completed([ex.submit(one, i) for i in todo]):
                idx, key, url, chain = fut.result()
                if url:
                    cache[key] = [url, chain]
                    chosen[idx] = (raw_structure_path(url, args.cache_dir), chain)
                done += 1
                if done % 500 == 0:
                    log.info("  %d/%d processed", done, len(todo))
                    save_json(cache, sel_file)
        save_json(cache, sel_file)
    return chosen


# Header records that describe one chain; only those for the kept chain are retained.
# Value: 0-based column of the chain ID in that record.
_CHAIN_HEADER_COL = {"DBREF ": 12, "DBREF1": 12, "DBREF2": 12, "SEQRES": 11, "SEQADV": 16,
                     "MODRES": 16, "HET   ": 12}
# Coordinate-section and bookkeeping records; never copied from the header pass.
_COORD_RECORDS = {"MODEL ", "ATOM  ", "HETATM", "ANISOU", "TER   ", "ENDMDL", "CONECT", "MASTER", "END   ",
                  "END"}


def clean_pdb(raw: Path, chain: str, dest: Path) -> None:
    """Write the header and the first model's ATOM records of one chain (altloc blank/A only)
    to dest. Header records are kept as-is, except per-chain records (DBREF, SEQRES, SEQADV,
    MODRES, HET), which are kept only for this chain, so e.g. the DBREF accession is the kept
    chain's. HETATM/CONECT records are dropped, and MASTER, whose counts would be wrong."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = chain if chain else " "  # check_structure_match reports a blank chain ID as ""
    out = []
    in_coords = False
    with open(raw, errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            rec = line[:6].ljust(6)
            if rec == "ENDMDL":
                break
            if rec in _COORD_RECORDS or line.strip() == "END":
                in_coords = True
                if rec == "ATOM  " and len(line) > 26 and line[21] == want and line[16] in (" ", "A"):
                    out.append(line[:16] + " " + line[17:] + "\n")
                continue
            if in_coords:
                continue  # non-coordinate records after the coordinates start are not header
            col = _CHAIN_HEADER_COL.get(rec)
            if col is not None and (line[col] if len(line) > col else " ") != want:
                continue
            out.append(line + "\n")
    out += ["TER\n", "END\n"]
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text("".join(out))
    tmp.replace(dest)


def link_or_copy(src_file: Path, dst: Path) -> str:
    """Hard-link src_file to dst, falling back to a copy. Returns 'link' or 'copy'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src_file, dst)
        return "link"
    except OSError:
        shutil.copy2(src_file, dst)
        return "copy"


# --- compactness -------------------------------------------------------------


def require_numpy(what: str) -> None:
    # Module-level check: main() imports numpy locally later on, which makes `np` a local name there.
    if np is None:
        sys.exit(f"{what} needs numpy: pip install numpy")


V_RES_NM3 = 0.135  # mean residue volume: ~110 Da at a protein density of ~1.35 g/cm^3


def compactness_geometry(path: Path, chain: str, up_seq: str, start: int, end: int) -> dict:
    """Extents (nm) along the C-alpha principal axes of the residues of `chain` that align to
    UniProt positions [start, end), measured on all their heavy atoms. Residues are mapped by
    sequence alignment, so gaps, offsets, tags and point mutations in experimental chains are
    handled; unaligned residues (tags, mutated positions) are left out.
    Returns {"ext", "n"} or {"note"} when the structure cannot be measured."""
    chains = read_chains(path)
    res = chains.get(chain)
    if res is None and len(chains) == 1:
        res = next(iter(chains.values()))
    if not res:
        return {"note": "chain not found"}
    mseq = "".join(r[0] for r in res)
    sm = difflib.SequenceMatcher(None, mseq, up_seq, autojunk=False)
    sel = [res[a + k] for a, b, size in sm.get_matching_blocks() if size >= 3
           for k in range(size) if start <= b + k < end]
    ca = [r[1] for r in sel if r[1] is not None]
    if len(ca) < 3:
        return {"note": "fewer than 3 aligned C-alpha atoms after trimming"}
    ca = np.array(ca, dtype=float)
    center = ca.mean(axis=0)
    _, _, vt = np.linalg.svd(ca - center, full_matrices=False)
    proj = (np.array([xyz for r in sel for xyz in r[2]], dtype=float) - center) @ vt.T
    ext = (proj.max(axis=0) - proj.min(axis=0)) / 10.0
    return {"ext": [round(float(e), 4) for e in ext], "n": int(len(ca))}


def box_ratio(ext: Sequence[float], n_res: int, pad: float) -> Tuple[float, float]:
    """(V_box, V_box / V_ref) in nm^3 for a rectangular cell with padding pad (nm)."""
    v_box = math.prod(e + 2 * pad for e in ext)
    d = (6.0 * V_RES_NM3 * n_res / math.pi) ** (1.0 / 3.0)
    return v_box, v_box / (d + 2 * pad) ** 3


# ---------------------------------------------------------------------------
# 8. Class imbalance
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
    f.add_argument("--monomer-mode", choices=["strict", "lenient", "supported"], default="strict",
                   help="strict: UniProt says monomer and nothing else; "
                        "lenient: also 'monomer and homodimer'-style mixed annotations; "
                        "supported: no complex evidence AND monomer support (tiers 1-3, see top of file; "
                        "tiers 1 and 3 need --pdb-oligomer-check). All three are evaluated and written "
                        "to annotations.tsv either way")
    f.add_argument("--pdb-hetero-as-complex", action="store_true",
                   help="supported mode: treat PDB assemblies with a real protein partner as complex "
                        "evidence (off by default: PDB over-represents crystallised transient complexes)")
    f.add_argument("--min-partner-len", type=int, default=50,
                   help="minimum length of a PDB partner protein for --pdb-hetero-as-complex (default 50)")
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
    f.add_argument("--max-disordered", type=int, default=10,
                   help="drop proteins with a non-terminal run of more than this many consecutive residues "
                        "whose disorder score exceeds --disorder-threshold (default 10; -1 disables)")
    f.add_argument("--max-terminal-disordered", type=int, default=25,
                   help="optionally also drop proteins whose N- or C-terminal disordered tail is longer than this (default 25; -1 disables)")
    f.add_argument("--disorder-threshold", "--disorder-threshhold", dest="disorder_threshold",
                   type=float, default=0.5,
                   help="metapredict score above which a residue counts as disordered (default 0.5)")
    f.add_argument("--box-padding", type=float, default=1.5,
                   help="solvent padding (nm) on each side of the rectangular cell for the compactness "
                        "filter (default 1.5)")
    f.add_argument("--max-box-volume", type=float, default=1200.0,
                   help="drop proteins whose padded principal-axis box volume exceeds this value (default 1200 nm^3)")
    f.add_argument("--min-seq-id", type=float, default=0.4,
                   help="MMseqs2 identity threshold for redundancy clustering (default 0.4; 0 disables)")
    f.add_argument("--cluster-coverage", type=float, default=0.8,
                   help="MMseqs2 -c alignment coverage (default 0.8)")
    f.add_argument("--cluster-cov-mode", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                   help="MMseqs2 --cov-mode (default 0: coverage of both query and target)")
    f.add_argument("--mmseqs", default="mmseqs", help="path to the MMseqs2 binary")
    f.add_argument("--skip-identity-verify", action="store_true",
                   help="skip the all-vs-all check that no two kept proteins exceed the identity threshold")
    f.add_argument("--dedup-name-prefix", action="store_true",
                   help="also keep at most one protein per UniProt entry-name prefix (CATA_HUMAN, "
                        "CATA_MOUSE -> one), chosen by the same priority as cluster representatives")
    f.add_argument("--verify-sensitivity", type=float, default=7.5,
                   help="MMseqs2 -s for the verification search (default 7.5, the most sensitive preset)")
    
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
    # <outdir>/structures only holds links/copies of cached files, so it is rebuilt every run;
    # this keeps proteins dropped by the current settings out of it.
    struct_dir = args.outdir / "structures"
    if struct_dir.exists():
        if any(Path(d).resolve() == struct_dir.resolve() for d in args.local_structure_dir):
            log.warning("Not clearing %s: it is listed in --local-structure-dir", struct_dir)
        else:
            log.info("Clearing %s", struct_dir)
            shutil.rmtree(struct_dir)
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

    ssc = man["cc_subunit"].map(subunit_self_class)
    man["subunit_self_class"] = ssc.map(lambda x: x[0])
    man["subunit_self_clause"] = ssc.map(lambda x: x[1])
    man["complex_portal"] = man["xref_complexportal"].map(lambda s: ";".join(split_terms(s)))

    man["pdb_oligomer"] = ""
    man["pdb_oligomer_votes"] = ""
    man["pdb_complex"] = ""
    man["pdb_complex_votes"] = ""
    if args.pdb_oligomer_check:
        alive = man["drop_reason"].eq("") & man["pdb_ids"].ne("")
        ids = sorted({p for s in man.loc[alive, "pdb_ids"] for p in split_terms(s)[:args.max_pdb_per_protein]})
        entries = fetch_pdb_entries(ids, args.cache_dir / "rcsb_entries_v2.json")
        states = {k: v["states"] for k, v in entries.items()}
        votes = man.loc[alive, "pdb_ids"].map(
            lambda s: pdb_oligomer_vote(split_terms(s), states, args.max_pdb_per_protein))
        man.loc[alive, "pdb_oligomer"] = votes.map(lambda x: x[0])
        man.loc[alive, "pdb_oligomer_votes"] = votes.map(lambda x: x[1])
        cvotes = man.loc[alive].apply(
            lambda r: pdb_complex_vote(split_terms(r["pdb_ids"]), entries, r["accession"],
                                       args.max_pdb_per_protein, args.min_partner_len,
                                       args.pdb_hetero_as_complex), axis=1)
        man.loc[alive, "pdb_complex"] = cvotes.map(lambda x: x[0])
        man.loc[alive, "pdb_complex_votes"] = cvotes.map(lambda x: x[1])

    def supported_tier(r) -> Tuple[int, str]:
        """0 = drop; 1-3 = monomer tier (lower is stronger evidence)."""
        cls, cp, pdbv = r["subunit_self_class"], r["complex_portal"], r["pdb_complex"]
        # vetoes: positive evidence of a stable complex
        if cls == "self_oligomer":
            return 0, "UniProt: " + r["subunit_self_clause"]
        if cp:
            return 0, f"Complex Portal: {cp}"
        if pdbv == "homo-oligomer":
            return 0, "PDB assemblies mostly homo-oligomeric"
        if pdbv == "hetero-complex":
            return 0, "PDB assemblies mostly with a protein partner"
        # monomer support
        if cls == "self_monomer":
            if pdbv == "monomer":
                return 1, "UniProt monomer, PDB monomer"
            return 2, "UniProt monomer" + ("" if pdbv in ("", "none") else f", PDB {pdbv} (neutral)")
        if pdbv == "monomer":
            what = {"mixed": "UniProt monomer/oligomer equilibrium", "partner_only": "UniProt lists partners only",
                    "unknown": "no UniProt subunit annotation"}[cls]
            return 3, f"PDB monomer; {what}"
        if cls == "mixed":
            return 0, "UniProt monomer/oligomer equilibrium without PDB monomer support"
        return 0, "no monomer evidence" + ("" if args.pdb_oligomer_check else " (PDB check off)")

    tiers = man.apply(supported_tier, axis=1)
    man["monomer_tier"] = tiers.map(lambda x: x[0] if x[0] else "")
    if args.monomer_mode == "supported" and not args.pdb_oligomer_check:
        log.warning("--monomer-mode supported without --pdb-oligomer-check: only tier 2 "
                    "(UniProt monomer) can be reached")

    def is_monomer(r, mode: str) -> Tuple[bool, str]:
        if mode == "supported":
            t, why = supported_tier(r)
            return t > 0, (f"tier {t}: {why}" if t else why)
        cls, pdbv = r["subunit_class"], r["pdb_oligomer"]
        ok_text = cls == "monomer" or (mode == "lenient" and cls == "mixed")
        if ok_text:
            if args.pdb_oligomer_check and pdbv == "homo-oligomer" and mode == "strict":
                return False, "UniProt monomer but PDB assemblies homo-oligomeric"
            return True, "uniprot" + ("+pdb" if pdbv == "monomer" else "")
        if cls in ("unknown", "interacts") and args.allow_unknown_subunit:
            if not args.pdb_oligomer_check:
                return True, "no subunit annotation (allowed)"
            if pdbv == "monomer":
                return True, "pdb assembly monomer"
            return False, "no subunit annotation and no PDB monomer evidence"
        return False, f"subunit annotation: {cls}"

    alive = man["drop_reason"].eq("")
    mode_comparison = {}
    for mode in ("strict", "lenient", "supported"):
        res = man.apply(lambda r: is_monomer(r, mode), axis=1)
        man[f"monomer_keep_{mode}"] = res.map(lambda x: x[0])
        if mode == args.monomer_mode:
            man["monomer_evidence"] = res.map(lambda x: x[1])
        mode_comparison[mode] = int((alive & man[f"monomer_keep_{mode}"]).sum())
    mode_comparison["supported_by_tier"] = {
        str(t): int((alive & man["monomer_tier"].eq(t)).sum()) for t in (1, 2, 3)}
    log.info("Monomer step would keep (of %d): %s", int(alive.sum()), mode_comparison)
    drop(~man[f"monomer_keep_{args.monomer_mode}"], "not monomer")
    record(f"monomer ({args.monomer_mode})", man[man.drop_reason.eq("")])

    # ---------------- 5. disorder ----------------
    for c in ("disorder_internal_max", "disorder_nterm", "disorder_cterm"):
        man[c] = float("nan")
    alive = man["drop_reason"].eq("")
    if args.max_disordered >= 0 and alive.any():
        seqs = man.loc[alive, "sequence"]
        scores = predict_disorder_scores(seqs.tolist(), args.cache_dir)
        runs = seqs.map(lambda s: disorder_runs(scores[seq_key(s)], args.disorder_threshold))
        man.loc[alive, "disorder_internal_max"] = runs.map(lambda r: r[0])
        man.loc[alive, "disorder_nterm"] = runs.map(lambda r: r[1])
        man.loc[alive, "disorder_cterm"] = runs.map(lambda r: r[2])
        drop(man["disorder_internal_max"].fillna(0) > args.max_disordered,
             f"internal disordered region > {args.max_disordered} aa")
        if args.max_terminal_disordered >= 0:
            tail = man[["disorder_nterm", "disorder_cterm"]].fillna(0).max(axis=1)
            drop(tail > args.max_terminal_disordered,
                 f"terminal disordered tail > {args.max_terminal_disordered} aa")
    record("disorder", man[man.drop_reason.eq("")])

    # ---------------- 5b. compactness ----------------
    # Each protein's structure is chosen here (same preference order as the download step, which
    # reuses the choice), so the filter measures the structure that will be written out.
    man["box_size"] = ""
    selected: Dict = {}
    alive = man["drop_reason"].eq("")
    if args.max_box_volume > 0 and alive.any():
        require_numpy("the size filter")
        if args.max_disordered < 0:
            log.warning("Disorder filter disabled: size is measured without trimming terminal tails")
        sub = man.loc[alive]
        selected.update(select_structures(sub, args))
        geom_file = args.cache_dir / "compactness_geometry_v2.json"
        geom = load_json(geom_file)
        unmeasured = Counter()
        for idx, r in sub.iterrows():
            if idx not in selected:
                unmeasured["no structure"] += 1
                continue
            raw, chain = selected[idx]
            seq = r["sequence"]
            nt = 0 if pd.isna(r["disorder_nterm"]) else int(r["disorder_nterm"])
            ct = 0 if pd.isna(r["disorder_cterm"]) else int(r["disorder_cterm"])
            start, end = nt, len(seq) - ct
            key = f"{raw.name}|{chain}|{start}|{end}|{seq_key(seq)}"
            if key not in geom:
                geom[key] = compactness_geometry(raw, chain, seq, start, end)
            g = geom[key]
            if "ext" not in g:
                unmeasured[g.get("note", "unmeasurable")] += 1
                continue
            man.at[idx, "box_size"] = f"{box_ratio(g['ext'], g['n'], args.box_padding)[0]:.3f}"
        save_json(geom, geom_file)
        volume = pd.to_numeric(man["box_size"], errors="coerce")
        log.info("Size: box volume measured for %d of %d proteins (median %.2f)",
                 int(volume[alive].notna().sum()), int(alive.sum()),
                 float(volume[alive].median()) if volume[alive].notna().any() else float("nan"))
        if unmeasured:
            log.info("Size: kept without measurement: %s", dict(unmeasured))
        # _sl = alive & (volume > args.max_box_volume)
        # for _r,_id in zip(volume[_sl], man[_sl].protein_id):
        #     log.info("Size: %s (%.2f)", _id, _r)
        drop(alive & (volume > args.max_box_volume),
             f"box volume > {args.max_box_volume:g} nm^3 (padding {args.box_padding:g} nm)")
    record("size", man[man.drop_reason.eq("")])

    # ---------------- 6. identity ----------------
    man["identity_cluster"] = ""
    man["identity_rep"] = ""
    alive = man["drop_reason"].eq("")
    if args.min_seq_id > 0 and alive.any():
        sub = man.loc[alive]
        member2cluster = run_mmseqs_cluster(dict(zip(sub["protein_id"], sub["sequence"])), args.cache_dir,
                                            args.min_seq_id, args.cluster_coverage, args.cluster_cov_mode,
                                            args.workers, args.mmseqs)
        man.loc[alive, "identity_cluster"] = sub["protein_id"].map(member2cluster)
        keep = pick_cluster_representatives(man.loc[alive], "identity_cluster")
        rep_of = man.loc[alive][keep].set_index("identity_cluster")["protein_id"]
        man.loc[alive, "identity_rep"] = man.loc[alive, "identity_cluster"].map(rep_of)
        drop(alive & man["identity_rep"].ne(man["protein_id"]),
             f"redundant (>= {args.min_seq_id:.0%} identity to a kept protein)")
        record(f"non-redundant ({args.min_seq_id:.0%} id)", man[man.drop_reason.eq("")])

        # Verification: MMseqs2 only guarantees members resemble their own cluster's MMseqs2
        # representative, and we pick our own representatives, so two kept proteins from
        # different clusters can still pass the thresholds. Search the kept set against itself
        # and greedily drop the lower-priority protein of each remaining pair.
    else:
        record("non-redundant (skipped)", man[man.drop_reason.eq("")])

    # Verification pass, plus optional entry-name-prefix deduplication, resolved together in one
    # greedy pass in priority order. MMseqs2 only guarantees members resemble their own cluster's
    # MMseqs2 representative, and we pick our own representatives, so two kept proteins from
    # different clusters can still pass the thresholds. The search finds such pairs; same-prefix
    # pairs (CATA_HUMAN / CATA_MOUSE) are added as extra edges. No two survivors share an edge.
    man["name_prefix"] = man["id"].where(man["id"].ne(""), man["protein_id"]).map(
        lambda s: s.rsplit("_", 1)[0] if "_" in s else s)
    do_verify = args.min_seq_id > 0 and not args.skip_identity_verify
    if (do_verify or args.dedup_name_prefix) and man["drop_reason"].eq("").any():
        kept_df = man[man["drop_reason"].eq("")]
        edges = [pd.DataFrame(columns=["query", "target"])]
        seq_pairs: Set[frozenset] = set()
        if do_verify:
            hits = run_mmseqs_pairs(dict(zip(kept_df["protein_id"], kept_df["sequence"])), args.cache_dir,
                                    args.min_seq_id, args.cluster_coverage, args.cluster_cov_mode,
                                    args.workers, args.mmseqs, sensitivity=args.verify_sensitivity)
            edges.append(hits[["query", "target"]])
            seq_pairs = {frozenset(p) for p in zip(hits["query"], hits["target"])}
            log.info("Identity verification: %d redundant pairs among kept proteins", len(seq_pairs))
        if args.dedup_name_prefix:
            groups = kept_df.groupby("name_prefix")["protein_id"].apply(list)
            pref_pairs = [pr for ids in groups if len(ids) > 1 for pr in itertools.combinations(ids, 2)]
            edges.append(pd.DataFrame(pref_pairs, columns=["query", "target"]))
            log.info("Entry-name prefix: %d prefixes shared by >1 kept protein (%d pairs)",
                     int((groups.map(len) > 1).sum()), len(pref_pairs))
        dropped = resolve_redundant_pairs(kept_df, pd.concat(edges, ignore_index=True))
        if dropped:
            idx = list(dropped)
            man.loc[idx, "identity_rep"] = [dropped[i] for i in idx]
            by_seq = [i for i in idx if frozenset((man.at[i, "protein_id"], dropped[i])) in seq_pairs]
            drop(man.index.isin(by_seq),
                 f"redundant (>= {args.min_seq_id:.0%} identity to a kept protein; verification pass)")
            drop(man.index.isin(idx), "same entry-name prefix as a kept protein")
        log.info("Verification / prefix pass dropped %d proteins", len(dropped))
        record("verification" + (" + name prefix" if args.dedup_name_prefix else ""),
               man[man.drop_reason.eq("")])


    # ---------------- 7. structures ----------------
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
        _has_af = has_af if args.structure_source_pref == 'alphafold' else ~has_pdb
        drop( _has_af & low.notna() & (low < args.min_plddt), f"Mean pLDDT < {args.min_plddt}")
    record(f"structure ({req})", man[man.drop_reason.eq("")])

    # ---------------- 7b. optional re-splitting ----------------
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

    # ---------------- 8. labels & imbalance ----------------
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
                "membrane_evidence", "subunit_class", "subunit_self_class", "subunit_self_clause", "subunit_text",
                "complex_portal", "pdb_oligomer", "pdb_oligomer_votes", "pdb_complex", "pdb_complex_votes",
                "monomer_keep_strict", "monomer_keep_lenient", "monomer_keep_supported", "monomer_tier", "monomer_evidence", "disorder_internal_max", "disorder_nterm", "disorder_cterm",
                "box_size",
                "identity_cluster", "identity_rep", "name_prefix", "structure_source", "pdb_ids", "alphafold_id", "af_plddt", "local_structure",
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
    extra = ["accession", "length", "monomer_tier", "structure_source", "pdb_ids", "alphafold_id", "af_plddt",
             "sample_weight"]
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
        need = out.index.difference(list(selected))
        if len(need):
            selected.update(select_structures(man.loc[need], args))
        link_modes = Counter()
        for idx in out.index:
            if idx not in selected:
                continue
            raw, chain = selected[idx]
            pid = out.at[idx, "protein_id"]
            if raw.suffix == ".pdb":
                clean = args.cache_dir / "structures" / "clean" / f"{pid}__{raw.stem}_{chain or 'blank'}.pdb"
                if not clean.exists():
                    clean_pdb(raw, chain, clean)
            else:  # mmCIF is passed through uncleaned
                clean = raw
            dest = struct_dir / f"{pid}{raw.suffix}"
            link_modes[link_or_copy(clean, dest)] += 1
            out.at[idx, "pdb_file"] = str(dest)
        log.info("Structures ready for %d / %d proteins in %s (%d hard-linked, %d copied from the cache)",
                 sum(link_modes.values()), len(out), struct_dir, link_modes["link"], link_modes["copy"])

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
        "subunit_self_class_after_membrane_filter": man.loc[
            ~man.drop_reason.isin(["not mapped to current UniProtKB", "membrane protein"])
            & ~man.drop_reason.str.startswith("length"), "subunit_self_class"].value_counts().to_dict(),
        "monomer_mode_comparison": mode_comparison,
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
