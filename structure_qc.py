#!/usr/bin/env python3
"""Quality report for the structures of a filtered manifest (PDB and AlphaFold alike).

  python structure_qc.py --manifest filtered_mf/filtered_manifest.tsv --root filtered_mf/.. --out structure_qc.tsv

One row per protein with the C-alpha checks below. A structure that fails any hard check is flagged
`fail` with the reasons; --drop writes a manifest without them. Thresholds are options.

  coverage      residues in the file / UniProt length               hard: >= --min-coverage (0.8)
  seq_match     identity of the file's residues to the UniProt      hard: >= --min-seq-match (0.95), positions compared
                sequence at the aligned positions (same numbering,   after trimming; a file with a different numbering
                or best offset of -50..50 for numbering shifts)      scheme is reported as low, not dropped silently
  breaks        consecutive C-alpha pairs > 4.2 A apart              hard: <= --max-breaks (3)
  clashes       non-neighbouring C-alpha pairs < 2.5 A               hard: 0
  plddt_mean    mean pLDDT                                           hard (AlphaFold only): >= --min-plddt (70)
  plddt_low     fraction of residues with pLDDT < 50                 soft: reported, used for the warning column
  resolution    X-ray / EM resolution from RCSB (--rcsb)             hard (PDB only): <= --max-resolution (3.0 A);
                                                                     NMR has none and is allowed
  af_vs_pdb     C-alpha RMSD of AlphaFold vs PDB on shared residues  soft: >= --warn-rmsd (4 A) is a warning
                (--af-root: folder with AF-<acc>-F1-model_v*.pdb)

Needs numpy and pandas; --rcsb also needs requests and internet access. The C-alpha reader below follows the
first model and the chain named in the manifest.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
       'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M'}


def read_ca(path, chain):
    """[(resseq, aa, xyz, bfactor)] for one chain of a PDB file (first model, altloc blank/A)."""
    out, seen, is_model = [], set(), False
    for line in open(path, errors='ignore'):
        rec = line[:6]
        if rec == 'ENDMDL':
            break
        if rec not in ('ATOM  ', 'HETATM') or line[12:16].strip() != 'CA' or line[16] not in (' ', 'A'):
            continue
        if chain and line[21] != chain:
            continue
        res = line[17:20].strip()
        if rec == 'HETATM' and res != 'MSE':
            continue
        key = line[22:27]
        if key in seen:
            continue
        seen.add(key)
        out.append((int(line[22:26]), AA3.get(res, 'X'), (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                    float(line[60:66] or 0)))
    return out


def is_alphafold(path):
    try:
        return 'ALPHAFOLD' in open(path, errors='ignore').read(4000).upper()
    except OSError:
        return False


def seq_match(recs, seq):
    """Fraction of file residues equal to the UniProt residue at the same number, best over offsets -50..50."""
    best = 0.0
    for off in range(-50, 51):
        hit = tot = 0
        for n, aa, _, _ in recs:
            i = n - 1 + off
            if 0 <= i < len(seq):
                tot += 1
                hit += aa == seq[i]
        if tot:
            best = max(best, hit / len(recs))
    return best


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    return float(np.sqrt(max((a ** 2).sum() + (b ** 2).sum() - 2 * s.sum(), 0) / len(a)))


def rcsb_resolution(pdb_ids, cache_file):
    import requests
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    for p in pdb_ids:
        if p in cache:
            continue
        r = requests.get(f'https://data.rcsb.org/rest/v1/core/entry/{p}', timeout=30)
        d = r.json() if r.ok else {}
        res = (d.get('rcsb_entry_info') or {}).get('resolution_combined') or [None]
        cache[p] = {'resolution': res[0], 'method': (d.get('exptl') or [{}])[0].get('method', '')}
    cache_file.write_text(json.dumps(cache))
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('.'), help='pdb_file paths in the manifest are relative to this folder')
    ap.add_argument('--out', type=Path, default=Path('structure_qc.tsv'))
    ap.add_argument('--drop', type=Path, help='also write the manifest without failing proteins here')
    ap.add_argument('--rcsb', action='store_true', help='look up resolution / method of the PDB entries (internet)')
    ap.add_argument('--af-root', type=Path, help='folder with AlphaFold models to compare experimental structures against')
    ap.add_argument('--min-coverage', type=float, default=0.8)
    ap.add_argument('--min-seq-match', type=float, default=0.95)
    ap.add_argument('--max-breaks', type=int, default=3)
    ap.add_argument('--min-plddt', type=float, default=70.0)
    ap.add_argument('--max-resolution', type=float, default=3.0)
    ap.add_argument('--warn-rmsd', type=float, default=4.0)
    ap.add_argument('--sequences', type=Path, help='TSV with accession and sequence columns (annotations.tsv works); '
                    'without it coverage uses the manifest length and seq_match is skipped')
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, sep='\t', dtype=str).fillna('')
    seqs = {}
    if args.sequences:
        s = pd.read_csv(args.sequences, sep='\t', dtype=str).fillna('')
        seqs = dict(zip(s['accession'], s['sequence']))
    res_info = {}
    if args.rcsb:
        ids = sorted({p for v in man.get('pdb_ids', pd.Series(dtype=str)) for p in v.split(';') if p})
        res_info = rcsb_resolution(ids, args.out.with_suffix('.rcsb.json'))

    rows = []
    for _, r in man.iterrows():
        row = {'protein_id': r['protein_id'], 'split': r['split'], 'structure_source': r.get('structure_source', ''),
               'status': 'ok', 'reasons': '', 'warnings': ''}
        fails, warns = [], []
        path = args.root / r['pdb_file'] if r['pdb_file'] else None
        if path is None or not path.exists():
            row.update(status='no_structure')
            rows.append(row)
            continue
        recs = read_ca(path, r.get('chain_id', ''))
        n = len(recs)
        length = int(r.get('length') or 0) or len(seqs.get(r.get('accession', ''), ''))
        if n == 0:
            row.update(status='fail', reasons='no C-alpha atoms for the chain')
            rows.append(row)
            continue
        xyz = np.array([x[2] for x in recs])
        num = np.array([x[0] for x in recs])
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        adjacent = np.diff(num) == 1
        row['coverage'] = round(n / length, 3) if length else ''
        row['breaks'] = int((d[adjacent] > 4.2).sum())
        dm = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        far = np.abs(num[:, None] - num[None]) > 1
        row['clashes'] = int(((dm < 2.5) & far).sum() // 2)
        af = is_alphafold(path)
        row['kind'] = 'alphafold' if af else 'experimental'
        if af:
            pl = np.array([x[3] for x in recs])
            row['plddt_mean'] = round(float(pl.mean()), 2)
            row['plddt_low'] = round(float((pl < 50).mean()), 3)
            if row['plddt_mean'] < args.min_plddt:
                fails.append(f"mean pLDDT {row['plddt_mean']} < {args.min_plddt}")
            if row['plddt_low'] > 0.3:
                warns.append(f"{row['plddt_low']:.0%} of residues have pLDDT < 50")
        else:
            infos = [res_info[p] for p in r.get('pdb_ids', '').split(';') if p in res_info]
            if infos:
                res = [i['resolution'] for i in infos if i['resolution']]
                row['resolution'] = min(res) if res else ''
                if res and min(res) > args.max_resolution:
                    fails.append(f'best resolution {min(res)} A > {args.max_resolution}')
        seq = seqs.get(r.get('accession', ''))
        if seq:
            row['seq_match'] = round(seq_match(recs, seq), 3)
            if row['seq_match'] < args.min_seq_match:
                fails.append(f"sequence match {row['seq_match']} < {args.min_seq_match}")
        if row['coverage'] != '' and row['coverage'] < args.min_coverage:
            fails.append(f"coverage {row['coverage']} < {args.min_coverage}")
        if row['breaks'] > args.max_breaks:
            fails.append(f"{row['breaks']} chain breaks")
        if row['clashes']:
            fails.append(f"{row['clashes']} C-alpha clashes")
        if args.af_root and not af and r.get('accession'):
            hits = sorted(args.af_root.glob(f"AF-{r['accession']}-F1-model_v*.pdb"))
            if hits:
                af_recs = {x[0]: x[2] for x in read_ca(hits[-1], 'A')}
                shared = [(x[2], af_recs[x[0]]) for x in recs if x[0] in af_recs]
                if len(shared) >= 20:
                    row['af_vs_pdb_rmsd'] = round(kabsch_rmsd(np.array([a for a, _ in shared]), np.array([b for _, b in shared])), 2)
                    if row['af_vs_pdb_rmsd'] >= args.warn_rmsd:
                        warns.append(f"AlphaFold and PDB differ by {row['af_vs_pdb_rmsd']} A RMSD")
        row.update(status='fail' if fails else 'ok', reasons='; '.join(fails), warnings='; '.join(warns))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep='\t', index=False)
    print(out['status'].value_counts().to_string())
    if args.drop:
        keep = set(out.loc[out.status != 'fail', 'protein_id'])
        man[man.protein_id.isin(keep)].to_csv(args.drop, sep='\t', index=False)
        print(f'wrote {args.drop} ({len(keep)} of {len(man)} proteins)')
    print(f'report: {args.out}')


if __name__ == '__main__':
    sys.exit(main())
