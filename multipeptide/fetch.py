"""Resolve one 3D structure per protein from the CSV: local file, PDB id, PDB sequence search,
AlphaFold DB (UniProt accession) or ESMFold. Every candidate is checked against the CSV sequence;
a structure that does not cover the sequence is rejected and the next source is tried."""
import difflib
import os
import re
import time

import requests

from .structure import load_chains, plddt_percent, predicted_by

MIN_STRUCT_FRACTION = 0.5          # the structure may not be dominated by residues that are not in your sequence
PDB_SEARCH_IDENTITY, PDB_SEARCH_ROWS, PDB_SEARCH_MAX_CHAINS = 0.95, 5, 12   # sequence search: strictness; best-ranked entries downloaded; skip huge complexes
ESMFOLD_MAX_LEN = 400                # ESMFold API limit
# Experimental structures first (file you gave -> pdb_id -> PDB sequence search), then AlphaFold DB; ESMFold only when asked to generate.
ORDER = ['file', 'pdb_id', 'pdb_search', 'alphafold']
RCSB_FILES = 'https://files.rcsb.org/download/{}.{}'
RCSB_ENTRY = 'https://data.rcsb.org/rest/v1/core/entry/{}'
RCSB_SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
AF_API = 'https://alphafold.ebi.ac.uk/api/prediction/{}'
ESMFOLD = 'https://api.esmatlas.com/foldSequence/v1/pdb/'


def match(query, struct_seq):
    """(coverage of the query, fraction of the structure explained) from longest-matching-block alignment."""
    sm = difflib.SequenceMatcher(None, query, struct_seq, autojunk=False)
    n = sum(b.size for b in sm.get_matching_blocks())
    return n / max(1, len(query)), n / max(1, len(struct_seq))


def _http(method, url, retries=3, **kw):
    err = None
    for i in range(retries):
        try:
            r = requests.request(method, url, timeout=60, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # noqa: PERF203
            err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f'{url}: {err}')


class Resolver:
    def __init__(self, cfg, base_dirs):
        s = cfg['structure']
        self.order = ORDER + (['esmfold'] if s.get('generate') else [])
        self.min_cov, self.min_frac = s['min_coverage'], MIN_STRUCT_FRACTION
        self.search_identity, self.search_rows, self.search_max_chains = PDB_SEARCH_IDENTITY, PDB_SEARCH_ROWS, PDB_SEARCH_MAX_CHAINS
        self.min_plddt = s.get('min_plddt', 0.0)
        self.esm_max = ESMFOLD_MAX_LEN
        self.dir = os.path.join(cfg['data']['cache_dir'], 'structures')
        self.base_dirs = base_dirs
        os.makedirs(self.dir, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _cached(self, source, acc, ext, url):
        path = os.path.join(self.dir, source, f'{acc}.{ext}')
        if not os.path.exists(path):
            r = _http('GET', url)
            if r is None:
                return None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'wb').write(r.content)
        return path

    def _check(self, path, seq, chain=None):
        """Best chain of `path` for `seq` -> dict or None."""
        best = None
        for c, (s, xyz, bfac) in load_chains(path).items():
            if chain and c != chain:
                continue
            cov, frac = match(seq, s)
            if best is None or cov > best['coverage']:
                best = dict(chain=c, coverage=cov, struct_fraction=frac, xyz=xyz, struct_len=len(s), bfac=bfac)
        if not best:
            return None
        if best['coverage'] < self.min_cov or best['struct_fraction'] < self.min_frac:
            return dict(rejected=f"coverage {best['coverage']:.2f}, struct_fraction {best['struct_fraction']:.2f}")
        # Model predictions (AlphaFold / ESMFold) carry their confidence (pLDDT) in the B-factor column; experimental files do not.
        kind = predicted_by(path)
        best['plddt'] = float(plddt_percent(best.pop('bfac')).mean()) if kind else None
        if kind and self.min_plddt and best['plddt'] < self.min_plddt:
            return dict(rejected=f"{kind} model confidence too low: mean pLDDT {best['plddt']:.1f} < {self.min_plddt:g} (sequence model only)")
        return best

    @staticmethod
    def _accept(hit, source, accession):
        """Outcome of a candidate check -> (resolved structure, '') or (None, reason). `accession` may depend on the hit."""
        if hit and 'xyz' in hit:
            return dict(hit, source=source, accession=accession(hit) if callable(accession) else accession), ''
        return None, hit['rejected'] if hit else 'no CA atoms'

    # ------------------------------------------------------------------ sources
    def from_file(self, row, seq):
        f = row.get('structure_file')
        if not f:
            return None, 'no structure_file'
        for b in self.base_dirs:
            p = f if os.path.isabs(f) else os.path.join(b, f)
            if os.path.exists(p):
                return self._accept(self._check(p, seq, row.get('chain_id') or None), 'file', os.path.basename(p))
        return None, f'file not found: {f}'

    def from_pdb(self, pdb, chain, seq):
        pdb = pdb.upper()
        for ext in ('pdb', 'cif'):
            p = self._cached('pdb', pdb, ext, RCSB_FILES.format(pdb, ext))
            if p:
                return self._accept(self._check(p, seq, chain), 'pdb', lambda h: f'{pdb}.{h["chain"]}')
        return None, f'{pdb} not found on RCSB'

    def _entry_info(self, pdb):
        r = _http('GET', RCSB_ENTRY.format(pdb))
        if r is None:
            return None
        i = r.json().get('rcsb_entry_info', {})
        res = i.get('resolution_combined') or [99.0]
        return dict(chains=i.get('deposited_polymer_entity_instance_count') or 99, resolution=min(res))

    def pdb_search(self, seq):
        """RCSB sequence search. Hits are ranked toward small, high-resolution entries before anything is downloaded, and the chain is
        picked by sequence match (the search reports label chain ids, files use author ids)."""
        if len(seq) < 25:
            return None, 'sequence < 25 aa (RCSB sequence search minimum)'
        body = {'query': {'type': 'terminal', 'service': 'sequence', 'parameters': {
            'evalue_cutoff': 1e-5, 'identity_cutoff': self.search_identity, 'sequence_type': 'protein', 'value': seq}},
            'return_type': 'polymer_entity',
            'request_options': {'paginate': {'start': 0, 'rows': 40}, 'scoring_strategy': 'sequence'}}
        r = _http('POST', RCSB_SEARCH, json=body)
        if r is None or not r.content:
            return None, 'no PDB sequence hit'
        entries = list(dict.fromkeys(h['identifier'].split('_')[0] for h in r.json().get('result_set', [])))
        info = {e: self._entry_info(e) for e in entries}
        cand = sorted((e for e in entries if info[e] and info[e]['chains'] <= self.search_max_chains),
                      key=lambda e: (info[e]['resolution'], info[e]['chains']))[:self.search_rows]
        if not cand:
            return None, f'{len(entries)} PDB hits, none with <= {self.search_max_chains} chains'
        why = []
        for e in cand:
            hit, w = self.from_pdb(e, None, seq)
            if hit:
                return dict(hit, source='pdb_search'), ''
            why.append(f'{e}: {w}')
        return None, '; '.join(why)

    def from_alphafold(self, uniprot, seq):
        r = _http('GET', AF_API.format(uniprot))
        if r is None:
            return None, f'{uniprot} not in AlphaFold DB'
        entry = r.json()[0]
        p = self._cached('alphafold', uniprot, 'pdb', entry['pdbUrl'])
        return self._accept(self._check(p, seq), 'alphafold', uniprot)

    def esmfold(self, seq, key):
        if len(seq) > self.esm_max:
            return None, f'sequence > {self.esm_max} aa (ESMFold API limit)'
        path = os.path.join(self.dir, 'esmfold', f'{key}.pdb')
        if not os.path.exists(path):
            r = _http('POST', ESMFOLD, data=seq, headers={'Content-Type': 'text/plain'})
            if r is None:
                return None, 'ESMFold returned nothing'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'wb').write(r.content)
        return self._accept(self._check(path, seq), 'esmfold', key)

    # ------------------------------------------------------------------ driver
    def resolve(self, row, seq, key):
        notes = []
        for step in self.order:
            try:
                if step == 'file':
                    hit, w = self.from_file(row, seq)
                elif step == 'pdb_id':
                    ids = [x for x in re.split(r'[;,\s]+', row.get('pdb_id') or '') if x]
                    hit, w = None, 'no pdb_id'
                    for i in ids:
                        hit, w = self.from_pdb(i.split('.')[0], row.get('chain_id') or (i.split('.')[1] if '.' in i else None), seq)
                        if hit:
                            break
                elif step == 'pdb_search':
                    hit, w = self.pdb_search(seq)
                elif step == 'alphafold':
                    hit, w = self.from_alphafold(row['uniprot_id'], seq) if row.get('uniprot_id') else (None, 'no uniprot_id')
                elif step == 'esmfold':
                    hit, w = self.esmfold(seq, key)
                else:
                    raise ValueError(f'unknown structure source {step!r}')
            except Exception as e:  # noqa: BLE001  network / parse problems must not kill the batch
                hit, w = None, f'error: {e}'
            if hit:
                return hit, notes
            notes.append(f'{step}: {w}')
        return None, notes
