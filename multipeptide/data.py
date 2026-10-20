import hashlib
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .columns import clean_seq, detect
from .features import encode_tabular, fit_array, fit_tabular, load_array, resolve_path
from .fetch import Resolver
from .plm import seq_key
from .targets import Target, _clean, infer_kind

VALID_AA = set('ACDEFGHIKLMNPQRSTVWYXBZUO')


# ----------------------------------------------------------------------------- table
def read_table(cfg, log=None):
    """Load any CSV/TSV and work out the role of each column (see columns.py). Explicit data.columns settings win."""
    d = cfg['data']
    assert d.get('csv') and os.path.exists(d['csv']), f"CSV not found: {d.get('csv')!r}"
    df = pd.read_csv(d['csv'], sep=None, engine='python', dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith('Unnamed:')]]
    cols, notes = detect(df, d.get('columns') or {})
    if cols['id'] is None:                                   # no usable id column: make one
        df.insert(0, 'row_id', [f'row{i}' for i in range(len(df))])
        cols['id'] = 'row_id'; notes.append(f"{'id':<15}= none found, using row numbers")
    assert not df[cols['id']].duplicated().any(), \
        f"ids in column {cols['id']!r} are not unique: {df[cols['id']][df[cols['id']].duplicated()].tolist()[:5]}"
    if log:
        for n in notes:
            log('[columns] ' + n)
    return df, cols


def usable_targets(df, cols, cfg):
    """-> ([(name, kind)], [(name, reason)]): target columns whose type is given or can be inferred, and the ones skipped."""
    d, ok, skipped = cfg['data'], [], []
    for name, kind in target_columns(df, cols, cfg):
        try:
            ok.append((name, kind or infer_kind(df[name], d['delimiter'], d['max_classes'])))
        except ValueError as e:
            skipped.append((name, str(e)))
    return ok, skipped


def feature_columns(cfg):
    out = set()
    for sp in cfg.get('features') or []:
        out |= set(sp.get('columns') or []) | ({sp['file_column']} if sp.get('file_column') else set())
    return out


def target_columns(df, cols, cfg):
    """[(name, kind|None)] from data.targets ('col' or 'col:kind'); default = every column that is not a reserved one."""
    spec = cfg['data'].get('targets')
    if spec:
        out = []
        for t in spec:
            name, _, kind = str(t).partition(':')
            assert name in df.columns, f'target column {name!r} not in CSV'
            out.append((name, kind or None))
        return out
    reserved = {v for v in cols.values() if v} | feature_columns(cfg)
    return [(c, None) for c in df.columns if c not in reserved]


# ----------------------------------------------------------------------------- structures
def resolve_structures(df, cols, cfg, log=print):
    """One CA trace per protein (or None). Cached on disk by (sequence, structure-hint columns, source order)."""
    d, s = cfg['data'], cfg['structure']
    os.makedirs(d['cache_dir'], exist_ok=True)
    path = os.path.join(d['cache_dir'], 'structures.pt')
    cache = torch.load(path) if os.path.exists(path) else {}
    resolver = Resolver(cfg, [os.path.dirname(os.path.abspath(d['csv'])), d.get('root') or '.'])
    get = lambda r, k: (r[cols[k]] if cols.get(k) else '').strip()
    out, report = [], []
    for i, r in df.iterrows():
        seq = clean_seq(r[cols['sequence']])
        row = {k: get(r, k) for k in ('structure_file', 'pdb_id', 'chain_id', 'uniprot_id')}
        key = hashlib.md5('|'.join([seq, *row.values(), *resolver.order, str(s.get('min_plddt'))]).encode()).hexdigest()
        if key not in cache:
            hit, notes = resolver.resolve(row, seq, key)
            cache[key] = dict(hit=hit, notes=notes)
        c = cache[key]
        out.append(c['hit'])
        h = c['hit']
        report.append(dict(id=r[cols['id']], status='ok' if h else 'missing', source=h and h['source'],
                           accession=h and h['accession'], chain=h and h['chain'], coverage=h and round(h['coverage'], 3), plddt=h and (None if h.get('plddt') is None else round(h['plddt'], 1)),
                           struct_len=h and h['struct_len'], seq_len=len(seq), notes=' | '.join(c['notes'])))
    torch.save(cache, path)
    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(d['cache_dir'], 'structure_report.tsv'), sep='\t', index=False)
    log('[structures] ' + ', '.join(f'{k}={v}' for k, v in rep.source.fillna('MISSING').value_counts().items()))
    return out


# ----------------------------------------------------------------------------- groups / split
def _kmers(s, k=4):
    return {s[i:i + k] for i in range(len(s) - k + 1)}


GROUP_KMER_THRESHOLD = 0.08      # 4-mer containment above which two proteins count as relatives (~>50-60% identity)


def assign_groups(recs, thr=GROUP_KMER_THRESHOLD):
    """Union-find groups of related proteins: user `group` column if given; else shared PDB entry
    or 4-mer containment >= thr (approx. >50-60% identity; use a group column from MMseqs2 for a strict split)."""
    if all(r.get('user_group') for r in recs):
        for r in recs:
            r['group'] = r['user_group']
        return recs
    n = len(recs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    seen = {}
    for i, r in enumerate(recs):
        keys = []
        if r['struct'] and r['struct']['source'] in ('pdb', 'pdb_search'):
            keys.append(r['struct']['accession'].split('.')[0])
        for k in keys:
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    km = [_kmers(r['seq']) for r in recs]
    inv = {}
    for i, ks in enumerate(km):
        for k in ks:
            inv.setdefault(k, []).append(i)
    for i, ks in enumerate(km):
        for j, c in Counter(j for k in ks for j in inv[k] if j > i).items():
            if c / max(1, min(len(ks), len(km[j]))) >= thr:
                parent[find(i)] = find(j)
    for i, r in enumerate(recs):
        r['group'] = f'g{find(i)}'
    return recs


def assign_split(recs, split_col_values, fractions, seed):
    if split_col_values is not None:
        norm = {'train': 'train', 'training': 'train', 'valid': 'valid', 'val': 'valid', 'validation': 'valid', 'dev': 'valid',
                'test': 'test', 'testing': 'test'}
        for r, v in zip(recs, split_col_values):
            assert v.lower() in norm, f"split value {v!r} not in train/valid/test"
            r['split'] = norm[v.lower()]
        return recs
    rng = np.random.default_rng(seed)
    groups = {}
    for r in recs:
        groups.setdefault(r['group'], []).append(r)
    order = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))
    names, tgt, cnt = ('train', 'valid', 'test'), np.array(fractions) * len(recs), np.zeros(3)
    for g in order:
        k = int(np.argmax((tgt - cnt) / tgt))
        for r in g:
            r['split'] = names[k]
        cnt[k] += len(g)
    return recs


# ----------------------------------------------------------------------------- build records
def attach_features(recs, df, states, base_dirs):
    """Encode every feature group for every protein (tabular now, array files lazily)."""
    for r in recs:
        r['feat'], r['ffile'] = {}, {}
    for st in states:
        if st['kind'] == 'tabular':
            x, ok = encode_tabular(st, df)
            for r in recs:
                r['feat'][st['name']] = x[r['row']] if ok[r['row']] else None
        else:
            for r in recs:
                r['ffile'][st['name']] = resolve_path(df[st['file_column']][r['row']], base_dirs) if st['file_column'] in df.columns else ''


def build_records(cfg, log=print, with_targets=True, feature_state=None, lookup_structures=None):
    """CSV -> (records, fitted targets, fitted feature groups). `feature_state` re-uses statistics fitted by a trained run;
    structures are looked up when structure.use is on unless `lookup_structures` says otherwise."""
    d = cfg['data']
    df, cols = read_table(cfg, log)
    seqs = df[cols['sequence']].map(clean_seq)
    bad = seqs.map(lambda s: len(s) < 5 or not set(s) <= VALID_AA)
    long_ = seqs.str.len() > d['max_len']
    if (bad | long_).any():
        log(f'[data] dropping {int(bad.sum())} invalid and {int(long_.sum())} sequences longer than {d["max_len"]} aa')
    keep = ~(bad | long_)
    df, seqs = df[keep].reset_index(drop=True), seqs[keep].reset_index(drop=True)
    use_struct = cfg['structure']['use'] if lookup_structures is None else lookup_structures
    structs = resolve_structures(df, cols, cfg, log) if use_struct else [None] * len(df)

    recs = []
    for i, r in df.iterrows():
        h = structs[i]
        recs.append(dict(row=i, id=r[cols['id']], seq=seqs[i],
                         struct=None if h is None else dict(source=h['source'], accession=h['accession'], plddt=h.get('plddt')),
                         sxyz=None if h is None else torch.from_numpy(h['xyz']),
                         user_group=r[cols['group']] if cols['group'] else '',
                         ipr=[x for x in r[cols['interpro']].split(d['delimiter']) if x] if cols['interpro'] else []))
    recs = assign_groups(recs)
    base_dirs = [os.path.dirname(os.path.abspath(d['csv'])), d.get('root') or '.']
    if not with_targets:
        attach_features(recs, df, feature_state or [], base_dirs)
        return recs, [], feature_state or []
    assign_split(recs, df[cols['split']].tolist() if cols['split'] else None, d['fractions'], cfg['train']['seed'])
    train_mask = np.array([r['split'] == 'train' for r in recs])
    assert train_mask.sum() > 0, 'no training rows'

    targets = []
    found, skipped = usable_targets(df, cols, cfg)
    for name, why in skipped:
        log(f'[targets] skipping {name!r}: {why}')
    for name, kind in found:
        try:
            t = Target(name, kind)
            t.fit(df[name], train_mask, d['delimiter'], d['min_label_count'])
            targets.append(t)
        except AssertionError as e:                  # e.g. no label frequent enough in the training split
            log(f'[targets] skipping {name!r}: {e}')
    assert targets, 'no usable target columns'
    fstates = feature_state
    if fstates is None:
        fstates = []
        for spec in cfg.get('features') or []:
            fstates.append(fit_tabular(spec, df, train_mask, d['max_classes']) if spec.get('columns')
                           else fit_array(spec, df, train_mask, base_dirs))
    attach_features(recs, df, fstates, base_dirs)
    for st in fstates:
        have = sum(r['feat'][st['name']] is not None for r in recs) if st['kind'] == 'tabular' else sum(bool(r['ffile'][st['name']]) for r in recs)
        log(f"[features] {st['name']}: {st['kind']}, dim={st['dim']}, present for {have}/{len(recs)} proteins")
    enc = {t.name: t.encode(df[t.name], d['delimiter']) for t in targets if t.name in df.columns}
    for r in recs:
        r['y'] = {k: v[0][r['row']] for k, v in enc.items()}
        r['m'] = {k: bool(v[1][r['row']]) for k, v in enc.items()}
    log('[targets] ' + '; '.join(f"{t.name}={t.kind}" + (f"[{t.dim}]" if t.dim > 1 else '') +
                                 f" ({int(sum(r['m'][t.name] for r in recs if t.name in r['m']))} labelled)" for t in targets))
    log('[split] ' + ', '.join(f"{s}={sum(r['split'] == s for r in recs)}" for s in ('train', 'valid', 'test')) +
        f', groups={len({r["group"] for r in recs})}, with structure={sum(r["sxyz"] is not None for r in recs)}/{len(recs)}')
    return recs, targets, fstates


# ----------------------------------------------------------------------------- dataset
def domain_vocab(recs, min_count):
    """InterPro vocabulary from the *training* split only (no leakage); empty when the table has no `interpro` column."""
    c = Counter(i for r in recs if r['split'] == 'train' for i in set(r['ipr']))
    return sorted(i for i, n in c.items() if n >= min_count)


class ProteinSet(Dataset):
    def __init__(self, recs, split, plm=None, train=False, coord_noise=0.0, dom_vocab=None, fstates=()):
        self.recs = [r for r in recs if split is None or r['split'] == split]
        self.plm, self.train, self.noise, self.fstates = plm, train, coord_noise, {st['name']: st for st in fstates}
        self.dix = {x: i for i, x in enumerate(dom_vocab)} if dom_vocab else None

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        sx = r['sxyz']
        if sx is not None and self.train and self.noise > 0:
            sx = sx + self.noise * torch.randn_like(sx)
        it = dict(id=r['id'], group=r['group'], n=len(r['seq']), sxyz=sx, y=r.get('y', {}), m=r.get('m', {}), feat={})
        if self.dix is not None:                               # multi-hot InterPro vector (all zeros for a protein with no known domain)
            v = torch.zeros(len(self.dix))
            for x in r['ipr']:
                if x in self.dix:
                    v[self.dix[x]] = 1
            it['domain'] = v
        if self.plm is not None:
            it['plm'] = self.plm[seq_key(r['seq'])]
        for name, st in self.fstates.items():
            if st['kind'] == 'tabular':
                it['feat'][name] = r['feat'].get(name)
            elif r['ffile'].get(name):
                if sx is not None:                       # arrays are indexed on the structure's residues
                    it['feat'][name] = load_array(st, r['ffile'][name], len(sx))
        return it


def collate(items, targets=(), fstates=(), pad_to=64):
    # lengths are padded up to a multiple of `pad_to`: few distinct tensor shapes keeps GPU kernels cached (large speed-up on Apple MPS)
    up = lambda n: -(-n // pad_to) * pad_to
    B, L = len(items), up(max(it['n'] for it in items))
    Ls = up(max([len(it['sxyz']) for it in items if it['sxyz'] is not None] or [1]))
    out = dict(ids=[it['id'] for it in items], groups=[it['group'] for it in items],
               mask=torch.zeros(B, L, dtype=torch.bool),
               sxyz=torch.zeros(B, Ls, 3), smask=torch.zeros(B, Ls, dtype=torch.bool),
               has_struct=torch.zeros(B, dtype=torch.bool), feat={}, has={})
    if 'plm' in items[0]:
        out['plm'] = torch.zeros(B, L, items[0]['plm'].shape[-1])
    if 'domain' in items[0]:
        out['domain'] = torch.stack([it['domain'] for it in items])
    out['y'] = {t.name: torch.stack([it['y'].get(t.name, torch.zeros(max(t.dim, 1))) for it in items]) for t in targets}
    out['m'] = {t.name: torch.tensor([it['m'].get(t.name, False) for it in items]) for t in targets}
    for st in fstates:
        n = st['name']
        out['has'][n] = torch.tensor([it['feat'].get(n) is not None for it in items])
        if st['kind'] == 'tabular':
            out['feat'][n] = dict(x=torch.zeros(B, 2 * st['dim']))
            continue
        out['feat'][n] = dict(x=torch.zeros(B, Ls, st['dim']), xmask=torch.zeros(B, Ls, dtype=torch.bool), prior=torch.ones(B, Ls))
        if st['has_pair']:
            out['feat'][n]['pair'] = torch.zeros(B, Ls, Ls)
        if st['vec_dim']:
            out['feat'][n]['vec'] = torch.zeros(B, st['vec_dim'])
    for b, it in enumerate(items):
        n_ = it['n']
        out['mask'][b, :n_] = True
        if 'plm' in it:
            out['plm'][b, :n_] = it['plm']
        sx = it['sxyz']
        if sx is None:
            out['smask'][b, 0] = True                      # dummy residue keeps the encoder finite; masked via has_struct
        else:
            m = len(sx)
            out['has_struct'][b] = True
            out['sxyz'][b, :m], out['smask'][b, :m] = sx, True
        for st in fstates:
            f = it['feat'].get(st['name'])
            if f is None:
                continue
            o = out['feat'][st['name']]
            if st['kind'] == 'tabular':
                o['x'][b] = f
                continue
            m = len(f['x'])
            o['x'][b, :m], o['xmask'][b, :m], o['prior'][b, :m] = f['x'], f['xmask'], f['prior']
            if f.get('pair') is not None and 'pair' in o:
                o['pair'][b, :m, :m] = f['pair']
            if f.get('vec') is not None and 'vec' in o:
                o['vec'][b] = f['vec']
    return out


class LengthBuckets(Sampler):
    """Batches of near-equal sequence length (lengths are jittered by `jitter` before sorting so buckets differ every epoch, and
    batch order is shuffled). In-batch negatives of a contrastive loss then cannot be told apart by length -- the one feature that
    a protein's sequence and structure trivially share -- and padding is minimal, which makes training faster."""
    def __init__(self, lengths, batch_size, jitter=0.03, seed=0, drop_last=True):
        self.len, self.bs, self.jit, self.drop = np.asarray(lengths, float), batch_size, jitter, drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = np.argsort(self.len * (1 + self.jit * self.rng.standard_normal(len(self.len))))
        batches = [order[i:i + self.bs].tolist() for i in range(0, len(order), self.bs)]
        if self.drop and len(batches) > 1 and len(batches[-1]) < max(2, self.bs // 2):
            batches.pop()
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = len(self.len)
        return max(1, n // self.bs if self.drop and n >= self.bs else -(-n // self.bs))


def inspect_csv(cfg, log=print):
    """Offline dry run: what was detected, which columns become targets (and of what type), what will be skipped and why."""
    d = cfg['data']
    df, cols = read_table(cfg, log)
    seqs = df[cols['sequence']].map(clean_seq)
    log(f"\n[table] {len(df)} rows, {len(df.columns)} columns; sequence length {seqs.str.len().min()}-{seqs.str.len().max()} "
        f"(median {int(seqs.str.len().median())}); {int((seqs.str.len() > d['max_len']).sum())} longer than max_len={d['max_len']}")
    if cols['interpro']:
        log(f"[domains] InterPro ids in column {cols['interpro']!r}: they guide the structure readout when structure is used")
    hints = [k for k in ('structure_file', 'pdb_id', 'uniprot_id') if cols[k]]
    log('[structures] hints given: ' + (', '.join(hints) if hints else 'none') +
        f" -> sources tried in order: {', '.join(Resolver(cfg, []).order)}")
    if cols['split']:
        log('[split] from column ' + repr(cols['split']) + ': ' + str(df[cols['split']].str.lower().value_counts().to_dict()))
    else:
        log('[split] no split column: group-aware random split ' + str(d['fractions']))
    for sp in cfg.get('features') or []:
        log(f"[features] input group {sp['name']!r}: " + (f"columns {sp['columns']}" if sp.get('columns') else f"files in column {sp['file_column']!r}"))
    found, skipped = usable_targets(df, cols, cfg)
    given = dict(target_columns(df, cols, cfg))
    for name, k in found:
        filled = int((df[name].astype('string').str.strip().fillna('') != '').sum())
        extra = ' ' + str(_clean(df[name]).value_counts().head(5).to_dict()) if k == 'multiclass' else \
            f" positives={int(df[name].astype('string').str.lower().isin(['1', 'true', 'yes', 'y']).sum())}" if k == 'binary' else ''
        log(f"[target] {name!r:<28} {k:<10} {filled}/{len(df)} filled{extra}" + ('' if given[name] else '   (inferred; override with name:type)'))
        if filled < 20:
            log(f"         WARNING: only {filled} labelled rows for {name!r}: expect an unreliable model")
    for name, why in skipped:
        log(f"[skip]   {name!r:<28} {why}")
    ok = [n for n, _ in found]
    if len(df) < 100:
        log('WARNING: fewer than 100 rows. Results will be noisy; rely on the confidence intervals.')
    log(f"\n=> {len(ok)} target column(s) will be modelled: {ok}")
    return ok, len(df)
