"""Extra input modalities from any columns of the table -- nothing here is specific to one kind of data.

A feature group is declared in config (`features:`) or on the command line (`--features`, `--feature-files`):
  tabular : {name, columns: [colA, colB, ...]}   numeric columns become standardised values (+ missing-value flags),
                                                  text columns become one-hot categories. One vector per protein.
  array   : {name, file_column: col}             a column of paths to .npz files, one per protein, indexed on the structure's C-alpha residues:
              X     [L, D]    (required)   per-residue values (summarise repeated measurements into channels, e.g. mean and std)
              pair  [L, L]    (optional)   pairwise residue values -> extra graph edges
              prior [L]       (optional)   residue weights in [0,1] -> weighted pooling
              vec   [Dv]      (optional)   protein-level values added to the readout
            NaN in X marks an invalid residue. Needs the structure modality (it supplies the residue graph).
Statistics are fitted on the training split only and stored with the model, so new tables are encoded identically."""
import os

import numpy as np
import pandas as pd
import torch


def parse_cli(features, feature_files):
    """--features chem=logp,charge x y   --feature-files md=md_file"""
    specs, loose = [], []
    for item in features or []:
        if '=' in item:
            n, cols = item.split('=', 1)
            specs.append(dict(name=n, columns=[c for c in cols.split(',') if c]))
        else:
            loose.append(item)
    if loose:
        specs.append(dict(name='features', columns=loose))
    for item in feature_files or []:
        n, _, col = item.partition('=')
        specs.append(dict(name=n, file_column=col))
    return specs


def _num(s):
    return pd.to_numeric(s.astype('string').str.strip().replace('', pd.NA), errors='coerce')


# ----------------------------------------------------------------------------- tabular
def fit_tabular(spec, df, train_mask, max_classes=20):
    cols, kinds, mean, std, cats = spec['columns'], {}, {}, {}, {}
    for c in cols:
        assert c in df.columns, f"feature group {spec['name']!r}: column {c!r} not in the table"
        v = _num(df[c])
        filled = df[c].astype('string').str.strip().replace('', pd.NA).notna()
        if v[filled].notna().all() and filled.any():
            kinds[c] = 'num'
            t = v[train_mask & filled.values]
            mean[c], std[c] = (float(t.mean()), float(t.std() + 1e-8)) if len(t) > 1 else (0.0, 1.0)
        else:
            vals = sorted(df[c].astype('string').str.strip().replace('', pd.NA).dropna().unique().tolist())
            assert 0 < len(vals) <= max_classes, f"feature column {c!r}: {len(vals)} distinct text values (max {max_classes}); not usable as a feature"
            kinds[c], cats[c] = 'cat', vals
    dim = sum(1 if kinds[c] == 'num' else len(cats[c]) for c in cols)
    return dict(kind='tabular', name=spec['name'], columns=cols, kinds=kinds, mean=mean, std=std, cats=cats, dim=dim)


def encode_tabular(st, df):
    """-> values+missing-flags [N, 2*dim] float, available [N] bool (at least one column filled)."""
    parts, miss, avail = [], [], np.zeros(len(df), bool)
    for c in st['columns']:
        if c not in df.columns:                                   # column absent in this table: all missing
            n = 1 if st['kinds'][c] == 'num' else len(st['cats'][c])
            parts.append(np.zeros((len(df), n))); miss.append(np.ones((len(df), n))); continue
        s = df[c].astype('string').str.strip().replace('', pd.NA)
        ok = s.notna().values
        avail |= ok
        if st['kinds'][c] == 'num':
            v = _num(df[c]).values
            ok = ok & ~np.isnan(v)
            parts.append(np.where(ok, (np.nan_to_num(v) - st['mean'][c]) / st['std'][c], 0.0)[:, None])
            miss.append((~ok)[:, None].astype(float))
        else:
            oh = np.stack([(s == k).fillna(False).values.astype(float) for k in st['cats'][c]], 1)
            parts.append(oh); miss.append(np.repeat((~ok)[:, None], oh.shape[1], 1).astype(float))
    x = np.concatenate([np.concatenate(parts, 1), np.concatenate(miss, 1)], 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(avail)


# ----------------------------------------------------------------------------- arrays
def _read(path):
    z = np.load(path, allow_pickle=False)
    return {'X': z} if isinstance(z, np.ndarray) else {k: z[k] for k in z.files}


def fit_array(spec, df, train_mask, base_dirs):
    col = spec['file_column']
    assert col in df.columns, f"feature group {spec['name']!r}: column {col!r} not in the table"
    paths = [resolve_path(p, base_dirs) for p in df[col]]
    ex = next((p for p in paths if p), None)
    assert ex, f"feature group {spec['name']!r}: no existing files found in column {col!r}"
    z = _read(ex)
    assert z['X'].ndim == 2, f"{os.path.basename(ex)}: X must be [residues, channels], got shape {z['X'].shape}"
    D = int(z['X'].shape[-1])
    vals = [_read(p)['X'] for p, tr in zip(paths, train_mask) if p and tr]
    v = np.concatenate(vals) if vals else np.zeros((1, D))
    v = v[np.isfinite(v).all(1)]
    return dict(kind='array', name=spec['name'], file_column=col, D=D, dim=D, has_pair='pair' in z,
                vec_dim=int(z['vec'].size) if 'vec' in z else 0,
                mean=v.mean(0).astype(np.float32), std=(v.std(0) + 1e-6).astype(np.float32))


def resolve_path(p, base_dirs):
    p = (p or '').strip()
    if not p:
        return ''
    for b in ([''] if os.path.isabs(p) else base_dirs):
        q = os.path.join(b, p)
        if os.path.exists(q):
            return q
    return ''


def load_array(st, path, L):
    """-> dict(x [L,D], xmask [L], pair, prior, vec) for one protein."""
    z = _read(path)
    X = np.asarray(z['X'], dtype=np.float32)
    assert X.shape == (L, st['D']), f"{os.path.basename(path)}: X is {X.shape} but this protein's structure has {L} residues and D={st['D']}"
    valid = np.isfinite(X).all(-1)
    x = (np.where(np.isfinite(X), X, 0) - st['mean']) / st['std']
    pair, prior, vec = z.get('pair'), z.get('prior'), z.get('vec')
    return dict(x=torch.from_numpy(x.astype(np.float32)), xmask=torch.from_numpy(valid),
                pair=None if pair is None else torch.from_numpy(np.nan_to_num(np.asarray(pair, np.float32))),
                prior=torch.ones(L) if prior is None else torch.from_numpy(np.clip(np.nan_to_num(prior, nan=1.0), 0, 1).astype(np.float32)),
                vec=None if vec is None else torch.from_numpy(np.nan_to_num(vec.astype(np.float32)).ravel()))
