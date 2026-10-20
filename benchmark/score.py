"""Score GO-term predictions from any method against one benchmark table, with the same code and the same label set.

  python benchmark/score.py --table benchmark_tables/full.csv \\
      --pred sequence=runs/full/seq_probs.npz --pred "sequence+structure"=runs/full/staged_probs.npz \\
      --pred dpfunc=dpfunc_predictions.tsv --out runs/full/scores

A prediction file is either
  * .npz written by `mp.py predict` (arrays ids, classes, probs), or
  * a table (.tsv/.csv) in long format with the columns protein_id, go_term, score (what you export from DPFunc).
Terms a method does not predict score 0, proteins it does not predict are left out for every method.

Labels: terms seen at least --min-count times among the table's training proteins, GO roots removed.
Metrics: protein-centric Fmax (CAFA style; precision averaged over proteins with at least one prediction above the
threshold) and AUPR over all protein-term pairs (micro) and averaged over terms (macro), with 95% bootstrap intervals over
test proteins.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from multipeptide.targets import bootstrap_ci, metrics  # noqa: E402

ROOTS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def split_terms(cell):
    return [t for t in str(cell).split(';') if t and t != 'nan' and t not in ROOTS]


def vocabulary(table, min_count):
    c = Counter(t for cell in table[table.split == 'train'].go_terms for t in set(split_terms(cell)))
    return sorted(t for t, n in c.items() if n >= min_count)


def read_predictions(path, terms):
    """-> {protein_id: vector over `terms`}"""
    index = {t: i for i, t in enumerate(terms)}
    out = {}
    if path.endswith('.npz'):
        z = np.load(path, allow_pickle=True)
        cols = [(j, index[t]) for j, t in enumerate(z['classes']) if t in index]
        src, dst = [j for j, _ in cols], [i for _, i in cols]
        for pid, row in zip(z['ids'], z['probs']):
            v = np.zeros(len(terms), dtype=np.float32)
            v[dst] = row[src]
            out[str(pid)] = v
        return out
    df = pd.read_csv(path, sep=None, engine='python', dtype={'protein_id': str, 'go_term': str})
    df = df.rename(columns={'term': 'go_term', 'prob': 'score', 'probability': 'score'})
    missing = {'protein_id', 'go_term', 'score'} - set(df.columns)
    if missing:
        raise SystemExit(f'{path}: needs columns protein_id, go_term, score (missing {sorted(missing)}; found {list(df.columns)})')
    df = df[df.go_term.isin(index)]
    for pid, g in df.groupby('protein_id'):
        v = np.zeros(len(terms), dtype=np.float32)
        v[[index[t] for t in g.go_term]] = g.score.to_numpy(dtype=np.float32)
        out[str(pid)] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', required=True, help='benchmark table (protein_id, split, go_terms)')
    ap.add_argument('--pred', action='append', default=[], metavar='NAME=FILE')
    ap.add_argument('--min-count', type=int, default=10, help='keep terms seen at least this often in the training proteins')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', help='write <out>.csv and <out>.md')
    args = ap.parse_args()

    table = pd.read_csv(args.table, dtype=str).fillna('')
    terms = vocabulary(table, args.min_count)
    test = table[table.split == 'test'].reset_index(drop=True)
    index = {t: i for i, t in enumerate(terms)}
    methods = {}
    for item in args.pred:
        name, _, path = item.partition('=')
        methods[name] = read_predictions(path, terms)

    # frequency baseline: every protein gets the training frequency of each term
    n_train = (table.split == 'train').sum()
    freq = np.zeros(len(terms), dtype=np.float32)
    for cell in table[table.split == 'train'].go_terms:
        for t in set(split_terms(cell)):
            if t in index:
                freq[index[t]] += 1
    methods['train frequency (reference)'] = {pid: freq / n_train for pid in test.protein_id}

    ids = [p for p in test.protein_id if all(p in m for m in methods.values())]
    test = test.set_index('protein_id').loc[ids]
    y = np.zeros((len(ids), len(terms)), dtype=np.float32)
    for r, cell in enumerate(test.go_terms):
        for t in split_terms(cell):
            if t in index:
                y[r, index[t]] = 1
    has_label = y.sum(1) > 0
    print(f'{len(terms)} terms, {len(table[table.split == "test"])} test proteins in the table, {len(ids)} scored by every method '
          f'({int(has_label.sum())} with at least one label in the vocabulary)')
    if len(ids) < 50:
        print(f'WARNING: only {len(ids)} test proteins - scores and intervals are very noisy; read them as a rough indication only.')
    y, ids = y[has_label], np.array(ids)[has_label]

    rows = []
    for name, preds in methods.items():
        p = np.stack([preds[i] for i in ids])
        m = metrics('multilabel', y, p)
        ci = bootstrap_ci('multilabel', y, p, np.arange(len(ids)), n_boot=args.n_boot)
        rows.append(dict(method=name, n_proteins=len(ids), n_terms=len(terms), fmax=m['fmax'],
                         fmax_ci_low=ci[0] if ci else np.nan, fmax_ci_high=ci[1] if ci else np.nan,
                         aupr_micro=m['aupr_micro'], aupr_macro=m['aupr_macro']))
    res = pd.DataFrame(rows)
    text = res.round(3).to_string(index=False)
    print(text)
    if args.out:
        res.to_csv(args.out + '.csv', index=False)
        with open(args.out + '.md', 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
