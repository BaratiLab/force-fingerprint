"""Train and score the sequence-only and the sequence+structure model on one benchmark table.

  python benchmark/run.py --table benchmark_tables/full.csv --out runs/full

Four steps, each one skipped when its output already exists (delete the file to redo it):
  1. train   sequence only            (--structure off)
  2. train   sequence + structure     (frozen structure teacher, sequence student; the default configuration)
  3. predict on the test proteins     sequence only | teacher/student, sequence only at inference | teacher/student, structure used at inference
  4. score   every prediction file, plus any --pred NAME=FILE you add (e.g. the DPFunc export), with benchmark/score.py

--dry-run prints the commands without running them.  Extra config overrides go after --set, e.g.
  --set train.batch_size=64 train.epochs=60
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TARGET = 'go_terms'


def run(cmd, dry):
    print('$', ' '.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', required=True, help='benchmark table from build_benchmark_tables.py (data branch)')
    ap.add_argument('--out', required=True, help='folder for the runs, predictions and scores')
    ap.add_argument('--min-count', type=int, default=10, help='GO terms seen fewer times in the training proteins are not predicted')
    ap.add_argument('--pred', action='append', default=[], metavar='NAME=FILE', help='predictions of another method, passed on to score.py')
    ap.add_argument('--set', nargs='*', default=[], help='config overrides, key=value')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    table, out = os.path.abspath(args.table), os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    sets = [f'data.min_label_count={args.min_count}', *args.set]
    common = ['--targets', f'{TARGET}:multilabel', '--mode', 'per_target']

    # the predict command scores every row of the csv it is given, so hand it only the test proteins
    test_csv = os.path.join(out, 'test_proteins.csv')
    if not args.dry_run:
        df = pd.read_csv(table, dtype=str).fillna('')
        df[df.split == 'test'].to_csv(test_csv, index=False)
        n = df.split.value_counts().to_dict()
        print(f"table: {n.get('train', 0)} train / {n.get('valid', 0)} valid / {n.get('test', 0)} test proteins")
        if n.get('valid', 0) < 30:
            print('NOTE: very few validation proteins; model selection and early stopping on them are noisy.')

    models = {'seq': ['--structure', 'off'], 'staged': ['--structure', 'on']}
    for name, flags in models.items():
        d = os.path.join(out, name)
        if os.path.isdir(d) and os.listdir(d):
            print(f'skip training {name}: {d} exists')
            continue
        run([py, 'mp.py', 'train', '--csv', table, *common, *flags, '--out', d, '--set', *sets], args.dry_run)

    preds = {}
    for label, model, structure in (('sequence only', 'seq', 'off'),
                                    ('sequence + structure (teacher; sequence only at inference)', 'staged', 'off'),
                                    ('sequence + structure (structure used at inference)', 'staged', 'on')):
        stem = os.path.join(out, f'pred_{model}_{structure}')
        probs = f'{stem}_{TARGET}_probs.npz'
        preds[label] = probs
        if os.path.exists(probs):
            print(f'skip prediction {label}: {probs} exists')
            continue
        run([py, 'mp.py', 'predict', '--csv', test_csv, '--run', os.path.join(out, model), '--structure', structure,
             '--out', stem + '.csv', '--set', *sets], args.dry_run)

    cmd = [py, 'benchmark/score.py', '--table', table, '--min-count', str(args.min_count), '--out', os.path.join(out, 'scores')]
    for label, path in {**preds, **dict(p.split('=', 1) for p in args.pred)}.items():
        cmd += ['--pred', f'{label}={path}']
    run(cmd, args.dry_run)


if __name__ == '__main__':
    main()
