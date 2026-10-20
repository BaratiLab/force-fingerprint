"""MultiPeptide command line.
  python mp.py inspect --csv data.csv                         # offline: detected columns, target types, problems
  python mp.py prepare --csv data.csv                         # resolve structures + embed sequences, print a report
  python mp.py train   --csv data.csv --mode per_target       # one model PER target column (recommended); --mode joint = one model for all
  python mp.py train   --csv data.csv --structure off         # pure sequence model: no structure lookup at all
  python mp.py train   --csv data.csv --features chem=logp,charge --feature-files md=md_file    # extra INPUT columns / array files
  python mp.py train   --csv data.csv --targets a b:regression --columns id=name sequence=seq split=fold
  python mp.py predict --run runs/default --csv new.csv --out preds.csv                       # sequence model only
  python mp.py predict --run runs/default --csv new.csv --out preds.csv --structure on        # also use PDB/AlphaFold structures
Any other setting:  --set section.key=value   (e.g. --set structure.min_plddt=80 train.epochs=30)"""
import argparse
import os
import sys

import yaml

from multipeptide.data import build_records, inspect_csv
from multipeptide.features import parse_cli
from multipeptide.pipeline import apply_overrides, get_plm, predict, run_training


SEC_PER_ROW_EPOCH = {True: 0.024, False: 0.006}      # measured on Apple M5: sequence+structure / sequence only (see README)


def choose_mode(cfg):
    """The user decides: one model per target column (recommended) or one joint model for all targets."""
    if cfg.get('mode') in ('joint', 'per_target'):
        return
    targets, rows = inspect_csv(cfg, log=lambda *_: None)
    if len(targets) <= 1:
        cfg['mode'] = 'per_target'                       # identical for a single target
        return
    per_run_min = 60 * SEC_PER_ROW_EPOCH[cfg['structure']['use']] * rows / 60     # ~60 epochs
    print(f"""
You have {len(targets)} target columns: {targets}   ({rows} rows)

  [1] one model PER target column   (RECOMMENDED)
        each target gets its own encoders, early stopping and metric, so a hard or unrelated target cannot drag down an easy one
        and each stops at its own best epoch.   Cost: {len(targets)} trainings, roughly {len(targets) * per_run_min:.0f} min in total.
  [2] ONE model for all targets combined
        one shared model with a head per target: about {len(targets)}x cheaper (roughly {per_run_min:.0f} min) and lets related targets share what they learn,
        but a single early-stopping point is a compromise and unrelated targets can interfere.
""")
    if sys.stdin.isatty():
        ans = input('Choose 1 or 2 [Enter = 1, recommended]: ').strip()
        cfg['mode'] = 'joint' if ans == '2' else 'per_target'
    else:
        cfg['mode'] = 'per_target'
        print('No --mode given and no terminal to ask: using the recommended per_target. (Pass --mode joint for one combined model.)')
    print(f"=> mode: {cfg['mode']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['inspect', 'prepare', 'train', 'predict'])
    ap.add_argument('--csv', required=True)
    ap.add_argument('--mode', choices=['joint', 'per_target'])
    ap.add_argument('--targets', nargs='*', help='column or column:kind (regression|binary|multiclass|multilabel); default: all other columns')
    ap.add_argument('--columns', nargs='*', help='tell it which column is what, e.g. id=name sequence=seq split=fold (default: detected)')
    ap.add_argument('--features', nargs='*', help='extra INPUT columns: name=colA,colB (a group = one modality) or bare column names')
    ap.add_argument('--feature-files', nargs='*', help='extra INPUT arrays from files: name=path_column (one .npz per protein, indexed on the structure residues)')
    ap.add_argument('--structure', choices=['off', 'on', 'generate'],
                    help='off = pure sequence model; on = use PDB/AlphaFold structures; generate = also fold missing ones with ESMFold. '
                         'Default: train on, predict off (sequence model only)')
    ap.add_argument('--out', help='train: run dir; predict: output csv')
    ap.add_argument('--run', help='predict: trained run directory')
    ap.add_argument('--set', nargs='*', action='extend', default=[], help='any config key, e.g. train.epochs=30')
    a, extra = ap.parse_known_args()
    bad = [x for x in extra if '=' not in x]
    if bad:
        ap.error(f'unrecognized arguments: {bad}')
    a.set = list(a.set) + extra          # key=value tokens left over (e.g. after a single-value flag) are config overrides

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')))
    cfg['data']['csv'] = a.csv
    if a.mode: cfg['mode'] = a.mode
    if a.targets: cfg['data']['targets'] = a.targets
    for item in a.columns or []:
        role, _, col = item.partition('=')
        cfg['data'].setdefault('columns', {})['sequence' if role == 'seq' else role] = col
    if a.features or a.feature_files:
        cfg['features'] = (cfg.get('features') or []) + parse_cli(a.features, a.feature_files)
    if a.structure and a.cmd != 'predict':
        cfg['structure'].update(use=a.structure != 'off', generate=a.structure == 'generate')
    if a.cmd == 'train' and a.out: cfg['train']['out_dir'] = a.out
    cfg = apply_overrides(cfg, a.set)

    if a.cmd == 'inspect':
        inspect_csv(cfg)
    elif a.cmd == 'prepare':
        recs, _, _ = build_records(cfg)
        get_plm(cfg, recs)
        print('structure report: ', cfg['data']['cache_dir'] + '/structure_report.tsv')
    elif a.cmd == 'train':
        choose_mode(cfg)
        run_training(cfg)
    else:
        assert a.run, '--run <trained run dir> is required for predict'
        predict(a.run, a.csv, a.out or 'predictions.csv', a.structure or 'off', a.set)

if __name__ == '__main__':
    try:
        main()
    except (AssertionError, ValueError, FileNotFoundError, KeyError) as e:      # user-fixable problems: message, not traceback
        raise SystemExit(f'\nError: {e}\n(see README.md "CSV format"; run `python mp.py inspect --csv <file>` to check your table)')
