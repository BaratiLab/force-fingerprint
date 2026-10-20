"""End-to-end smoke test on the demo data (real proteins and structure files; its targets are only for exercising the code; no network, no mock data): every supported option is exercised by one of
two scenarios, plus a few cheap checks. Each model trains 1 epoch through the real command line.  Run:  python tests/smoke.py  (about 10 minutes)

  scenario A  "sequence at inference"   the DEFAULT configuration (InterPro domain column on): frozen structure teacher -> sequence student, one model per target
                                        (chosen automatically because --mode is omitted), prediction on the sequence alone
  scenario B  "sequence + structure at  the other flags together: one joint model, a tabular feature group, per-residue array features
              inference"                (X, pair, prior, vec), a user `group` column; prediction run both on the sequence alone and
                                        with structures looked up and used (--structure on)
  plus        a pure sequence model (--structure off), the pLDDT filter on real AlphaFold / experimental files, and error handling"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
DEMO = 'examples/demo_only_data'
CSV = DEMO + '/demo_only_curated_mf.csv'
SETS = ['train.epochs=1', 'train.warmup_epochs=1', 'train.n_boot=20', 'train.teacher_epochs=1']
OUT = tempfile.mkdtemp(prefix='mp_smoke_')
T3 = ['--targets', 'af_plddt', 'is_catalytic', 'go_terms']
T2 = ['--targets', 'is_catalytic', 'go_terms']


def mp(*args, stdin=None):
    return subprocess.run([sys.executable, 'mp.py', *args], cwd=ROOT, capture_output=True, text=True, stdin=stdin)


def train(name, flags, sets=(), csv=CSV, stdin=None):
    out = os.path.join(OUT, name)
    r = mp('train', '--csv', csv, *flags, '--set', *SETS, f'train.out_dir={out}', *sets, stdin=stdin)
    assert r.returncode == 0, f'{name} FAILED:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}'
    return out, r.stdout


def scores(out):
    """Test scores of a run (every per-target model for a per_target run), all finite."""
    files = glob.glob(os.path.join(out, 'results.json')) or sorted(glob.glob(os.path.join(out, 'per_target', '*', 'results.json')))
    got = {}
    for f in files:
        res = json.load(open(f))['results']['test']
        got.update({k: v['score'] for k, v in res.items() if not k.startswith('_')})
    assert got and all(np.isfinite(list(got.values()))), f'bad scores {got}'
    return got


def show(label, out):
    print(f'  ok  {label:<58} {{{", ".join(f"{k}: {v:.2f}" for k, v in scores(out).items())}}}', flush=True)


def predict(label, run_dir, csv, structure, expect):
    p = os.path.join(OUT, f'{label.split()[0]}_{len(os.listdir(OUT))}.csv')
    r = mp('predict', '--run', run_dir, '--csv', csv, '--out', p, *(['--structure', 'on'] if structure else []))
    assert r.returncode == 0 and expect in r.stdout, f'{label} FAILED:\n{r.stdout[-800:]}\n{r.stderr[-800:]}'
    used = pd.read_csv(p).structure_used
    assert (used != 'none (sequence-only)').any() == structure, f'{label}: structure_used = {used.value_counts().to_dict()}'
    print(f'  ok  {label:<58} {expect}', flush=True)


def extended_csv():
    """The demo table plus (a) a `group` column (gene prefix of the id) and (b) a column of per-residue array files built from the
    structure files themselves: X = B-factor, pair = exp(-CA distance/8) (proteins <= 200 residues), prior = B-factor scaled to [0,1],
    vec = [mean, std] of the B-factors; every 7th training protein has no file (that modality is then masked for it)."""
    from multipeptide.structure import load_chains
    df = pd.read_csv(os.path.join(ROOT, CSV))
    d = os.path.join(OUT, 'arrays')
    os.makedirs(d, exist_ok=True)
    files = []
    for r in df.itertuples():
        _, xyz, b = load_chains(os.path.join(ROOT, DEMO, r.structure_file))[r.chain_id]
        if r.split == 'train' and r.Index % 7 == 0:
            files.append('')
            continue
        arr = dict(X=b[:, None].astype(np.float32), prior=(b / max(float(b.max()), 1e-6)).astype(np.float32),
                   vec=np.array([b.mean(), b.std()], dtype=np.float32))
        if len(b) <= 200:
            arr['pair'] = np.exp(-np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) / 8).astype(np.float32)
        files.append(os.path.join(d, f'{r.protein_id}.npz'))
        np.savez(files[-1], **arr)
    df['bfac_file'] = files
    df['structure_file'] = [os.path.join(ROOT, DEMO, f) for f in df.structure_file]      # this table lives in a temp folder, so use absolute paths
    df['group'] = df.protein_id.str.split('_').str[0]
    df = df.drop(columns='interpro')                                                          # scenario B covers the no-domain path
    out = os.path.join(OUT, 'extended.csv')
    df.to_csv(out, index=False)
    return out


def structure_confidence():
    """AlphaFold/ESMFold models below the pLDDT threshold are ignored; experimental structures are never filtered (offline, real files)."""
    from multipeptide.fetch import Resolver
    from multipeptide.structure import load_chains, predicted_by
    files = sorted(glob.glob(os.path.join(ROOT, DEMO, 'structures', '*.pdb')))
    af = next(f for f in files if predicted_by(f) == 'alphafold')
    exp = next(f for f in files if predicted_by(f) is None)
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'config.yaml')))
    cfg['data']['cache_dir'] = os.path.join(OUT, 'conf_cache')

    def resolve(path, min_plddt):
        cfg['structure']['min_plddt'] = min_plddt
        chain, (seq, _, _) = next(iter(load_chains(path).items()))
        return Resolver(cfg, ['/']).resolve(dict(structure_file=path, chain_id=chain), seq, 'k')
    hit, _ = resolve(af, 0)
    assert hit and hit['plddt'] is not None, 'AlphaFold file not recognised as a prediction'
    assert resolve(af, hit['plddt'] - 1)[0], 'structure above the threshold must be kept'
    bad, notes = resolve(af, hit['plddt'] + 1)
    assert bad is None and 'confidence too low' in ' '.join(notes), f'structure below the threshold must be ignored: {notes}'
    assert resolve(exp, 99.9)[0]['plddt'] is None, 'experimental structures must not be gated on B-factors'
    print(f"  ok  pLDDT filter: mean pLDDT {hit['plddt']:.1f} kept at {hit['plddt'] - 1:.1f}, ignored at {hit['plddt'] + 1:.1f}; experimental file never filtered", flush=True)


if __name__ == '__main__':
    print('offline checks')
    r = mp('inspect', '--csv', CSV, '--features', 'plddt=af_plddt')
    assert r.returncode == 0 and 'af_plddt' not in r.stdout.split('=>')[-1] and 'is_catalytic' in r.stdout, r.stdout[-800:]
    print('  ok  inspect: an input feature column is removed from the targets')
    structure_confidence()
    r = mp('train', '--csv', CSV, '--mode', 'joint', *T2, '--set', 'train.epoch=1')
    assert r.returncode != 0 and "unknown setting 'train.epoch'" in r.stdout + r.stderr, r.stdout + r.stderr
    print('  ok  a mistyped setting fails loudly instead of being ignored')

    print('scenario A: sequence at inference (default configuration)')
    a, out = train('A_default', T3, stdin=subprocess.DEVNULL)               # no --mode, no terminal -> recommended per_target; structure on -> staged
    assert 'using the recommended per_target' in out and os.path.isdir(os.path.join(a, 'per_target')), out[-800:]
    assert '[domain] InterPro guide on' in out, 'the demo table has an interpro column, so the domain guide must be on'
    show('A  default: staged, --mode omitted -> per_target', a)
    predict('A  predict, default', a, CSV, False, 'sequence-only')

    print('scenario B: sequence + structure at inference (every extra input)')
    ext = extended_csv()
    b, out = train('B_staged_joint', [*T2, '--mode', 'joint', '--features', 'plddt=af_plddt', '--feature-files', 'bfac=bfac_file'], csv=ext)
    assert '[domain]' not in out, 'no interpro column -> no domain guide'
    show('B  joint, tabular + array features, group column', b)
    predict('B  predict, sequence only', b, ext, False, 'sequence-only')
    predict('B  predict, --structure on', b, ext, True, 'sequence + structure')

    print('pure sequence model')
    s, out = train('seq_only', [*T2, '--mode', 'joint', '--structure', 'off'])
    assert '[domain]' not in out, 'domains guide the structure readout, so they are ignored without structure'
    show('--structure off (no structure lookup, no graph network)', s)
    print('ALL SMOKE CHECKS PASSED  (outputs in %s)' % OUT)
