import copy
import time
import functools
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .data import LengthBuckets, ProteinSet, build_records, collate, domain_vocab
from .diagnostics import alignment_report
from .losses import align_loss, expert_loss, task_loss
from .model import MultiPeptide
from .plm import embed_sequences
from .targets import PRIMARY, Target, bootstrap_ci, metrics, score


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def to_dev(b, dev):
    """Move a (possibly nested) batch to the device; non-tensors (ids, groups) pass through."""
    if torch.is_tensor(b):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: to_dev(v, dev) for k, v in b.items()}
    return b


def apply_overrides(cfg, sets):
    for kv in sets:
        k, v = kv.split('=', 1)
        node = cfg
        *path, last = k.split('.')
        for p in path:
            if not isinstance(node, dict) or p not in node:
                raise ValueError(f'unknown setting {k!r} (see config.yaml for the available settings)')
            node = node[p]
        if not isinstance(node, dict) or (last not in node and path[-1:] != ['columns']):
            raise ValueError(f'unknown setting {k!r} (see config.yaml for the available settings)')
        node[last] = yaml.safe_load(v)
    return cfg


def get_plm(cfg, recs, log=print):
    s = cfg['seq_encoder']
    path = os.path.join(cfg['data']['cache_dir'], 'plm_' + s['plm_model'].split('/')[-1] + '.pt')
    os.makedirs(cfg['data']['cache_dir'], exist_ok=True)
    return embed_sequences([r['seq'] for r in recs], s['plm_model'], path, log)


# ----------------------------------------------------------------------------- predict / evaluate
@torch.no_grad()
def forward_all(model, loader, targets, dev, use=None, control=None, skip_unused=False):
    """-> dict(ids, groups, pred{t: [N,...] natural scale}, y{t}, m{t}, z{mod}, avail{mod})."""
    model.eval()
    acc = dict(ids=[], groups=[], plen=[], slen=[], pred={t.name: [] for t in targets}, y={t.name: [] for t in targets},
               m={t.name: [] for t in targets}, z={}, avail={})
    gsum = gn = 0
    for b in loader:
        if control == 'shuffle_struct':     # give every protein another protein's structure: heads that use structure must get worse
            for k in ('sxyz', 'smask', 'has_struct'):
                b[k] = b[k].roll(1, 0)
        o = model(to_dev(b, dev), use, skip_unused)
        acc['ids'] += b['ids']; acc['groups'] += b['groups']
        acc['plen'].append(b['mask'].sum(1)); acc['slen'].append(b['smask'].sum(1))
        for t in targets:
            acc['pred'][t.name].append(t.to_pred(o['logits'][t.name]).cpu())
            acc['y'][t.name].append(b['y'][t.name]); acc['m'][t.name].append(b['m'][t.name])
        for k in o['z']:
            acc['z'].setdefault(k, []).append(o['z'][k].cpu()); acc['avail'].setdefault(k, []).append(o['avail'][k].cpu())
        av = torch.stack([o['avail'][k] for k in model.mods], 1).float()                      # [B,M]
        gsum, gn = gsum + (o['gate'] * av.unsqueeze(1)).sum(0).cpu(), gn + av.sum(0).cpu()
    cat = lambda d: {k: torch.cat(v).numpy() for k, v in d.items()}
    for k in ('pred', 'y', 'm', 'z', 'avail'):
        acc[k] = cat(acc[k])
    acc['plen'], acc['slen'] = torch.cat(acc['plen']).numpy(), torch.cat(acc['slen']).numpy()
    acc['mlen'] = {k: acc['slen'] if k == 'struct' or (k in model.fstates and model.fstates[k]['kind'] == 'array') else acc['plen'] for k in model.mods}
    w = (gsum / gn.clamp(min=1)).numpy()       # mean learned gate weight of each modality, per target, over the proteins that have it
    acc['gates'] = {t.name: {k: float(w[ti, mi]) for mi, k in enumerate(model.mods)} for ti, t in enumerate(model.targets)}
    return acc


def score_split(acc, targets, with_ci=0):
    out = {}
    for t in targets:
        m = acc['m'][t.name]
        if m.sum() < 3:
            continue
        y, p = acc['y'][t.name][m], acc['pred'][t.name][m]
        y = t.true_natural(torch.from_numpy(y)).numpy()
        r = metrics(t.kind, y, p)
        r['score'] = score(t.kind, y, p)
        if with_ci:
            r['score_ci95'] = bootstrap_ci(t.kind, y, p, np.asarray(acc['groups'])[m], with_ci)
        out[t.name] = r
    return out


def mean_score(res):
    return float(np.mean([v['score'] for v in res.values()])) if res else -1.0


# ----------------------------------------------------------------------------- training
def module_params(model, names):
    """All parameters owned by the given modalities (encoder, projection, norm, expert head; the domain embedding belongs to struct)."""
    return [p for k in names for m in (model.enc[k], model.proj[k], model.pre_norm[k], model.expert[k]) + ((model.dom,) if k == 'struct' and model.dom is not None else ())
            for p in m.parameters()]


def make_opt(model, params, t_, l_):
    """AdamW with four groups: weights (decayed), biases/norms (not decayed), and two scalar-like control parameters that need a larger
    step than the encoders to move within a few hundred updates -- the contrastive temperature and the modality gates."""
    special = {id(model.logit_scale): ('temp', l_['control_lr']), id(model.gate): ('gate', l_['control_lr'])}
    ps = [p for p in {id(p): p for p in params}.values() if id(p) not in special]
    groups = [dict(params=[p for p in ps if p.ndim > 1], weight_decay=t_['weight_decay']),
              dict(params=[p for p in ps if p.ndim <= 1], weight_decay=0.0)]
    groups += [dict(params=[p], weight_decay=0.0, lr=lr) for p in params if id(p) in special for _, lr in [special[id(p)]]]
    return torch.optim.AdamW([g for g in groups if g['params']], lr=t_['lr'])


def fit(model, name, params, epochs, loss_fn, val_use, loaders, targets, cfg, dev, log, frozen=()):
    """Generic training phase: AdamW + warm-up/cosine, early stopping on the mean validation score of path `val_use`.
    Restores the best weights. `frozen` modules are kept in eval mode (deterministic teacher)."""
    t_, l_ = cfg['train'], cfg['loss']
    tr_l, va_l = loaders
    opt = make_opt(model, params, t_, l_)
    total, warm = epochs * len(tr_l), t_['warmup_epochs'] * len(tr_l)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, best_state, bad, hist = -1e9, copy.deepcopy(model.state_dict()), 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for m in frozen:
            m.eval()
        agg, nb = np.zeros(4), 0
        for b in tr_l:
            bd = to_dev(b, dev)
            gid = torch.tensor([hash(g) % (2 ** 31) for g in b['groups']], device=dev)
            loss, parts = loss_fn(model, bd, gid, ep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            opt.step(); sched.step()
            agg += [loss.item(), *parts]; nb += 1
        agg /= max(nb, 1)
        if dev.type == 'mps':          # the MPS allocator fragments over epochs with varying shapes; releasing keeps epoch time flat
            torch.mps.empty_cache()
        t_train = time.time() - t0
        va = score_split(forward_all(model, va_l, targets, dev, val_use), targets)
        sc = mean_score(va)
        hist.append(dict(phase=name, epoch=ep, loss=agg[0], task=agg[1], align=agg[2], expert=agg[3], val_score=sc, val={k: v['score'] for k, v in va.items()}))
        log(f'[{name}] ep{ep:03d} loss {agg[0]:.3f} (task {agg[1]:.3f} align {agg[2]:.3f} expert {agg[3]:.3f}) | val ' +
            ' '.join(f"{k}={v['score']:.3f}" for k, v in va.items()) + f' | T={1 / model.logit_scale.exp().item():.3f} | {t_train:.0f}s train + {time.time() - t0 - t_train:.0f}s val')
        if sc > best:
            best, bad, best_state = sc, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= t_['patience']:
                log(f'[{name}] early stop'); break
    model.load_state_dict(best_state)
    return hist, best


def train_model(cfg, recs, targets, plm, fstates, out_dir, log=print):
    """With structure: (A) the structure GNN (the teacher) is trained alone on the targets and then FROZEN; (B) the sequence side (and the
       heads) are trained with the frozen teacher space as a fixed contrastive target, so structure knowledge is distilled into the
       sequence model; the teacher is randomly hidden from the head so the structure-free path is trained too. Selection uses that
       structure-free path (sequence-first: new proteins need no structure). Without structure: a single phase."""
    t_, l_ = cfg['train'], cfg['loss']
    random.seed(t_['seed']); np.random.seed(t_['seed']); torch.manual_seed(t_['seed'])
    dev = device()
    os.makedirs(out_dir, exist_ok=True)
    col = functools.partial(collate, targets=targets, fstates=fstates)
    dv = domain_vocab(recs, cfg['data']['domain_min_count']) if cfg['structure']['use'] and any(r['ipr'] for r in recs) else []
    if dv:
        log(f'[domain] InterPro guide on: {len(dv)} domains (train split, seen >= {cfg["data"]["domain_min_count"]}x)')

    def mk(split, tr):
        ds = ProteinSet(recs, split, plm, tr, t_['coord_noise'] if tr else 0, dv, fstates)
        if tr and len(ds) > t_['batch_size']:
            return DataLoader(ds, batch_sampler=LengthBuckets([len(r['seq']) for r in ds.recs], t_['batch_size'], seed=t_['seed']), collate_fn=col)
        if not tr:        # evaluation: length-sorted batches (same results, far less padding); order is carried by the ids
            o = np.argsort([len(r['seq']) for r in ds.recs]).tolist()
            return DataLoader(ds, batch_sampler=[o[i:i + t_['batch_size']] for i in range(0, len(o), t_['batch_size'])], collate_fn=col)
        return DataLoader(ds, batch_size=t_['batch_size'], shuffle=True, drop_last=len(ds) > t_['batch_size'], collate_fn=col)
    tr_l, va_l, te_l = mk('train', True), mk('valid', False), mk('test', False)
    model = MultiPeptide(cfg, targets, len(dv) if dv else 0, fstates).to(dev)
    path_scores = lambda ld, **kw: {k: v['score'] for k, v in score_split(forward_all(model, ld, targets, dev, **kw), targets).items()}
    mods = model.mods
    teacher = ['struct'] if model.use_struct else []
    student = tuple(m for m in mods if m not in teacher)        # = the deployed, structure-free path
    log(f'[train] {out_dir}: targets={[t.name for t in targets]} modalities={list(mods)} '
        + (f'teacher={teacher} student={list(student)} ' if teacher else 'single phase ') +
        f'params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M device={dev}')
    hist, frozen = [], []
    if teacher:
        # ---- phase A: the teacher alone, supervised on the targets, then frozen
        tparams = module_params(model, teacher)

        def teacher_loss(m_, bd, gid, ep):
            out = m_(bd, use=tuple(teacher), skip_unused=True)
            lu = expert_loss(out, bd, targets, only=teacher)
            return lu, [0.0, 0.0, lu.item()]
        h, sc = fit(model, 'teacher', tparams, t_['teacher_epochs'], teacher_loss, tuple(teacher), (tr_l, va_l), targets, cfg, dev, log)
        hist += h
        log(f'[teacher] best validation score {sc:.3f} -> frozen')
        for p in tparams:
            p.requires_grad_(False)
        snap = [p.detach().clone() for p in tparams]
        frozen = [model.enc[k] for k in teacher] + [model.proj[k] for k in teacher] + ([model.dom] if model.dom is not None else [])
        model.set_drop({k: t_['teacher_dropout'] for k in teacher})

    # ---- phase B (or the only phase): everything that is not frozen, contrastively aligned to the teacher space
    def student_loss(m_, bd, gid, ep):
        out = m_(bd)
        lt = task_loss(out, bd, targets)
        la = align_loss(out, gid) if l_['lambda_align'] > 0 else lt * 0
        lu = expert_loss(out, bd, targets, only=student)
        return lt + l_['lambda_expert'] * lu + l_['lambda_align'] * la, [lt.item(), la.item(), lu.item()]
    h, sc = fit(model, 'student' if teacher else 'train', [p for p in model.parameters() if p.requires_grad], t_['epochs'], student_loss,
                student, (tr_l, va_l), targets, cfg, dev, log, frozen=frozen)
    hist += h
    if teacher:
        model.set_drop(None)
        assert all(torch.equal(a.cpu(), b.detach().cpu()) for a, b in zip(snap, tparams)), 'BUG: the frozen teacher changed during phase B'
        log('[teacher] verified: teacher weights identical after phase B (frozen)')

    torch.save(dict(model=model.state_dict(), cfg=cfg, targets=[t.state() for t in targets], features=fstates, domains=dv), os.path.join(out_dir, 'best.pt'))
    # Reports. The headline (and summary.csv, and test_predictions.csv) is the DEPLOYED, structure-free path `student`; everything else is context.
    res = {}
    for name, ld in (('valid', va_l), ('test', te_l)):
        if len(ld.dataset) == 0:
            continue
        full = forward_all(model, ld, targets, dev)                  # every modality, structure where a protein has one
        head = forward_all(model, ld, targets, dev, student)         # the deployed path
        res[name] = score_split(head, targets, t_['n_boot'])
        if model.use_struct:
            res[name]['_with_structure'] = score_split(full, targets, t_['n_boot'])
        res[name]['_only'] = {u: path_scores(ld, use=(u,)) for u in mods}      # what each modality carries on its own
        ctrl = {}
        if model.use_struct:     # give every protein another (similar-length) protein's structure: if the heads use structure this must hurt
            ctrl['struct_shuffled'] = path_scores(ld, control='shuffle_struct')
        for u in mods:           # the full model with one extra feature group removed: what it adds on top of the rest
            if u not in ('seq', 'struct'):
                ctrl[f'without_{u}'] = path_scores(ld, use=tuple(m_ for m_ in mods if m_ != u))
        res[name]['_controls'] = ctrl
        res[name]['_alignment'] = alignment_report(full)
        res[name]['_gates'] = full['gates']
        if name == 'test':
            cols = {'id': head['ids']}
            for t in targets:
                v = head['pred'][t.name]
                if v.ndim == 1:
                    cols[t.name] = v
                else:
                    cols.update({f'{t.name}[{t.classes[j]}]': v[:, j] for j in range(v.shape[1])})
            pd.DataFrame(cols).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)
    json.dump(dict(results=res, history=hist), open(os.path.join(out_dir, 'results.json'), 'w'), indent=1, default=float)
    return res


def run_training(cfg, log=print):
    """mode joint: one model, one head per target column.  mode per_target: one independent model per target column."""
    recs, targets, fstates = build_records(cfg, log)
    plm = get_plm(cfg, recs, log)
    out = cfg['train']['out_dir']
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, 'config.yaml'), 'w'))
    rows = []
    groups = [(None, targets)] if cfg['mode'] == 'joint' else [(t.name, [t]) for t in targets]
    for name, ts in groups:
        d = out if name is None else os.path.join(out, 'per_target', name.replace('/', '_'))
        res = train_model(cfg, recs, ts, plm, fstates, d, log)
        for split, r in res.items():
            for tn, v in r.items():
                if not tn.startswith('_'):
                    rows.append(dict(model=name or 'joint', split=split, target=tn, kind=next(t.kind for t in ts if t.name == tn),
                                     primary=PRIMARY[next(t.kind for t in ts if t.name == tn)], **{k: v[k] for k in v if not isinstance(v[k], (list, tuple, dict))}))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, 'summary.csv'), index=False)
    log('\n' + summary[summary.split == 'test'][['model', 'target', 'kind', 'primary', 'score', 'n']].to_string(index=False))
    return summary


# ----------------------------------------------------------------------------- inference on a new CSV
def load_run(run_dir, dev):
    """A joint run dir (best.pt) or a per_target run dir (per_target/<t>/best.pt) -> list of (model, targets, ckpt)."""
    paths = [os.path.join(run_dir, 'best.pt')] if os.path.exists(os.path.join(run_dir, 'best.pt')) else \
        sorted(os.path.join(run_dir, 'per_target', d, 'best.pt') for d in os.listdir(os.path.join(run_dir, 'per_target')))
    out = []
    for p in paths:
        ck = torch.load(p, map_location='cpu')
        targets = [Target.from_state(s) for s in ck['targets']]
        model = MultiPeptide(ck['cfg'], targets, len(ck['domains'] or []), ck['features']).to(dev)
        model.load_state_dict(ck['model']); model.eval()
        out.append((model, targets, ck))
    return out


def predict(run_dir, csv, out_csv, structure='off', sets=(), log=print):
    """structure 'off' (default): sequence model only -- no structure lookup, no graph network. 'on': also look up PDB/AlphaFold structures
    and use those that pass the pLDDT filter. 'generate': additionally fold proteins with no structure using ESMFold. A protein without an
    accepted structure always falls back to the sequence model. `sets` are config overrides such as structure.min_plddt=80."""
    dev = device()
    members = load_run(run_dir, dev)
    ck0 = members[0][2]
    cfg = apply_overrides(copy.deepcopy(ck0['cfg']), sets)
    cfg['data']['csv'] = csv
    cfg['structure']['generate'] = structure == 'generate'
    seq_only = structure == 'off' or not members[0][0].use_struct
    recs, _, _ = build_records(cfg, log, with_targets=False, feature_state=ck0['features'], lookup_structures=not seq_only)
    for r in recs:
        r['split'] = 'test'
    plm = get_plm(cfg, recs, log)
    res = pd.DataFrame({'id': [r['id'] for r in recs]})
    res['structure_used'] = ['none (sequence-only)' if (seq_only or not r['struct']) else r['struct']['source'] for r in recs]
    for model, targets, ck in members:
        c = ck['cfg']
        use = tuple(m for m in model.mods if m != 'struct') if seq_only else None
        ld = DataLoader(ProteinSet(recs, None, plm, False, 0, ck['domains'], ck['features']),
                        batch_size=c['train']['batch_size'], collate_fn=functools.partial(collate, targets=targets, fstates=ck['features']))
        acc = forward_all(model, ld, targets, dev, use, skip_unused=seq_only)
        for t in targets:
            p = acc['pred'][t.name]
            if t.kind in ('regression', 'binary'):
                res[f'{t.name}_pred'] = p
            elif t.kind == 'multiclass':
                res[f'{t.name}_pred'] = [t.classes[i] for i in p.argmax(1)]
                for i, c_ in enumerate(t.classes):
                    res[f'{t.name}_p[{c_}]'] = p[:, i]
            else:
                res[f'{t.name}_pred'] = [';'.join(t.classes[j] for j in np.where(row >= 0.5)[0]) for row in p]
                np.savez_compressed(out_csv.rsplit('.', 1)[0] + f'_{t.name}_probs.npz', ids=acc['ids'], classes=t.classes, probs=p)
    res.to_csv(out_csv, index=False)
    log(f'[predict] wrote {len(res)} rows -> {out_csv}  ({"sequence-only" if seq_only else "sequence + structure when available"})')
    return res
