"""Target columns of any type -> encoding, masked losses, metrics.
kinds: regression | binary | multiclass | multilabel (delimited set of labels per cell)."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score)

KINDS = ('regression', 'binary', 'multiclass', 'multilabel')
PRIMARY = {'regression': 'spearman', 'binary': 'auroc', 'multiclass': 'macro_f1', 'multilabel': 'fmax'}
_TRUE, _FALSE = {'1', 'true', 'yes', 'y', 'positive', 'active', 't'}, {'0', 'false', 'no', 'n', 'negative', 'inactive', 'f'}


def _clean(series):
    s = series.astype('string').str.strip()
    return s.mask(s.isin(['', 'nan', 'NaN', 'NA', 'None', 'null']))


def infer_kind(series, delimiter, max_classes):
    s = _clean(series).dropna()
    if s.empty:
        raise ValueError('column has no values')
    low = set(s.str.lower())
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().all():
        u = set(num)
        if u <= {0, 1} or (len(u) == 2 and low <= (_TRUE | _FALSE)):
            return 'binary'
        if all(float(x).is_integer() for x in u) and len(u) <= min(10, max_classes) and len(u) > 2:
            return 'multiclass'
        return 'regression' if len(u) > 1 else 'regression'
    if low <= (_TRUE | _FALSE):
        return 'binary'
    if s.str.contains(delimiter, regex=False).any():
        return 'multilabel'
    if s.nunique() <= max_classes and len(s) / s.nunique() >= 2:
        return 'multiclass'
    if s.nunique() <= max_classes:
        raise ValueError(f'{s.nunique()} distinct values in {len(s)} rows (almost every value unique): looks like free text or ids, not a class label')
    raise ValueError(f'{s.nunique()} distinct non-numeric values and no "{delimiter}" delimiter: looks like free text; '
                     f'set the type explicitly (name:multiclass / name:multilabel) or drop the column')


class Target:
    def __init__(self, name, kind):
        assert kind in KINDS, f'{name}: unknown kind {kind!r} (choose from {KINDS})'
        self.name, self.kind = name, kind
        self.classes, self.mean, self.std, self.pos_weight = [], 0.0, 1.0, None

    @property
    def dim(self):
        return 1 if self.kind in ('regression', 'binary') else len(self.classes)

    # ---- fitting / encoding -------------------------------------------------
    def fit(self, series, train_mask, delimiter, min_count):
        s = _clean(series)
        if self.kind == 'regression':
            v = pd.to_numeric(s[train_mask], errors='coerce').dropna().values.astype(float)
            self.mean, self.std = float(v.mean()), float(v.std() + 1e-8)
        elif self.kind == 'multiclass':
            self.classes = sorted(s.dropna().unique().tolist())
        elif self.kind == 'multilabel':
            from collections import Counter
            c = Counter(t for x in s[train_mask].dropna() for t in {y.strip() for y in x.split(delimiter) if y.strip()})
            self.classes = sorted(t for t, n in c.items() if n >= min_count)
            assert self.classes, f'{self.name}: no label occurs >= {min_count} times in the training split'
        Y, M = self.encode(series, delimiter)
        obs = Y[M & torch.as_tensor(np.asarray(train_mask))]
        if self.kind == 'binary' and len(obs):
            n = float(obs.sum()); self.pos_weight = torch.tensor([min(20.0, max(1.0, ((len(obs) - n) / max(n, 1)) ** .5))])
        if self.kind == 'multilabel' and len(obs):
            n = obs.sum(0); self.pos_weight = ((len(obs) - n) / n.clamp(min=1)).sqrt().clamp(1, 20)

    def encode(self, series, delimiter):
        s = _clean(series)
        N = len(s)
        M = torch.tensor(s.notna().values)
        Y = torch.zeros(N, max(self.dim, 1))
        if self.kind == 'regression':
            v = pd.to_numeric(s, errors='coerce')
            M &= torch.tensor(v.notna().values)
            Y[:, 0] = torch.tensor(((v.fillna(self.mean) - self.mean) / self.std).values, dtype=torch.float)
        elif self.kind == 'binary':
            low = s.str.lower().fillna('0')
            num = pd.to_numeric(low, errors='coerce')
            v = np.where(low.isin(_TRUE), 1.0, np.where(low.isin(_FALSE), 0.0, num.values))
            M &= torch.tensor(~np.isnan(v))
            Y[:, 0] = torch.tensor(np.nan_to_num(v), dtype=torch.float)
        elif self.kind == 'multiclass':
            ix = {c: i for i, c in enumerate(self.classes)}
            for i, x in enumerate(s):
                if x is not pd.NA and x in ix:
                    Y[i, ix[x]] = 1
                else:
                    M[i] = False
        else:
            ix = {c: i for i, c in enumerate(self.classes)}
            for i, x in enumerate(s):
                if x is pd.NA:
                    continue
                for t in x.split(delimiter):
                    if t.strip() in ix:
                        Y[i, ix[t.strip()]] = 1
        return Y, M

    # ---- model output <-> natural scale ---------------------------------------
    def to_pred(self, logits):
        if self.kind == 'regression':
            return logits[:, 0] * self.std + self.mean
        if self.kind == 'binary':
            return torch.sigmoid(logits[:, 0])
        return F.softmax(logits, -1) if self.kind == 'multiclass' else torch.sigmoid(logits)

    def true_natural(self, y):
        if self.kind == 'regression':
            return y[:, 0] * self.std + self.mean
        if self.kind == 'binary':
            return y[:, 0]
        return y.argmax(1) if self.kind == 'multiclass' else y

    # ---- loss -----------------------------------------------------------------
    def loss(self, logits, y, m):
        if not m.any():
            return None
        l, y = logits[m], y[m]
        if self.kind == 'regression':
            return F.smooth_l1_loss(l[:, 0], y[:, 0])
        if self.kind == 'binary':
            pw = self.pos_weight.to(l.device) if self.pos_weight is not None else None
            return F.binary_cross_entropy_with_logits(l[:, 0], y[:, 0], pos_weight=pw)
        if self.kind == 'multiclass':
            return F.cross_entropy(l, y.argmax(1))
        w = y * self.pos_weight.to(l.device) + (1 - y)
        return (w * F.binary_cross_entropy_with_logits(l, y, reduction='none')).mean()

    # ---- (de)serialisation ----------------------------------------------------
    def state(self):
        return dict(name=self.name, kind=self.kind, classes=self.classes, mean=self.mean, std=self.std,
                    pos_weight=None if self.pos_weight is None else self.pos_weight.tolist())

    @classmethod
    def from_state(cls, d):
        t = cls(d['name'], d['kind'])
        t.classes, t.mean, t.std = d['classes'], d['mean'], d['std']
        t.pos_weight = None if d['pos_weight'] is None else torch.tensor(d['pos_weight'])
        return t


# ----------------------------------------------------------------------------- metrics
def fmax(y, p, steps=100):
    best, thr = 0.0, 0.5
    for t in np.linspace(0.01, 1.0, steps):
        pred = p >= t
        tp = (pred & (y > 0)).sum(1)
        npred = pred.sum(1)
        has = npred > 0
        if not has.any():
            continue
        prec, rec = (tp[has] / npred[has]).mean(), (tp / np.maximum(y.sum(1), 1)).mean()
        if prec + rec > 0 and 2 * prec * rec / (prec + rec) > best:
            best, thr = 2 * prec * rec / (prec + rec), float(t)
    return best, thr


def metrics(kind, y, p):
    """y, p numpy in natural scale (see Target.true_natural / to_pred). Returns dict; PRIMARY[kind] is the selection score."""
    n = len(y)
    if kind == 'regression':
        sst = ((y - y.mean()) ** 2).sum()
        return dict(n=n, spearman=float(spearmanr(y, p)[0]) if n > 2 and np.std(p) > 0 else 0.0,
                    pearson=float(pearsonr(y, p)[0]) if n > 2 and np.std(p) > 0 else 0.0,
                    r2=float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else float('nan'),
                    rmse=float(np.sqrt(((y - p) ** 2).mean())), mae=float(np.abs(y - p).mean()))
    if kind == 'binary':
        out = dict(n=n, pos_rate=float(y.mean()), accuracy=float(accuracy_score(y, p >= .5)),
                   mcc=float(matthews_corrcoef(y, p >= .5)))
        ok = 0 < y.sum() < n
        out.update(auroc=float(roc_auc_score(y, p)) if ok else float('nan'),
                   auprc=float(average_precision_score(y, p)) if ok else float('nan'))
        return out
    if kind == 'multiclass':
        pred = p.argmax(1)
        return dict(n=n, accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, average='macro')),
                    majority_baseline=float((y == np.bincount(y.astype(int)).argmax()).mean()))
    f, t = fmax(y, p)
    keep = (y.sum(0) > 0) & (y.sum(0) < n)
    return dict(n=n, fmax=f, fmax_thr=t, aupr_micro=float(average_precision_score(y.ravel(), p.ravel())),
                aupr_macro=float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in np.where(keep)[0]])) if keep.any() else float('nan'))


def score(kind, y, p):
    v = metrics(kind, y, p)[PRIMARY[kind]]
    return 0.0 if v != v else v


def bootstrap_ci(kind, y, p, groups, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    if len(ug) < 5:
        return None
    mem = {g: np.where(groups == g)[0] for g in ug}
    v = []
    for _ in range(n_boot):
        ix = np.concatenate([mem[g] for g in rng.choice(ug, len(ug))])
        if kind == 'binary' and not 0 < y[ix].sum() < len(ix):
            continue
        v.append(score(kind, y[ix], p[ix]))
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) > 20 else None
