import itertools
import torch
import torch.nn.functional as F


def info_nce(za, zb, scale, groups):
    """Symmetric InfoNCE between two modalities of the same proteins (positive = same protein).
    Proteins of the same group (close homologs) are masked out of the negatives."""
    n = za.shape[0]
    if n < 2:
        return za.new_zeros(())
    same = (groups.unsqueeze(0) == groups.unsqueeze(1)) & ~torch.eye(n, dtype=torch.bool, device=za.device)
    logits = (scale * za @ zb.T).masked_fill(same, -1e4)
    tgt = torch.arange(n, device=za.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


def align_loss(out, groups):
    """Mean symmetric InfoNCE over every pair of available modalities (only proteins having both)."""
    losses = []
    for a, b in itertools.combinations(list(out['z']), 2):
        ok = out['avail'][a] & out['avail'][b]
        if ok.sum() >= 2:
            losses.append(info_nce(out['z'][a][ok], out['z'][b][ok], out['scale'], groups[ok]))
    return torch.stack(losses).mean() if losses else groups.new_zeros((), dtype=torch.float)


def task_loss(out, batch, targets):
    """Mean over the targets that have labelled rows in the batch (missing labels are masked, not imputed)."""
    ls = [l for t in targets if (l := t.loss(out['logits'][t.name], batch['y'][t.name], batch['m'][t.name])) is not None]
    return torch.stack(ls).mean() if ls else out['scale'] * 0


def expert_loss(out, batch, targets, only=None):
    """Auxiliary loss: every modality's expert head must predict the targets *on its own* (only rows where that modality exists).
    This keeps a gradient flowing into each encoder even when the other modalities already fit the training set."""
    ls = [l for m, lg in out.get('expert', {}).items() if only is None or m in only for t in targets
          if (l := t.loss(lg[t.name], batch['y'][t.name], batch['m'][t.name] & out['avail'][m])) is not None]
    return torch.stack(ls).mean() if ls else out['scale'] * 0
