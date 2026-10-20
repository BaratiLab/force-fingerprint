import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def view_mask(mask, p, training):
    """Random residue dropout used as a data-augmentation 'view' (training only; always keeps >= 1 residue)."""
    if not training or p <= 0:
        return mask
    keep = mask & (torch.rand(mask.shape, device=mask.device) > p)
    return torch.where(keep.any(1, keepdim=True), keep, mask)


# ----------------------------------------------------------------------------- pooling
INIT_TEMPERATURE, MAX_LOGIT_SCALE = 0.07, 100.0      # contrastive temperature start (CLIP's value) and cap on the learned logit scale


class AttnPool(nn.Module):
    """Masked attention pooling (+ masked mean). Optional guidance:
       * `prior` [B,L]: per-residue weights in [0,1] (any per-residue annotation), added as a log-prior to the attention logits;
       * `guide` [B,G]: a per-protein vector (the InterPro-domain embedding) that queries the residue states, so the pooling
         attends to the residues that fit the protein's known domains (DPFunc-style domain-guided attention)."""
    def __init__(self, dim, guide_dim=None):
        super().__init__()
        self.dim = dim
        self.score = nn.Linear(dim, 1)
        self.key = nn.Linear(dim, dim) if guide_dim else None
        self.query = nn.Linear(guide_dim, dim) if guide_dim else None
        self.out = nn.Linear(2 * dim, dim)

    def forward(self, h, mask, prior=None, guide=None):
        s = self.score(h).squeeze(-1)
        if guide is not None and self.key is not None:
            s = s + (self.key(h) * self.query(guide).unsqueeze(1)).sum(-1) / math.sqrt(self.dim)
        if prior is not None:
            s = s + torch.log(prior.clamp(min=1e-3))
        a = s.masked_fill(~mask, float('-inf')).softmax(-1)
        att = (a.unsqueeze(-1) * h).sum(1)
        mean = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.out(torch.cat([att, mean], -1))


# ----------------------------------------------------------------------------- sequence
class SeqEncoder(nn.Module):
    """Frozen protein-language-model residue embeddings -> projection -> Transformer -> pooled vector."""
    def __init__(self, dim, layers, heads, dropout, plm_dim, max_len=1024, view_drop=0.0):
        super().__init__()
        self.view_drop = view_drop
        self.inp = nn.Sequential(nn.LayerNorm(plm_dim), nn.Linear(plm_dim, dim), nn.Dropout(dropout))
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.pool = AttnPool(dim)

    def forward(self, batch):
        mask = view_mask(batch['mask'], self.view_drop, self.training)
        x = self.inp(batch['plm'])
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.pool(self.norm(self.enc(x, src_key_padding_mask=~mask)), mask)


# ----------------------------------------------------------------------------- graph
def rbf(d, n, max_dist):
    mu = torch.linspace(0, max_dist, n, device=d.device)
    return torch.exp(-((d.unsqueeze(-1) - mu) / (max_dist / n)) ** 2)


def ca_geometry(xyz, mask):
    """SE(3)-invariant per-residue backbone features from the CA trace (angle & torsion sin/cos)."""
    v = F.normalize(xyz[:, 1:] - xyz[:, :-1], dim=-1, eps=1e-6)
    B, L, _ = xyz.shape
    feats = xyz.new_zeros(B, L, 4)
    ang = torch.acos((-v[:, :-1] * v[:, 1:]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
    feats[:, 1:-1, 0], feats[:, 1:-1, 1] = torch.sin(ang), torch.cos(ang)
    b0, b1, b2 = -v[:, :-2], v[:, 1:-1], v[:, 2:]
    b1n = F.normalize(b1, dim=-1, eps=1e-6)
    p = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    q = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    tor = torch.atan2((torch.cross(b1n, p, dim=-1) * q).sum(-1), (p * q).sum(-1))
    feats[:, 1:-2, 2], feats[:, 1:-2, 3] = torch.sin(tor), torch.cos(tor)
    ok1 = mask & F.pad(mask[:, 1:], (0, 1))
    ok2 = ok1 & F.pad(mask[:, 2:], (0, 2))
    feats[..., :2] *= ok1.unsqueeze(-1)
    feats[..., 2:] *= ok2.unsqueeze(-1)
    return feats


class EdgeBuilder(nn.Module):
    """Dense neighbour lists [B,L,K] with invariant edge features.
    Edge set = structural contacts (k nearest CA within `cutoff` A; DPFunc uses 10 A)
             U optional top-`kc` coupling partners from a pairwise matrix C (typed, so long-range
               couplings are not lost when they are not structural contacts).
    Edge feature = RBF(distance) | clipped sequence offset | coupling value | edge type."""
    def __init__(self, k, cutoff, n_rbf, max_dist, kc=0):
        super().__init__()
        self.k, self.cutoff, self.n_rbf, self.max_dist, self.kc = k, cutoff, n_rbf, max_dist, kc
        self.off = nn.Embedding(65, 16)
        self.typ = nn.Embedding(2, 4)
        self.edge_dim = n_rbf + 16 + 1 + 4

    def forward(self, xyz, mask, C=None):
        B, L, _ = xyz.shape
        d = torch.cdist(xyz, xyz).masked_fill(~(mask.unsqueeze(1) & mask.unsqueeze(2)), float('inf'))
        k = min(self.k, L)
        dist, nbr = d.topk(k, dim=-1, largest=False)
        nmask = torch.isfinite(dist) & mask.unsqueeze(-1)
        if self.cutoff:
            nmask = nmask & ((dist <= self.cutoff) | (nbr == torch.arange(L, device=xyz.device).view(1, L, 1)))
        etype = torch.zeros_like(nbr)
        if C is not None and self.kc:
            c = C.abs().masked_fill(~(mask.unsqueeze(1) & mask.unsqueeze(2)), -1.0)
            c = c.masked_fill(torch.eye(L, dtype=torch.bool, device=xyz.device).unsqueeze(0), -1.0)
            kc = min(self.kc, L)
            cv, cn = c.topk(kc, dim=-1)
            bi = torch.arange(B, device=xyz.device).view(B, 1, 1)
            cd = d[bi, torch.arange(L, device=xyz.device).view(1, L, 1), cn]
            nbr = torch.cat([nbr, cn], 2)
            dist = torch.cat([dist, cd], 2)
            nmask = torch.cat([nmask, (cv >= 0) & mask.unsqueeze(-1) & torch.isfinite(cd)], 2)
            etype = torch.cat([etype, torch.ones_like(cn)], 2)
        dist = torch.where(nmask, dist, torch.zeros_like(dist))
        idx = torch.arange(L, device=xyz.device).view(1, L, 1)
        off = (nbr - idx).clamp(-32, 32) + 32
        bi = torch.arange(B, device=xyz.device).view(B, 1, 1)
        cval = C[bi, idx.expand(B, L, 1), nbr].unsqueeze(-1) if C is not None else torch.zeros_like(dist).unsqueeze(-1)
        e = torch.cat([rbf(dist, self.n_rbf, self.max_dist), self.off(off), cval, self.typ(etype)], -1)
        return e, nbr, nmask


class GeoLayer(nn.Module):
    """Edge-conditioned attention message passing over dense neighbour lists."""
    def __init__(self, dim, edge_dim, heads, dropout):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.e = nn.Linear(edge_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, nbr, nmask):
        B, L, K = nbr.shape
        x = self.n1(h)
        bi = torch.arange(B, device=h.device).view(B, 1, 1)
        eh = self.e(e)
        q = self.q(x).view(B, L, 1, self.heads, self.dh)
        k = (self.k(x)[bi, nbr] + eh).view(B, L, K, self.heads, self.dh)
        v = (self.v(x)[bi, nbr] + eh).view(B, L, K, self.heads, self.dh)
        s = (q * k).sum(-1) / math.sqrt(self.dh)
        s = torch.nan_to_num(s.masked_fill(~nmask.unsqueeze(-1), float('-inf')).softmax(2))
        h = h + self.drop(self.o((s.unsqueeze(-1) * v).sum(2).reshape(B, L, -1)))
        return h + self.drop(self.ff(self.n2(h)))


class StructEncoder(nn.Module):
    """Residue contact graph (CA-CA < cutoff) -> invariant geometric GNN -> pooled readout (domain-guided when a domain embedding is given).
    Geometry only (distances, sequence offsets, backbone angles/torsions): the branch cannot see residue identity, so
    sequence<->structure alignment is a genuine cross-modal task and the structure may come from any source."""
    def __init__(self, g, dropout, guide_dim=None, view_drop=0.0):
        super().__init__()
        self.view_drop = view_drop
        dim = g['dim']
        self.geo = nn.Linear(4, dim)
        self.node0 = nn.Parameter(torch.zeros(dim))
        self.edges = EdgeBuilder(g['k_neighbors'], g.get('cutoff', 10.0), g['rbf'], g['max_dist'])
        self.layers = nn.ModuleList(GeoLayer(dim, self.edges.edge_dim, g['heads'], dropout) for _ in range(g['layers']))
        self.norm = nn.LayerNorm(dim)
        self.pool = AttnPool(dim, guide_dim)

    def forward(self, batch, guide=None):
        xyz, mask = batch['sxyz'], view_mask(batch['smask'], self.view_drop, self.training)
        e, nbr, nmask = self.edges(xyz, mask)
        h = self.geo(ca_geometry(xyz, batch['smask'])) + self.node0
        for layer in self.layers:
            h = layer(h, e, nbr, nmask)
        return self.pool(self.norm(h), mask, guide=guide)


class TabularEncoder(nn.Module):
    """Vector of table columns (standardised values + missing flags) -> readout."""
    def __init__(self, in_dim, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, f, batch=None):
        return self.net(f['x'])


class ArrayEncoder(nn.Module):
    """Per-residue values from files (indexed on the structure's residues) -> readout. Values are embedded, refined by a GNN on the structure's
    contact graph (+ typed edges to the strongest `pair` partners when given), then pooled with the optional residue `prior`;
    an optional protein-level `vec` is added to the readout. Which keys exist is decided by the data, not by the code."""
    def __init__(self, st, dim, g, dropout=0.1, layers=2, heads=4, k_pair=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(st['dim'] + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edges = EdgeBuilder(g['k_neighbors'], g.get('cutoff', 10.0), g['rbf'], g['max_dist'], kc=k_pair if st['has_pair'] else 0)
        self.layers = nn.ModuleList(GeoLayer(dim, self.edges.edge_dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.pool = AttnPool(dim)
        self.vec = nn.Sequential(nn.Linear(st['vec_dim'], dim), nn.GELU()) if st['vec_dim'] else None

    def forward(self, f, batch):
        mask = batch['smask']
        h = self.inp(torch.cat([f['x'], f['xmask'].unsqueeze(-1).float()], -1))
        e, nbr, nmask = self.edges(batch['sxyz'], mask, f.get('pair'))
        for layer in self.layers:
            h = layer(h, e, nbr, nmask)
        m = mask & f['xmask']
        m = torch.where(m.any(1, keepdim=True), m, mask)               # never an all-masked row
        out = self.pool(self.norm(h), m, prior=f['prior'])
        return out if self.vec is None else out + self.vec(f['vec'])


# ----------------------------------------------------------------------------- heads / model
class Projection(nn.Module):
    """Residual projection head -> L2-normalised alignment embedding."""
    def __init__(self, din, dout, dropout):
        super().__init__()
        self.proj, self.fc = nn.Linear(din, dout), nn.Linear(dout, dout)
        self.drop, self.norm = nn.Dropout(dropout), nn.LayerNorm(dout)

    def forward(self, x):
        p = self.proj(x)
        return F.normalize(self.norm(p + self.drop(self.fc(F.gelu(p)))), dim=-1)


class MultiPeptide(nn.Module):
    """Modalities: seq | struct | any feature groups (tabular columns / array files).  Each modality m gives a readout h_m.

    Contrastive space : z_m = normalise(Projection(h_m)); symmetric InfoNCE between every pair of available modalities.
    Prediction        : logits = MLP_int([h_1..h_M]) + sum_m w_m * E_m(h_m)
                          E_m : per-modality expert head, supervised on its own (auxiliary loss) so every encoder receives gradient
                                even when another modality already fits the training set;
                          w_m : per-target softmax gates over the *available* modalities (learned, interpretable);
                          MLP_int : interaction term, zero-initialised, so training starts from the expert ensemble and learns
                                    only the residual (an additive model with interactions)."""
    def __init__(self, cfg, targets, n_domains=0, fstates=()):
        super().__init__()
        s, g, m, tr = cfg['seq_encoder'], cfg['struct_encoder'], cfg['model'], cfg['train']
        vd = tr.get('view_dropout', 0.0)
        self.use_struct = bool(cfg['structure']['use'])
        self.fstates = {st['name']: st for st in fstates}
        dd = m.get('domain_dim', 64)       # InterPro domains guide the structure readout, so they exist only with structure
        self.dom = nn.Sequential(nn.Linear(n_domains, dd), nn.GELU()) if n_domains > 0 and self.use_struct else None
        self.enc, dims = nn.ModuleDict(), {}
        fdim = m.get('feature_dim', 64)
        self.enc['seq'] = SeqEncoder(s['dim'], s['layers'], s['heads'], m['dropout'], s['plm_dim'], view_drop=vd)
        dims['seq'] = s['dim']
        if self.use_struct:
            self.enc['struct'] = StructEncoder(g, m['dropout'], dd if self.dom is not None else None, view_drop=vd)
            dims['struct'] = g['dim']
        for name, st in self.fstates.items():
            assert st['kind'] == 'tabular' or self.use_struct, f"feature group {name!r} holds per-residue arrays, which need the structure modality (model.modalities must include struct)"
            self.enc[name] = TabularEncoder(2 * st['dim'], fdim, m['head_dropout']) if st['kind'] == 'tabular' else \
                ArrayEncoder(st, fdim, g, m['head_dropout'])
            dims[name] = fdim
        self.mods = tuple(dims)
        self.dims = dims
        self.proj = nn.ModuleDict({k: Projection(d, m['proj_dim'], m['head_dropout']) for k, d in dims.items()})
        self.pre_norm = nn.ModuleDict({k: nn.LayerNorm(d) for k, d in dims.items()})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / INIT_TEMPERATURE)))
        self.max_logit_scale = math.log(MAX_LOGIT_SCALE)
        self.mod_drop = m['modality_dropout']
        self.drop_p = None
        self.targets = list(targets)
        self.outs = [max(t.dim, 1) for t in targets]
        tot = sum(dims.values())
        self.trunk = nn.Sequential(nn.Dropout(m['head_dropout']), nn.Linear(tot, tot), nn.GELU(), nn.Dropout(m['head_dropout']))
        self.int_head = nn.Linear(tot, sum(self.outs))
        nn.init.zeros_(self.int_head.weight); nn.init.zeros_(self.int_head.bias)
        self.expert = nn.ModuleDict({k: nn.Sequential(nn.Dropout(m['head_dropout']), nn.Linear(d, sum(self.outs))) for k, d in dims.items()})
        self.gate = nn.Parameter(torch.zeros(len(targets), len(dims)))

    def _split(self, x):
        return {t.name: v for t, v in zip(self.targets, torch.split(x, self.outs, -1))}

    def available(self, batch):
        B = batch['mask'].shape[0]
        av = {'seq': torch.ones(B, dtype=torch.bool, device=batch['mask'].device)}
        if self.use_struct:
            av['struct'] = batch['has_struct']
        for name, st in self.fstates.items():
            av[name] = batch['has'][name] & (batch['has_struct'] if st['kind'] == 'array' else True)
        return av

    def set_drop(self, probs=None):
        """Per-modality probability of hiding a modality during training (default: model.modality_dropout for all).
        Used in the student phase so the head also learns the structure-free path (probs={'struct': 0.5})."""
        self.drop_p = None if probs is None else torch.tensor([float(probs.get(k, 0.0)) for k in self.mods])

    def forward(self, batch, use=None, skip_unused=False):
        """use: modalities allowed to reach the prediction (ablation / deployment path). Default: all.
        skip_unused: do not even run the encoders of modalities outside `use` (fast sequence-only inference; no z for them)."""
        use = tuple(use) if use is not None else self.mods
        active = [k for k in self.mods if (k in use or not skip_unused)]
        B = batch['mask'].shape[0]
        h = {}
        for k in active:
            if k == 'seq':
                h[k] = self.enc[k](batch)
            elif k == 'struct':
                h[k] = self.enc[k](batch, self.dom(batch['domain']) if self.dom is not None else None)
            else:
                h[k] = self.enc[k](batch['feat'][k], batch)
        z = {k: self.proj[k](v) for k, v in h.items()}
        av = self.available(batch)
        keep = torch.stack([av[k] & (k in use) for k in self.mods], 1)
        p = torch.full((len(self.mods),), float(self.mod_drop)) if self.drop_p is None else self.drop_p
        if self.training and float(p.max()) > 0:
            kept = keep & ~(torch.rand_like(keep.float()) < p.to(keep.device))
            keep = torch.where(kept.any(1, keepdim=True), kept, keep)               # never drop everything
        hn = [self.pre_norm[k](h[k]) if k in h else torch.zeros(B, self.dims[k], device=keep.device) for k in self.mods]
        x = self.trunk(torch.cat([a * keep[:, i:i + 1] for i, a in enumerate(hn)], -1))
        logit = self.int_head(x)
        ex = torch.stack([self.expert[k](hn[i]) if k in h else torch.zeros(B, sum(self.outs), device=keep.device)
                          for i, k in enumerate(self.mods)], 1)                                      # [B,M,sum(outs)]
        gl = self.gate.unsqueeze(0).expand(len(keep), -1, -1).masked_fill(~keep.unsqueeze(1), float('-inf'))
        w = torch.nan_to_num(torch.softmax(gl, -1))                                              # [B,T,M]; 0 if nothing available
        per_t = torch.split(ex, self.outs, -1)
        logit = logit + torch.cat([(w[:, ti].unsqueeze(-1) * per_t[ti]).sum(1) for ti in range(len(self.outs))], -1)
        return dict(h=h, z=z, avail=av, scale=self.logit_scale.clamp(max=self.max_logit_scale).exp(), gate=w,
                    expert={k: self._split(ex[:, i]) for i, k in enumerate(self.mods) if k in h}, logits=self._split(logit))
