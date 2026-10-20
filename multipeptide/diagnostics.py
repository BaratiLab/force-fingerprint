"""Is the contrastive space really cross-modal? Retrieval that cannot be explained by sequence length, plus embedding geometry."""
import numpy as np


def alignment_report(acc, tol=0.10):
    """For every pair of modalities: top-k retrieval, its chance level, retrieval restricted to length-matched candidates
    (|len_j - len_i| <= tol*len_i, the honest test, since sequence and structure of one protein have the same length),
    the length-only baseline, and positive/negative cosine."""
    out = {}
    mods = list(acc['z'])
    lens = acc['mlen']
    for i, a in enumerate(mods):
        for b in mods[i + 1:]:
            ok = acc['avail'][a] & acc['avail'][b]
            n = int(ok.sum())
            if n < 5:
                continue
            za, zb = acc['z'][a][ok], acc['z'][b][ok]
            la, lb = lens[a][ok].astype(float), lens[b][ok].astype(float)
            sim = za @ zb.T
            diag = np.diag(sim)
            rank = (sim > diag[:, None]).sum(1)
            cand = np.abs(lb[None, :] - la[:, None]) <= tol * la[:, None]
            np.fill_diagonal(cand, True)
            k = cand.sum(1)
            use = k >= 3
            rm = np.array([(sim[r, cand[r]] > diag[r]).sum() for r in range(n)])
            lrank = (-np.abs(lb[None, :] - la[:, None]) > -np.abs(lb - la)[:, None]).sum(1)
            off = ~np.eye(n, dtype=bool)
            out[f'{a}-{b}'] = dict(
                n=n,
                R1=float((rank < 1).mean()), R5=float((rank < 5).mean()), chance_R1=1.0 / n,
                R1_length_matched=float((rm[use] < 1).mean()) if use.any() else None,
                chance_R1_length_matched=float((1.0 / k[use]).mean()) if use.any() else None,
                length_only_R1=float((lrank < 1).mean()),
                cos_positive=float(diag.mean()), cos_negative=float(sim[off].mean()),
                # Wang & Isola (2020): alignment = E||z_a - z_b||^2 over positives (lower is better, 0..4);
                # uniformity = log E exp(-2||z - z'||^2) per modality (lower = more uniform on the sphere, i.e. less collapse);
                # modality gap (Liang et al. 2022) = distance between the two modality centroids.
                alignment=float(((za - zb) ** 2).sum(1).mean()),
                uniformity={a: _uniformity(za), b: _uniformity(zb)},
                modality_gap=float(np.linalg.norm(za.mean(0) - zb.mean(0))))
    return out


def _uniformity(z, t=2.0):
    d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
    off = ~np.eye(len(z), dtype=bool)
    return float(np.log(np.exp(-t * d2[off]).mean()))
