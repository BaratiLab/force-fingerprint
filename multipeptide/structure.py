"""Minimal PDB / mmCIF C-alpha reader (no Biopython dependency)."""
import numpy as np

AA3 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'C', 'PYL': 'K',
}
_MODRES = ('MSE', 'SEC', 'PYL')


def _pdb_records(path):
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            rec = line[:6]
            if rec == 'ENDMDL':
                break  # first model only
            if rec not in ('ATOM  ', 'HETATM') or line[12:16].strip() != 'CA' or line[16] not in (' ', 'A', '1'):
                continue
            res = line[17:20].strip()
            if rec == 'HETATM' and res not in _MODRES:
                continue
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            out.append((line[21], res, float(line[30:38]), float(line[38:46]), float(line[46:54]), float(line[60:66] or 0)))
    return out


def _cif_records(path):
    cols, rows, in_loop = [], [], False
    with open(path) as f:
        for line in f:
            if line.startswith('_atom_site.'):
                cols.append(line.strip().split('.')[1]); in_loop = True; continue
            if in_loop and cols and not line.startswith('_'):
                if line.startswith(('#', 'loop_')):
                    break
                rows.append(line.split())
    ix = {c: i for i, c in enumerate(cols)}
    cid = 'auth_asym_id' if 'auth_asym_id' in ix else 'label_asym_id'
    seen, out = set(), []
    for r in rows:
        if len(r) < len(cols) or r[ix['group_PDB']] not in ('ATOM', 'HETATM') or r[ix['label_atom_id']].strip('"') != 'CA':
            continue
        if r[ix['pdbx_PDB_model_num']] != '1' or r[ix['label_alt_id']] not in ('.', 'A', '1'):
            continue
        res = r[ix['label_comp_id']]
        if r[ix['group_PDB']] == 'HETATM' and res not in _MODRES:
            continue
        key = (r[ix[cid]], r[ix['auth_seq_id']] if 'auth_seq_id' in ix else len(out),
               r[ix['pdbx_PDB_ins_code']] if 'pdbx_PDB_ins_code' in ix else '')
        if key in seen:
            continue
        seen.add(key)
        out.append((r[ix[cid]], res, float(r[ix['Cartn_x']]), float(r[ix['Cartn_y']]), float(r[ix['Cartn_z']]),
                    float(r[ix['B_iso_or_equiv']]) if 'B_iso_or_equiv' in ix else 0.0))
    return out


def load_chains(path):
    """{chain_id: (sequence str, coords[L,3] float32, bfactor[L] float32)} for every polymer chain with C-alpha atoms."""
    recs = _cif_records(path) if str(path).endswith(('.cif', '.mmcif')) else _pdb_records(path)
    chains = {}
    for c, res, x, y, z, b in recs:
        chains.setdefault(c, []).append((AA3.get(res, 'X'), x, y, z, b))
    return {c: (''.join(r[0] for r in v), np.array([r[1:4] for r in v], dtype=np.float32), np.array([r[4] for r in v], dtype=np.float32))
            for c, v in chains.items()}


def predicted_by(path):
    """'alphafold' / 'esmfold' if the file header says it is a model prediction (its B-factor column is then pLDDT), else None.
    Experimental files keep thermal B-factors there, which say nothing about confidence."""
    try:
        with open(path, errors='ignore') as f:
            head = f.read(4000).upper()
    except OSError:
        return None
    return 'alphafold' if 'ALPHAFOLD' in head else 'esmfold' if 'ESMFOLD' in head else None


def plddt_percent(b):
    """pLDDT on a 0-100 scale (ESMFold writes 0-1)."""
    b = np.asarray(b, dtype=np.float32)
    return b * 100.0 if b.size and float(b.max()) <= 1.0 else b
