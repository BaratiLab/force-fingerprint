"""Work out what each column of an uploaded table is. Explicit settings (data.columns / CLI) always win; otherwise
roles are detected from column names AND values, and every decision is reported so the user can correct it."""
import re

import pandas as pd

AA20 = set('ACDEFGHIKLMNPQRSTVWY')
NAMES = dict(
    id=['protein_id', 'id', 'name', 'accession', 'entry', 'uniprot', 'uid', 'protein', 'gene', 'identifier', 'seq_id', 'sample'],
    sequence=['sequence', 'seq', 'aa_sequence', 'protein_sequence', 'aaseq', 'peptide', 'primary_sequence'],
    structure_file=['structure_file', 'pdb_file', 'structure', 'structure_path', 'pdb_path', 'cif_file'],
    pdb_id=['pdb_id', 'pdb', 'pdbid', 'pdb_code'],
    chain_id=['chain_id', 'chain'],
    uniprot_id=['uniprot_id', 'uniprot', 'uniprot_accession', 'accession', 'uniprotkb'],
    split=['split', 'set', 'fold', 'subset', 'partition', 'dataset'],
    group=['group', 'cluster', 'cluster_id', 'family'],
    interpro=['interpro', 'interpro_terms', 'interpro_ids', 'domains'],
)
UNIPROT = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$')
SPLIT_VALUES = {'train', 'training', 'valid', 'val', 'validation', 'dev', 'test', 'testing'}


def clean_seq(s):
    return re.sub(r'[\s\-\*\.]', '', str(s)).upper()


def _looks_like_sequence(col):
    v = col[col.str.strip() != ''].head(200).map(clean_seq)
    if len(v) < max(1, 0.5 * min(len(col), 200)):
        return 0.0
    ok = v.map(lambda s: len(s) >= 5 and len(set(s) & AA20) / max(1, len(set(s))) > 0.6 and sum(c in AA20 for c in s) / len(s) > 0.9)
    return float(ok.mean()) * min(1.0, v.str.len().median() / 20)


def detect(df, explicit):
    """-> (roles {role: column|None}, notes [str]). `explicit` maps role -> column name given by the user."""
    cols = {c.lower(): c for c in df.columns}
    roles, notes, used = {}, [], set()
    for role, c in (explicit or {}).items():
        if c in df.columns:
            roles[role] = c; used.add(c); notes.append(f'{role:<15}= {c!r} (set by you)')
        elif c:
            raise ValueError(f'You set {role} = {c!r}, but the table has no such column. Columns: {list(df.columns)}')

    def pick(role, test=None):
        if role in roles:
            return
        for n in NAMES[role]:
            c = cols.get(n)
            if c and c not in used and (test is None or test(df[c])):
                roles[role] = c; used.add(c); notes.append(f'{role:<15}= {c!r} (by name)'); return

    # sequence: by name if the values look like proteins, otherwise by values alone
    pick('sequence', lambda s: _looks_like_sequence(s) > 0.8)
    if 'sequence' not in roles:
        scores = {c: _looks_like_sequence(df[c]) for c in df.columns if c not in used}
        best = max(scores, key=scores.get, default=None)
        if best and scores[best] > 0.8:
            roles['sequence'] = best; used.add(best); notes.append(f"{'sequence':<15}= {best!r} (values look like amino-acid sequences)")
    pick('id', lambda s: s.is_unique and (s.str.strip() != '').all())
    if 'id' not in roles:
        for c in df.columns:
            if c not in used and df[c].is_unique and (df[c].str.strip() != '').all() and not pd.to_numeric(df[c], errors='coerce').notna().all():
                roles['id'] = c; used.add(c); notes.append(f"{'id':<15}= {c!r} (first column of unique text values)"); break
    pick('structure_file', lambda s: s.str.lower().str.contains(r'\.(pdb|cif|ent|mmcif)(?:\.gz)?$').mean() > 0.5)
    pick('pdb_id', lambda s: s[s.str.strip() != ''].str.match(r'^[0-9][A-Za-z0-9]{3}([.:_][A-Za-z0-9])?(;|$)').mean() > 0.8)
    pick('uniprot_id', lambda s: s[s.str.strip() != ''].map(lambda x: bool(UNIPROT.match(x.strip()))).mean() > 0.8)
    pick('chain_id', lambda s: s.str.strip().str.len().max() <= 2)
    pick('split', lambda s: set(s.str.strip().str.lower()) <= SPLIT_VALUES and len(set(s.str.strip().str.lower())) >= 2)
    pick('interpro', lambda s: s[s.str.strip() != ''].str.match(r'^IPR[0-9]{6}').mean() > 0.8)
    pick('group')
    if 'sequence' not in roles:
        raise ValueError('Could not find a sequence column. Name it "sequence" (or pass --columns sequence=<name>). '
                         f'Columns seen: {list(df.columns)}')
    for role in NAMES:
        roles.setdefault(role, None)
    return roles, notes
