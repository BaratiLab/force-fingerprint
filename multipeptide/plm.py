"""Frozen protein-language-model residue embeddings, cached per sequence."""
import hashlib
import os
import re

import torch


def seq_key(seq):
    return hashlib.md5(seq.encode()).hexdigest()


def embed_sequences(seqs, model_name, cache_path, log=print):
    cache = torch.load(cache_path) if os.path.exists(cache_path) else {}
    todo = [s for s in dict.fromkeys(seqs) if seq_key(s) not in cache]
    if todo:
        from transformers import AutoModel, AutoTokenizer
        dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        net = AutoModel.from_pretrained(model_name).to(dev).eval()
        bert_style = 'bert' in model_name.lower()
        log(f'[plm] embedding {len(todo)} sequences with {model_name} on {dev}')
        with torch.no_grad():
            for s in todo:
                x = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', s)
                enc = tok(' '.join(x) if bert_style else x, return_tensors='pt', truncation=True, max_length=1024).to(dev)
                h = net(**enc).last_hidden_state[0][1:-1]
                assert h.shape[0] == len(s), f'embedding length {h.shape[0]} != sequence length {len(s)}'
                cache[seq_key(s)] = h.cpu().half()
        torch.save(cache, cache_path)
    return cache
