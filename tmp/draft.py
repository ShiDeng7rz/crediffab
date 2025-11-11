#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-check chains_emb.lmdb (features) against structures.lmdb (sequences).
- Build md5 sets for AB(heavy/light) and AG(antigen) from structures.lmdb
- Probe chains_emb.lmdb with correct keys:
    AB  -> model=antiberty_base, layer=last, tokenizer=antiberty-v1
    AG  -> model=esm2_t33_650M, layer=33,   tokenizer=ur50d
- Report hit ratio and dim histogram for each role.
"""

from __future__ import annotations

import argparse
import hashlib
import lmdb
import pickle
from collections import defaultdict, Counter
from typing import Dict, Tuple, Iterable, Optional

AA_INDEX_TO_LETTER = list("ACDEFGHIKLMNPQRSTVWYX")  # 20='X'


def aa_ids_to_string(ids) -> str:
    try:
        import torch
        if hasattr(torch, "is_tensor") and torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
    except Exception:
        pass
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ''.join(AA_INDEX_TO_LETTER[int(i)] if 0 <= int(i) < len(AA_INDEX_TO_LETTER) else 'X' for i in ids)


def _seqs_from_data_block(block) -> Dict[str, str]:
    seqs = defaultdict(list)
    chain_ids = getattr(block, 'chain_id', None)
    aa = getattr(block, 'aa', None)
    if chain_ids is None or aa is None:
        return {}
    for i in range(len(aa)):
        seqs[str(chain_ids[i])].append(int(aa[i]))
    return {cid: aa_ids_to_string(v) for cid, v in seqs.items()}


def _open_any_lmdb(path: str) -> lmdb.Environment:
    try:
        return lmdb.open(path, readonly=True, subdir=True, lock=False, readahead=True, max_dbs=1)
    except Exception:
        return lmdb.open(path, readonly=True, subdir=False, lock=False, readahead=True, max_dbs=1)


def md5_of_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def collect_md5_by_role(struct_lmdb: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (ab_dict, ag_dict): {md5: seq} for antibodies and antigens."""
    ab, ag = {}, {}
    env = _open_any_lmdb(struct_lmdb)
    with env.begin() as txn:
        for _k, v in txn.cursor():
            try:
                rec = pickle.loads(bytes(v))
            except Exception:
                continue
            # heavy
            h_seq = rec.get('heavy_seq')
            if h_seq is None and rec.get('heavy') is not None:
                h_seqs = _seqs_from_data_block(rec['heavy'])
                h_chain = str(rec.get('entry', {}).get('H_chain', '') or rec.get('H_chain') or next(iter(h_seqs), ''))
                h_seq = h_seqs.get(h_chain)
            if h_seq:
                ab.setdefault(md5_of_text(h_seq), h_seq)
            # light
            l_seq = rec.get('light_seq')
            if l_seq is None and rec.get('light') is not None:
                l_seqs = _seqs_from_data_block(rec['light'])
                l_chain = str(rec.get('entry', {}).get('L_chain', '') or rec.get('L_chain') or next(iter(l_seqs), ''))
                l_seq = l_seqs.get(l_chain)
            if l_seq:
                ab.setdefault(md5_of_text(l_seq), l_seq)
            # antigen (may be multiple)
            ag_seqs = rec.get('antigen_seqs')
            if ag_seqs is None and rec.get('antigen') is not None:
                ag_map = _seqs_from_data_block(rec['antigen'])
                ag_seqs = list(ag_map.values())
            if ag_seqs:
                for s in ag_seqs:
                    if s:
                        ag.setdefault(md5_of_text(s), s)
    env.close()
    return ab, ag


def make_key(model: str, layer: str, tokenizer: str, seq_md5: str) -> bytes:
    pool, norm = "last", "none"
    s = f"{model}:{layer}:{pool}:{norm}:{tokenizer}:{seq_md5}"
    return s.encode("utf-8")


def read_feature(env_feat: lmdb.Environment, key: bytes):
    with env_feat.begin(write=False) as txn:
        blob = txn.get(key)
    if blob is None:
        return None, None
    obj = pickle.loads(blob)  # {"emb": np.float16[L,D], "meta": {...}}
    return obj.get("emb"), obj.get("meta", {})


def probe_role(env_feat, md5s: Iterable[str], model: str, layer: str, tok: str,
               limit: Optional[int] = 5000, tag: str = ""):
    md5_list = list(md5s) if limit is None else list(md5s)[:limit]
    hits = 0
    dim_hist = Counter()
    meta_mismatch = 0
    for m in md5_list:
        emb, meta = read_feature(env_feat, make_key(model, layer, tok, m))
        if emb is None:
            continue
        hits += 1
        d = int(emb.shape[1]) if emb.ndim == 2 else -1
        dim_hist[d] += 1
        if meta and "dim" in meta and int(meta["dim"]) != d:
            meta_mismatch += 1
    total = len(md5_list)
    ratio = (hits / total * 100) if total else 0.0
    print(f"[{tag}] hit {hits}/{total} ({ratio:.1f}%), dim_hist={dict(sorted(dim_hist.items()))}, "
          f"meta_dim_mismatch={meta_mismatch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-lmdb", default="../data/processed_com/structures.lmdb")
    ap.add_argument("--feat-lmdb", default="../data/processed_com/chains_emb.lmdb")
    ap.add_argument("--limit", type=int, default=6000)
    # keys used when building features:
    ap.add_argument("--ab-model", default="antiberty_base")
    ap.add_argument("--ab-layer", default="last")
    ap.add_argument("--ab-tokenizer", default="antiberty-v1")
    ap.add_argument("--ag-model", default="esm2_t33_650M")
    ap.add_argument("--ag-layer", default="33")
    ap.add_argument("--ag-tokenizer", default="ur50d")
    args = ap.parse_args()

    ab, ag = collect_md5_by_role(args.struct_lmdb)
    print(f"[collect] structures.lmdb → AB={len(ab)}, AG={len(ag)}")

    env_feat = _open_any_lmdb(args.feat_lmdb)
    probe_role(env_feat, ab.keys(), args.ab_model, args.ab_layer, args.ab_tokenizer,
               limit=(None if args.limit <= 0 else args.limit),
               tag="AB→AntiBERTy")
    probe_role(env_feat, ag.keys(), args.ag_model, args.ag_layer, args.ag_tokenizer,
               limit=(None if args.limit <= 0 else args.limit),
               tag="AG→ESM")
    env_feat.close()


if __name__ == "__main__":
    main()
