import datetime
import os
import pickle

import faiss
import numpy as np
import torch

from cdr_graft_pipeline import to_numpy_safe, graft_one_in_batch


# ---------- 工具函数 ----------
def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
    return torch.distributed.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def unwrap(model):
    import torch.nn as nn
    return model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model


def ddp_setup(arguments):
    """
    根据 torchrun 注入的环境变量进行 DDP 初始化。
    需要使用: torchrun --nproc_per_node=8 your_script.py ...
    """
    # torchrun 会设置 LOCAL_RANK/RANK/WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # 单机多卡：NCCL + env:// 初始化
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=120))

    # 把信息写回 arguments（可选）
    arguments.local_rank = local_rank
    arguments.world_size = world_size
    arguments.rank = rank

    return local_rank, world_size, rank


def ddp_cleanup():
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _load_indices_and_meta(save_dir):
    idx_ab = faiss.read_index(os.path.join(save_dir, 'ab_index.faiss'))
    with open(os.path.join(save_dir, 'ab_meta.pkl'), 'rb') as f:
        ab_meta = pickle.load(f)
    with open(os.path.join(save_dir, 'ag_meta.pkl'), 'rb') as f:
        ag_meta = pickle.load(f)
    return idx_ab, ab_meta, ag_meta


def _self_indices_and_seqs(ag_data):
    S = ag_data['aa']
    M = ag_data['mask'].to(torch.bool)
    seqs = [S[b][M[b]].detach().cpu().numpy() for b in range(S.size(0))]
    return seqs


def _match_self_indices(seqs_per_query, ag_meta):
    """用长度+序列在 ag_meta 中匹配“自身索引”"""
    self_indices = []
    for seq_q in seqs_per_query:
        self_idx = -1
        for j, meta in enumerate(ag_meta):
            if len(meta['aa']) == len(seq_q) and np.array_equal(meta['aa'].detach().cpu(), seq_q):
                self_idx = j
                continue
        self_indices.append(self_idx)
    return self_indices


def _search_and_filter(idx_ab, q_ag, ag_meta, ab_meta, self_indices, seqs_per_query, k=5, topn=3,
                       self_sim_thresh=0.999):
    """kNN 检索 + 排除自身（与原逻辑一致）"""
    D, I = idx_ab.search(q_ag, k)  # (B, k)
    results = []
    for qi, self_idx in enumerate(self_indices):
        I_row, D_row, seq_q = I[qi], D[qi], seqs_per_query[qi]
        kept = []
        for idx, sc in zip(I_row, D_row):
            # 已知自身 → 排除
            if 0 <= self_idx == idx:
                continue
            # 未知自身 → 用相似度+同序列兜底排除
            if self_idx < 0 and sc > self_sim_thresh and len(ag_meta[idx]['S']) == len(seq_q) \
                    and np.array_equal(ag_meta[idx]['S'], seq_q):
                continue
            kept.append((idx, sc))

        if not kept:
            results.append([])
            continue
        kept.sort(key=lambda x: -x[1])
        sel = kept[:topn]

        results.append([
            {
                'index': int(idx),
                'score': float(sc),
                'ab_meta': {k: ag_or_ab[k] for k in
                            ['pos_heavyatom', 'aa', 'fragment_type', 'mask_heavyatom', 'res_nb', 'chain_nb', 'cdr_flag',
                             'generate_flag']
                            if k in (ag_or_ab := ab_meta[idx])},  # 取 ab_meta
                'ag_meta': {k: ag_or_ab2[k] for k in
                            ['pos_heavyatom', 'aa', 'fragment_type', 'mask_heavyatom', 'res_nb', 'chain_nb', 'cdr_flag',
                             'generate_flag']
                            if k in (ag_or_ab2 := ag_meta[idx])},  # 取 ag_meta
            }
            for idx, sc in sel
        ])
    return results


def _build_complex_items_from_batch(batch, ab_data, results):
    """与原逻辑一致：对每个样本做 CDR 移植，收集 rt_metas 与 complex_list"""
    rt_metas, complex_list = [], []
    ab_batch_ids = to_numpy_safe(ab_data['batch_id'])
    for i, sid in enumerate(np.unique(ab_batch_ids)):
        candidates = results[i]
        if not candidates:
            continue
        cand = candidates[0]  # 取最相似
        ab_src = cand['ab_meta']

        rt_meta = graft_one_in_batch(
            batch, sid, ab_src,
            chain_label=1,  # 与原逻辑保持一致
            k_anchor=2,
            atom_indices={'N': 0, 'CA': 1, 'C': 2, 'O': 3},
            prefer_full_backbone=True
        )
        rt_metas.append(rt_meta)

        batch_ids_np = batch['batch_id'].detach().cpu().numpy()
        mask_ = (batch_ids_np == sid)
        exclude_keys = {'edges', 'edge_features', 'lengths', 'batch_id', 'fasta_pack', 'icode', 'chain_tag'}
        sample = {k: v[mask_] for k, v in batch.items() if k not in exclude_keys}
        complex_list.append(sample)
    return rt_metas, complex_list


def split_antigen_antibody_batch(batch: dict, device):
    """
    输入：
      batch: collate_fn 拼好的一个 mini-batch 字典
        - batch['tag']: LongTensor([N_total])
        - 其他 key 对应的是 [N_total, ...] 或 [E_total, ...]
    返回：
      two dicts ag_batch, ab_batch，结构同 batch 但只保留对应 tag 的条目
    """
    tags = batch['tag']  # LongTensor([N_total])
    ag_mask = tags == 0  # BoolTensor([N_total])
    ab_mask = tags > 0  # BoolTensor([N_total])

    ag_idx = torch.nonzero(ag_mask, as_tuple=False).view(-1)  # 原始抗原节点 idx
    ab_idx = torch.nonzero(ab_mask, as_tuple=False).view(-1)  # 原始抗体节点 idx

    # 2. 构造重映射表：原始 idx -> 子图新 idx
    N_total = tags.size(0)
    ag_map = torch.full((N_total,), -1, dtype=torch.long, device=device)
    ab_map = torch.full((N_total,), -1, dtype=torch.long, device=device)
    ag_map[ag_idx] = torch.arange(ag_idx.size(0), device=device)
    ab_map[ab_idx] = torch.arange(ab_idx.size(0), device=device)

    ag_batch = {}
    ab_batch = {}
    for key, val in batch.items():
        if key in ('edges', 'edge_features', 'lengths', 'batch_id', 'fasta_pack', 'icode', 'chain_tag'):
            continue
        # 假设 val.shape[0] 对应节点数 N_total
        ag_batch[key] = val[ag_mask]
        ab_batch[key] = val[ab_mask]

    # 4. 过滤并重映射边和边特征
    edges = batch['edges']  # [2, E_total]
    e_feats = batch['edge_features']  # [E_total, F]

    # 对每个边端点都要属于同一子图
    ag_edge_mask = ag_mask[edges[0]] & ag_mask[edges[1]]  # [E_total]
    ab_edge_mask = ab_mask[edges[0]] & ab_mask[edges[1]]

    # 取出并重映射
    ag_edges = edges[:, ag_edge_mask]
    ab_edges = edges[:, ab_edge_mask]
    ag_e_feats = e_feats[ag_edge_mask]
    ab_e_feats = e_feats[ab_edge_mask]

    # 重映射端点编号
    ag_edges = ag_map[ag_edges]
    ab_edges = ab_map[ab_edges]

    # 存回子 batch
    ag_batch['edges'] = ag_edges
    ag_batch['edge_features'] = ag_e_feats

    ab_batch['edges'] = ab_edges
    ab_batch['edge_features'] = ab_e_feats

    batch_ids = batch['batch_id']
    ag_batch['batch_id'] = batch_ids[ag_mask]
    ab_batch['batch_id'] = batch_ids[ab_mask]

    ag_batch['fasta_pack'] = batch['fasta_pack']
    ab_batch['fasta_pack'] = batch['fasta_pack']
    return ag_batch, ab_batch
