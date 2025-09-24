import numpy as np
import torch

from micro_repair import micro_repair_reference
from cdr_graft_pipeline import graft_one_in_batch


# ========= 工具 =========
def to_numpy_safe(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


BB_N, BB_CA, BB_C = 0, 1, 2  # 主链索引（如顺序不同改这里）


def _ensure_seq_idx(S):
    S = to_numpy_safe(S)
    if S.ndim == 1:
        return S.astype(np.int64)
    elif S.ndim == 2:
        return np.argmax(S, axis=-1).astype(np.int64)
    else:
        raise ValueError("S must be [L] or [L,21]")


def _get_mask_res_or_true(d):
    """若无 mask_res，则返回全 True 的 [L]。"""
    if 'mask' in d and d['mask'] is not None:
        return to_numpy_safe(d['mask']).astype(bool).reshape(-1)
    # 优先从 X 推出 L，否则从 smask 推
    if 'pos_heavyatom' in d:
        L = int(to_numpy_safe(d['pos_heavyatom']).shape[0])
    elif 'generate_flag' in d:
        L = int(to_numpy_safe(d['generate_flag']).shape[0])
    else:
        raise ValueError("Cannot infer L: need X or smask when mask_res is absent.")
    return np.ones((L,), dtype=bool)


def _get_xloss_mask_or_true(d):
    """若无 xloss_mask，则返回全 True 的 [L,A]；如果 d['pos_heavyatom'] 只有 CA 坐标也可退化。"""
    if 'mask_heavyatom' in d and d['mask_heavyatom'] is not None:
        return to_numpy_safe(d['mask_heavyatom'])
    # 推断 L,A
    X = to_numpy_safe(d['pos_heavyatom'])
    if X.ndim == 3:  # [L,A,3]
        L, A = X.shape[0], X.shape[1]
    elif X.ndim == 2:  # [L,3] (仅CA)
        L, A = X.shape[0], 1
    else:
        raise ValueError(f"Unexpected X shape {X.shape} for generating xloss_mask.")
    return np.ones((L, A), dtype=bool)


def _extract_ca_mask_maybe_no_masks(d):
    """
    生成每残基 CA 可用的 [L]。兼容：
    - 缺 mask_res: 全 True
    - 缺 xloss_mask: 全 True
    - xloss_mask 为 [L,A] 或 [A,L]
    """
    mr = _get_mask_res_or_true(d)  # [L]
    xm = _get_xloss_mask_or_true(d)  # [L,A] or [A,L]
    if xm.ndim != 2:
        raise ValueError(f"xloss_mask must be 2D, got {xm.shape}")
    # [L,A]
    if xm.shape[0] == mr.shape[0]:
        ca_vec = xm[:, BB_CA if xm.shape[1] > BB_CA else 0]
    # [A,L]
    elif xm.shape[1] == mr.shape[0]:
        ca_vec = xm[BB_CA if xm.shape[0] > BB_CA else 0, :]
    else:
        raise ValueError(f"mask_res length {mr.shape[0]} not matching xloss_mask {xm.shape}")
    return ca_vec.astype(bool) & mr


def kabsch(P, Q):
    P = to_numpy_safe(P)
    Q = to_numpy_safe(Q)
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = P.mean(axis=0, keepdims=True) - Q.mean(axis=0, keepdims=True) @ R.T
    return R, t


def rmsd_after_kabsch(P, Q):
    R, t = kabsch(P, Q)
    Q_aln = Q @ R.T + t
    diff = P - Q_aln
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1)) + 1e-12))


def contact_map_ca(X_ab, X_ag, thr=8.0):
    X_ab = to_numpy_safe(X_ab)
    X_ag = to_numpy_safe(X_ag)
    A_ab = X_ab[:, BB_CA, :] if X_ab.ndim == 3 else X_ab
    A_ag = X_ag[:, BB_CA, :] if X_ag.ndim == 3 else X_ag
    D = np.linalg.norm(A_ab[:, None, :] - A_ag[None, :, :], axis=-1)
    return D < thr


def f1_score(y_true, y_pred):
    y_true = to_numpy_safe(y_true).astype(bool).ravel()
    y_pred = to_numpy_safe(y_pred).astype(bool).ravel()
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp / denom))


def clash_score(X_ab, mask_ab_atoms, X_ag, mask_ag_atoms, thr=2.0, sample_cap=3000):
    X_ab = to_numpy_safe(X_ab)
    X_ag = to_numpy_safe(X_ag)
    Mab = to_numpy_safe(mask_ab_atoms).astype(bool)
    Mag = to_numpy_safe(mask_ag_atoms).astype(bool)
    Ab = X_ab[Mab]
    Ag = X_ag[Mag]
    if Ab.shape[0] == 0 or Ag.shape[0] == 0:
        return 0.0
    if Ab.shape[0] > sample_cap:
        Ab = Ab[np.random.choice(Ab.shape[0], sample_cap, replace=False)]
    if Ag.shape[0] > sample_cap:
        Ag = Ag[np.random.choice(Ag.shape[0], sample_cap, replace=False)]

    def pairwise_min(A, B, chunk=2000):
        mins = []
        for i in range(0, A.shape[0], chunk):
            Ai = A[i:i + chunk]
            d2 = np.sum((Ai[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            mins.append(np.sqrt(np.min(d2, axis=1)))
        return np.concatenate(mins, axis=0)

    dmin = pairwise_min(Ab, Ag)
    return float(np.mean(dmin < thr))


def cdr_compat_length_only(target_smask, ref_smask):
    tgt_idx = np.where(to_numpy_safe(target_smask).astype(bool))[0]
    ref_idx = np.where(to_numpy_safe(ref_smask).astype(bool))[0]
    len_t, len_r = int(tgt_idx.size), int(ref_idx.size)
    # 可按需要调节尺度（3.0 是经验值）
    return float(np.exp(-abs(len_t - len_r) / 3.0))


def ag_sequence_identity(target_S, ref_S, target_tag, ref_tag):
    tS = _ensure_seq_idx(target_S)
    rS = _ensure_seq_idx(ref_S)
    t_ag = (to_numpy_safe(target_tag).astype(int) == 0)
    r_ag = (to_numpy_safe(ref_tag).astype(int) == 0)
    tS = tS[t_ag]
    rS = rS[r_ag]
    L = min(tS.shape[0], rS.shape[0])
    return float(0.0 if L == 0 else np.mean(tS[:L] == rS[:L]))


def fr_ca_rmsd(target, ref, mapping_target2ref=None):
    X_t = to_numpy_safe(target['pos_heavyatom'])
    X_r = to_numpy_safe(ref['pos_heavyatom'])
    tgt_valid_ca = _extract_ca_mask_maybe_no_masks(target)  # [L]
    ref_valid_ca = _extract_ca_mask_maybe_no_masks(ref)  # [L]
    tgt_fr = (~to_numpy_safe(target['generate_flag'])) & tgt_valid_ca
    ref_fr = (~to_numpy_safe(ref['generate_flag'])) & ref_valid_ca
    tgt_idx = np.where(tgt_fr)[0]
    ref_idx = np.where(ref_fr)[0]
    if mapping_target2ref is None:
        L = min(tgt_idx.shape[0], ref_idx.shape[0])
        if L < 3: return 999.0
        P = X_t[tgt_idx[:L], BB_CA, :]
        Q = X_r[ref_idx[:L], BB_CA, :]
    else:
        mp = to_numpy_safe(mapping_target2ref).astype(np.int64)
        mapped = mp[tgt_idx]
        keep = mapped >= 0
        if np.sum(keep) < 3: return 999.0
        P = X_t[tgt_idx[keep], BB_CA, :]
        Q = X_r[mapped[keep], BB_CA, :]
    return rmsd_after_kabsch(P, Q)


def interface_f1_fr_only(target, ref, thr=8.0):
    """
    目标侧：只用 抗体FR(~smask 且 tag in {1,2}) 与 抗原(tag==0) 的 CA 接触图
    参考侧：同理（其 CDR 可用，用不用都行，这里也只取 FR 以保持一致）
    """

    def tag_masks_FR(d):
        tag = to_numpy_safe(d['fragment_type']).astype(int)  # [L]
        is_ag = (tag == 3)
        is_ab = np.logical_or(tag == 1, tag == 2)
        valid = _get_mask_res_or_true(d)  # 缺省则全 True
        # 只保留 Framework（~smask）
        fr_mask = (~to_numpy_safe(d['generate_flag'])) & valid
        ab_fr = is_ab & fr_mask
        ag_all = is_ag & valid
        return ab_fr, ag_all

    X_t = to_numpy_safe(target['pos_heavyatom'])
    X_r = to_numpy_safe(ref['pos_heavyatom'])
    t_ab_fr, t_ag = tag_masks_FR(target)
    r_ab_fr, r_ag = tag_masks_FR(ref)

    if np.sum(t_ab_fr) == 0 or np.sum(t_ag) == 0 or np.sum(r_ab_fr) == 0 or np.sum(r_ag) == 0:
        return 0.0

    cm_t = contact_map_ca(X_t[t_ab_fr], X_t[t_ag], thr=thr)
    cm_r = contact_map_ca(X_r[r_ab_fr], X_r[r_ag], thr=thr)

    L_ab = min(cm_t.shape[0], cm_r.shape[0])
    L_ag = min(cm_t.shape[1], cm_r.shape[1])
    if L_ab == 0 or L_ag == 0:
        return 0.0
    return f1_score(cm_t[:L_ab, :L_ag], cm_r[:L_ab, :L_ag])


def compute_clash_ref(ref, thr=2.0):
    tag = to_numpy_safe(ref['fragment_type']).astype(int)
    valid = _get_mask_res_or_true(ref)
    is_ag = (tag == 0) & valid
    is_ab = np.logical_or(tag == 1, tag == 2) & valid
    if np.sum(is_ab) == 0 or np.sum(is_ag) == 0:
        return 0.0
    X = to_numpy_safe(ref['pos_heavyatom'])
    M = _get_xloss_mask_or_true(ref)
    return clash_score(X[is_ab], M[is_ab], X[is_ag], M[is_ag], thr=thr)


def score_bundle(target, ref, weights=None, mapping_target2ref=None):
    if weights is None:
        weights = dict(fr=0.45, if_f1=0.35, clash=0.15, cdr=0.03, ag=0.02)
    fr = fr_ca_rmsd(target, ref, mapping_target2ref)  # 低好
    if1 = interface_f1_fr_only(target, ref)  # 高好
    cls = compute_clash_ref(ref)  # 低好
    cdr = cdr_compat_length_only(target['generate_flag'], ref['generate_flag'])
    agm = ag_sequence_identity(target['aa'], ref['aa'], target['fragment_type'], ref['fragment_type'])
    fr_pos = float(np.exp(-max(fr, 0.0) / 2.0))
    clash_pos = float(np.exp(-5.0 * cls))
    total = (weights['fr'] * fr_pos +
             weights['if_f1'] * if1 +
             weights['clash'] * clash_pos +
             weights['cdr'] * cdr +
             weights['ag'] * agm)
    return dict(
        fr_ca_rmsd=float(fr), fr_pos=float(fr_pos),
        interface_f1=float(if1),
        clash=float(cls), clash_pos=float(clash_pos),
        cdr_compat=float(cdr), ag_match=float(agm),
        score=float(total),
    )


def slice_target_sample(batch, sid):

    exclude_keys = {'edges', 'edge_features', 'lengths', 'batch_id', 'fasta_pack', 'icode', 'chain_tag'}
    sample = {k: (to_numpy_safe(v[sid]) if hasattr(v, "__getitem__") else v)
              for k, v in batch.items() if k not in exclude_keys}
    return sample


# ========= 主函数：在你的流程中，对所有 candidates 打分并排序 =========
def _build_and_score_candidates_from_batch(
        batch, ab_data, results,
        chain_label=1, k_anchor=2,
        atom_indices=None,
        prefer_full_backbone=True,
        top_k=1, weights=None, mapping_target2ref=None
):
    """返回 ranked_scores_per_sample, best_rt_metas, complex_list"""
    if atom_indices is None:
        atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}

    ranked_scores_per_sample, best_rt_metas, complex_list = [], [], []
    batch_size = ab_data['aa'].size(0)

    for i in range(batch_size):
        target_sample = slice_target_sample(batch, i)

        ranked = []
        if i < len(results) and results[i]:
            for j, cand in enumerate(results[i]):
                ab_src = cand['ab_meta']
                try:
                    rt_meta = graft_one_in_batch(
                        batch, i, ab_src,
                        chain_label=chain_label,
                        k_anchor=k_anchor,
                        atom_indices=atom_indices,
                        prefer_full_backbone=prefer_full_backbone
                    )
                except RuntimeError:
                    continue

                scr = score_bundle(target_sample, rt_meta, weights=weights, mapping_target2ref=mapping_target2ref)
                ranked.append(dict(cand_idx=j, scores=scr, rt_meta=rt_meta, cand=cand, score=scr['score']))
        ranked.sort(key=lambda d: d['score'], reverse=True)
        best_ref = ranked[0]['rt_meta']
        best_ref_rep = micro_repair_reference(best_ref, target=target_sample)  # 轻量修复
        best_rt_metas.append(best_ref_rep if ranked else None)
        ranked_scores_per_sample.append([
            dict(cand_idx=it['cand_idx'], **it['scores']) for it in ranked[:max(1, top_k)]
        ])
        complex_list.append(target_sample)

    return ranked_scores_per_sample, best_rt_metas, complex_list
