from typing import Dict, Tuple, Any

import numpy as np
import torch

from renumber import imgt_relabel_to_target_window

FIELDS = ['pos_heavyatom', 'aa', 'mask_heavyatom', 'generate_flag', 'res_nb', 'chain_nb', 'fragment_type']


def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_tensor_like(x_np, like_tensor):
    t = torch.as_tensor(x_np, device=like_tensor.device, dtype=like_tensor.dtype)
    return t


def make_sample_from_batch(ab_data, sid):
    sample = {}
    for k in FIELDS:
        if k in ab_data:
            sample[k] = to_numpy_safe(ab_data[k])[sid]
    return sample, sid


def write_back_sample_into_batch(ab_data, idxs, ab_init_sample):
    """把 graft 后的单样本（numpy）写回到 batch（torch）原位。只写会被修改的字段：X/S/xloss_mask。"""
    # X
    if 'pos_heavyatom' in ab_init_sample:
        X_new = ab_init_sample['pos_heavyatom']  # (N,14,3) numpy
        X_t = ab_data['pos_heavyatom']
        if isinstance(X_t, torch.Tensor):
            ab_data['pos_heavyatom'][idxs] = to_tensor_like(X_new, X_t)
        else:
            ab_data['pos_heavyatom'][idxs] = X_new
    # S
    if 'aa' in ab_init_sample:
        S_new = ab_init_sample['aa']
        S_t = ab_data['aa']
        if isinstance(S_t, torch.Tensor):
            ab_data['aa'][idxs] = to_tensor_like(S_new, S_t)
        else:
            ab_data['aa'][idxs] = S_new
    # xloss_mask（可选）
    if 'mask_heavyatom' in ab_init_sample and 'mask_heavyatom' in ab_data:
        XL_new = ab_init_sample['mask_heavyatom']
        XL_t = ab_data['mask_heavyatom']
        if isinstance(XL_t, torch.Tensor):
            ab_data['mask_heavyatom'][idxs] = to_tensor_like(XL_new, XL_t)
        else:
            ab_data['mask_heavyatom'][idxs] = XL_new


def _to_chain_val(chain_label):
    """将 'H'/'L'/1/2/'1'/'2' 统一转换为 1/2，其他值报错。"""
    if chain_label in ('H', 'h'):
        return 1
    if chain_label in ('L', 'l'):
        return 2
    if chain_label in (1, 2, '1', '2'):
        return int(chain_label)
    raise ValueError(f"Unsupported chain_label: {chain_label}")


def _normalize_chain_labels(ch_array):
    """
    输入: chain_id 数组 (N,)
         其中 1=重链(H)，2=轻链(L)，0=抗原
    输出: 同样 shape 的 numpy 数组 (int)
    """
    ch_array = np.asarray(ch_array.detach().cpu()) if torch.is_tensor(ch_array) else np.asarray(ch_array)

    # 直接返回 int，不需要额外映射
    return ch_array.astype(int)


def _get_cdr_indices_from_smask(chain_id, smask, chain_label):
    """
    返回 smask==1 且链别为 chain_label 的全局下标。
    chain_label 可为 'H'/'L' 或 1/2。
    """
    # ch = _normalize_chain_labels(chain_id)  # 0/1/2
    lab = _to_chain_val(chain_label)
    mask = (chain_id == lab) & smask
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        raise RuntimeError(f"未在链 {chain_label} 上找到 CDR（smask==1）")
    return idxs


def _get_chain_span_indices(chain_id, chain_label):
    """
    返回指定链别（'H'/'L' 或 1/2）在整条分子中的所有残基下标。
    """
    lab = _to_chain_val(chain_label)
    return np.where(chain_id == lab)[0]


def _get_anchor_indices(cdr_idxs, k_anchor, chain_id):
    """返回 CDR 左/右各 k_anchor 个【框架】残基（不与 CDR 重叠）的全局下标。"""
    ch = _normalize_chain_labels(chain_id)  # 0/1/2
    lab = ch[cdr_idxs[0]]  # CDR 所在链的数值标签
    chain_all = np.where(ch == lab)[0]
    # 在该链中的相对位置
    pos = np.searchsorted(chain_all, cdr_idxs)
    left_last = max(pos[0] - 1, 0)
    right_first = min(pos[-1] + 1, len(chain_all) - 1)
    left = chain_all[max(0, left_last - k_anchor + 1): left_last + 1]
    right = chain_all[right_first: min(len(chain_all), right_first + k_anchor)]
    return left, right


def _kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    V, S, Wt = np.linalg.svd(Q0.T @ P0)
    R = V @ Wt
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ Wt
    t = Pc - R @ Qc
    return R, t


def _apply_rt(X, R, t):  # X: (...,3)

    device = X.device
    dtype = X.dtype

    if not torch.is_tensor(R):
        R = torch.as_tensor(R, device=device, dtype=dtype)
    else:
        R = R.to(device=device, dtype=dtype)

    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=device, dtype=dtype)
    else:
        t = t.to(device=device, dtype=dtype)
    return X @ R.T + t


def _extract_points(X, idxs, atom_indices=None, prefer_full_backbone=True):
    pts = []
    if atom_indices and prefer_full_backbone:
        for ridx in idxs:
            for name in ('N', 'CA', 'C', 'O'):
                ai = atom_indices[name]
                pts.append(X[ridx, ai, :])
    else:
        ca = atom_indices.get('CA', 1) if atom_indices else (1 if X.shape[1] > 1 else 0)
        pts = X[idxs, ca, :]
    if torch.is_tensor(pts):
        return pts.detach().cpu().to(torch.float64).numpy()
    if isinstance(pts, (list, tuple)) and pts and torch.is_tensor(pts[0]):
        # 形状一致 → stack；如果是不规则列表请改成 cat 或逐个 .cpu().numpy() 再 pad
        x = torch.stack([t.detach().cpu() for t in pts], dim=0).to(torch.float64)
        return x.numpy()
    return np.asarray(pts, dtype=np.float64)


def graft_cdr_single(
        ab_tgt_np: dict,  # 单样本（numpy）
        ab_src_np: dict,  # 单样本（numpy）
        chain_label='H',
        k_anchor=2,
        atom_indices=None,
        prefer_full_backbone=True
):
    if atom_indices is None:
        atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    X_t, S_t = ab_tgt_np['pos_heavyatom'], ab_tgt_np['aa']
    XL_t, sm_t = ab_tgt_np['mask_heavyatom'], ab_tgt_np['generate_flag']
    ch_t = ab_tgt_np['fragment_type']

    X_s, S_s = ab_src_np['pos_heavyatom'], ab_src_np['aa']
    XL_s, sm_s = ab_src_np['mask_heavyatom'], ab_src_np['generate_flag']
    ch_s = ab_src_np['fragment_type']

    # 目标/源 CDR 索引（按 smask）
    cdr_t = _get_cdr_indices_from_smask(ch_t, sm_t, chain_label)
    cdr_s = _get_cdr_indices_from_smask(ch_s, sm_s, chain_label)
    if cdr_t.size != cdr_s.size:
        ab_new_np, info = graft_cdr_variable_length(
            ab_tgt_np, ab_src_np,
            chain_label=chain_label,
            k_anchor=k_anchor,
            atom_indices=atom_indices,
            prefer_full_backbone=prefer_full_backbone
        )
        return ab_new_np, info

    # 目标锚点（左右各 k 个框架）
    left_t, right_t = _get_anchor_indices(cdr_t, k_anchor, ch_t)
    tgt_anc = np.concatenate([left_t, right_t], 0)

    # 源锚点：用 CDR 片段头/尾各 k 个
    Lc = cdr_s.size
    left_s_local = np.arange(0, min(k_anchor, Lc))
    right_s_local = np.arange(max(0, Lc - k_anchor), Lc)
    Xs_cdr = X_s[cdr_s]

    # 构造 A(目标)、B(源)点集
    P = _extract_points(X_t, tgt_anc, atom_indices, prefer_full_backbone)
    # 将源 CDR 的局部锚点转成点集
    if atom_indices and prefer_full_backbone:
        Q = []
        for ridx in np.concatenate([left_s_local, right_s_local], 0):
            for name in ('N', 'CA', 'C', 'O'):
                ai = atom_indices[name]
                Q.append(Xs_cdr[ridx, ai, :].detach().cpu().numpy())
        Q = np.asarray(Q, dtype=np.float64)
    else:
        ca = atom_indices.get('CA', 1) if atom_indices else (1 if Xs_cdr.shape[1] > 1 else 0)
        Q = Xs_cdr[np.concatenate([left_s_local, right_s_local], 0), ca, :].astype(np.float64)

    if P.shape[0] != Q.shape[0] or P.shape[0] < 3:
        raise RuntimeError(f"锚点点数不匹配或过少：P={P.shape}, Q={Q.shape}")

    # 刚体对齐
    R, t = _kabsch(P, Q)
    Xs_cdr_t = _apply_rt(Xs_cdr.reshape(-1, 3), R, t).reshape(Xs_cdr.shape)

    # 组装输出（只替换 CDR 残基的 X/S/xloss_mask）
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ab_tgt_np.items()}
    out['pos_heavyatom'][cdr_t] = Xs_cdr_t.detach().cpu().numpy()
    out['aa'][cdr_t] = S_s[cdr_s].detach().cpu().numpy()
    out['mask_heavyatom'][cdr_t] = XL_s[cdr_s].detach().cpu().numpy()
    return out, {'R': R, 't': t, 'cdr_t': cdr_t, 'cdr_s': cdr_s}


# ----------------------
# 非等长 graft（新增）
# ----------------------

def _rot_to_quat(R):
    """3x3 -> (4,) [w,x,y,z]"""
    m = R
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=np.float64)


def _slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return s0 * q0 + s1 * q1


def reorder_by_chain(
        ab_out: Dict[str, Any],
        chain_key: str = "tag",
        order: Tuple[int, ...] = (0, 1, 2),
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    将 ab_out 中以残基为第0维的数据，按 chain_id 的顺序 0→1→2 重排，组内顺序保持原样。
    返回 (新的字典, perm 索引)
    """
    ch = ab_out[chain_key]
    # 统一成 numpy 便于找索引
    if torch.is_tensor(ch):
        ch_np = ch.detach().cpu().numpy()
    else:
        ch_np = np.asarray(ch)

    L = ch_np.shape[0]

    # 构造重排索引（稳定：各组内原顺序不变）
    groups = []
    for g in order:
        idx = np.where(ch_np == g)[0]
        if idx.size > 0:
            groups.append(idx)
    perm = np.concatenate(groups, axis=0) if groups else np.arange(L)

    def reindex_value(v):
        # 只重排第0维长度等于 L 的对象；其他保持不动
        if isinstance(v, np.ndarray) and v.shape[0] == L:
            if v.ndim >= 2 and v.shape[1] == L:
                # [L, L, ...] 情况：双轴重排
                return v[perm][:, perm]
            else:
                return v[perm]
        elif torch.is_tensor(v) and v.size(0) == L:
            idx = torch.as_tensor(perm, device=v.device, dtype=torch.long)
            if v.dim() >= 2 and v.size(1) == L:
                return v.index_select(0, idx).index_select(1, idx)
            else:
                return v.index_select(0, idx)
        elif isinstance(v, list) and len(v) == L:
            return [v[i] for i in perm]
        else:
            return v  # 形状与 L 不对齐的键不处理

    new_out = {k: reindex_value(v) for k, v in ab_out.items()}
    # chain_id 自身也要重排（上面的字典推导已处理，这里只是确保）
    new_out[chain_key] = reindex_value(ab_out[chain_key])

    return new_out, perm


def graft_cdr_variable_length(
        ab_tgt_np: dict,
        ab_src_np: dict,
        chain_label='H',
        k_anchor=2,
        atom_indices=None,  # 例如 {'N':0,'CA':1,'C':2,'O':3} 不确定时可 None/只给 CA
        prefer_full_backbone=True,
        tag_map=None,  # 写入新残基的 tag
        chain_id_map_rev=None  # 写入新残基的 chain_id（数字）
):
    """非等长 CDR：两端局部对齐 + 端到端形变（slerp）+ 插/删 + 字段同步。返回新单样本（numpy）。"""
    if tag_map is None:
        tag_map = {'H': 1, 'L': 2}
    if atom_indices is None:
        atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}

    X_t, S_t = ab_tgt_np['pos_heavyatom'], ab_tgt_np['aa']
    XL_t, sm_t = ab_tgt_np['mask_heavyatom'], ab_tgt_np['generate_flag']
    pos_t, ch_t = ab_tgt_np['res_nb'], ab_tgt_np['fragment_type']
    tag_t = ab_tgt_np.get('fragment_type', None)

    X_s, S_s = ab_src_np['pos_heavyatom'], ab_src_np['aa']
    XL_s, sm_s = ab_src_np['mask_heavyatom'], ab_src_np['generate_flag']
    pos_s, ch_s = ab_src_np['res_nb'], ab_src_np['fragment_type']
    lab = _to_chain_val(chain_label)
    # ① 按链别与 smask 取出目标/源的 CDR 残基全局下标
    cdr_t = _get_cdr_indices_from_smask(ch_t, sm_t, lab)  # target 的 CDR 索引（在整分子中的下标）
    cdr_s = _get_cdr_indices_from_smask(ch_s, sm_s, lab)  # source 的 CDR 索引（在整分子中的下标）
    Lt, Ls = cdr_t.size, cdr_s.size

    # ② 目标侧：找出 CDR 左/右两端各 k_anchor 个“框架锚点”（不与 CDR 重叠）
    left_t, right_t = _get_anchor_indices(cdr_t, k_anchor, ch_t)
    P_left = _extract_points(X_t, left_t, atom_indices, prefer_full_backbone)
    P_right = _extract_points(X_t, right_t, atom_indices, prefer_full_backbone)

    # ③ 源侧：取出 CDR 段的坐标；并各自截取“头/尾各 k_anchor 个”作为源端局部锚点
    Xs_cdr = X_s[cdr_s]
    kL = min(k_anchor, Ls)
    kR = min(k_anchor, Ls)
    src_left_local = np.arange(0, kL)
    src_right_local = np.arange(Ls - kR, Ls)
    Q_left = _extract_points(Xs_cdr, src_left_local, atom_indices, prefer_full_backbone)
    Q_right = _extract_points(Xs_cdr, src_right_local, atom_indices, prefer_full_backbone)

    # ④ 分别对“左端/右端”做 Kabsch 刚体对齐，得到两个刚体变换（R, t）
    Rl, tl = _kabsch(P_left, Q_left)
    try:
        Rr, tr = _kabsch(P_right, Q_right)
    except Exception:
        # 右端锚点可能过少（例如 CDR3 很短时），退化为与左端相同的刚体变换
        print(P_right, Q_right)

    # ⑤ 端到端形变（只用于“非等长”）：在左/右刚体变换之间进行插值
    ql, qr = _rot_to_quat(Rl), _rot_to_quat(Rr)
    Xs_cdr_deform = torch.empty_like(Xs_cdr)
    for i in range(Ls):
        alpha = 0.5 if Ls == 1 else i / (Ls - 1)
        qi = _slerp(ql, qr, alpha)
        Ri = _quat_to_rot(qi)
        ti = (1.0 - alpha) * tl + alpha * tr
        Xs_cdr_deform[i] = _apply_rt(Xs_cdr[i], Ri, ti)  # 对第 i 个 CDR 残基施加刚体变换

    # 组装目标链：pre | CDR_new | post
    chain_all_t = _get_chain_span_indices(ch_t, lab)
    pre_idxs = np.arange(chain_all_t[0], cdr_t[0], dtype=int)
    post_idxs = np.arange(cdr_t[-1] + 1, chain_all_t[-1] + 1, dtype=int)

    A = X_t.shape[1]
    new_chain_len = pre_idxs.size + Ls + post_idxs.size

    def _alloc_1d_like(arr, n):
        return np.zeros((n,), dtype=arr.dtype)

    def _alloc_like_imgt(pos_t, n):
        # 如果原 pos_t 里含有字符串（带插入码），就用 object dtype；否则仍可用原 dtype
        try:
            sample = str(pos_t[0])
            use_obj = True
        except Exception:
            use_obj = False
        if use_obj:
            return np.empty((n,), dtype=object)
        else:
            # 即使原来是整数，这里也建议存成字符串，兼容插入码
            return np.empty((n,), dtype=object)

    def _alloc_2d_like(arr, n):
        return np.zeros((n, arr.shape[1]), dtype=arr.dtype)

    def _alloc_3d_like(arr, n):
        return np.zeros((n, arr.shape[1], arr.shape[2]), dtype=arr.dtype)

    X_chain = _alloc_3d_like(X_t, new_chain_len)
    S_chain = _alloc_1d_like(S_t, new_chain_len)
    XL_chain = _alloc_2d_like(XL_t, new_chain_len)
    SM_chain = _alloc_1d_like(sm_t, new_chain_len)
    POS_chain = _alloc_like_imgt(pos_t, new_chain_len)
    CH_chain = _alloc_1d_like(ch_t, new_chain_len)
    TG_chain = _alloc_1d_like(tag_t, new_chain_len) if tag_t is not None else None

    # pre
    X_chain[:pre_idxs.size] = X_t[pre_idxs]
    S_chain[:pre_idxs.size] = S_t[pre_idxs]
    XL_chain[:pre_idxs.size] = XL_t[pre_idxs]
    SM_chain[:pre_idxs.size] = 0
    POS_chain[:pre_idxs.size] = pos_t[pre_idxs]
    CH_chain[:pre_idxs.size] = ch_t[pre_idxs]
    if TG_chain is not None:
        TG_chain[:pre_idxs.size] = tag_t[pre_idxs]

    # CDR_new
    i0 = pre_idxs.size
    i1 = i0 + Ls
    X_chain[i0:i1] = Xs_cdr_deform.detach().cpu().numpy()
    S_chain[i0:i1] = S_s[cdr_s].detach().cpu().numpy()
    XL_chain[i0:i1] = XL_s[cdr_s].detach().cpu().numpy()
    SM_chain[i0:i1] = 1
    # 简单重新编号：延续 pre，逐一 +1（如果你用 IMGT/插入码，替换这里）
    tgt_cdr_pos_slice = pos_t[cdr_t]  # 目标的原 IMGT 编号（字符串或带插入码）
    src_cdr_pos_slice = pos_s[cdr_s]  # 源 CDR 的 residue_pos（可能包含插入码）
    POS_chain[i0:i1] = imgt_relabel_to_target_window(src_cdr_pos_slice, tgt_cdr_pos_slice)

    ch_val = lab
    CH_chain[i0:i1] = ch_val
    if TG_chain is not None:
        TG_chain[i0:i1] = (tag_map['H'] if lab == 1 else tag_map['L'])

    # post
    X_chain[i1:] = X_t[post_idxs]
    S_chain[i1:] = S_t[post_idxs]
    XL_chain[i1:] = XL_t[post_idxs]
    SM_chain[i1:] = 0
    last = POS_chain[i1 - 1]
    Lp = post_idxs.size
    POS_chain[i1:] = pos_t[post_idxs]
    CH_chain[i1:] = ch_t[post_idxs]  # post 的 chain_id 也保持原值更稳妥
    if TG_chain is not None:
        TG_chain[i1:] = tag_t[post_idxs]

    # 合并非该链的残基
    ch_all = _normalize_chain_labels(ch_t)
    others = np.where(ch_all != lab)[0]
    new_total = others.size + new_chain_len

    def _alloc_like(arr, n):
        if arr.ndim == 3:
            return np.zeros((n, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
        if arr.ndim == 2:
            return np.zeros((n, arr.shape[1]), dtype=arr.dtype)
        # residue_pos 需要容纳 IMGT 字符串（如 '105A'）
        if arr is pos_t:
            return np.empty((n,), dtype=arr.dtype)
        return np.zeros((n,), dtype=arr.dtype)

    X_new = _alloc_like(X_t, new_total)
    S_new = _alloc_like(S_t, new_total)
    XL_new = _alloc_like(XL_t, new_total)
    SM_new = _alloc_like(sm_t, new_total)
    POS_new = _alloc_like(pos_t, new_total)
    CH_new = _alloc_like(ch_t, new_total)
    TG_new = _alloc_like(tag_t, new_total) if tag_t is not None else None

    X_new[:others.size] = X_t[others]
    S_new[:others.size] = S_t[others]
    XL_new[:others.size] = XL_t[others]
    SM_new[:others.size] = sm_t[others]
    POS_new[:others.size] = pos_t[others]
    CH_new[:others.size] = ch_t[others]
    if TG_new is not None: TG_new[:others.size] = tag_t[others]

    X_new[others.size:] = X_chain
    S_new[others.size:] = S_chain
    XL_new[others.size:] = XL_chain
    SM_new[others.size:] = SM_chain
    POS_new[others.size:] = POS_chain
    CH_new[others.size:] = CH_chain
    if TG_new is not None: TG_new[others.size:] = TG_chain

    ab_out = {**ab_tgt_np, 'pos_heavyatom': X_new, 'aa': S_new, 'mask_heavyatom': XL_new,
              'generate_flag': SM_new, 'res_nb': POS_new, 'fragment_type': CH_new, 'chain_nb': CH_new}
    if TG_new is not None: ab_out['fragment_type'] = TG_new
    ab_out_sorted, perm = reorder_by_chain(ab_out, chain_key='fragment_type', order=(1, 2, 3, 0))

    info = {'Lt': int(Lt), 'Ls': int(Ls),
            'left_anchors': left_t.tolist(), 'right_anchors': right_t.tolist(),
            'Rl': Rl.tolist(), 'tl': tl.tolist(), 'Rr': Rr.tolist(), 'tr': tr.tolist()}
    return ab_out_sorted, info


def graft_one_in_batch(ab_data, sid, retrieved_ab_meta_single,
                       chain_label='H', k_anchor=2,
                       atom_indices=None,
                       prefer_full_backbone=True):
    """
    ab_data: batch 级 dict（torch 或 numpy 混合），含 batch_id
    sid: 该 batch 中的一个样本 id（来自 ag/ab 的 batch_id）
    retrieved_ab_meta_single: 从检索得到的某个 ab_meta[i]（numpy）
    """
    # 1) 切出目标抗体单样本（numpy）
    if atom_indices is None:
        atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    ab_tgt_np, idxs = make_sample_from_batch(ab_data, sid)

    # 2) 调 graft（numpy→numpy）
    ab_init_np, rt_meta = graft_cdr_single(
        ab_tgt_np, retrieved_ab_meta_single,
        chain_label=chain_label,
        k_anchor=k_anchor,
        atom_indices=atom_indices,
        prefer_full_backbone=prefer_full_backbone
    )

    # 3) 写回 batch（保持设备与 dtype）
    # write_back_sample_into_batch(ab_data, idxs, ab_init_np)

    return ab_init_np  # 可选：返回对齐信息
