import numpy as np

BB_N, BB_CA, BB_C = 0, 1, 2  # 如果你的通道顺序不同，改这里


def _np(x):  # torch/np 统一
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _mask_res_or_true(d):
    if 'mask_res' in d and d['mask_res'] is not None:
        m = _np(d['mask_res']).astype(bool).reshape(-1)
        return m
    return np.ones((_np(d['pos_heavyatom']).shape[0],), dtype=bool)


def _xmask_or_true(d):
    if 'mask_heavyatom' in d and d['mask_heavyatom'] is not None:
        xm = _np(d['mask_heavyatom'])
        return xm
    L, A = _np(d['pos_heavyatom']).shape[:2]
    return np.ones((L, A), dtype=bool)


def _is_ab(d):  # 1/2 是抗体
    tag = _np(d['fragment_type']).astype(int)
    return (tag == 1) | (tag == 2)


def _is_ag(d):  # 0 是抗原
    tag = _np(d['fragment_type']).astype(int)
    return tag == 3


def _fr_mask(d):  # Framework = 非 CDR
    return ~_np(d['generate_flag'])


def _ca_coords(d):  # [L,3]，取 CA；若只有单原子，也能兼容
    X = _np(d['pos_heavyatom'])
    if X.ndim == 3 and X.shape[1] > BB_CA:
        return X[:, BB_CA, :]
    elif X.ndim == 2 and X.shape[1] == 3:
        return X
    else:
        raise ValueError(f"Unexpected X shape for extracting CA: {X.shape}")


def _kabsch(P, Q):
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = P.mean(0, keepdims=True) - Q.mean(0, keepdims=True) @ R.T
    return R, t


def rigid_align_ref_fr_to_target_fr(ref, target):
    """仅用 Ab-FR 的 CA 做刚体对齐；不读取 target CDR。"""
    ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ref.items()}
    valid_t = _mask_res_or_true(target)
    valid_r = _mask_res_or_true(ref)
    ab_t = _is_ab(target) & valid_t
    ab_r = _is_ab(ref) & valid_r
    fr_t = _fr_mask(target) & ab_t
    fr_r = _fr_mask(ref) & ab_r
    Ct = _ca_coords(target)[fr_t]
    Cr = _ca_coords(ref)[fr_r]
    L = min(Ct.shape[0], Cr.shape[0])
    if L < 3:
        return ref  # 数据太少，不对齐
    R, t = _kabsch(Ct[:L], Cr[:L])
    X = _np(ref['pos_heavyatom']).copy()
    X = X @ R.T  # [L,A,3] 每个原子都旋转
    X = X + t  # 平移
    ref['pos_heavyatom'] = X
    return ref


def _pairwise_min_dist(A, B, chunk=4096):
    """返回每个 A 点到所有 B 的最小距离及对应向量（用于方向）。"""
    mins = []
    vecs = []
    for i in range(0, A.shape[0], chunk):
        Ai = A[i:i + chunk]  # [m,3]
        diff = Ai[:, None, :] - B[None, :, :]  # [m,n,3]
        d2 = np.sum(diff * diff, axis=-1)  # [m,n]
        idx = np.argmin(d2, axis=1)  # [m]
        v = Ai - B[idx]  # [m,3]
        d = np.sqrt(np.maximum(np.sum(v * v, axis=1), 1e-12))  # [m]
        mins.append(d)
        vecs.append(v / d[:, None])
    return np.concatenate(mins, 0), np.concatenate(vecs, 0)


def _adjacent_pairs(L):
    """生成 (i,i+1) 相邻 CA 索引对。"""
    i = np.arange(0, L - 1, dtype=int)
    j = i + 1
    return i, j


def micro_repair_reference(
        ref,
        target=None,
        clash_thr=2.1,  # 判定为碰撞的距离阈值（Å）
        step_size=0.15,  # 每次推开的步长（Å）
        n_iter=50,  # 迭代次数
        keep_fr_weight=0.5,  # FR/Ag 的“骨架回拉”权重（越大越不动）
        ca_bond_len=3.8,  # CA-CA 目标距离
        ca_len_lambda=0.05  # CA-CA 回拉强度
):
    """
    对“参考结构 ref”做轻量修复：
      1) （可选）用 target 的 Ab-FR 对齐（不读 target CDR）
      2) Ab↔Ag 去碰撞（小步互推），且 FR/Ag 有“骨架回拉”保护
      3) 参考内部 CA-CA 距离轻微回拉到 3.8Å（抑制畸形）

    仅修改 ref，自成一体；不会碰 target 的任何坐标。
    """
    ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ref.items()}

    # (0) 可选：只用 FR 对齐
    if target is not None:
        ref = rigid_align_ref_fr_to_target_fr(ref, target)

    # 基本掩码
    valid = _mask_res_or_true(ref)
    is_ab = _is_ab(ref) & valid
    is_ag = _is_ag(ref) & valid
    fr_ab = _fr_mask(ref) & is_ab  # 抗体-FR
    # cdr_ab = (~_fr_mask(ref)) & is_ab      # 抗体-CDR（不读取 target 的，只在 ref 内部用）
    X = _np(ref['pos_heavyatom']).copy()  # [L,A,3]

    # 预存一份“骨架参考位姿”（FR/Ag）用于回拉
    CA0 = _ca_coords(ref).copy()  # [L,3]
    CA0_fr = CA0[fr_ab].copy()
    CA0_ag = CA0[is_ag].copy()

    # 邻接 CA 对（参考内部平滑）
    L = X.shape[0]
    ip, jp = _adjacent_pairs(L)

    for it in range(n_iter):
        # --- 1) 计算 Ab↔Ag 最近距离，用于判定碰撞并推开 ---
        CA = X[:, BB_CA, :]  # [L,3]
        Ab = CA[is_ab]  # [Na,3]
        Ag = CA[is_ag]  # [Ng,3]
        if Ab.shape[0] and Ag.shape[0]:
            dmin_ab, dir_ab = _pairwise_min_dist(Ab, Ag)  # 对 Ab: 最近 Ag
            dmin_ag, dir_ag = _pairwise_min_dist(Ag, Ab)  # 对 Ag: 最近 Ab

            # 需要推开的索引
            push_ab = dmin_ab < clash_thr
            push_ag = dmin_ag < clash_thr
            # 推开的位移（朝远离方向）
            disp_ab = np.zeros_like(Ab)
            disp_ag = np.zeros_like(Ag)
            disp_ab[push_ab] += step_size * dir_ab[push_ab]
            disp_ag[push_ag] += step_size * dir_ag[push_ag]

            # 把 CA 位移传播到所有原子（同一残基刚体平移）
            # 这里只做平移，不做旋转（轻量且稳定）
            # 抗体
            if np.any(push_ab):
                res_idx_ab = np.where(is_ab)[0]
                moved_ab = res_idx_ab[push_ab]
                X[moved_ab, :, :] += disp_ab[push_ab][:, None, :]
            # 抗原
            if np.any(push_ag):
                res_idx_ag = np.where(is_ag)[0]
                moved_ag = res_idx_ag[push_ag]
                X[moved_ag, :, :] += disp_ag[push_ag][:, None, :]

        # --- 2) 骨架回拉（只约束 FR/Ag 的 CA，避免漂移；CDR 给更大自由度） ---
        CA = X[:, BB_CA, :]
        # FR 抗体
        if np.any(fr_ab):
            idx_fr = np.where(fr_ab)[0]
            CA[idx_fr] = (1.0 - keep_fr_weight) * CA[idx_fr] + keep_fr_weight * CA0_fr
        # 抗原
        if np.any(is_ag):
            idx_ag = np.where(is_ag)[0]
            CA[idx_ag] = (1.0 - keep_fr_weight) * CA[idx_ag] + keep_fr_weight * CA0_ag
        X[:, BB_CA, :] = CA

        # --- 3) CA-CA 轻微回拉到 3.8Å（参考内部的几何平滑） ---
        # （对所有相邻残基的 CA）
        v = CA[jp] - CA[ip]  # [L-1,3]
        d = np.sqrt((v * v).sum(-1, keepdims=True)) + 1e-8
        corr = (ca_bond_len - d) * (v / d)  # 纠正向量，长度 ~ (3.8-d)
        CA[ip] -= ca_len_lambda * corr
        CA[jp] += ca_len_lambda * corr
        X[:, BB_CA, :] = CA

    ref['pos_heavyatom'] = X
    return ref
