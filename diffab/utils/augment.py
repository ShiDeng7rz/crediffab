# diffab/utils/augment.py
import torch

# ===== 常量（按你项目的 BBHeavyAtom 下标调整） =====
from diffab.utils.protein.constants import BBHeavyAtom

BB_N, BB_CA, BB_C, BB_O = BBHeavyAtom.N, BBHeavyAtom.CA, BBHeavyAtom.C, BBHeavyAtom.O


def se3_transform(pos, R, t):
    # pos: [..., 3],  R:[3,3], t:[1,3]
    return pos @ R.transpose(-1, -2) + t


def _same_dtype(t_like, ref):
    return t_like.to(dtype=ref.dtype, device=ref.device)


# ===== 位姿增强 =====


def edge_dropout(edge_index, p=0.1):
    """
    edge_index: [2, E]
    """
    E = edge_index.size(1)
    keep = (torch.rand(E, device=edge_index.device) > p)
    return edge_index[:, keep]


def interface_crop_mask(pos_ca_ab, pos_ca_ag, radius=12.0):
    """
    以 CA–CA 距离阈值做界面选择
    返回 keep_ab:[La], keep_ag:[Lg]
    """
    d = torch.cdist(pos_ca_ab, pos_ca_ag)  # [La, Lg]
    keep_ab = (d.min(dim=1).values <= radius)
    keep_ag = (d.min(dim=0).values <= radius)
    return keep_ab, keep_ag


# ===== 组装视图（适配你数据字典）=====
def _select_by_mask(data, keep_res_mask):
    """按残基掩码挑子图（保持 key 统一）"""
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            if v.dim() == 2 and v.size(1) == 3:  # [L,3] (如果你有这种)
                out[k] = v[keep_res_mask]
            elif v.dim() == 3 and v.size(-1) == 3:  # [L,A,3]
                out[k] = v[keep_res_mask]
            elif v.dim() >= 1 and v.size(0) == keep_res_mask.size(0):  # 第0维是 L
                out[k] = v[keep_res_mask]
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _apply_pos_replace(data, new_pos_atoms, new_mask_atoms=None):
    data = dict(data)
    data['pos_heavyatom'] = new_pos_atoms
    if new_mask_atoms is not None:
        data['mask_heavyatom'] = new_mask_atoms
    return data


# ===== 基础工具：随机旋转/平移（支持 half/dtype 对齐）=====
def _rand_quat(device, dtype):
    q = torch.randn(4, device=device, dtype=dtype)
    q = q / (q.norm() + 1e-8)
    return q


def random_rotation_matrix(device, dtype, batch: int | None = None):
    """
    返回 3x3 或 (B,3,3) 的旋转矩阵，均匀采样于 SO(3)。
    """
    if batch is None:
        w, x, y, z = _rand_quat(device, dtype)
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=0),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=0),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=0),
        ], dim=0)
        return R
    else:
        q = torch.randn(batch, 4, device=device, dtype=dtype)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)  # (B,4)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ], dim=-2)  # (B,3,3)
        return R


def se3_transform_atoms(x, R, t):
    """
    x: (B, L, A, 3) 或 (L, A, 3)
    R: (3,3) 或 (B,3,3)
    t: 形状可广播到 x 的向量，如 (3,), (1,3), (B,3), (B,1,1,3) 等
    返回：与 x 同形
    """
    assert x.size(-1) == 3
    # 先统一成 (B, L, A, 3)
    squeeze_batch = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    B = x.size(0)

    # 旋转
    if R.dim() == 2:
        x_rot = x @ R.T
    else:
        # (B,3,3)
        x_rot = torch.matmul(x, R.transpose(-1, -2))

    # 平移：做成 (B,1,1,3)
    if t.dim() == 1 and t.shape[-1] == 3:
        t = t.view(1, 1, 1, 3).to(device=x.device, dtype=x.dtype)
    elif t.dim() == 2 and t.shape == (1, 3):
        t = t.view(1, 1, 1, 3).to(device=x.device, dtype=x.dtype)
    elif t.dim() == 2 and t.shape == (B, 3):
        t = t.view(B, 1, 1, 3).to(device=x.device, dtype=x.dtype)
    else:
        t = t.to(device=x.device, dtype=x.dtype)
    x_new = x_rot + t

    if squeeze_batch:
        x_new = x_new[0]
    return x_new


# ===== 位姿增强：对“全原子”一次性变换（替换原 se3_independent / se3_joint）=====
def se3_independent(ab_pos, ag_pos, t_sigma=3.0):
    """
    Ab 与 Ag 各自独立的刚体变换；直接对全原子 (… ,3) 作用，不再先只对 CA。
    支持 (B,L,A,3) 或 (L,A,3)。
    """
    # 统一 dtype/device
    Ra = random_rotation_matrix(ab_pos.device, ab_pos.dtype)
    Rg = random_rotation_matrix(ag_pos.device, ag_pos.dtype)
    ta = torch.randn(3, device=ab_pos.device, dtype=ab_pos.dtype) * t_sigma
    tg = torch.randn(3, device=ag_pos.device, dtype=ag_pos.dtype) * t_sigma
    ab_new = se3_transform_atoms(ab_pos, Ra, ta)
    ag_new = se3_transform_atoms(ag_pos, Rg, tg)
    return ab_new, ag_new


def se3_joint(ab_pos, ag_pos, t_sigma=3.0):
    """
    Ab 与 Ag 使用同一 R/t（适合保持复合物几何关系的视图）
    """
    R = random_rotation_matrix(ab_pos.device, ab_pos.dtype)
    t = torch.randn(3, device=ab_pos.device, dtype=ab_pos.dtype) * t_sigma
    ab_new = se3_transform_atoms(ab_pos, R, t)
    ag_new = se3_transform_atoms(ag_pos, R, t)
    return ab_new, ag_new


# ===== 轻噪声 / Dropout（修主链索引，避免硬编码 0..3）=====
def sidechain_jitter(pos_atoms, mask_atoms, std=0.15, keep_mainchain=True):
    """
    pos_atoms: [B, L, A, 3] 或 [L, A, 3]
    mask_atoms: 同维度 [B, L, A] or [L, A]
    """
    squeeze_batch = False
    if pos_atoms.dim() == 3:
        pos_atoms = pos_atoms.unsqueeze(0)
        mask_atoms = mask_atoms.unsqueeze(0)
        squeeze_batch = True

    noise = torch.randn_like(pos_atoms) * pos_atoms.new_tensor(std)
    if keep_mainchain:
        main_idx = torch.tensor([BB_N, BB_CA, BB_C, BB_O], device=pos_atoms.device)
        noise[:, :, main_idx] = 0.0
    pos_out = pos_atoms + noise * mask_atoms.unsqueeze(-1)

    if squeeze_batch:
        pos_out = pos_out[0]
    return pos_out


def atom_dropout_mask(mask_atoms, p=0.1, keep_ca=True):
    drop = (torch.rand_like(mask_atoms.float()) < p)
    if keep_ca:
        drop[..., BB_CA] = False
    return mask_atoms & (~drop)


# ===== 视图构建（仅改动用法：不再“先CA再套回全原子”）=====
def build_two_views_pose_invariant(ab_data, ag_data,
                                   atom_drop_p=0.1, edge_drop_p=0.1, jitter_std=0.1):
    """
    返回两组视图：(ab_v1, ag_v1), (ab_v2, ag_v2)
    Ab/Ag 各自独立 R/t（位姿不变对比）
    """

    def one_view(ab, ag):
        # 1) 直接对全原子做独立 SE(3)
        ab_pos2, ag_pos2 = se3_independent(ab['pos_heavyatom'], ag['pos_heavyatom'], t_sigma=3.0)

        # 2) 侧链轻噪
        ab_pos2 = sidechain_jitter(ab_pos2, ab['mask_heavyatom'], std=jitter_std, keep_mainchain=True)
        ag_pos2 = sidechain_jitter(ag_pos2, ag['mask_heavyatom'], std=jitter_std, keep_mainchain=True)

        # 3) 原子/边 Dropout
        ab_mask_atom2 = atom_dropout_mask(ab['mask_heavyatom'], p=atom_drop_p, keep_ca=True)
        ag_mask_atom2 = atom_dropout_mask(ag['mask_heavyatom'], p=atom_drop_p, keep_ca=True)

        ab_view = _apply_pos_replace(ab, ab_pos2, ab_mask_atom2)
        ag_view = _apply_pos_replace(ag, ag_pos2, ag_mask_atom2)

        if 'edges_index' in ab_view:
            ab_view['edges_index'] = edge_dropout(ab_view['edges_index'], p=edge_drop_p)
        if 'edges_index' in ag_view:
            ag_view['edges_index'] = edge_dropout(ag_view['edges_index'], p=edge_drop_p)
        return ab_view, ag_view

    v1 = one_view(ab_data, ag_data)
    v2 = one_view(ab_data, ag_data)
    return v1, v2


def build_two_views_pose_aware(ab_data, ag_data,
                               radius=12.0, atom_drop_p=0.05, edge_drop_p=0.05, jitter_std=0.1):
    """
    返回两组视图：(ab_v1, ag_v1), (ab_v2, ag_v2)
    同一 R/t（保持复合物相对构型）+ 界面裁剪 + 轻扰动
    """

    def one_view(ab, ag):
        # 0) 同步 SE3（整复合体）
        ab_pos2, ag_pos2 = se3_joint(ab['pos_heavyatom'], ag['pos_heavyatom'], t_sigma=3.0)

        # 1) 界面裁剪（按 CA–CA 半径；此处假设无 batch 维）
        keep_ab, keep_ag = interface_crop_mask(ab_pos2[..., BB_CA, :], ag_pos2[..., BB_CA, :], radius=radius)
        ab_crop = _select_by_mask(_apply_pos_replace(ab, ab_pos2), keep_ab)
        ag_crop = _select_by_mask(_apply_pos_replace(ag, ag_pos2), keep_ag)

        # 2) 轻噪 + Dropout
        ab_crop['pos_heavyatom'] = sidechain_jitter(ab_crop['pos_heavyatom'], ab_crop['mask_heavyatom'],
                                                    std=jitter_std, keep_mainchain=True)
        ag_crop['pos_heavyatom'] = sidechain_jitter(ag_crop['pos_heavyatom'], ag_crop['mask_heavyatom'],
                                                    std=jitter_std, keep_mainchain=True)
        ab_crop['mask_heavyatom'] = atom_dropout_mask(ab_crop['mask_heavyatom'], p=atom_drop_p, keep_ca=True)
        ag_crop['mask_heavyatom'] = atom_dropout_mask(ag_crop['mask_heavyatom'], p=atom_drop_p, keep_ca=True)

        if 'edges_index' in ab_crop:
            ab_crop['edges_index'] = edge_dropout(ab_crop['edges_index'], p=edge_drop_p)
        if 'edges_index' in ag_crop:
            ag_crop['edges_index'] = edge_dropout(ag_crop['edges_index'], p=edge_drop_p)
        return ab_crop, ag_crop

    v1 = one_view(ab_data, ag_data)
    v2 = one_view(ab_data, ag_data)
    return v1, v2
