#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.pdb_utils import VOCAB


def compute_dihedrals(X, eps=1e-7):
    """
    N1       Ca1      C1        N2       Ca2      C2       N3       Ca3      C3
   *---------*--------*---------*--------*--------*--------*--------*--------*
       Residue i-1         |         Residue i           |        Residue i+1
                          (计算 φ 角)
                               ↓
                      4个连续原子： C(i-1) - N(i) - CA(i) - C(i)    → φ

                                 (计算 ψ 角)
                             4个连续原子： N(i) - CA(i) - C(i) - N(i+1)   → ψ

                                 (计算 ω 角)
                          4个连续原子： CA(i) - C(i) - N(i+1) - CA(i+1)  → ω
    """
    # 提取 N、CA、C 的坐标并重塑
    X = X.unsqueeze(0)
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

    # 计算相邻原子之间的单位向量
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]

    # 计算法向量

    n_2 = F.normalize(torch.linalg.cross(u_2, u_1, axis=-1), dim=-1)
    n_1 = F.normalize(torch.linalg.cross(u_1, u_0, axis=-1), dim=-1)

    # 计算二面角
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    D = F.pad(D, (3, 0), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    # 将二面角提升到圆上
    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    return D_features


def rbf(distance, centers, width):
    """
    计算径向基函数（RBF）
    Args:
        distance (float): 两个点之间的欧几里得距离
        centers (np.ndarray): RBF 的中心（μ 的数组）
        width (float): RBF 的宽度（σ）
    Returns:
        np.ndarray: RBF 值数组
    """
    distance = np.array(distance)
    centers = np.array(centers)
    return np.exp(-((distance - centers) ** 2) / (2 * width ** 2))


def to_numpy(v):
    return v.detach().cpu().numpy().astype(np.float32)


def construct_local_frame(X):
    """
    构造局部坐标系 O_i。

    利用残基的主链原子 N、Cα、C 的三维坐标构造局部参考系，
    其中：
      - 以 (ca - n) 作为局部 x 轴（e1），
      - 以 (c - ca) 与 e1 的叉乘生成局部 z 轴（e3），
      - 以 e3 与 e1 的叉乘获得局部 y 轴（e2）。

    Args:
        X

    Returns:
        np.ndarray: 局部坐标系矩阵 O_i，形状 (3, 3)，每列分别为 e1, e2, e3。
    """

    X = X.detach().cpu().numpy()
    X = np.array(X, dtype=np.float32)
    n = X[0]  # N
    ca = X[1]  # CA
    c = X[2]  # C
    v1 = ca - n  # N -> CA
    v2 = c - ca  # CA -> C
    e1 = v1 / np.linalg.norm(v1)  # 单位向量作为局部 x 轴
    e3 = np.cross(e1, v2)  # 计算垂直于 e1 与 v2 的向量
    e3 = e3 / np.linalg.norm(e3)  # 单位化得到局部 z 轴
    e2 = np.cross(e3, e1)  # 局部 y 轴保证正交性
    return np.stack([e1, e2, e3], axis=1)  # 每列是一个基向量


def compute_local_direction(O_i, x_i_alpha, x_j_alpha):
    """
    计算残基 j 相对于残基 i 的局部方向编码。

    具体做法是先计算残基 i 的 Cα 到残基 j 的 Cα 的方向向量，
    然后将该向量投影到残基 i 的局部坐标系中，输出 3 维编码。

    Args:
        O_i (torch.tensor): 残基 i 的局部坐标系矩阵，形状 (3, 3)
        x_i_alpha (torch.tensor): 残基 i 的 Cα 原子坐标，形状 (3, )
        x_j_alpha (torch.tensor): 残基 j 的 Cα 原子坐标，形状 (3, )

    Returns:
        np.ndarray: 残基 j 在残基 i 局部坐标系中的方向编码，形状 (3, )
    """
    """
        计算残基 j 相对于残基 i 的局部方向编码
        """
    # 转换为 NumPy 干净数组并指定 dtype
    if isinstance(O_i, torch.Tensor):
        O_i = O_i.detach().cpu().numpy()
    if isinstance(x_i_alpha, torch.Tensor):
        x_i_alpha = x_i_alpha.detach().cpu().numpy()
    if isinstance(x_j_alpha, torch.Tensor):
        x_j_alpha = x_j_alpha.detach().cpu().numpy()

    # 显式指定 float32 类型
    O_i = np.asarray(O_i, dtype=np.float32)
    x_i_alpha = np.asarray(x_i_alpha, dtype=np.float32)
    x_j_alpha = np.asarray(x_j_alpha, dtype=np.float32)

    direction_vector = x_j_alpha - x_i_alpha
    norm = np.linalg.norm(direction_vector)
    if norm < 1e-8:
        direction_vector = np.zeros_like(direction_vector)
    else:
        direction_vector = direction_vector / norm

    # 确保方向向量也是干净的 float32
    direction_vector = np.asarray(direction_vector, dtype=np.float32)

    # 使用 @ 或 np.matmul
    local_direction = np.matmul(O_i.T, direction_vector)
    return local_direction


def compute_rotation_encoding(O_i, O_j):
    """
    计算从残基 i 的局部坐标系到残基 j 的局部坐标系的旋转编码，并以四元数形式表示。

    具体操作为先计算旋转矩阵 R = O_i^T * O_j，
    然后根据旋转矩阵的对角线和反对称部分转换为四元数 q(R) 表示。

    Args:
        O_i (np.ndarray or torch.Tensor): 残基 i 的局部坐标系矩阵，形状 (3, 3)
        O_j (np.ndarray or torch.Tensor): 残基 j 的局部坐标系矩阵，形状 (3, 3)

    Returns:
        torch.Tensor: 四元数表示的旋转编码，形状 (4,)
    """
    # 若输入为 numpy 数组，则转换为 torch.Tensor
    if isinstance(O_i, np.ndarray):
        O_i = torch.tensor(O_i, dtype=torch.float32)
    if isinstance(O_j, np.ndarray):
        O_j = torch.tensor(O_j, dtype=torch.float32)

    # 计算旋转矩阵 R = O_i^T * O_j
    R = torch.matmul(O_i.transpose(0, 1), O_j)  # (3, 3)

    # 为了统一计算，扩展 batch 维度：变为 (1, 1, 3, 3)
    R_expanded = R.unsqueeze(0).unsqueeze(0)

    # 计算旋转矩阵的对角线元素
    diag = torch.diagonal(R_expanded, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)

    # 根据旋转矩阵转换为四元数
    qw = torch.sqrt(torch.clamp(1.0 + Rxx + Ryy + Rzz, min=0)) / 2.0
    qx = torch.sqrt(torch.clamp(1.0 + Rxx - Ryy - Rzz, min=0)) / 2.0
    qy = torch.sqrt(torch.clamp(1.0 - Rxx + Ryy - Rzz, min=0)) / 2.0
    qz = torch.sqrt(torch.clamp(1.0 - Rxx - Ryy + Rzz, min=0)) / 2.0

    # 调整符号，确保四元数表示正确的旋转方向
    qx = qx * torch.sign(R_expanded[0, 0, 2, 1] - R_expanded[0, 0, 1, 2])
    qy = qy * torch.sign(R_expanded[0, 0, 0, 2] - R_expanded[0, 0, 2, 0])
    qz = qz * torch.sign(R_expanded[0, 0, 1, 0] - R_expanded[0, 0, 0, 1])

    # 拼接四元数向量 [qx, qy, qz, qw]
    q = torch.cat([qx, qy, qz, qw], dim=-1)  # (1, 1, 4)
    q = q.squeeze(0).squeeze(0)  # 变为 (4,)
    q = F.normalize(q, dim=-1)
    return q


def rotation_matrix_to_euler(R, eps=1e-6):
    """
    将旋转矩阵 R (3x3) 转换为欧拉角（ZYX 顺序）：[yaw, pitch, roll]
    其中 yaw 表示绕 Z 轴旋转角度，pitch 表示绕 Y 轴旋转角度，roll 表示绕 X 轴旋转角度。
    返回的角度单位为弧度。
    """
    # 取旋转矩阵的分量（假设 R 为 numpy 数组）
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    # 计算 sy = sqrt(r00^2 + r10^2)
    sy = np.sqrt(r00 ** 2 + r10 ** 2)
    singular = sy < eps

    if not singular:
        # yaw: 以 r10 和 r00 计算（绕 Z 轴）
        yaw = np.arctan2(r10, r00)
        # pitch: 以 -r20 和 sy 计算（绕 Y 轴）
        pitch = np.arctan2(-r20, sy)
        # roll: 以 r21 和 r22 计算（绕 X 轴）
        roll = np.arctan2(r21, r22)
    else:
        # 特殊情况，当 sy 非常接近 0
        yaw = np.arctan2(-r01, r11)
        pitch = np.arctan2(-r20, sy)
        roll = 0

    return np.array([yaw, pitch, roll])


def compute_relative_euler_angles(O_i, O_j):
    """
    计算残基 j 的局部坐标系相对于残基 i 的局部坐标系的相对旋转，
    并将该旋转分解为欧拉角 (yaw, pitch, roll)，
    即计算欧拉角 q(O_i^T * O_j) 的三个分量。

    输入:
        O_i: 残基 i 的局部坐标系 (3x3)，可以是 numpy 数组或 torch.Tensor
        O_j: 残基 j 的局部坐标系 (3x3)
    输出:
        一个 numpy 数组，表示 [yaw, pitch, roll]，角度单位为弧度。
    """
    # 如果输入为 torch.Tensor，则转换成 numpy 数组
    if isinstance(O_i, torch.Tensor):
        O_i = O_i.detach().cpu().numpy()
    if isinstance(O_j, torch.Tensor):
        O_j = O_j.detach().cpu().numpy()

    # 计算相对旋转矩阵 R = O_i^T * O_j
    R = np.dot(O_i.T, O_j)

    # 将 R 转换为欧拉角（ZYX 顺序）
    euler_angles = rotation_matrix_to_euler(R)
    return euler_angles


def compute_spherical_angles(Ca_i, Ca_j, O_i, r):
    """
    计算从残基 i (P_i) 到残基 j (P_j) 的向量在球坐标中的表示，
    包括：距离 r、方位角 azimuth、俯仰角 elevation (均为弧度制)。

    参数：
        P_i (np.ndarray): 残基 i 的 Cα 坐标, shape = (3,)
        P_j (np.ndarray): 残基 j 的 Cα 坐标, shape = (3,)
    返回：
        (r, azimuth, elevation)
        - r: 两个残基间的欧几里得距离
        - azimuth: 方位角, 在 XY 平面内相对于 X 轴的夹角 (范围 -π~π)
        - elevation: 俯仰角, 与 XY 平面的夹角 (范围 -π/2~π/2)
    """
    # 1) 计算向量 v
    v = Ca_i - Ca_j  # shape=(3,)
    v_local = O_i.T.dot(v)
    # 若 r=0 表示两点重合，后续角度可按需求返回 0 或特殊值
    if r < 1e-12:
        return 0.0, 0.0, 0.0

    # 3) 方位角 azimuth = atan2(v_y, v_x)
    azimuth = np.arctan2(v_local[1], v_local[0])

    # 4) 俯仰角 elevation = atan2(v_z, sqrt(v_x^2 + v_y^2))
    #   也有定义为: elevation = arc-tan2( sqrt(v_x^2 + v_y^2), v_z )
    #   视实际需求而定，这里采用常见的 "与 XY 平面的夹角" 定义
    xy_proj = np.sqrt(v_local[0] ** 2 + v_local[1] ** 2)
    elevation = np.arctan2(v_local[2], xy_proj)

    return azimuth, elevation


def _cal_backbone_bond_lengths(X):
    # loss of backbone (...N-CA-C(O)-N...) bond length
    # N-CA, CA-C, C=O
    bl = torch.norm(torch.tensor(X[:, 1:4] - X[:, :3], dtype=torch.float32), dim=-1)  # [N, 3], (N-CA), (CA-C), (C=O)
    return bl


def get_sidechain_bonds(S_symbol, max_k=10):
    """
    参数:
      S_symbol: list, 长度为 N，每个元素为一个残基的单字母代码，如 ["A", "V", "G", ...]
      VOCAB: 一个包含一些预定义字典的对象，里面存储了：
             - VOCAB.aas: 一个列表，每个元素为 (单字母, 三字母) 的元组，用于构建辅助字典；
             - VOCAB.sidechain_bonds: dict，键为三字母代码（例如 "ALA", "VAL"），值是侧链键的定义字典；
             - VOCAB.sidechain_map: dict，键为单字母代码（例如 'A', 'V'），值为该残基侧链原子的顺序列表，如 ['CB'] 或 ['CB','CG1','CG2'] 等。

    返回:
      bonds: LongTensor, 形状 [N, max_k, 2]，每个有效的键对给出完整原子列表中的 [src_index, dst_index]；
      bonds_mask: BoolTensor, 形状 [N, max_k]，有效键条目为 True，不存在键的条目为 False。
    """
    N = len(S_symbol)
    bonds_all = []  # 存放每个残基的键列表，列表中每个元素为 [[src, dst], ...]
    # self.sidechain_bonds: dict, 键为三字母代码，如 "ALA", "VAL" 等；
    # self.sidechain_map: dict, 键为单字母代码，如 'A', 'V', ...；值为 list，例如 ['CB']、['CB','CG1','CG2'] 等。
    aas_dict = dict(VOCAB.aas)
    for i, res in enumerate(S_symbol):
        res_three = aas_dict.get(res, None)
        bond_list = []  # 存放当前残基的所有键对 [src, dst]
        # 如果残基没有侧链键定义（例如 Glycine），则返回空列表
        if res_three not in VOCAB.sidechain_bonds:
            bonds_all.append(bond_list)
            continue

        bonds_dict = VOCAB.sidechain_bonds[res_three]  # 例如对于 ALA: {"CA": ["CB"]}
        # 获得单字母码
        # 获取该残基侧链原子的顺序列表
        sidechain_atoms = VOCAB.sidechain_map.get(res, [])

        # 定义一个辅助函数计算原子名称到索引的映射
        def atom_index(atom_name):
            # 如果 atom_name 属于 backbone (N, CA, C, O)
            if atom_name == "N":
                return 0
            elif atom_name == "CA":
                return 1
            elif atom_name == "C":
                return 2
            elif atom_name == "O":
                return 3
            else:
                # 在侧链中查找位置，侧链原子从索引4开始
                # 如果找不到，抛出异常或返回 None
                try:
                    idx = sidechain_atoms.index(atom_name)
                    return 4 + idx
                except ValueError:
                    return None

        # 对 bonds_dict 中的每个键值对建立键对
        for src_atom, dst_atoms in bonds_dict.items():
            src_idx = atom_index(src_atom)
            # 如果未找到对应索引，则跳过
            if src_idx is None:
                continue
            for dst_atom in dst_atoms:
                dst_idx = atom_index(dst_atom)
                if dst_idx is None:
                    continue
                bond_list.append([src_idx, dst_idx])
        bonds_all.append(bond_list)

    # 初始化张量和 mask
    bonds_tensor = torch.zeros((N, max_k, 2), dtype=torch.long)
    mask_tensor = torch.zeros((N, max_k), dtype=torch.bool)

    for i, bond_list in enumerate(bonds_all):
        k = len(bond_list)
        if k > 0:
            bonds_tensor[i, :k, :] = torch.tensor(bond_list, dtype=torch.long)
            mask_tensor[i, :k] = True

    return bonds_tensor, mask_tensor


def _cal_sidechain_bond_lengths(X, S_symbol):
    bonds, bonds_mask = get_sidechain_bonds(S_symbol)
    # bonds 的形状为 [N, max_k, 2]，其中每个键以 [src_index, dst_index] 表示
    # bonds_mask 的形状为 [N, max_k]，表示哪些键对是有效的

    # 分别取出源原子的索引和目标原子的索引，形状均为 [N, max_k]
    src_idx = bonds[..., 0]
    dst_idx = bonds[..., 1]

    # 利用 gather 在 X 中提取源、目标原子的坐标
    # X 的形状为 [N, m, 3]，我们沿着第1维（原子维度）聚合，得到 shape [N, max_k, 3]
    # 确保 X 是一个 torch.Tensor
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    src_coords = torch.gather(X, 1, src_idx.unsqueeze(-1).expand(-1, -1, 3))
    dst_coords = torch.gather(X, 1, dst_idx.unsqueeze(-1).expand(-1, -1, 3))

    # 计算对应键的向量差和欧几里得距离（L2 范数）
    diff = dst_coords - src_coords
    bond_lengths = torch.norm(diff, p=2, dim=-1)  # shape [N, max_k]

    # 对无效（mask为False）的键，将距离置为 0
    bond_lengths = bond_lengths * bonds_mask.to(bond_lengths.dtype)
    return bond_lengths


def get_sidechain_chi_angles_atoms(S, max_chi=4):
    """
    根据残基的三字母代码 S（list of str）以及 VOCAB 中预定义的 chi_angles_atoms，
    构造每个残基侧链二面角所用的原子索引。

    VOCAB 中必须包含两个字典：
      - VOCAB.chi_angles_atoms: dict, 键为三字母代码（如 "ARG", "ALA", ...），值为一个列表，
                                每个元素为一组由 4 个原子名称组成的二面角定义。
      - VOCAB.sidechain_map: dict, 键为单字母残基代码（如 'A', 'R', ...），值为该残基侧链原子的顺序列表。
        注意：backbone 原子固定为 N, CA, C, O 对应索引 0,1,2,3，侧链原子从索引 4 开始。

    返回:
      chi_atoms: LongTensor, 形状 [N, max_chi, 4]，每个有效的二面角给出完整原子列表中的 4 个索引；
      chi_mask: BoolTensor, 形状 [N, max_chi]，标记哪些二面角为有效。
    """
    N = len(S)
    chi_all = []  # 每个残基的二面角列表，每个元素为 list of [a, b, c, d] (索引对)

    # 假设我们有一个辅助字典将三字母转换为单字母代码
    # 可以从 VOCAB.aas（格式为 [(单字母, 三字母), ...]）生成
    aas_dict = dict(VOCAB.aas)

    for res_letter in S:
        # 获取该残基的单字母代码（用于查询 sidechain_map）
        res = aas_dict.get(res_letter, None)
        # 获取该残基的二面角定义，若未定义（例如 Glycine），则返回空列表
        chi_defs = VOCAB.chi_angles_atoms.get(res, [])

        # 若未找到或没有侧链原子列表，直接返回空列表
        if res is None or res_letter not in VOCAB.sidechain_map:
            chi_all.append([])
            continue
        # 获取侧链原子的顺序列表，如对于 "R" 可能为 ['CB','CG','CD','NE','CZ','NH1','NH2']
        sidechain_atoms = VOCAB.sidechain_map[res_letter]

        # 定义一个辅助函数，将原子名称转换为残基内的原子索引（backbone固定，侧链从索引4开始）
        def atom_index(atom_name):
            if atom_name == "N":
                return 0
            elif atom_name == "CA":
                return 1
            elif atom_name == "C":
                return 2
            elif atom_name == "O":
                return 3
            else:
                try:
                    idx = sidechain_atoms.index(atom_name)
                    return 4 + idx
                except ValueError:
                    return None

        chi_list = []
        for chi_def in chi_defs:  # chi_def 为包含 4 个原子名称的列表
            indices = []
            valid = True
            for atom_name in chi_def:
                idx = atom_index(atom_name)
                if idx is None:
                    valid = False
                    break
                indices.append(idx)
            if valid:
                chi_list.append(indices)
        chi_all.append(chi_list)

    # 初始化张量，形状: [N, max_chi, 4]
    chi_atoms = torch.zeros((N, max_chi, 4), dtype=torch.long)
    chi_mask = torch.zeros((N, max_chi), dtype=torch.bool)

    for i, chi_list in enumerate(chi_all):
        k = len(chi_list)
        if k > 0:
            chi_atoms[i, :k, :] = torch.tensor(chi_list, dtype=torch.long)
            chi_mask[i, :k] = True
    return chi_atoms, chi_mask


def _cal_sidechain_chis(X, S_symbol):
    """
    参数：
      S: list of str, 长度为 N，每个元素为残基的三字母代码，例如 ["ALA", "ARG", ...]
      X: Tensor, 形状 [N, m, 3]，每个残基中 m 个原子的三维坐标
      aa_feature: 对象，包含侧链二面角定义信息（例如其属性 VOCAB 与 chi_angles_atoms）

    返回：
      chi_angles: Tensor, 形状 [N, max_chi]，每个有效二面角的弧度值；
                  对于无效位置，可置为 0，或使用 mask 进行区分。
    """
    # 得到每个残基侧链二面角所用原子索引及 mask，形状分别为 [N, max_chi, 4] 与 [N, max_chi]
    chi_atoms, chi_mask = get_sidechain_chi_angles_atoms(S_symbol)
    # chi_atoms 为 LongTensor, 形状 [N, max_chi, 4]
    N, max_chi, _ = chi_atoms.shape
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    # 利用 torch.gather 从 X 中获取对应的原子坐标
    # X.shape: [N, m, 3]
    # 对于每个残基与每个 chi 定义，获取四个原子坐标 A, B, C, D，形状均为 [N, max_chi, 3]
    A = torch.gather(X, 1, chi_atoms[..., 0].unsqueeze(-1).expand(-1, -1, 3))
    B = torch.gather(X, 1, chi_atoms[..., 1].unsqueeze(-1).expand(-1, -1, 3))
    C = torch.gather(X, 1, chi_atoms[..., 2].unsqueeze(-1).expand(-1, -1, 3))
    D = torch.gather(X, 1, chi_atoms[..., 3].unsqueeze(-1).expand(-1, -1, 3))

    # 计算各向量
    b1 = B - A  # [N, max_chi, 3]
    b2 = C - B  # [N, max_chi, 3]
    b3 = D - C  # [N, max_chi, 3]

    # 归一化 b2：防止除 0，加上一个很小的数 epsilon
    eps = 1e-6
    b2_norm = b2 / (torch.norm(b2, dim=-1, keepdim=True) + eps)

    # 计算法向量 n1 和 n2
    n1 = torch.cross(b1, b2, dim=-1)  # [N, max_chi, 3]
    n2 = torch.cross(b2, b3, dim=-1)  # [N, max_chi, 3]

    # 计算 m1 = n1 x b2_norm
    m1 = torch.cross(n1, b2_norm, dim=-1)  # [N, max_chi, 3]

    # 计算 x = dot(n1, n2) 和 y = dot(m1, n2)
    x = torch.sum(n1 * n2, dim=-1)  # [N, max_chi]
    y = torch.sum(m1 * n2, dim=-1)  # [N, max_chi]

    # 计算二面角，范围在 -pi 到 pi
    chi_angles = torch.atan2(y, x)  # [N, max_chi]

    # 对无效的二面角位置置 0（或你也可以返回 mask 供外层使用）
    chi_angles = chi_angles * chi_mask.to(chi_angles.dtype)

    return chi_angles


def mask_complex_data(batch, task_mode=1):
    """
    根据任务模式对抗体的不同区域进行掩码
    task_mode:
      1 - 全部抗体掩码（任务1：从头生成）
      2 - 掩码CDR区域全部，并对框架区进行10%-20%随机掩码（任务2：CDR优化）
      3 - 仅对框架中少量节点（约5%）进行掩码（任务3：结构微调）
    返回：掩码后的序列、坐标及布尔型mask向量
    """
    X = batch['X']
    S = batch['S']
    A = batch['A']
    edges = batch['edges']
    edge_features = batch['edge_features']
    xloss_mask = batch['xloss_mask']
    residue_pos = batch['residue_pos']
    cmask = batch['cmask']
    paratope_mask = batch['paratope_mask']

    # 1. 找出所有抗体节点索引（cmask == True）
    antibody_idx = torch.where(cmask)[0]
    # 根据任务模式确定需要掩码的抗体节点
    if task_mode == 1:
        # 模式1：全部抗体节点（cmask==True）掩码
        mask_node_idx = antibody_idx
    elif task_mode == 2:
        # 模式2：掩码抗体中所有CDR节点，并对抗体框架区域随机掩码10%-20%的节点（任务2：CDR优化）

        # 1. 查找所有抗体节点中属于CDR区域的节点索引
        #    条件为：cmask为True表示抗体节点，paratope_mask为True表示该节点在CDR区域
        cdr_idx = torch.where(cmask & paratope_mask)[0]

        # 2. 查找抗体节点中不在CDR区域的节点索引，即框架部分
        framework_idx = torch.where(cmask & (paratope_mask is False))[0]

        # 3. 计算框架区域节点的总数
        num_framework = framework_idx.numel()

        # 4. 从均匀分布中随机生成一个掩码比例，范围在10%到20%之间
        #    注意：torch.empty(1).uniform_(0.1, 0.2)生成一个大小为1的张量，然后item()提取标量
        rand_ratio = torch.empty(1).uniform_(0.1, 0.2).item()

        # 5. 根据随机比例计算需要掩码的框架节点数目（向下取整）
        num_mask_framework = int(num_framework * rand_ratio)

        # 6. 如果需要掩码的框架节点数大于零，则从框架节点中随机选择对应数量的节点索引
        if num_mask_framework > 0:
            # 生成一个从0到num_framework-1的随机排列索引，并取前num_mask_framework个
            perm = torch.randperm(num_framework)[:num_mask_framework]
            # 从框架节点索引中选取对应的节点
            selected_framework = framework_idx[perm]
        else:
            # 如果框架节点数量为0或随机比例导致没有节点被选，则创建一个空的张量
            selected_framework = framework_idx.new_empty(0, dtype=torch.long)

        # 7. 将CDR区域全部掩码的节点和随机选出的框架区域节点合并起来作为最终需要掩码的抗体节点
        mask_node_idx = torch.cat([cdr_idx, selected_framework])

    elif task_mode == 3:
        # 模式3：仅对抗体CDR区中的部分节点进行掩码（任务3：结构微调）

        # 1. 查找抗体节点中不在CDR区域的节点索引，即仅包含框架区域（抗体节点且paratope_mask为False）
        cdr_idx = torch.where(cmask & paratope_mask)[0]

        # 2. 计算框架区域节点的总数
        num_cdr = cdr_idx.numel()

        # 3. 根据框架节点总数计算需要掩码的节点数，这里比例为5%
        #    使用max(1, int(num_framework * 0.05))确保至少掩码1个节点（如果框架节点非空）
        #    如果框架节点总数为0，则num_mask_framework设为0
        num_mask_cdr = max(1, int(num_cdr * 0.5)) if num_cdr > 0 else 0

        # 4. 如果需要掩码的框架节点数大于零，则随机选择对应数量的节点索引
        if num_mask_cdr > 0:
            # 生成一个从0到num_framework-1的随机排列索引，并取前num_mask_framework个
            perm = torch.randperm(num_cdr)[:num_mask_cdr]
            mask_node_idx = cdr_idx[perm]
        else:
            # 如果没有框架节点需要掩码，则创建一个空的索引张量
            mask_node_idx = cdr_idx.new_empty(0, dtype=torch.long)
    else:
        raise ValueError("task_mode 只能为 1、2 或 3")
    # 2. 对节点数据（X, S, A, xloss_mask, residue_pos）进行掩码操作
    # 这里简单地将被掩码的节点对应项置 0（或特殊标记）
    masked_X = X.clone()  # [N, 14, 3]
    masked_S = S.clone()  # [N,]
    masked_A = A.clone()  # [N, d]
    masked_xloss_mask = xloss_mask.clone()  # [N, 14]
    masked_residue_pos = residue_pos.clone()  # [N,]

    # 常见的mask做法：对于数值型输入，通常赋值0；对于类别型，可以用 -1 作为 mask 标记
    masked_X[mask_node_idx] = 0.0  # 坐标置为0
    masked_S[mask_node_idx] = 21  # 类别置为 21表示为mask
    masked_A[mask_node_idx] = 0.0  # 节点特征置为0
    masked_xloss_mask[mask_node_idx] = 0.0  # 将原子有效性掩码置0
    masked_residue_pos[mask_node_idx] = -1  # 位置编码置为 -1
    residue_atom_type = nn.parameter.Parameter(
        torch.tensor(VOCAB.residue_atom_type, dtype=torch.long).to(X.device),
        requires_grad=False)
    residue_atom_pos = nn.parameter.Parameter(
        torch.tensor(VOCAB.residue_atom_pos, dtype=torch.long).to(X.device),
        requires_grad=False)
    mask_atom_type = residue_atom_type[masked_S]
    mask_atom_pos = residue_atom_pos[masked_S]
    # 3. 对边进行掩码：若边的任一端点被掩码，则该边也需要掩码
    # 构造一个 [N,] 的布尔张量标记哪些节点被掩码
    node_mask = torch.zeros(X.size(0), dtype=torch.bool).to(X.device)
    node_mask[mask_node_idx] = True

    # edges 为 [2, E]，对于每条边只要有任一端点掩码，则将其 mask
    edge_mask = node_mask[edges[0]] | node_mask[edges[1]]
    keep_edge_mask = ~edge_mask  # 保留的边，即两端节点都未被掩码

    filtered_edges = edges[:, keep_edge_mask]
    filtered_edge_feature = edge_features[keep_edge_mask]

    return (masked_X, mask_atom_type, mask_atom_pos, masked_S, masked_A, masked_xloss_mask, masked_residue_pos,
            filtered_edges, filtered_edge_feature)


def euler_angles_to_rotation_matrix(angles, device='cpu'):
    """
    根据输入欧拉角（弧度制）生成旋转矩阵。
    假设旋转顺序为：先绕 x 轴，再绕 y 轴，最后绕 z 轴（或 R = Rz * Ry * Rx）。

    参数：
        angles: tensor, 形状 (3,)，分别表示绕 x, y, z 的旋转角度（弧度）
        device: 使用的设备

    返回：
        R: tensor, 形状 (3, 3)，旋转矩阵
    """
    rx, ry, rz = angles[0], angles[1], angles[2]

    # 计算每个角度对应的 cos 和 sin
    cosx = torch.cos(rx)
    sinx = torch.sin(rx)
    cosy = torch.cos(ry)
    siny = torch.sin(ry)
    cosz = torch.cos(rz)
    sinz = torch.sin(rz)

    # 绕 x 轴旋转的矩阵
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cosx, -sinx],
        [0, sinx, cosx]
    ], device=device)

    # 绕 y 轴旋转的矩阵
    R_y = torch.tensor([
        [cosy, 0, siny],
        [0, 1, 0],
        [-siny, 0, cosy]
    ], device=device)

    # 绕 z 轴旋转的矩阵
    R_z = torch.tensor([
        [cosz, -sinz, 0],
        [sinz, cosz, 0],
        [0, 0, 1]
    ], device=device)

    # 合成旋转矩阵，注意矩阵乘法的次序
    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R


def perturb_coords(X, cmask, paratope_mask, task_mode=1):
    """
    根据任务模式对抗体的原子坐标进行扰动。

    参数：
        X: tensor, 形状 [N, 14, 3]，原子坐标（N个残基，每个残基14个原子）
        cmask: tensor, 形状 [N]，布尔类型，True 表示该残基属于抗体
        paratope_mask: tensor, 形状 [N]，布尔类型，对于抗体：True 表示CDR区域，False 表示框架区域
        task_mode: int，任务模式。1：从头生成；2：CDR优化；3：结构精调
        device: 使用的设备

    返回：
        X_new: tensor, [N, 14, 3]，经过扰动后的坐标
    """
    # 复制原始坐标，避免修改原输入
    device = X.device
    X_new = X.clone().to(device)
    # 获取抗体节点（cmask==True）的索引（抗原节点保持不变）
    antibody_idx = torch.where(cmask)[0]
    if task_mode == 1:
        # 任务1：从头生成任务
        # 对抗体节点进行完全随机初始化，坐标范围较大（例如[-20, 20] Å）
        # 这种完全随机化模拟未知起始构象
        # 抗体节点的个数
        num_antibody = antibody_idx.numel()
        if num_antibody > 0:
            # 生成随机坐标，[num_antibody, 14, 3]
            random_coords = torch.empty((num_antibody, X.size(1), 3), device=device).uniform_(-20, 20)
            X_new[antibody_idx] = random_coords
    elif task_mode == 2:
        # 任务2：CDR优化任务
        # 整体先对抗体施加较小的刚体变换（假定框架部分可锚定于抗原附近）
        # 随后对CDR区域（paratope_mask==True）施加较大扰动，模拟较高的不确定性

        # 1. 全局刚体变换（适用于整个抗体）
        # 随机生成较小幅度的旋转角（单位：度，转换为弧度），例如 [-30°, 30°]
        angles_deg = torch.empty(3, device=device).uniform_(-10, 10)
        angles = angles_deg * math.pi / 180.0  # 转换为弧度
        # 生成旋转矩阵
        R = euler_angles_to_rotation_matrix(angles, device=device)
        # 随机平移量，较小范围，例如 [-8, 8] Å
        translation = torch.empty(3, device=device).uniform_(-2, 2)

        # 对所有抗体节点应用全局刚体变换
        # 注意：对每个残基的所有原子进行变换
        X_new[antibody_idx] = torch.matmul(X_new[antibody_idx], R.t()) + translation

        # 2. 对CDR区域的残基施加额外扰动
        # 筛选出抗体中属于CDR区域的残基索引
        cdr_idx = torch.where(cmask & paratope_mask)[0]
        if cdr_idx.numel() > 0:
            # 对CDR区域，进一步施加较大扰动
            # 例如，可以在原有全局变换的基础上增加一个较大的随机平移
            # 这里采用较大范围的随机平移（例如 [-4, 4] Å）
            extra_translation = torch.empty((cdr_idx.numel(), X.size(1), 3), device=device).uniform_(-5, 5)
            X_new[cdr_idx] = X_new[cdr_idx] + extra_translation

            # 另一种选择是直接重新初始化CDR区域的坐标，
            # 例如将其设置为沿框架延伸的拉直链或者完全打乱（此处仅做示例，可根据实际需要修改）
            # random_cdr = torch.empty((cdr_idx.numel(), X.size(1), 3), device=device).uniform_(-10, 10)
            # X_new[cdr_idx] = random_cdr
    elif task_mode == 3:
        # 任务3：结构精调任务
        # 对抗体施加微小扰动，模拟实验结构与最佳构型的细微差异
        # 这里采用在各原子坐标上加上均值为0、标准差约1Å 的高斯噪声
        if antibody_idx.numel() > 0:
            noise = torch.randn((antibody_idx.numel(), X.size(1), 3), device=device) * 1.0  # std=1 Å
            X_new[antibody_idx] = X_new[antibody_idx] + noise

    else:
        raise ValueError("未知的任务模式，task_mode 必须为 1、2 或 3。")
    return X_new


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


# 若作为独立模块运行，则给出示例
if __name__ == "__main__":
    # 示例1：构造残基 i 的局部坐标系，并计算局部方向编码
    n_i = np.array([1.03, 2.03, 3.0], dtype=np.float32)  # 残基 i 的 N 原子
    ca_i = np.array([2.02, 3.023, 4.02], dtype=np.float32)  # 残基 i 的 Cα 原子
    c_i = np.array([3.01, 4.0, 5.03], dtype=np.float32)  # 残基 i 的 C 原子
    # ---------------------------
    # 示例数据：残基 j 的坐标
    # ---------------------------
    n_j = np.array([4.031, 52.0, 6.0], dtype=np.float32)  # 残基 j 的 N 原子坐标
    ca_j = np.array([5.0, 6.0, 7.03], dtype=np.float32)  # 残基 j 的 Cα 原子坐标
    c_j = np.array([6.02, 7.0, 8.02], dtype=np.float32)  # 残基 j 的 C 原子坐标

    # 构造残基 i 的局部坐标系 O_i
    X = torch.tensor([[20.4690, 30.7340, 19.2860],
                      [20.1930, 31.5470, 20.4840],
                      [19.3060, 30.8170, 21.4770],
                      [18.5060, 29.9800, 21.0750]])
    O_i = construct_local_frame(X)
    print("残基 i 的局部坐标系 O_i:")
    print(O_i)

    # 计算残基 j 在残基 i 局部坐标系中的方向编码
    local_dir = compute_local_direction(O_i, ca_i, ca_j)
    print("残基 j 相对于残基 i 的局部方向编码:")
    print(local_dir)

    # 示例2：计算旋转编码 q(O_i^T O_j)

    # 利用 construct_local_frame 构造残基 i 的局部坐标系 O_i，
    # 以及残基 j 的局部坐标系 O_j。
    O_i = construct_local_frame(X, )  # O_i 的形状为 (3, 3)
    O_j = construct_local_frame(X, )  # O_j 的形状为 (3, 3)

    # ---------------------------
    # 计算旋转编码
    # ---------------------------
    # compute_rotation_encoding 函数会计算从残基 i 局部坐标系到残基 j 局部坐标系的旋转差异，
    # 表示为一个 4 维四元数 q，即 q(O_i^T O_j)
    q_encoding = compute_rotation_encoding(O_i, O_j)

    print("残基 j 相对于残基 i 的旋转编码 (四元数):", q_encoding)
