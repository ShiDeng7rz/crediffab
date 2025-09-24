# -*- coding: utf-8 -*-
from typing import Tuple

# =========================
# 固化的基础映射（保持与你现有语义一致）
# =========================
IDX2ATOM: Tuple[str, ...] = ('p', 'C', 'N', 'O', 'S')  # 0:mask, 1:pad, 2:C, 3:N, 4:O, 5:S
IDX2ATOM_POS: Tuple[str, ...] = ('p', 'b', 'B', 'G', 'D', 'E', 'Z', 'H')  # 0:mask,1:pad,2:主链,3~8:侧链位点
BACKBONE_ATOMS: Tuple[str, ...] = ('N', 'CA', 'C', 'O')
MAX_ATOM_NUMBER: int = 15

# 你的真实顺序（与 AA 枚举一致）：
# ALA=0, CYS=1, ASP=2, GLU=3, PHE=4, GLY=5, HIS=6, ILE=7, LYS=8, LEU=9,
# MET=10, ASN=11, PRO=12, GLN=13, ARG=14, SER=15, THR=16, VAL=17, TRP=18, TYR=19, UNK=20
AAS_ORDER: Tuple[Tuple[str, str], ...] = (
    ('A', 'ALA'),
    ('C', 'CYS'),
    ('D', 'ASP'),
    ('E', 'GLU'),
    ('F', 'PHE'),
    ('G', 'GLY'),
    ('H', 'HIS'),
    ('I', 'ILE'),
    ('K', 'LYS'),
    ('L', 'LEU'),
    ('M', 'MET'),
    ('N', 'ASN'),
    ('P', 'PRO'),
    ('Q', 'GLN'),
    ('R', 'ARG'),
    ('S', 'SER'),
    ('T', 'THR'),
    ('V', 'VAL'),
    ('W', 'TRP'),
    ('Y', 'TYR'),
    ('X', 'UNK'),
)


ORDERED_SYMBOLS: Tuple[str, ...] = tuple([s for s, _ in AAS_ORDER])
ORDERED_ABRVS: Tuple[str, ...] = tuple([a for _, a in AAS_ORDER])

# 与现有 sidechain_map 完全一致（务必保持）
SIDECHAIN_MAP = {
    'G': [],
    'A': ['CB'],
    'V': ['CB', 'CG1', 'CG2'],
    'L': ['CB', 'CG', 'CD1', 'CD2'],
    'I': ['CB', 'CG1', 'CG2', 'CD1'],
    'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'D': ['CB', 'CG', 'OD1', 'OD2'],
    'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'N': ['CB', 'CG', 'OD1', 'ND2'],
    'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'M': ['CB', 'CG', 'SD', 'CE'],
    'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'S': ['CB', 'OG'],
    'T': ['CB', 'OG1', 'CG2'],
    'C': ['CB', 'SG'],
    'P': ['CB', 'CG', 'CD'],
    'X': [],  # UNK 最小占位
    # specials 不展开
}

# 字符 → 索引映射（固化）
ATOM2IDX = {a: i for i, a in enumerate(IDX2ATOM)}  # 'C'->2, 'N'->3, 'O'->4, 'S'->5, 'm'->0, 'p'->1
ATOMPOS2IDX = {p: i for i, p in enumerate(IDX2ATOM_POS)}  # 'b'->2, 'B'->3, 'G'->4, ...


def _atom_to_type_idx(atom_name: str) -> int:
    """原子名首字母决定类型：C/N/O/S；否则回退 pad。"""
    head = atom_name[0] if atom_name else 'p'
    return ATOM2IDX.get(head, ATOM2IDX['p'])


def _atom_to_pos_idx(atom_name: str) -> int:
    """主链固定为 'b'；侧链按第二字符映射 B/G/D/E/Z/H；未知回退 pad。"""
    if atom_name in BACKBONE_ATOMS:
        return ATOMPOS2IDX['b']
    if len(atom_name) >= 2:
        return ATOMPOS2IDX.get(atom_name[1], ATOMPOS2IDX['p'])
    return ATOMPOS2IDX['p']


def _build_residue_tables():
    """构建与 ORDERED_SYMBOLS 对齐的 residue_atom_type / residue_atom_pos（每行长度=14）。"""
    residue_atom_type, residue_atom_pos = [], []

    bb_type = [_atom_to_type_idx(a) for a in BACKBONE_ATOMS]  # [N(3), CA(2), C(2), O(4)]
    bb_pos = [_atom_to_pos_idx(a) for a in BACKBONE_ATOMS]  # [2, 2, 2, 2]

    # 这些 token + MASK(*) → 全 mask；PAD(#) 与 NANO(~) → 全 pad（与你原逻辑保持）
    global_mask_nodes = {'&', '+', '-', '*'}

    for sym in ORDERED_SYMBOLS:
        if sym in global_mask_nodes:
            row_type = [ATOM2IDX['m']] * MAX_ATOM_NUMBER
            row_pos = [ATOMPOS2IDX['m']] * MAX_ATOM_NUMBER
        else:
            # 标准氨基酸
            row_type = bb_type.copy()
            row_pos = bb_pos.copy()
            for sc_atom in SIDECHAIN_MAP.get(sym, []):
                row_type.append(_atom_to_type_idx(sc_atom))
                row_pos.append(_atom_to_pos_idx(sc_atom))
            need = MAX_ATOM_NUMBER - len(row_type)
            if need < 0:
                # 极端保护（理论不会发生）
                row_type = row_type[:MAX_ATOM_NUMBER]
                row_pos = row_pos[:MAX_ATOM_NUMBER]
            else:
                row_type += [ATOM2IDX['p']] * need
                row_pos += [ATOMPOS2IDX['p']] * need

        residue_atom_type.append(row_type)
        residue_atom_pos.append(row_pos)

    return residue_atom_type, residue_atom_pos
