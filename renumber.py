#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from anarci import run_anarci


def renumber_seq(seq, scheme='imgt'):
    _, numbering, details, _ = run_anarci([('A', seq)], scheme=scheme, allowed_species=['mouse', 'human'])
    numbering = numbering[0]
    fv, position = [], []
    if not numbering:  # not antibody
        return None
    chain_type = details[0][0]['chain_type']
    numbering = numbering[0][0]
    for pos, res in numbering:
        if res == '-':
            continue
        fv.append(res)
        position.append(pos)
    return ''.join(fv), position, chain_type


def renumber_pdb(pdb, out_pdb, scheme='imgt', mute=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('anonym', pdb)
    for chain in structure.get_chains():
        seq = []
        for residue in chain:
            hetero_flag, _, _ = residue.get_id()
            if hetero_flag != ' ':
                continue
            seq.append(seq1(residue.get_resname()))
        seq = ''.join(seq)
        res = renumber_seq(seq, scheme)
        if res is None:
            continue
        fv, position, chain_type = res
        if not mute:
            print(f'chain {chain.id} type: {chain_type}')
        start = seq.index(fv)
        end = start + len(fv)
        assert start != -1, 'fv not found'
        seq_index, pos_index = -1, 0
        for r in list(chain.get_residues()):
            hetero_flag, _, _ = r.get_id()
            if hetero_flag != ' ':
                continue
            seq_index += 1
            if seq_index < start or seq_index >= end:
                chain.__delitem__(r.get_id())
                continue
            assert fv[pos_index] == seq1(r.get_resname()), f'Inconsistent residue in Fv {fv[pos_index]} at {r._id}'
            r._id = (' ', *position[pos_index])
            pos_index += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


# ----------------------
# IMGT 编号工具
# ----------------------
def _imgt_parse_one(x):
    """把一个 IMGT 标签（如 '105', '105A'）解析成 (base:int, ins:str)。x 也可能是 int。"""
    if isinstance(x, (bytes, np.bytes_)):
        s = x.decode()
    else:
        s = str(x)
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    base = int(s[:i]) if i > 0 else 0
    ins = s[i:] if i < len(s) else ''
    return base, ins


def _imgt_compose(base: int, ins: str) -> str:
    return f"{int(base)}{ins or ''}"


def _letters():
    """无限字母生成器：A,B,C,...,Z,AA,AB,...（实际基本用不到两位，但做了兜底）"""
    import string
    alphabet = string.ascii_uppercase
    L = len(alphabet)
    n = 1
    while True:
        # 长度为 n 的所有组合
        if n == 1:
            for c in alphabet:
                yield c
        else:
            # 递归生成 n 位
            from itertools import product
            for tup in product(alphabet, repeat=n):
                yield ''.join(tup)
        n += 1


def imgt_relabel_to_target_window(src_labels, tgt_cdr_pos_slice):
    """
    将【源 CDR 的 residue_pos 序列】重映射到目标的 IMGT 窗口（例如 105..107）。
    规则：
      1) 目标窗口的整数序列来自 tgt_cdr_pos_slice 的整数部分（例如 [105,106,107]）
      2) 源中与同一整数位相同的残基：第一个保留整数位，其余按 A,B,C… 递增
      3) 超过窗口的 base（<start 或 >end），夹到两端并作为插入码（很少发生，但兜底）
    返回：np.ndarray[str]，长度与 src_labels 相同
    """
    src_labels = np.asarray(src_labels.detach().cpu().numpy())
    # 目标窗口整数基数列表
    tgt_bases = [_imgt_parse_one(x)[0] for x in np.asarray(tgt_cdr_pos_slice)]
    bases_sorted = sorted(set(tgt_bases))
    start, end = bases_sorted[0], bases_sorted[-1]

    # 统计每个 base 的数量（按照源的顺序填充）
    bins = {b: [] for b in range(start, end + 1)}
    left_bin = []  # <start 的落这里
    right_bin = []  # >end   的落这里

    for lab in src_labels:
        b, ins = _imgt_parse_one(lab)
        if b < start:
            left_bin.append((b, ins))
        elif b > end:
            right_bin.append((b, ins))
        else:
            bins[b].append((b, ins))

    # 开始按窗口输出：每个 base 依次输出
    out = []
    for b in range(start, end + 1):
        items = bins[b]
        if not items:
            continue
        # 第一位用整数 b
        out.extend([b] * len(items))

    # 左右越界 -> 夹到边界整数位
    if left_bin:
        out = [start] * len(left_bin) + out
    if right_bin:
        out.extend([end] * len(right_bin))

    return np.asarray(out, dtype=np.int32)


if __name__ == '__main__':
    import sys

    infile, outfile, scheme = sys.argv[1:4]
    if len(sys.argv) > 4:
        mute = bool(sys.argv[4])
    else:
        mute = False
    renumber_pdb(infile, outfile, scheme, mute)
