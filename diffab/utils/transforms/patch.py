import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants


@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):

    def __init__(self, initial_patch_size=128, antigen_size=128):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size

    def _center(self, data, origin):
        origin = origin.reshape(1, 1, 3)
        data['pos_heavyatom'] -= origin  # (L, A, 3)
        data['pos_heavyatom'] = data['pos_heavyatom'] * data['mask_heavyatom'][:, :, None]
        data['origin'] = origin.reshape(3)
        return data

    def __call__(self, data):
        anchor_flag = data['anchor_flag']  # (L,)
        anchor_points = data['pos_heavyatom'][anchor_flag, constants.BBHeavyAtom.CA]  # (n_anchors, 3)
        antigen_mask = (data['fragment_type'] == constants.Fragment.Antigen)
        antibody_mask = torch.logical_not(antigen_mask)

        if anchor_flag.sum().item() == 0:
            # Generating full antibody-Fv, no antigen given
            data_patch = _mask_select_data(
                data=data,
                mask=antibody_mask,
            )
            data_patch = self._center(
                data_patch,
                origin=data_patch['pos_heavyatom'][:, constants.BBHeavyAtom.CA].mean(dim=0)
            )
            return data_patch

        pos_alpha = data['pos_heavyatom'][:, constants.BBHeavyAtom.CA]  # (L, 3)
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]  # (L, )
        initial_patch_idx = torch.topk(
            dist_anchor,
            k=min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]  # (initial_patch_size, )

        dist_anchor_antigen = dist_anchor.masked_fill(
            mask=antibody_mask,  # Fill antibody with +inf
            value=float('+inf')
        )  # (L, )
        antigen_patch_idx = torch.topk(
            dist_anchor_antigen,
            k=min(self.antigen_size, antigen_mask.sum().item()),
            largest=False, sorted=True
        )[1]  # (ag_size, )

        patch_mask = torch.logical_or(
            data['generate_flag'],
            data['anchor_flag'],
        )
        patch_mask[initial_patch_idx] = True
        patch_mask[antigen_patch_idx] = True

        patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]

        data_patch = _mask_select_data(data, patch_mask)
        data_patch = self._center(
            data_patch,
            origin=anchor_points.mean(dim=0)
        )
        data_patch['patch_idx'] = patch_idx
        return data_patch


def _generate_graph_edges(
        X,  # [N, A, 3] 所有原子坐标
        tags,  # [N]       链/类型标签（例如: 0=heavy,1=light,3=antigen）
        mask,  # [N, A]    原子有效掩码
        *,
        ca_index=1,  # Cα 在 A 维的索引；按你项目里的常量替换
        c1=4.5,  # 同链半径阈值（Å）
        c2=6.0,  # 跨链半径阈值（Å）
        c3=2.0,  # 序列相邻阈值（Å）
        k_intra=48,  # 同链每点最多保留邻居数
        k_inter=32,  # 跨链每点最多保留邻居数
        bidir=True,  # 是否复制反向边
):
    """
    返回:
        edge_index: LongTensor [2, E]
    说明:
        - 只用代表性坐标（默认 Cα）来近似残基间距离，极大降低计算和显存。
        - 每个节点按同链/跨链各自做 top-k，边数受控 -> 避免 OOM。
        - 默认不复制反向边（bidir=False）；如模型需要双向，设 True。
    """
    device = X.device
    N, A, _ = X.shape
    tags = tags.to(device)
    mask = mask.to(device)

    # 1) 代表性坐标：优先取 Cα；若该残基 Cα 缺失，用该残基第一颗有效原子兜底
    #    （避免因缺 Cα 产生 NaN）
    has_ca = mask[:, ca_index] if ca_index is not None else torch.zeros(N, dtype=torch.bool, device=device)
    if ca_index is not None:
        x_rep = X[:, ca_index]  # [N,3]
    else:
        x_rep = X[:, 0]  # 万一不给 index，就先取第 0 个

    # 如果 Cα 缺失，用第一个有效原子顶上
    if has_ca.any().item() is False or (~has_ca).any().item():
        # 找每个残基第一个有效原子下标
        # idx_first_valid[i] = 该残基第一个 True 的原子位置，否则 0
        idx_first_valid = torch.argmax(mask.float(), dim=1)  # 没有 True 时返回 0，但下面再用 has_any 过滤
        has_any = mask.any(dim=1)
        x_fallback = X[torch.arange(N, device=device), idx_first_valid]  # [N,3]
        x_rep = torch.where(has_ca[:, None], x_rep, x_fallback)
        # 对完全无原子的残基，直接屏蔽（后续不连边）
        valid_node = has_any
    else:
        valid_node = has_ca  # 至少有 Cα

    # 如果全是无效残基
    if valid_node.sum() <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    # 2) 计算代表点两两距离 (N x N)
    #    注意：用 torch.cdist 比你之前的 (N,N,14,14) 好得多
    xv = x_rep
    D = torch.cdist(xv, xv, p=2)  # [N,N]

    # 3) 构造序列相邻边（非抗原 & 同链 & 距离<=c3）
    idx = torch.arange(N - 1, device=device)
    # 非抗原掩码（保留你原来的规则）

    same_chain_adj = (tags[:-1] == tags[1:])
    adj_ok = same_chain_adj & (D[idx, idx + 1] <= c3)
    i_seq = idx[adj_ok]
    seq_edges = torch.stack([i_seq, i_seq + 1], dim=0)  # [2, E_seq]
    if bidir:
        seq_edges = torch.cat([seq_edges, seq_edges.flip(0)], dim=1)

    # 4) 同链近邻：每个 i 选 <=k_intra 个最近的同链邻居（不含自己）且距离<=c1
    tags_i = tags.unsqueeze(1)  # [N,1]
    tags_j = tags.unsqueeze(0)  # [1,N]
    same_chain = (tags_i == tags_j)  # [N,N]
    # 有效节点才参与
    valid_mask_row = valid_node.unsqueeze(1)  # [N,1]
    valid_mask_col = valid_node.unsqueeze(0)  # [1,N]
    intra_mask = same_chain & valid_mask_row & valid_mask_col
    # 排除对角线
    intra_mask.fill_diagonal_(False)
    # 半径约束
    intra_mask = intra_mask & (D <= c1)

    # 对每一行做 top-k（把无效位置置 +inf，再 topk 最小 k 个）
    # 注意：若有效邻居少于 k_intra，topk 会返回全体；我们再过滤 inf
    D_intra = D.clone()
    D_intra[~intra_mask] = float('+inf')
    # 为了拿到最小 k 个，把负号取反：取 topk(-D)
    k_intra = min(k_intra, N - 1)
    vals_i, idxs_i = torch.topk(-D_intra, k=k_intra, dim=1)  # [N,k]
    # 过滤掉 inf（即原来是无效位置）
    keep_i = (vals_i != float('-inf'))
    # 构造 (src,dst)
    src_i = torch.arange(N, device=device).unsqueeze(1).expand_as(idxs_i)[keep_i]
    dst_i = idxs_i[keep_i]
    intra_edges = torch.stack([src_i, dst_i], dim=0)
    if bidir:
        intra_edges = torch.cat([intra_edges, intra_edges.flip(0)], dim=1)

    # 5) 跨链近邻：每个 i 选 <=k_inter 个最近的跨链邻居且距离<=c2
    inter_mask = (~same_chain) & valid_mask_row & valid_mask_col & (D <= c2)
    D_inter = D.clone()
    D_inter[~inter_mask] = float('+inf')
    k_inter = min(k_inter, N)
    vals_e, idxs_e = torch.topk(-D_inter, k=k_inter, dim=1)
    keep_e = (vals_e != float('-inf'))
    src_e = torch.arange(N, device=device).unsqueeze(1).expand_as(idxs_e)[keep_e]
    dst_e = idxs_e[keep_e]
    inter_edges = torch.stack([src_e, dst_e], dim=0)
    if bidir:
        inter_edges = torch.cat([inter_edges, inter_edges.flip(0)], dim=1)

    # 6) 合并并去重
    edge_index = torch.cat([seq_edges, intra_edges, inter_edges], dim=1)  # [2, E]
    # 去掉自环（双保险）
    self_loop = edge_index[0] == edge_index[1]
    if self_loop.any():
        edge_index = edge_index[:, ~self_loop]

    # 去重（对无向图：只保留 src<dst；若 bidir=True，可先做无向去重、再复制反向）
    if not bidir:
        # 把 (i,j) 和 (j,i) 统一为 (min,max) 后 uniq
        u = torch.minimum(edge_index[0], edge_index[1])
        v = torch.maximum(edge_index[0], edge_index[1])
        uv = torch.stack([u, v], dim=0)
        # unique 需要转置到 [E,2]
        uv2 = uv.t().contiguous()
        uv2 = torch.unique(uv2, dim=0)
        edge_index = uv2.t().contiguous()
    else:
        # bidir: 用 unique 去重精确相等的重复边
        ei2 = edge_index.t().contiguous()
        ei2 = torch.unique(ei2, dim=0)
        edge_index = ei2.t().contiguous()

    return edge_index


@register_transform('patch_cdr_epitope')
class PatchCDREpitope(object):

    def __init__(self, initial_patch_size=96, antigen_size=128):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size

    def __call__(self, data):
        antibody = data['antibody']
        antigen = data['antigen']
        complex = data['complex']

        anchor_flag = complex['anchor_flag']  # (L,)
        anchor_points = complex['pos_heavyatom'][anchor_flag, constants.BBHeavyAtom.CA]  # (n_anchors, 3)
        antigen_mask = (complex['fragment_type'] == constants.Fragment.Antigen)
        antibody_mask = torch.logical_not(antigen_mask)

        pos_alpha = complex['pos_heavyatom'][:, constants.BBHeavyAtom.CA]  # (L, 3)
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]  # (L, )
        initial_patch_idx = torch.topk(
            dist_anchor,
            k=min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]  # (initial_patch_size, )

        dist_anchor_antigen = dist_anchor.masked_fill(
            mask=antibody_mask,  # Fill antibody with +inf
            value=float('+inf')
        )  # (L, )
        antigen_patch_idx = torch.topk(
            dist_anchor_antigen,
            k=min(self.antigen_size, antigen_mask.sum().item()),
            largest=False, sorted=True
        )[1]  # (ag_size, )

        patch_mask = torch.logical_or(
            complex['generate_flag'],
            complex['anchor_flag'],
        )
        patch_mask[initial_patch_idx] = True
        patch_mask[antigen_patch_idx] = True

        # 复合物（全局）索引
        complex_patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]

        # === 2) 把 complex 的 patch 映射到 antibody / antigen 的局部掩码 ===
        # 说明：若 complex 是按某个顺序拼的（常见是先抗体再抗原），
        # 下面这种“先用 fragment mask 过滤，再用 patch_mask 取子掩码”的方式
        # 与具体拼接顺序无关，安全且通用。
        antibody_local_mask = patch_mask[antibody_mask]  # 长度 == antibody 的残基数
        antigen_local_mask = patch_mask[antigen_mask]  # 长度 == antigen  的残基数

        # === 3) 分别裁剪 antibody / antigen / complex ===
        antibody_patch = _mask_select_data(antibody, antibody_local_mask)
        antigen_patch = _mask_select_data(antigen, antigen_local_mask)
        complex_patch = _mask_select_data(complex, patch_mask)

        # === 4) 记录各自的 patch_idx（局部与全局都给，便于追踪）===
        # 全局（复合物）索引
        complex_patch['patch_idx'] = complex_patch_idx

        # 抗体/抗原的“局部”索引（相对各自链起点的索引）
        La = antibody['aa'].size(0) if antibody is not None else 0
        Lg = antigen['aa'].size(0) if antigen is not None else 0
        if La > 0:
            antibody_patch['patch_idx'] = torch.arange(La)[antibody_local_mask]
        if Lg > 0:
            antigen_patch['patch_idx'] = torch.arange(Lg)[antigen_local_mask]

        antibody_patch['edges'] = _generate_graph_edges(
            antibody_patch['pos_heavyatom'], antibody_patch['fragment_type'], antibody_patch['mask_heavyatom']
        )
        antigen_patch['edges'] = _generate_graph_edges(
            antigen_patch['pos_heavyatom'], antigen_patch['fragment_type'], antigen_patch['mask_heavyatom']
        )

        # === 5) 组装返回 ===
        data_patch = {
            'antibody': antibody_patch,
            'antigen': antigen_patch,
            'complex': complex_patch,
        }
        return data_patch
