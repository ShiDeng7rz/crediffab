import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_model.egnn_clean import EGNN
from cl_model.position_embedding import SinusoidalPositionEmbedding
from diffab.datasets.AminoAcidVocab import _build_residue_tables, IDX2ATOM, IDX2ATOM_POS
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom

LOGIT_MIN = math.log(1 / 100.0)  # 1/T ∈ [0.01, 100]
LOGIT_MAX = math.log(100.0)

PRECOMP_RESIDUE_ATOM_TYPE, PRECOMP_RESIDUE_ATOM_POS = _build_residue_tables()
MAX_ATOM_NUMBER = 15


class WeightedAttnReadout(nn.Module):
    def __init__(self, dim, heads=4, init_cdr_bias=0.6, init_iface_bias=0.8, p_drop=0.1,
                 mean_res_scale=0.2):  # ← 新增：残差缩放
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0
        self.ln_kv = nn.LayerNorm(dim)
        self.ln_q = nn.LayerNorm(dim)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(2 * dim, dim, bias=False)
        self.res_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.cdr_bias = nn.Parameter(torch.tensor(float(init_cdr_bias)))
        self.iface_bias = nn.Parameter(torch.tensor(float(init_iface_bias)))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(p_drop)
        self.mean_res_scale = mean_res_scale  # ← 保存

    def forward(self, h, batch, cdr_mask=None, iface_mask=None):
        B, D = int(batch.max().item()) + 1, h.size(-1)
        outs = []
        for b in range(B):
            m = (batch == b)
            hb_raw = h[m]  # [Nb, D]
            mean_raw = hb_raw.mean(0, keepdim=True)  # [1, D]
            max_raw = hb_raw.max(0, keepdim=True).values  # [1, D]
            q = self.to_q(torch.cat([mean_raw, max_raw], -1))
            q = self.drop(self.ln_q(q))
            k = self.to_k(self.ln_kv(hb_raw)).view(-1, self.heads, self.head_dim)
            v = self.to_v(self.ln_kv(hb_raw)).view(-1, self.heads, self.head_dim)
            q = q.view(self.heads, self.head_dim)

            logit = (k * q.unsqueeze(0)).sum(-1) / (self.head_dim ** 0.5)
            logit = logit * self.logit_scale
            if cdr_mask is not None:
                logit = logit + self.cdr_bias * cdr_mask[m].float().view(-1, 1)
            if iface_mask is not None:
                logit = logit + self.iface_bias * iface_mask[m].float().view(-1, 1)

            attn = logit.softmax(dim=0).unsqueeze(-1)  # [Nb, H, 1]
            g_attn = (attn * v).sum(0).reshape(1, D)  # [1, D]
            out = self.out(g_attn) + self.mean_res_scale * self.res_proj(mean_raw)
            outs.append(out)
        return torch.cat(outs, dim=0)  # [B, D]


class ContrastiveLearningModel(nn.Module):
    """
    A contrastive learning model to learn latent representations for antibodies and antigens.
    """

    def __init__(self, input_node_dim=10, input_edge_dim=1, hidden_dim=128, num_node_attr=6, vocab_size=None,
                 max_len=2048, temperature=1.0, device='cpu', use_esm_in_forward=True,
                 esm_model_name='esm2_t33_650M_UR50D', max_relpos=64, feat_dim=32):
        super(ContrastiveLearningModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size if vocab_size is not None else 22
        # ---- 残基/位置编码统一到 hidden_dim ----
        self.residue_type_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.residue_pos_embedding = SinusoidalPositionEmbedding(hidden_dim, max_len)

        self.temperature = temperature
        # “原子词典”嵌入（C1）
        self.num_atom_pos = len(IDX2ATOM_POS)
        self.num_atom_types = len(IDX2ATOM)
        self.atom_embed_dim = max(8, hidden_dim // 4)  # 小一些更稳
        self.atom_pos_embedding = nn.Embedding(self.num_atom_pos, self.atom_embed_dim)
        self.atom_type_embedding = nn.Embedding(self.num_atom_types, self.atom_embed_dim)
        self.num_node_attr = num_node_attr + hidden_dim
        self.atom_proj = nn.Linear(self.atom_embed_dim, hidden_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)
        # ---- 外部节点属性投影到 hidden_dim ----
        self.node_attr_proj = nn.Linear(num_node_attr, hidden_dim)

        # Shared or separate encoders for antibody and antigen
        self.encoder = EGNN(in_node_nf=hidden_dim, hidden_nf=hidden_dim, out_node_nf=hidden_dim,
                            in_edge_nf=input_edge_dim, device=device)
        self.residue_embed = ResidueEmbedding(hidden_dim, MAX_ATOM_NUMBER)
        # 残基对嵌入（编码残基之间的几何和序列关系）
        self.pair_embed = PairEmbedding(hidden_dim // 2, MAX_ATOM_NUMBER)
        # Projection heads for contrastive learning
        self.proj_head_ab = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        )

        self.proj_head_ag = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        )
        self.max_relpos = 64
        # ---- 读出 ----
        self.readout = WeightedAttnReadout(hidden_dim)
        # ---- 可学习温度（供外部 InfoNCE 调用）----
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        # ---- 把 VOCAB 中的查表缓存成 buffer（非训练参数）----
        self.register_buffer('residue_atom_type_tbl', torch.tensor(PRECOMP_RESIDUE_ATOM_TYPE, dtype=torch.long))
        self.register_buffer('residue_atom_pos_tbl', torch.tensor(PRECOMP_RESIDUE_ATOM_POS, dtype=torch.long))

    def _masked_mean(self, x, mask, dim=1, eps=1e-6):
        """
        x:   [N, 14, H] 或 [N, 14, D]
        mask:[N, 14] (0/1 or float)
        return: [N, H]
        """
        if mask is None:
            return x.mean(dim=dim)
        m = mask.float().unsqueeze(-1)  # [N,14,1]
        num = (x * m).sum(dim=dim)  # [N,H]
        den = m.sum(dim=dim).clamp_min(eps)  # [N,1]
        return num / den

    @torch.no_grad()
    def pack_to_big_graph(self, data):
        """
        输入
          data['aa']              : [B, N]
          data['pos_heavyatom']   : [B, N, 15, 3]
          data['res_nb']          : [B, N]
          data['mask_heavyatom']  : [B, N, 15]    # 有效节点: mask.any(-1) 为 True
          data['edges']           : List[B] of [2, E_b]  (每个样本边数不同)
          data['edge_features']   : List[B] of [E_b, Fe] (同上, 可为 None)

        输出（拼成一张大图）
          out = {
            'aa'            : [∑N_b],
            'pos_heavyatom' : [∑N_b, 15, 3],
            'res_nb'        : [∑N_b],
            'xloss_mask'    : [∑N_b, 15],
            'edge_index'    : [2, ∑E_b_valid],     # 已重映射并做了节点偏移
            'edge_attr'     : [∑E_b_valid, Fe] or None,
            'batch_id'      : [∑N_b],              # 节点所属图 id ∈ [0, B)
            'num_nodes_per_graph': [B],            # 每个样本有效节点数
            'graph_ptr'     : [B+1],               # [0, n0, n0+n1, ...]
          }
        """
        s = data['aa']
        x = data['pos_heavyatom']
        A = data['A']
        residue_pos = data['res_nb']
        xloss_mask = data['mask_heavyatom']
        chain_nb = data['chain_nb']
        cdr_flag = data['generate_flag']
        edges_list = data['edges']

        B, N = s.shape
        device = s.device

        # 有效节点掩码：每个残基只要有任一原子是有效的，就认为这个节点有效
        node_mask = xloss_mask.any(dim=-1)  # [B, N] -> bool

        # 汇总每个样本有效节点数
        Ns = [int(node_mask[b].sum().item()) for b in range(B)]
        Ns_t = torch.tensor(Ns, dtype=torch.long, device=device)
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = torch.cumsum(Ns_t[:-1], dim=0)

        # —— 节点级字段拼接（只取有效节点）——
        aa_cat, x_cat, res_cat, chn_cat, mask_cat, batch_id, a_cat, cdr_cat = [], [], [], [], [], [], [], []
        for b in range(B):
            m = node_mask[b]  # [N]
            if m.any():
                aa_cat.append(s[b][m])
                a_cat.append(A[b][m])
                x_cat.append(x[b][m])
                res_cat.append(residue_pos[b][m])
                mask_cat.append(xloss_mask[b][m])
                chn_cat.append(chain_nb[b][m])
                cdr_cat.append(cdr_flag[b][m])
                batch_id.append(torch.full((Ns[b],), b, dtype=torch.long, device=device))
            # 如果整张图没有有效节点，也允许跳过；相应边也会被全部丢弃

        if aa_cat:
            aa_cat = torch.cat(aa_cat, dim=0).contiguous()
            a_cat = torch.cat(a_cat, dim=0).contiguous()
            x_cat = torch.cat(x_cat, dim=0).contiguous()
            res_cat = torch.cat(res_cat, dim=0).contiguous()
            mask_cat = torch.cat(mask_cat, dim=0).contiguous()
            cdr_cat = torch.cat(cdr_cat, dim=0).contiguous()
            chn_cat = torch.cat(chn_cat, dim=0).contiguous()
            batch_id = torch.cat(batch_id, dim=0).contiguous()
        else:
            # 全空兜底
            aa_cat = torch.empty(0, dtype=s.dtype, device=device)
            x_cat = torch.empty(0, x.size(2), x.size(3), dtype=x.dtype, device=device)
            res_cat = torch.empty(0, dtype=residue_pos.dtype, device=device)
            mask_cat = torch.empty(0, xloss_mask.size(2), dtype=torch.bool, device=device)
            batch_id = torch.empty(0, dtype=torch.long, device=device)

        # —— 边级字段拼接：重映射 + 节点偏移，丢弃无效边 ——
        e_all = []
        for b in range(B):
            e = edges_list[b]
            if e is None:
                continue
            # 标准化为 (2, E)
            if e.dim() == 2 and e.size(0) != 2 and e.size(1) == 2:
                e = e.t()
            assert e.dim() == 2 and e.size(0) == 2, "edge_index must be (2, E) or (E, 2)."
            e = e.to(device).long()

            # 为该样本构建旧索引 -> 新索引的映射（无效节点映射为 -1）
            m = node_mask[b]  # [N]
            old2new = torch.full((N,), -1, dtype=torch.long, device=device)
            if m.any():
                new_idx = torch.arange(int(m.sum().item()), dtype=torch.long, device=device)
                old2new[m] = new_idx + offsets[b]  # 直接加上全局偏移，得到全局新索引

            # 重映射
            src_new = old2new[e[0]]
            dst_new = old2new[e[1]]
            valid_edge = (src_new >= 0) & (dst_new >= 0)
            if valid_edge.any():
                e_new = torch.stack([src_new[valid_edge], dst_new[valid_edge]], dim=0).contiguous()
                e_all.append(e_new)

        if e_all:
            edge_index = torch.cat(e_all, dim=1).contiguous().clone()  # [2, sum_E]
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

        out = {
            'aa': aa_cat,
            'A': a_cat,
            'pos_heavyatom': x_cat,
            'res_nb': res_cat,
            'xloss_mask': mask_cat,
            'cdr_flag': cdr_cat,
            'chain_nb': chn_cat,
            'edges_index': edge_index,
            'batch_id': batch_id,
            'num_nodes_per_graph': Ns_t,
            'graph_ptr': torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                                    torch.cumsum(Ns_t, dim=0)], dim=0),
        }
        return out

    def _center_by_batch_mean(self, x, batch):
        # x: [N,3], batch: [N]
        x_center = x.clone()
        for b in torch.unique(batch):
            m = (batch == b)
            x_center[m] = x[m] - x[m].mean(dim=0, keepdim=True)
        return x_center

    def random_uniform_so3(self, device=None, dtype=None, eps: float = 1e-8):
        """返回均匀分布的 SO(3) 旋转矩阵，形状 [3,3]."""
        q = torch.randn(4, device=device, dtype=dtype)
        q = q / (q.norm() + eps)  # 单位四元数
        w, x, y, z = q
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])
        ], dim=0)
        return R  # [3,3]

    # cl_model/cl_model.py

    def _apply_shared_random_rotation(self, x, batch_id):
        """
        x: (..., 3)  例如 [N, 3] 或 [N, A, 3]
        batch_id: [N]，同一图共享一个旋转
        """
        x_dtype = x.dtype
        device = x.device

        # 生成与 x 同 dtype/device 的旋转矩阵（每个图一个 3x3）
        B = int(batch_id.max().item()) + 1
        R_list = []
        for _ in range(B):
            R = self.random_uniform_so3(device=device, dtype=x_dtype)  # 你原来的随机旋转函数
            R = R.to(device=device, dtype=x_dtype)  # ← 关键：和 x 同 dtype
            R_list.append(R)
        R_stack = torch.stack(R_list, dim=0)  # [B, 3, 3]

        # 若 x 形状为 [N, 3]（或 [N, A, 3]）
        x_rot = x.clone()

        # 向量化版本（更快，避免循环）：
        # 对 [N, 3]
        if x.dim() == 2:
            R_per_node = R_stack[batch_id]  # [N, 3, 3]
            x_rot = torch.einsum('nij,nj->ni', R_per_node, x).to(x_dtype)
            return x_rot

        # 对 [N, A, 3]
        if x.dim() == 3:
            N, A, _ = x.shape
            R_per_node = R_stack[batch_id]  # [N, 3, 3]
            x_flat = x.reshape(N * A, 3)
            R_rep = R_per_node.repeat_interleave(A, dim=0)  # [N*A, 3, 3]
            x_rot = torch.einsum('nij,nj->ni', R_rep, x_flat).reshape(N, A, 3).to(x_dtype)
            return x_rot
        return None

        # 如果你保留了原来的逐图循环写法，也可以只在赋值处 cast 一下（最小改动）：
        # x_rot[m] = (x[m] @ R_stack[b].T).to(x_rot.dtype)

    def rbf(self, d, D=32, cutoff=20.0):
        # 简单 RBF 距离展开
        centers = torch.linspace(0, cutoff, D, device=d.device)
        widths = (cutoff / D) * 0.8
        return torch.exp(-((d[..., None] - centers[None, :]) ** 2) / (2 * widths ** 2))

    def forward(self, data: dict, is_antibody=True):
        """
        Forward pass to compute embeddings for antibody and antigen inputs.
        """
        data = self.pack_to_big_graph(data)
        s = data['aa']
        cdr_flag = data['cdr_flag']
        x = data['pos_heavyatom']
        chain_nb = data['chain_nb']  # [N_res]
        residue_pos = data['res_nb']
        edge_index = data['edges_index']
        xloss_mask = data['xloss_mask']
        batch_id = data['batch_id']

        # 1) 残基坐标：用 CA 作为节点坐标（EGNN 需要 [N,3]）
        CA = BBHeavyAtom.CA  # 假定你在模块里定义过 CA 的索引
        x_res = x[:, CA, :]  # [N_res, 3]

        # 2) 每图中心化 + (训练时)随机旋转增强 → 去全局位姿
        x_res = self._center_by_batch_mean(x_res, batch_id)  # 每个图减去自身均值
        cdr_mask = cdr_flag.float()  # 先转成 float
        if self.training:
            x_res = self._apply_shared_random_rotation(x_res, batch_id)  # 同一图同一个R
            drop_p = 0.1  # 可调

            keep = (torch.rand_like(cdr_mask) > drop_p).float()  # 现在 OK
            cdr_mask = cdr_mask * keep

        # 残基/位置
        h = self.residue_type_embedding(s) + self.residue_pos_embedding(residue_pos)
        # --- 原子级嵌入（由残基映射得到），并入 h ---
        atom_type = self.residue_atom_type_tbl[s]  # [N]
        atom_pos = self.residue_atom_pos_tbl[s]  # [N]
        atom_emb = self.atom_type_embedding(atom_type) + self.atom_pos_embedding(atom_pos)  # [N,atom_dim]
        atom_feat = self.atom_proj(atom_emb)
        # 掩码平均（用 xloss_mask 选择有效原子）
        if xloss_mask is not None and xloss_mask.dim() == 1:
            xloss_mask = xloss_mask.view(-1, max_num_heavyatoms)
        atom_feat_res = self._masked_mean(atom_feat, xloss_mask, dim=1)  # [N, H]
        h = h + atom_feat_res
        ## 进化与序列特征（高级选项）：可以整合来自大型蛋白质语言模型（如ESM-3）的预训练嵌入 ———— 后续可进行优化。

        # 4) 边特征：用相对几何（RBF距离 + 同链/相对序号），避免绝对坐标泄漏
        if getattr(self, 'use_builtin_edge_feat', True):
            # 根据 x_res 动态构造
            dij = (x_res[edge_index[0]] - x_res[edge_index[1]]).pow(2).sum(-1).sqrt()  # [E]
            edge_attr = self.rbf(dij)  # [E, D_rbf]
            same_chain = (chain_nb[edge_index[0]] == chain_nb[edge_index[1]]).float().unsqueeze(-1)
            relpos = (residue_pos[edge_index[0]] - residue_pos[edge_index[1]]).clamp(-self.max_relpos, self.max_relpos)
            relpos_emb = self.relpos_embed(relpos + self.max_relpos)  # [E, D_rp]
            edge_attr = torch.cat([edge_attr, same_chain, relpos_emb], dim=-1)  # [E, D_e]
        else:
            edge_attr = data['edge_features']  # 确保它不包含绝对坐标
        h = h + 0.2 * (cdr_mask.unsqueeze(-1) * h)  # 给 CDR 位置一个小偏置
        h_embed, coord, edge_embed = self.encoder(h, x_res, edge_index, edge_attr)

        z_graph = self.readout(h_embed, batch_id, cdr_mask)

        # # 投影并归一化
        if is_antibody:
            z_proj = self.proj_head_ab(z_graph)
        else:
            z_proj = self.proj_head_ag(z_graph)

        z = F.normalize(z_proj, dim=1)

        logit_scale_safe = self.logit_scale.clamp(LOGIT_MIN, LOGIT_MAX)
        scale = torch.exp(logit_scale_safe)
        return z, scale


def contrastive_loss(z_ab: torch.Tensor, z_ag: torch.Tensor, logit_scale):
    # z_ab, z_ag: [B,D] (建议已 F.normalize)
    # logit_scale: 传入的是“log(1/T)”这个参数
    scale = torch.exp(logit_scale)  # 关键：取指数
    logits = (z_ab @ z_ag.t()) * scale
    labels = torch.arange(z_ab.size(0), device=z_ab.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))
    return loss


def supcon_samecluster(z, y, temp=0.07):
    # z: [B,D] 已 F.normalize；y: [B] (同簇为正), -1 表示无簇
    B = z.size(0)
    sim = (z @ z.t()) / temp  # [B,B]
    same = (y[:, None] == y[None, :]) & (y[:, None] >= 0)
    same.fill_diagonal_(False)
    row_mask = same.any(dim=1)
    if not row_mask.any():
        return z.new_tensor(0.)
    # SupCon:  log( sum_j∈P_i exp(sim_ij) / sum_k!=i exp(sim_ik) )
    pos = torch.where(same[row_mask], sim[row_mask], sim.new_full(sim[row_mask].shape, -1e9))
    log_pos = torch.logsumexp(pos, dim=1)
    log_all = torch.logsumexp(sim[row_mask] - torch.eye(B, device=z.device)[row_mask] * 1e9, dim=1)
    return -(log_pos - log_all).mean()


def info_nce_masked(z_ab, z_ag, temp, y_ab):
    # z_ab/z_ag: [B,D] 归一化；y_ab: [B]
    B = z_ab.size(0)
    S = (z_ab @ z_ag.t()) / temp  # [B,B]
    labels = torch.arange(B, device=z_ab.device)

    # 屏蔽同簇的“非配对项”
    same = (y_ab[:, None] == y_ab[None, :]) & (y_ab[:, None] >= 0)
    mask = same & (~torch.eye(B, dtype=torch.bool, device=z_ab.device))
    S = S.masked_fill(mask, -1e9)

    return 0.5 * (F.cross_entropy(S, labels) + F.cross_entropy(S.t(), labels))


def infonce_loss(z_ab, z_ag, temp=0.07):
    z_ab = F.normalize(z_ab, dim=1)
    z_ag = F.normalize(z_ag, dim=1)
    logits = (z_ab @ z_ag.t()) / temp
    labels = torch.arange(z_ab.size(0), device=z_ab.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def pair_align_loss(z_ab, z_ag, w=1.0):
    # 余弦直接拉拽（归一化后）
    return w * (1 - F.cosine_similarity(z_ab, z_ag, dim=1)).mean()
