from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from cl_model.egnn_clean import EGNN
from diffab.utils.protein.constants import BBHeavyAtom

LOGIT_MIN = -4  # 1/T ∈ [0.01, 100]
LOGIT_MAX = 4

MAX_ATOM_NUMBER = 15


@dataclass
class EncoderOutput:
    node_feat: torch.Tensor
    node_mask: torch.Tensor
    batch: torch.Tensor
    coords: torch.Tensor
    chain_id: torch.Tensor
    residue_index: torch.Tensor
    extras: Dict[str, torch.Tensor]


class ResidualProjection(nn.Module):
    """Projection head with residual scaling and final normalization."""

    def __init__(self, dim: int, scale: float = 0.5, p: float = 0.1):
        super().__init__()
        self.scale = scale
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(dim, dim, bias=True),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.out_norm(x + self.scale * self.mlp(x))


class FeatureProjector(nn.Module):
    """Project per-residue features (sequence + structure cues) to hidden dim."""

    def __init__(self, hidden_dim: int, trans_dim: int, out_dim: int, vocab_size: int, max_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trans_dim = 1024
        self.vocab_size = vocab_size
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.aa_embed = nn.Embedding(vocab_size, hidden_dim)

        self.esm_proj = nn.Sequential(
            nn.Linear(hidden_dim, trans_dim),  # ESM维度
            nn.LayerNorm(trans_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(trans_dim, out_dim),
        )

    def _encode_esm(self, esm: Optional[torch.Tensor]) -> torch.Tensor:
        esm_proj = self.esm_proj(esm.float())
        return F.normalize(esm_proj, dim=1)

    def forward(
            self,
            aa: torch.Tensor,
            residue_index: torch.Tensor,
            esm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # aa_feat = self.aa_embed(aa)
        # pos_feat = self.pos_embed(residue_index)
        esm_feat = self._encode_esm(esm)
        return esm_feat


class PMAReadout(nn.Module):
    def __init__(self, dim, k=1, heads=4, use_center=True):
        super().__init__()
        self.k = k
        self.h = heads
        self.use_center = use_center
        self.seeds = nn.Parameter(torch.randn(k, dim) * 0.02)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)  # 共享线性生成 K,V

        # 添加多样性约束
        self.diversity_weight = 0.1

    def forward(self, h, batch, node_mask, bias_prob=None):
        if h.numel() == 0:
            return h.new_zeros((0, h.size(-1)))
        B, D = int(batch.max().item()) + 1, h.size(-1)

        if self.use_center:
            m = node_mask.to(h.dtype).unsqueeze(-1)
            sum_per = torch.zeros((B, D), device=h.device, dtype=h.dtype)
            cnt_per = torch.zeros((B, 1), device=h.device, dtype=h.dtype)
            sum_per.scatter_add_(0, batch.unsqueeze(-1).expand(-1, D), h * m)
            cnt_per.scatter_add_(0, batch.unsqueeze(-1), m)
            mean_raw = sum_per / (cnt_per + 1e-12)
            h = h - mean_raw.index_select(0, batch)

        # per-graph seeds
        S = self.seeds.unsqueeze(0).expand(B, -1, -1)  # [B,k,D]
        Q = self.q(S)  # [B,k,D]
        KV = self.kv(h)  # [N,2D]
        K, V = KV.split(D, dim=-1)

        d_h = D // self.h
        out = torch.zeros(B, self.k, D, device=h.device, dtype=h.dtype)
        neg_inf = torch.finfo(h.dtype).min

        for b in range(B):
            mask_b = (batch == b) & node_mask
            if not mask_b.any():
                continue
            Kb, Vb = K[mask_b], V[mask_b]  # [Nb, D]
            Qb = Q[b]  # [k,  D]

            # 多头注意力（单图）
            q = Qb.view(self.k, self.h, d_h)  # [k,h,d]
            k = Kb.view(-1, self.h, d_h)  # [Nb,h,d]
            v = Vb.view(-1, self.h, d_h)
            attn = torch.einsum('khd,nhd->khn', q, k) / (d_h ** 0.5)  # [k,h,Nb]
            # 添加注意力熵正则化，防止过度集中
            attn_entropy = -torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1), dim=-1)
            attn = attn + self.diversity_weight * attn_entropy.unsqueeze(-1)

            attn = attn.softmax(dim=-1)
            oh = torch.einsum('khn,nhd->khd', attn, v)  # [k,h,d]
            out[b] = oh.reshape(self.k, D)

        # seed_weights = F.softmax(torch.randn(B, self.k, device=h.device) * 0.1, dim=-1)
        weighted_out = out.mean(dim=1)

        return weighted_out


# === NEW: 双线性互补核 W =====================================
class BilinearKernel(nn.Module):
    """
    s = z_ab^T W z_ag + b
    mode:
      - 'full'      : W \in R^{dxd}
      - 'lowrank'   : W = U V^T,  U,V \in R^{d x r}
      - 'block_diag': 把通道一分为二，s = z_sim^T W_sim z_sim  - z_cmp^T W_cmp z_cmp
                      其中 W_sim, W_cmp 为非负对角（softplus），便于解释“相似/互补”通道
    """

    def __init__(self, d: int,
                 mode: str = "block_diag",
                 rank: int = 64,
                 block_ratio: float = 0.5,
                 bias: bool = False):
        super().__init__()
        self.d = d
        self.mode = mode
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

        if mode == "full":
            self.W = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_uniform_(self.W)
        elif mode == "lowrank":
            self.U = nn.Parameter(torch.empty(d, rank))
            self.V = nn.Parameter(torch.empty(d, rank))
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        elif mode == "block_diag":
            d_sim = int(d * block_ratio)
            d_cmp = d - d_sim
            self.d_sim, self.d_cmp = d_sim, d_cmp
            self.w_sim = nn.Parameter(torch.zeros(d_sim))  # 对角参数
            self.w_cmp = nn.Parameter(torch.zeros(d_cmp))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def score(self, z_ab: torch.Tensor, z_ag: torch.Tensor) -> torch.Tensor:
        """
        z_ab: (B, d), z_ag: (B2, d) => returns (B, B2)
        """
        if self.mode == "full":
            S = (z_ab @ self.W) @ z_ag.t()
        elif self.mode == "lowrank":
            ua = z_ab @ self.U  # (B, r)
            va = z_ag @ self.V  # (B2, r)
            S = ua @ va.t()
        elif self.mode == "block_diag":
            d_sim = self.d_sim
            z_ab_sim, z_ab_cmp = z_ab[:, :d_sim], z_ab[:, d_sim:]
            z_ag_sim, z_ag_cmp = z_ag[:, :d_sim], z_ag[:, d_sim:]
            w_sim = F.softplus(self.w_sim)  # >=0
            w_cmp = F.softplus(self.w_cmp)  # >=0
            S_sim = (z_ab_sim * w_sim) @ z_ag_sim.t()  # 同向匹配
            S_cmp = (z_ab_cmp * w_cmp) @ z_ag_cmp.t()  # 互补通道 -> 减
            S = S_sim - S_cmp
        else:
            raise RuntimeError

        if self.bias is not None:
            S = S + self.bias
        return S

    @torch.no_grad()
    def transform_right(self, z_ag: torch.Tensor) -> torch.Tensor:
        """
        离线检索用：把右塔预变换成 \tilde{z}_ag = W z_ag
        - full/lowrank：严格等价
        - block_diag：给出按通道加权并在互补通道取负的等价展开
        """
        if self.mode == "full":
            return z_ag @ self.W.t()
        elif self.mode == "lowrank":
            return (z_ag @ self.V) @ self.U.t()
        elif self.mode == "block_diag":
            d_sim = self.d_sim
            w_sim = F.softplus(self.w_sim)
            w_cmp = F.softplus(self.w_cmp)
            z_sim, z_cmp = z_ag[:, :d_sim], z_ag[:, d_sim:]
            return torch.cat([z_sim * w_sim, -z_cmp * w_cmp], dim=-1)
        else:
            raise RuntimeError


class ContrastiveLearningModel(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            vocab_size: int = 22,
            max_len: int = 2048,
            radius: float = 8.0,  # 0为无边对照
            max_neighbors: int = 8,
            temperature: float = 0.07,
            device: str = "cpu",
            feat_dim: int = 32,
            max_relpos: int = 64,
            cutoff: float = 20.0
    ) -> None:
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.esm_dim = 1280
        self.antiberty_dim = 512
        self.vocab_size = vocab_size
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.temperature = temperature
        self.max_relpos = max_relpos
        self.alpha_ret = 0.1
        self.ab_projector = FeatureProjector(self.antiberty_dim, 512, 512, vocab_size, max_len)
        # 为抗原特征投影器增加更多非线性
        self.ag_projector = FeatureProjector(self.esm_dim, 1024, 512, vocab_size, max_len * 2)

        edge_dim = feat_dim + 1 + 32
        self.antibody_gnn = EGNN(
            in_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=edge_dim,
            device=device,
            n_layers=2,
            attention=True,
            tanh=True
        )
        self.antigen_gnn = EGNN(
            in_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=edge_dim,
            device=device,
            n_layers=2,
        )
        self.rbf_dim = 32
        self.cutoff = cutoff
        centers = torch.linspace(0, cutoff, self.rbf_dim)
        self.register_buffer("rbf_centers", centers, persistent=False)
        width = (cutoff / self.rbf_dim) * 0.8
        self.register_buffer("rbf_width", torch.tensor(width), persistent=False)

        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)
        self.paratope_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.epitope_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 把原来的 0 调大，让 paratope/epitope 概率能起作用
        # self.ab_readout = AttnReadout(self.hidden_dim, bias_scale=2.0, use_center=True, use_cls=False,
        #                               add_back_mean_alpha=0.0, init_logit_scale=1.0)
        # self.ag_readout = AttnReadout(self.hidden_dim, bias_scale=2.0, use_center=True, use_cls=False,
        #                               add_back_mean_alpha=0.0, init_logit_scale=1.0)

        # 替换原来的 ab_readout / ag_readout
        self.ab_readout = PMAReadout(self.hidden_dim, )
        self.ag_readout = PMAReadout(self.hidden_dim, )

        self.ab_projection = ResidualProjection(hidden_dim)
        self.ag_projection = ResidualProjection(hidden_dim)

        self.compat_kernel = BilinearKernel(
            d=hidden_dim,
            mode="block_diag",  # 可选： "full" / "lowrank" / "block_diag"
            rank=64,
            block_ratio=0.5,  # 相似/互补通道比例
            bias=False
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        self.post_readout_bn_ab = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-5)
        self.post_readout_bn_ag = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-5)
        self.register_buffer("ema_center_ab", torch.zeros(self.hidden_dim))  # 抗体端
        self.register_buffer("ema_center_ag", torch.zeros(self.hidden_dim))  # 抗原端
        self.center_m = 0.99  # 动量，可 0.98~0.995
        # ---- Utilities -------------------------------------------------------------------

    def _center_by_batch_mean(self, coords: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if coords.numel() == 0:
            return coords
        ones = torch.ones((coords.size(0), 1), device=coords.device, dtype=coords.dtype)
        sum_per = torch.zeros((batch.max().item() + 1, 3), device=coords.device, dtype=coords.dtype)
        cnt_per = torch.zeros((batch.max().item() + 1, 1), device=coords.device, dtype=coords.dtype)
        sum_per.scatter_add_(0, batch.unsqueeze(-1).expand_as(coords), coords)
        cnt_per.scatter_add_(0, batch.unsqueeze(-1).expand_as(ones), ones)
        mean = sum_per / (cnt_per + 1e-12)
        return coords - mean.index_select(0, batch)

    def _build_radius_graph(self, coords: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if coords.numel() == 0:
            return coords.new_empty((2, 0), dtype=torch.long)
        edges_src, edges_dst = [], []
        device = coords.device
        for b in torch.unique(batch):
            node_idx = torch.nonzero(batch == b, as_tuple=False).view(-1)
            if node_idx.numel() <= 1:
                continue
            sub_coords = coords[node_idx]
            dist = torch.cdist(sub_coords, sub_coords)
            mask = (dist <= self.radius) & (~torch.eye(dist.size(0), device=device, dtype=torch.bool))
            for i in range(mask.size(0)):
                nbr = torch.nonzero(mask[i], as_tuple=False).view(-1)
                if nbr.numel() == 0:
                    continue
                if self.max_neighbors is not None and nbr.numel() > self.max_neighbors:
                    dvals = dist[i, nbr]
                    keep = torch.topk(dvals, k=self.max_neighbors, largest=False).indices
                    nbr = nbr[keep]
                src_nodes = node_idx[i].repeat(nbr.numel())
                dst_nodes = node_idx[nbr]
                edges_src.append(src_nodes)
                edges_dst.append(dst_nodes)
        if not edges_src:
            return coords.new_empty((2, 0), dtype=torch.long)
        return torch.stack([torch.cat(edges_src), torch.cat(edges_dst)], dim=0)

    def rbf(self, dist: torch.Tensor) -> torch.Tensor:
        x = dist[..., None] - self.rbf_centers[None, :]
        return torch.exp(-(x * x) / (2 * (self.rbf_width ** 2)))

    # ---- Packing ---------------------------------------------------------------------
    def pack_to_big_graph(self, data: Dict[str, torch.Tensor]) -> EncoderOutput:
        aa = data['aa']
        coords = data['pos_heavyatom']
        heavy_mask = data['mask_heavyatom']
        residue_index = data['res_nb']
        chain_nb = data['chain_nb']
        batch_mask = data.get('mask')
        node_mask = heavy_mask.any(dim=-1)
        if batch_mask is not None:
            node_mask = node_mask & batch_mask.bool()

        B = aa.size(0)
        device = aa.device
        aa_out, coord_out, res_out, chain_out, batch_out = [], [], [], [], []
        extras_lists: Dict[str, list] = {}
        for optk in ('paratope_mask', 'paratope_ctx_mask', 'epitope_mask', 'epitope_ctx_mask', 'lang_feat',
                     'generate_flag'):
            if optk in data:
                extras_lists[optk] = []

        for b in range(B):
            m = node_mask[b]
            if not m.any():
                continue
            aa_out.append(aa[b][m])
            coord_out.append(coords[b][m])
            res_out.append(residue_index[b][m])
            chain_out.append(chain_nb[b][m])
            batch_out.append(torch.full((int(m.sum().item()),), b, device=device, dtype=torch.long))
            for k in extras_lists.keys():
                extras_lists[k].append(data[k][b][m])

        if aa_out:
            aa_cat = torch.cat(aa_out, 0)
            coord_cat = torch.cat(coord_out, 0)
            res_cat = torch.cat(res_out, 0)
            chain_cat = torch.cat(chain_out, 0)
            batch_cat = torch.cat(batch_out, 0)
        else:
            aa_cat = torch.empty(0, dtype=aa.dtype, device=device)
            coord_cat = torch.empty(0, coords.size(2), coords.size(3), dtype=coords.dtype, device=device)
            res_cat = torch.empty(0, dtype=residue_index.dtype, device=device)
            chain_cat = torch.empty(0, dtype=chain_nb.dtype, device=device)
            batch_cat = torch.empty(0, dtype=torch.long, device=device)

        extras = {
            k: (torch.cat(v, 0) if v else coord_cat.new_zeros((0,) + (() if data[k].dim() <= 2 else data[k].shape[2:])))
            for k, v in extras_lists.items()}

        return EncoderOutput(aa_cat, torch.ones(aa_cat.size(0), dtype=torch.bool, device=device),
                             batch_cat, coord_cat, chain_cat, res_cat, extras)

        # ---- Forward ---------------------------------------------------------------------

    def _per_graph_whiten(self, feat, batch):
        B = int(batch.max().item()) + 1 if batch.numel() else 0
        if B == 0 or feat.numel() == 0:
            return feat
        D = feat.size(-1)
        one = torch.ones((feat.size(0), 1), device=feat.device, dtype=feat.dtype)

        mu_sum = torch.zeros((B, D), device=feat.device, dtype=feat.dtype)
        cnt = torch.zeros((B, 1), device=feat.device, dtype=feat.dtype)
        mu_sum.scatter_add_(0, batch.unsqueeze(-1).expand(-1, D), feat)
        cnt.scatter_add_(0, batch.unsqueeze(-1), one)
        mu = mu_sum / cnt.clamp_min(1.0)
        xc = feat - mu.index_select(0, batch)

        sq_sum = torch.zeros((B, D), device=feat.device, dtype=feat.dtype)
        sq_sum.scatter_add_(0, batch.unsqueeze(-1).expand(-1, D), xc * xc)
        var = (sq_sum / cnt.clamp_min(1.0)).index_select(0, batch)
        return xc / (var.sqrt() + 1e-6)

    def pairnorm_per_graph(self, h, batch, scale=1.0, eps=1e-6):
        mu = scatter_mean(h, batch, dim=0)  # [B,D] 每图均值
        hc = h - mu[batch]  # 图内零均值
        sq = (hc * hc).sum(dim=1, keepdim=True)  # [N,1]
        msq = scatter_mean(sq, batch, dim=0)  # [B,1] 图内均值平方范数
        s = (scale / (msq.sqrt() + eps))[batch]  # [N,1]
        return hc * s

    def _prepare_encoder_inputs(
            self,
            packed: EncoderOutput,
            is_antibody: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        if packed.coords.numel() == 0:
            return (
                torch.empty(0, self.hidden_dim, device=packed.node_feat.device),
                torch.empty(0, 3, device=packed.coords.device),
                torch.empty((2, 0), dtype=torch.long, device=packed.coords.device),
                torch.empty(0, self.rbf_dim + 1 + self.relpos_embed.embedding_dim, device=packed.coords.device),
                packed.extras,
            )

        coords = packed.coords[:, BBHeavyAtom.CA]
        coords = self._center_by_batch_mean(coords, packed.batch)
        edge_index = self._build_radius_graph(coords, packed.batch)
        dist = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1)
        rbf_feat = self.rbf(dist)
        same_chain = (packed.chain_id[edge_index[0]] == packed.chain_id[edge_index[1]]).float().unsqueeze(-1)
        relpos = (packed.residue_index[edge_index[0]] - packed.residue_index[edge_index[1]]).clamp(
            -self.max_relpos, self.max_relpos
        )
        relpos_emb = self.relpos_embed(relpos + self.max_relpos)
        edge_attr = torch.cat([rbf_feat, same_chain, relpos_emb], dim=-1)

        esm_key_order = ('lang_feat',)
        esm_tensor = None
        for key in esm_key_order:
            if key in packed.extras:
                esm_tensor = packed.extras[key]
                break
        projector = self.ab_projector if is_antibody else self.ag_projector
        node_input = projector(
            packed.node_feat,
            packed.residue_index,
            esm=esm_tensor,
        )
        node_input = self._per_graph_whiten(node_input, packed.batch)
        return node_input, coords, edge_index, edge_attr, packed.extras

    def compat_scores(self, z_ab: torch.Tensor, z_ag: torch.Tensor) -> torch.Tensor:
        """无温度缩放的 s = z_ab^T W z_ag（调试/可视化用）"""
        return self.compat_kernel.score(z_ab, z_ag)

    @torch.no_grad()
    def transform_right_for_retrieval(self, z_ag: torch.Tensor) -> torch.Tensor:
        """
        离线检索：把右塔做 W 变换并保存，在线只需做 z_ab @ (W z_ag)^T
        """
        return self.compat_kernel.transform_right(z_ag)

    def forward(self, data: Dict[str, torch.Tensor], is_antibody: bool = True, use_bias: bool = False):
        packed = self.pack_to_big_graph(data)
        h0, coords0, edge_index, edge_attr, extras = self._prepare_encoder_inputs(packed, is_antibody)
        batch = packed.batch

        if is_antibody:
            gnn = self.antibody_gnn
        else:
            gnn = self.antigen_gnn

        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            h = h0  # 直接短路
        else:
            h, coords, _ = gnn(h0, coords0, edge_index, edge_attr=edge_attr)
        # h = (1.0 - self.alpha_ret) * h + self.alpha_ret * h0  # 先残差混合（alpha_ret=0.05~0.1）
        # h = self.pairnorm_per_graph(h, batch)  # 再按图 PairNorm
        h = F.normalize(h, dim=1)  # 最后 L2

        aux: Dict[str, torch.Tensor] = {}
        if h.numel() == 0:
            graph_embed = h.new_zeros((0, self.hidden_dim))
            paratope_logits = h.new_zeros((0,))
            paratope_prob = paratope_logits
            epitope_logits = h.new_zeros((0,))
            epitope_prob = epitope_logits
        else:
            if is_antibody:
                paratope_logits = self.paratope_head(h).squeeze(-1)  # 不用看
                paratope_prob = torch.sigmoid(paratope_logits)
                node_mask_eff = extras.get('paratope_mask') | extras.get('paratope_ctx_mask')
                bias = paratope_prob if use_bias else None
                graph_embed = self.ab_readout(
                    h,
                    batch,
                    node_mask_eff,
                    bias_prob=None,
                )

                epitope_logits = h.new_zeros((0,))
                aux.update({
                    'paratope_logits': paratope_logits,
                    'paratope_prob': paratope_prob,
                    'paratope_target': extras.get('paratope_mask'),
                })
            else:
                epitope_logits = self.epitope_head(h).squeeze(-1)
                epitope_prob = torch.sigmoid(epitope_logits)
                bias = epitope_prob if use_bias else None
                node_mask_eff = extras.get('epitope_mask') | extras.get('epitope_ctx_mask')
                graph_embed = self.ag_readout(
                    h,
                    batch,
                    node_mask_eff,
                    bias_prob=None,
                )
                paratope_logits = h.new_zeros((0,))
                paratope_prob = paratope_logits
                aux.update({
                    'epitope_logits': epitope_logits,
                    'epitope_prob': epitope_prob,
                    'epitope_target': extras.get('epitope_mask'),
                })
        graph_embed = F.normalize(graph_embed, dim=-1)
        if is_antibody:
            z = self.ab_projection(graph_embed)
            aux.setdefault('paratope_logits', paratope_logits)
            aux.setdefault('paratope_prob', paratope_prob)
        else:
            z = self.ag_projection(graph_embed)
            aux.setdefault('epitope_logits', epitope_logits)
            aux.setdefault('epitope_prob', epitope_prob)
        # z = z - z.mean(dim=0, keepdim=True)
        z = F.normalize(z, dim=-1)

        aux.update({
            'batch': batch,
            'node_embeddings': h,
        })
        return z, self.logit_scale, aux


def info_nce_masked(z_ab: torch.Tensor, z_ag: torch.Tensor, temp: float, y_ab: torch.Tensor) -> torch.Tensor:
    # z_ab/z_ag: [B,D] 归一化；y_ab: [B]
    B = z_ab.size(0)
    sim = (z_ab @ z_ag.t()) / temp  # [B,B]
    labels = torch.arange(B, device=z_ab.device)

    same = (y_ab[:, None] == y_ab[None, :]) & (y_ab[:, None] >= 0)
    mask = same & (~torch.eye(B, dtype=torch.bool, device=z_ab.device))
    sim = sim.masked_fill(mask, -1e9)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def infonce_loss(z_ab, z_ag, temp=0.07):
    z_ab = F.normalize(z_ab, dim=1)
    z_ag = F.normalize(z_ag, dim=1)
    logits = (z_ab @ z_ag.t()) / temp
    labels = torch.arange(z_ab.size(0), device=z_ab.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def pair_align_loss(z_ab, z_ag, w=1.0):
    # 余弦直接拉拽（归一化后）
    return w * (1 - F.cosine_similarity(z_ab, z_ag, dim=1)).mean()


def improved_contrastive_loss(
        z_ab: torch.Tensor,
        z_ag: torch.Tensor,
        logit_scale: torch.Tensor,
        margin: float = 0.1
) -> torch.Tensor:
    logit_scale = logit_scale.clamp(LOGIT_MIN, LOGIT_MAX)
    scale = torch.exp(logit_scale)

    # 计算相似度矩阵
    sim = (z_ab @ z_ag.t()) * scale

    # 添加边界margin，确保正样本明显高于负样本
    pos_mask = torch.eye(z_ab.size(0), device=z_ab.device).bool()
    neg_mask = ~pos_mask

    # 正样本损失
    pos_sim = sim[pos_mask]

    # 负样本挖掘：关注最难负样本
    neg_sim = sim[neg_mask].view(z_ab.size(0), -1)
    hardest_neg, _ = neg_sim.max(dim=1)

    # 带边界的对比损失
    loss = F.relu(hardest_neg - pos_sim + margin).mean()

    return loss


# === NEW: 兼容打分（z 已 L2 归一）=====================================
def compat_scores_with_scale(model: ContrastiveLearningModel,
                             z_ab: torch.Tensor, z_ag: torch.Tensor) -> torch.Tensor:
    """
    返回 S/τ 形式的打分矩阵（直接可送入 CE）
    """
    scale = torch.exp(model.logit_scale.clamp(LOGIT_MIN, LOGIT_MAX))
    S = model.compat_kernel.score(z_ab, z_ag) * scale  # (B,B)
    return S


# === NEW: 双向 InfoNCE（用兼容打分） + 反相似正则 =======================
def compat_infonce_loss(model: ContrastiveLearningModel,
                        z_ab: torch.Tensor, z_ag: torch.Tensor,
                        anti_margin: float = 0.1, anti_weight: float = 0.1) -> Tuple[
    torch.Tensor, Dict[str, torch.Tensor]]:
    """
    - 主损失：CLIP/InfoNCE，但把相似度换成 s = z_ab^T W z_ag
    - 反相似正则：鼓励正样本余弦 <= -margin，避免学成“同向相似”
    """
    S_over_tau = compat_scores_with_scale(model, z_ab, z_ag)  # (B,B)
    B = z_ab.size(0)
    labels = torch.arange(B, device=z_ab.device)

    loss_row = F.cross_entropy(S_over_tau, labels)  # ab -> ag
    loss_col = F.cross_entropy(S_over_tau.t(), labels)  # ag -> ab
    loss_main = 0.5 * (loss_row + loss_col)

    # anti-sim（z 已 L2N）：max(0, cos + margin)
    if anti_weight > 0:
        cos = (z_ab * z_ag).sum(dim=-1)
        loss_anti = F.relu(cos + anti_margin).mean() * anti_weight
    else:
        loss_anti = S_over_tau.new_tensor(0.0)

    return loss_main + loss_anti, {"main": loss_main.detach(), "anti": loss_anti.detach()}


def compat_contrastive_loss_mixed(
        model, z_ab, z_ag,
        margin=0.1, alpha=0.3, q=0.8,  # α: hinge权重，q: 分位数
        uniform_w=5e-3, var_w=5e-3
):
    z_ab = F.normalize(z_ab, dim=-1)
    z_ag = F.normalize(z_ag, dim=-1)

    S = compat_scores_with_scale(model, z_ab, z_ag)  # (B,B)

    # InfoNCE（对称）
    B = S.size(0);
    dev = S.device
    labels = torch.arange(B, device=dev)
    loss_i = 0.5 * (F.cross_entropy(S, labels) + F.cross_entropy(S.t(), labels))

    def quantile_hinge(S, margin=0.1, q=0.8):
        B = S.size(0);
        dev = S.device
        pos = S.diag()
        off = ~torch.eye(B, dtype=torch.bool, device=dev)
        S_off = S.masked_fill(~off, float('-inf'))

        # 行方向（ab->ag）
        row_vals, _ = torch.sort(S_off, dim=1, descending=True)
        k = torch.floor(torch.tensor(q * (B - 1), device=dev)).long().clamp_min(0).clamp_max(B - 2).item()

        row_q = row_vals[torch.arange(B, device=dev), k]  # 每行的 q 分位负样本
        loss_row = F.relu(row_q - pos + margin).mean()

        # 列方向（ag->ab）
        col_vals, _ = torch.sort(S_off, dim=0, descending=True)
        col_q = col_vals[k, torch.arange(B, device=dev)]
        loss_col = F.relu(col_q - pos + margin).mean()

        return 0.5 * (loss_row + loss_col), row_q.mean().detach(), col_q.mean().detach()

    # 分位数 Hinge（对称）
    loss_h, row_qm, col_qm = quantile_hinge(S, margin=margin, q=q)

    # Uniformity（可选）
    def uniformity(z):
        d = torch.cdist(z, z, p=2)
        m = ~torch.eye(d.size(0), dtype=torch.bool, device=d.device)
        return torch.log(torch.exp(-2 * (d[m] ** 2)).mean() + 1e-12)

    loss_u = 0.5 * (uniformity(z_ab) + uniformity(z_ag)) if uniform_w > 0 else S.new_tensor(0.)

    # Variance（可选）
    def variance_loss(z, eps=1e-4):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.relu(1.0 - std).mean()

    loss_v = 0.5 * (variance_loss(z_ab) + variance_loss(z_ag)) if var_w > 0 else S.new_tensor(0.)

    loss = (1 - alpha) * loss_i + alpha * loss_h + uniform_w * loss_u + var_w * loss_v

    with torch.no_grad():
        diag = S.diag().mean()
        offmax = (S + torch.diag(torch.full((B,), float('-inf'), device=dev))).amax(dim=1).mean()
    stats = {
        "loss": loss.detach(),
        "infonce": loss_i.detach(),
        "hinge_q": loss_h.detach(),
        "row_q_mean": row_qm, "col_q_mean": col_qm,
        "uniform": loss_u.detach(), "variance": loss_v.detach(),
        "diag_mean": diag, "offmax_mean": offmax,
    }
    return loss, stats


# === NEW: 可选核正则（防爆/可解释）======================================
def kernel_regularizer(model: ContrastiveLearningModel,
                       l2_w: float = 1e-4, l1_diag: float = 0.0) -> torch.Tensor:
    k = model.compat_kernel
    reg = z = torch.tensor(0.0, device=model.logit_scale.device)
    if isinstance(k, BilinearKernel):
        if k.mode == "full":
            reg = l2_w * k.W.norm(p=2)
        elif k.mode == "lowrank":
            reg = l2_w * (k.U.norm(p=2) + k.V.norm(p=2))
        elif k.mode == "block_diag":
            if l2_w > 0:
                reg = l2_w * (F.softplus(k.w_sim).pow(2).mean() + F.softplus(k.w_cmp).pow(2).mean())
            if l1_diag > 0:
                reg = reg + l1_diag * (F.softplus(k.w_sim).mean() + F.softplus(k.w_cmp).mean())
    return reg
