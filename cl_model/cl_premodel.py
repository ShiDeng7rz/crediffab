from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_model.egnn_clean import EGNN
from cl_model.position_embedding import SinusoidalPositionEmbedding
from diffab.utils.protein.constants import BBHeavyAtom

LOGIT_MIN = math.log(1 / 100.0)  # 1/T ∈ [0.01, 100]
LOGIT_MAX = math.log(100.0)

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

    def __init__(self, dim: int, scale: float = 0.5):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.scale * self.proj(x), dim=-1)


class FeatureProjector(nn.Module):
    """Project per-residue features (sequence + structure cues) to hidden dim."""

    def __init__(self, hidden_dim: int, vocab_size: int, max_len: int, dssp_states: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pos_embed = SinusoidalPositionEmbedding(hidden_dim, max_len)
        self.aa_proj = nn.Linear(vocab_size, hidden_dim)
        self.dssp_embed = nn.Embedding(dssp_states, hidden_dim)
        self.dssp_linear = nn.Sequential(
            nn.Linear(dssp_states, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sasa_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.esm_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 5 feature slots: aa, position, dssp, sasa, esm
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _encode_dssp(self, dssp: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if dssp is None:
            return torch.zeros_like(ref)
        if dssp.dtype in (torch.int32, torch.int64, torch.int16, torch.long):
            dssp_clamped = dssp.clamp(min=0, max=self.dssp_embed.num_embeddings - 1)
            return self.dssp_embed(dssp_clamped)
        # assume already one-hot or probabilities along last dim
        if dssp.dim() == ref.dim() and dssp.size(-1) != ref.size(-1):
            return self.dssp_linear(dssp.float())
        return dssp.float()

    def _encode_sasa(self, sasa: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if sasa is None:
            return torch.zeros_like(ref)
        if sasa.dim() == ref.dim() - 1:
            sasa = sasa.unsqueeze(-1)
        return self.sasa_proj(sasa.float())

    def _encode_esm(self, esm: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if esm is None:
            return torch.zeros_like(ref)
        return self.esm_proj(esm.float())

    def forward(
            self,
            aa: torch.Tensor,
            residue_index: torch.Tensor,
            esm: Optional[torch.Tensor] = None,
            dssp: Optional[torch.Tensor] = None,
            sasa: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        onehot = F.one_hot(aa.clamp(min=0), num_classes=self.vocab_size).float()
        aa_feat = self.aa_proj(onehot)
        pos_feat = self.pos_embed(residue_index)
        dssp_feat = self._encode_dssp(dssp, aa_feat)
        sasa_feat = self._encode_sasa(sasa, aa_feat)
        esm_feat = self._encode_esm(esm, aa_feat)
        feat = torch.cat([aa_feat, pos_feat, dssp_feat, sasa_feat, esm_feat], dim=-1)
        return self.combine(feat)


class ParatopeAwareReadout(nn.Module):
    def __init__(self, dim: int, beta: float = 1.0, gamma: float = 0.0):
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=False)
        self.beta = beta
        self.gamma = gamma

    def forward(
            self,
            h: torch.Tensor,
            batch: torch.Tensor,
            node_mask: torch.Tensor,
            paratope_prob: torch.Tensor,
            sasa_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        outs = []
        for b in range(B):
            m = (batch == b) & node_mask
            if not torch.any(m):
                outs.append(h.new_zeros((1, h.size(-1))))
                continue
            hb = h[m]
            logits = self.query(hb).squeeze(-1)
            logits = logits + self.beta * paratope_prob[m]
            if sasa_prior is not None and self.gamma != 0.0:
                logits = logits + self.gamma * sasa_prior[m]
            weights = torch.softmax(logits, dim=0)
            outs.append((weights.unsqueeze(-1) * hb).sum(dim=0, keepdim=True))
        if not outs:
            return h.new_zeros((0, h.size(-1)))
        return torch.cat(outs, dim=0)


class SurfaceAwareReadout(nn.Module):
    def __init__(self, dim: int, gamma: float = 0.5, delta: float = 0.5):
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=False)
        self.gamma = gamma
        self.delta = delta

    def forward(
            self,
            h: torch.Tensor,
            batch: torch.Tensor,
            node_mask: torch.Tensor,
            surface_prior: Optional[torch.Tensor],
            epitope_prob: torch.Tensor,
    ) -> torch.Tensor:
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        outs = []
        for b in range(B):
            m = (batch == b) & node_mask
            if not torch.any(m):
                outs.append(h.new_zeros((1, h.size(-1))))
                continue
            hb = h[m]
            logits = self.query(hb).squeeze(-1)
            if surface_prior is not None and self.gamma != 0.0:
                logits = logits + self.gamma * surface_prior[m]
            logits = logits + self.delta * epitope_prob[m]
            weights = torch.softmax(logits, dim=0)
            outs.append((weights.unsqueeze(-1) * hb).sum(dim=0, keepdim=True))
        if not outs:
            return h.new_zeros((0, h.size(-1)))
        return torch.cat(outs, dim=0)


class ContrastiveLearningModel(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            vocab_size: int = 22,
            max_len: int = 2048,
            radius: float = 8.0,
            max_neighbors: int = 24,
            temperature: float = 0.25,
            device: str = "cpu",
            feat_dim: int = 32,
            max_relpos: int = 64,
    ) -> None:
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.temperature = temperature
        self.max_relpos = max_relpos

        self.ab_projector = FeatureProjector(hidden_dim, vocab_size, max_len)
        self.ag_projector = FeatureProjector(hidden_dim, vocab_size, max_len)

        edge_dim = feat_dim + 1 + 32
        self.antibody_gnn = EGNN(
            in_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=edge_dim,
            device=device,
            n_layers=2,
        )
        self.antigen_gnn = EGNN(
            in_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=edge_dim,
            device=device,
            n_layers=1,
        )
        self.rbf_dim = 32
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

        self.ab_readout = ParatopeAwareReadout(hidden_dim, beta=1.0, gamma=0.0)
        self.ag_readout = SurfaceAwareReadout(hidden_dim, gamma=0.5, delta=0.5)

        self.ab_projection = ResidualProjection(hidden_dim)
        self.ag_projection = ResidualProjection(hidden_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

        # ---- Utilities -------------------------------------------------------------------

    def _center_by_batch_mean(self, coords: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        centered = coords.clone()
        if batch.numel() == 0:
            return centered
        for b in torch.unique(batch):
            m = batch == b
            centered[m] = coords[m] - coords[m].mean(dim=0, keepdim=True)
        return centered

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

    def rbf(self, dist: torch.Tensor, cutoff: float = 20.0) -> torch.Tensor:
        centers = torch.linspace(0, cutoff, self.rbf_dim, device=dist.device)
        widths = (cutoff / self.rbf_dim) * 0.8
        return torch.exp(-((dist[..., None] - centers[None, :]) ** 2) / (2 * widths ** 2))

    def _normalize_prior(self, prior: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        prior = prior.clone().float()
        if prior.numel() == 0:
            return prior
        for b in torch.unique(batch):
            m = batch == b
            if not torch.any(m):
                continue
            values = prior[m]
            max_val = values.max()
            if torch.isfinite(max_val) and max_val > 0:
                prior[m] = values / max_val
            else:
                prior[m] = 0.0
        return prior

    # ---- Packing ---------------------------------------------------------------------
    def pack_to_big_graph(self, data: Dict[str, torch.Tensor]) -> EncoderOutput:
        aa = data['aa']
        coords = data['pos_heavyatom']
        heavy_mask = data['mask_heavyatom']
        residue_index = data['res_nb']
        chain_nb = data['chain_nb']
        batch_mask = data.get('mask')

        optional_keys: Sequence[str] = (
            'paratope_mask',
            'epitope_mask',
            'sasa',
            'surface_prior',
            'dssp',
            'antiberty',
            'esm_if1',
            'esm2',
            'esm',
            'generate_flag',
        )
        optional_collect: Dict[str, list] = {k: [] for k in optional_keys if k in data}
        # 有效节点掩码：每个残基只要有任一原子是有效的，就认为这个节点有效
        node_mask = heavy_mask.any(dim=-1)  # [B, N] -> bool
        if batch_mask is not None:
            node_mask = node_mask & batch_mask.bool()

        B, L = aa.shape
        device = aa.device
        lengths = node_mask.sum(dim=1)
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

        aa_list, coord_list, res_idx_list, chain_list, mask_list, batch_list = [], [], [], [], [], []
        for b in range(B):
            mask = node_mask[b]
            if not mask.any():
                continue

            aa_list.append(aa[b][mask])
            coord_list.append(coords[b][mask])
            res_idx_list.append(residue_index[b][mask])
            chain_list.append(chain_nb[b][mask])
            mask_list.append(mask[mask])
            batch_list.append(torch.full((int(mask.sum().item()),), b, device=device, dtype=torch.long))
            for key, store in optional_collect.items():
                store.append(data[key][b][mask])

        if aa_list:
            aa_cat = torch.cat(aa_list, dim=0)
            coord_cat = torch.cat(coord_list, dim=0)
            res_idx_cat = torch.cat(res_idx_list, dim=0)
            chain_cat = torch.cat(chain_list, dim=0)
            node_mask_cat = torch.cat(mask_list, dim=0)
            batch_cat = torch.cat(batch_list, dim=0)
        else:
            aa_cat = torch.empty(0, dtype=aa.dtype, device=device)
            coord_cat = torch.empty(0, coords.size(2), coords.size(3), dtype=coords.dtype, device=device)
            res_idx_cat = torch.empty(0, dtype=residue_index.dtype, device=device)
            chain_cat = torch.empty(0, dtype=chain_nb.dtype, device=device)
            node_mask_cat = torch.empty(0, dtype=torch.bool, device=device)
            batch_cat = torch.empty(0, dtype=torch.long, device=device)

        extras = {}
        for key, store in optional_collect.items():
            if store:
                extras[key] = torch.cat(store, dim=0)
                continue
            value = data[key]
            if value.dim() <= 2:
                shape: Tuple[int, ...] = (0,)
            else:
                shape = (0, *value.shape[2:])
            extras[key] = value.new_zeros(shape)

        return EncoderOutput(
            node_feat=aa_cat,
            node_mask=node_mask_cat,
            batch=batch_cat,
            coords=coord_cat,
            chain_id=chain_cat,
            residue_index=res_idx_cat,
            extras=extras,
        )

        # ---- Forward ---------------------------------------------------------------------

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

        if is_antibody:
            esm_key_order = ('antiberty',)
        else:
            esm_key_order = ('esm2',)
        esm_tensor = None
        for key in esm_key_order:
            if key in packed.extras:
                esm_tensor = packed.extras[key]
                break
        dssp_tensor = packed.extras.get('dssp')
        sasa_tensor = packed.extras.get('sasa')
        projector = self.ab_projector if is_antibody else self.ag_projector
        node_input = projector(
            packed.node_feat,
            packed.residue_index,
            esm=esm_tensor,
            dssp=dssp_tensor,
            sasa=sasa_tensor,
        )
        return node_input, coords, edge_index, edge_attr, packed.extras

    def forward(self, data: Dict[str, torch.Tensor], is_antibody: bool = True):
        packed = self.pack_to_big_graph(data)
        h0, coords, edge_index, edge_attr, extras = self._prepare_encoder_inputs(packed, is_antibody)
        batch = packed.batch

        if is_antibody:
            gnn = self.antibody_gnn
        else:
            gnn = self.antigen_gnn
        if h0.numel() == 0 or edge_index.size(1) == 0:
            h = h0
        else:
            h, coords, _ = gnn(h0, coords, edge_index, edge_attr=edge_attr)

        aux: Dict[str, torch.Tensor] = {}
        if h.numel() == 0:
            graph_embed = h.new_zeros((0, self.hidden_dim))
            paratope_logits = h.new_zeros((0,))
            paratope_prob = paratope_logits
            epitope_logits = h.new_zeros((0,))
            epitope_prob = epitope_logits
        else:
            if is_antibody:
                paratope_logits = self.paratope_head(h).squeeze(-1)
                paratope_prob = torch.sigmoid(paratope_logits)
                sasa_prior = extras.get('sasa')
                if sasa_prior is not None and sasa_prior.numel() == paratope_prob.numel():
                    sasa_prior = self._normalize_prior(sasa_prior, batch)
                graph_embed = self.ab_readout(
                    h,
                    batch,
                    torch.ones(h.size(0), dtype=torch.bool, device=h.device),
                    paratope_prob,
                    sasa_prior=sasa_prior,
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
                surface_prior = extras.get('surface_prior')
                if surface_prior is None:
                    surface_prior = extras.get('sasa')
                if surface_prior is not None and surface_prior.numel() == epitope_prob.numel():
                    surface_prior = self._normalize_prior(surface_prior, batch)
                node_mask_eff = packed.node_mask if packed.node_mask.numel() > 0 else torch.ones(h.size(0),
                                                                                                 dtype=torch.bool,
                                                                                                 device=h.device)
                graph_embed = self.ag_readout(
                    h,
                    batch,
                    node_mask_eff,
                    surface_prior=surface_prior,
                    epitope_prob=epitope_prob,
                )
                paratope_logits = h.new_zeros((0,))
                paratope_prob = paratope_logits
                aux.update({
                    'epitope_logits': epitope_logits,
                    'epitope_prob': epitope_prob,
                    'epitope_target': extras.get('epitope_mask'),
                    'surface_prior': surface_prior,
                })

        if is_antibody:
            z = self.ab_projection(graph_embed)
            aux.setdefault('paratope_logits', paratope_logits)
            aux.setdefault('paratope_prob', paratope_prob)
        else:
            z = self.ag_projection(graph_embed)
            aux.setdefault('epitope_logits', epitope_logits)
            aux.setdefault('epitope_prob', epitope_prob)

        logit_scale_safe = self.logit_scale.clamp(LOGIT_MIN, LOGIT_MAX)
        scale = torch.exp(logit_scale_safe)
        aux.update({
            'batch': batch,
            'node_embeddings': h,
        })
        return z, scale, aux


def contrastive_loss(z_ab: torch.Tensor, z_ag: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(logit_scale)
    logits = (z_ab @ z_ag.t()) * scale
    labels = torch.arange(z_ab.size(0), device=z_ab.device)
    loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    return loss


def supcon_samecluster(z: torch.Tensor, y: torch.Tensor, temp: float = 0.25) -> torch.Tensor:
    B = z.size(0)
    sim = (z @ z.t()) / temp
    same = (y[:, None] == y[None, :]) & (y[:, None] >= 0)
    same.fill_diagonal_(False)
    row_mask = same.any(dim=1)
    if not row_mask.any():
        return z.new_tensor(0.0)
    pos = torch.where(same[row_mask], sim[row_mask], sim.new_full(sim[row_mask].shape, -1e9))
    log_pos = torch.logsumexp(pos, dim=1)
    log_all = torch.logsumexp(sim[row_mask] - torch.eye(B, device=z.device)[row_mask] * 1e9, dim=1)
    return -(log_pos - log_all).mean()


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
