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


def _compute_residue_representatives(pos_atoms, mask_atoms):
    """Return per-residue representative coordinates and validity mask."""
    ca_index = constants.BBHeavyAtom.CA
    device = pos_atoms.device

    has_any = mask_atoms.any(dim=1)
    if mask_atoms.size(1) == 0:
        return pos_atoms.new_zeros(pos_atoms.size(0), 3), has_any

    if ca_index < mask_atoms.size(1):
        has_ca = mask_atoms[:, ca_index]
        pos_rep = pos_atoms[:, ca_index]
    else:
        has_ca = torch.zeros_like(has_any)
        pos_rep = pos_atoms.new_zeros(pos_atoms.size(0), 3)

    # fallback to the first valid atom when CA is missing
    idx_first_valid = torch.argmax(mask_atoms.float(), dim=1)
    fallback = pos_atoms[torch.arange(pos_atoms.size(0), device=device), idx_first_valid]
    pos_rep = torch.where(has_ca[:, None], pos_rep, fallback)

    # zero-out invalid residues explicitly
    pos_rep = torch.where(has_any[:, None], pos_rep, torch.zeros_like(pos_rep))
    return pos_rep, has_any


def compute_interface_masks_complex(complex_data, distance_threshold=8.0):
    """Compute paratope/epitope masks for a complex without altering it."""
    fragment_type = complex_data['fragment_type']
    pos_atoms = complex_data['pos_heavyatom']
    mask_atoms = complex_data['mask_heavyatom']

    antibody_mask = fragment_type != constants.Fragment.Antigen
    antigen_mask = fragment_type == constants.Fragment.Antigen

    paratope_mask = torch.zeros_like(fragment_type, dtype=torch.bool)
    epitope_mask = torch.zeros_like(fragment_type, dtype=torch.bool)

    if not antibody_mask.any() or not antigen_mask.any():
        return paratope_mask, epitope_mask

    pos_rep, valid_mask = _compute_residue_representatives(pos_atoms, mask_atoms)

    valid_ab = antibody_mask & valid_mask
    valid_ag = antigen_mask & valid_mask

    if not valid_ab.any() or not valid_ag.any():
        return paratope_mask, epitope_mask

    pos_ab = pos_rep[valid_ab]
    pos_ag = pos_rep[valid_ag]

    dist = torch.cdist(pos_ab, pos_ag)

    contact_ab = (dist.min(dim=1).values <= distance_threshold)
    contact_ag = (dist.min(dim=0).values <= distance_threshold + 4)

    paratope_indices = torch.where(valid_ab)[0]
    epitope_indices = torch.where(valid_ag)[0]
    paratope_mask[paratope_indices] = contact_ab
    epitope_mask[epitope_indices] = contact_ag

    return paratope_mask, epitope_mask


def span_fill_by_chain(mask: torch.Tensor, chain_nb: torch.Tensor) -> torch.Tensor:
    """
    在每条链内，把该链上第一个 True 到最后一个 True 之间全部置 True。
    mask: [N]  布尔
    chain_nb: [N]  记录每个残基属于哪条链（你的约定里 1=重链，2=轻链 等）
    """
    out = mask.clone()
    for c in torch.unique(chain_nb).tolist():
        idx = torch.nonzero(chain_nb == c, as_tuple=False).squeeze(1)  # 当前链的全局位置
        if idx.numel() == 0:
            continue
        m = mask[idx]  # 当前链的局部掩码
        pos = torch.nonzero(m, as_tuple=False).squeeze(1)
        if pos.numel() >= 2:
            lo = int(pos.min().item())
            hi = int(pos.max().item())
            m[lo:hi + 1] = True
            out[idx] = m
        # pos.numel()==0 或 ==1 的情况保持原样（如需可再做±w的膨胀）
    return out


def _residue_reps(pos_atoms: torch.Tensor, mask_atoms: torch.Tensor, ca_index: int):
    """每残基取一个代表点（优先 CA，缺失则取该残基第一个有效原子）。"""
    N = pos_atoms.size(0)
    has_any = mask_atoms.any(dim=1)
    if ca_index < mask_atoms.size(1):
        has_ca = mask_atoms[:, ca_index]
        pos_rep = pos_atoms[:, ca_index]
    else:
        has_ca = torch.zeros_like(has_any)
        pos_rep = pos_atoms.new_zeros((N, 3))
    idx_first = torch.argmax(mask_atoms.float(), dim=1)
    fallback = pos_atoms[torch.arange(N, device=pos_atoms.device), idx_first]
    pos_rep = torch.where(has_ca[:, None], pos_rep, fallback)
    pos_rep = torch.where(has_any[:, None], pos_rep, torch.zeros_like(pos_rep))
    return pos_rep, has_any


def _ctx_by_radius_simple(pos_rep: torch.Tensor, valid: torch.Tensor,
                          core_mask: torch.Tensor, radius: float) -> torch.Tensor:
    """距核心≤radius 且非核心的作为上下文。"""
    device = pos_rep.device
    N = pos_rep.size(0)
    ctx = torch.zeros(N, dtype=torch.bool, device=device)
    if not (valid.any() and core_mask.any()):
        return ctx
    v_idx = torch.where(valid)[0]
    c_idx = torch.where(valid & core_mask)[0]
    if c_idx.numel() == 0:
        return ctx
    P = pos_rep[v_idx]  # [Nv,3]
    C = pos_rep[c_idx]  # [Nc,3]
    d = torch.cdist(P, C)  # [Nv,Nc]
    near = (d.min(dim=1).values <= radius)  # [Nv]
    near_idx = v_idx[near]
    # 去掉核心自身
    near_idx = near_idx[~core_mask[near_idx]]
    ctx[near_idx] = True
    return ctx


@register_transform('patch_cdr_epitope')
class PatchCDREpitope(object):
    def __init__(self, initial_patch_size=128, antigen_size=128,
                 r_ab_ctx: float = 7.0, r_ag_ctx: float = 9.0):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size
        self.r_ab_ctx = r_ab_ctx  # 抗体上下文半径
        self.r_ag_ctx = r_ag_ctx  # 抗原上下文半径

    def __call__(self, data):
        antibody = data['antibody']
        antigen = data['antigen']
        complex = data['complex']

        # 1) 距离得到的核心掩码（全复合物级）
        paratope_mask, epitope_mask = compute_interface_masks_complex(complex)
        complex['paratope_mask'] = paratope_mask
        complex['epitope_mask'] = epitope_mask

        fragment_type = complex['fragment_type']
        ab_mask = (fragment_type != constants.Fragment.Antigen)
        ag_mask = (fragment_type == constants.Fragment.Antigen)

        # 2) 抗体端：核心 + 锚点扩展 + 上下文环
        if isinstance(antibody, dict):
            par_ab = paratope_mask[ab_mask].clone()  # 抗体视图的核心
            # 锚点：若提供 anchor_flag，则在每条链内把两端 True 之间填满
            anchor_ab = None
            if 'anchor_flag' in antibody:
                anchor_ab = antibody['anchor_flag'].to(par_ab.device).bool()
            elif 'anchor_flag' in complex:
                anchor_ab = complex['anchor_flag'][ab_mask].to(par_ab.device).bool()
            if anchor_ab is not None and 'chain_nb' in antibody:
                anchor_ab = span_fill_by_chain(anchor_ab, antibody['chain_nb'])
                par_ab |= anchor_ab

            # 上下文环（基于 CA 半径）
            CA = constants.BBHeavyAtom.CA
            pos_rep_ab, valid_ab = _residue_reps(complex['pos_heavyatom'][ab_mask],
                                                 complex['mask_heavyatom'][ab_mask], CA)
            ctx_ab = _ctx_by_radius_simple(pos_rep_ab, valid_ab, par_ab, self.r_ab_ctx)

            antibody['paratope_mask'] = par_ab
            antibody['paratope_ctx_mask'] = ctx_ab
            antibody['epitope_mask'] = torch.zeros_like(par_ab)
            antibody['epitope_ctx_mask'] = torch.zeros_like(par_ab)

        # 3) 抗原端：核心 + 上下文环
        if isinstance(antigen, dict):
            epi_ag = epitope_mask[ag_mask].clone()  # 抗原视图的核心
            CA = constants.BBHeavyAtom.CA
            pos_rep_ag, valid_ag = _residue_reps(complex['pos_heavyatom'][ag_mask],
                                                 complex['mask_heavyatom'][ag_mask], CA)
            ctx_ag = _ctx_by_radius_simple(pos_rep_ag, valid_ag, epi_ag, self.r_ag_ctx)

            antigen['epitope_mask'] = epi_ag
            antigen['epitope_ctx_mask'] = ctx_ag
            antigen['paratope_mask'] = torch.zeros_like(epi_ag)
            antigen['paratope_ctx_mask'] = torch.zeros_like(epi_ag)

        return data
