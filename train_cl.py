import argparse
import pickle
import shutil
from typing import Tuple, Optional

import faiss
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.tensorboard
import wandb
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cl_model.cl_premodel import ContrastiveLearningModel, LOGIT_MIN, LOGIT_MAX, \
    compat_infonce_loss, kernel_regularizer
from cl_model.interaction_map import NodeInteractionBilinearLoss
from data.chain_feature_cache import make_key, gather_indices, ChainStore
from diffab.datasets import get_dataset
from diffab.utils.augment import build_two_views_pose_invariant
from diffab.utils.data import *
from diffab.utils.misc import *
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.utils.train import *
from tools import unwrap

mp.set_sharing_strategy('file_system')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# ================ Warmup + Cosine 调度器 ================
class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=5e-6, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                scale = step / float(self.warmup_steps)
                lrs.append(base_lr * scale)
            else:
                t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                cosine = 0.5 * (1 + math.cos(math.pi * t))
                lrs.append(self.min_lr + (base_lr - self.min_lr) * cosine)
        return lrs


# ====================== 早停器 ======================
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.num_bad = 0

    def step(self, val):
        improved = (self.best - val) > self.min_delta
        if improved:
            self.best = val
            self.num_bad = 0
        else:
            self.num_bad += 1
        return improved, (self.num_bad >= self.patience)


class XBM:
    def __init__(self, dim, capacity=8000, device='cuda'):
        self.dim = dim
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.mem_ab = torch.zeros(capacity, dim, device=device)
        self.mem_ag = torch.zeros(capacity, dim, device=device)

        self.mem_y = torch.full((capacity,), -1, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(self, z_ab, z_ag, y: torch.Tensor = None):
        n = z_ab.size(0)

        if y is None:
            y = torch.full((n,), -1, dtype=torch.long, device=self.device)
        else:
            y = y.to(self.mem_y.dtype)
        end = self.ptr + n
        if end <= self.capacity:
            self.mem_ab[self.ptr:end].copy_(z_ab)
            self.mem_ag[self.ptr:end].copy_(z_ag)
            self.mem_y[self.ptr:end].copy_(y)
        else:
            first = self.capacity - self.ptr
            self.mem_ab[self.ptr:].copy_(z_ab[:first])
            self.mem_ag[self.ptr:].copy_(z_ag[:first])
            self.mem_y[self.ptr:].copy_(y[:first])
            rest = end - self.capacity
            self.mem_ab[:rest].copy_(z_ab[first:])
            self.mem_ag[:rest].copy_(z_ag[first:])
            self.mem_y[:rest].copy_(y[first:])
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    @torch.no_grad()
    def get(self):
        if self.size == 0:
            return None, None
        return self.mem_ab[:self.size], self.mem_ag[:self.size], self.mem_y[:self.size]


def _get_valid_mask(batch_dict: dict, fallback_key: str = 'mask_heavyatom') -> torch.Tensor:
    if 'mask' in batch_dict:
        return batch_dict['mask'].bool()
    elif fallback_key in batch_dict:
        return batch_dict[fallback_key].any(dim=-1)
    raise KeyError('Cannot find valid residue mask in batch dictionary.')


def compute_interface_masks(ab_batch: dict, ag_batch: dict, cutoff: float = 6.0) -> Tuple[torch.Tensor, torch.Tensor]:
    device = ab_batch['pos_heavyatom'].device
    mask_ab = _get_valid_mask(ab_batch)
    mask_ag = _get_valid_mask(ag_batch)
    paratope = torch.zeros_like(mask_ab, dtype=torch.bool, device=device)
    epitope = torch.zeros_like(mask_ag, dtype=torch.bool, device=device)
    pos_ab = ab_batch['pos_heavyatom'][:, :, BBHeavyAtom.CA, :]
    pos_ag = ag_batch['pos_heavyatom'][:, :, BBHeavyAtom.CA, :]
    B = pos_ab.size(0)
    for b in range(B):
        m_ab = mask_ab[b]
        m_ag = mask_ag[b]
        if not m_ab.any() or not m_ag.any():
            continue
        pa = pos_ab[b][m_ab]
        pg = pos_ag[b][m_ag]
        dist = torch.cdist(pa, pg)
        paratope[b, m_ab] = dist.min(dim=1).values <= cutoff
        epitope[b, m_ag] = dist.min(dim=0).values <= cutoff
    return paratope, epitope


def normalize_surface_prior(prior: Optional[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    if prior is None:
        return torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    prior = prior.clone().float()
    out = torch.zeros_like(prior)
    for b in range(prior.size(0)):
        valid = mask[b]
        if not valid.any():
            continue
        values = prior[b][valid]
        max_val = values.max()
        if torch.isfinite(max_val) and max_val > 0:
            out[b, valid] = values / max_val
    return out


def assign_weak_labels(
        ab_views: Tuple[dict, dict],
        ag_views: Tuple[dict, dict],
        paratope_mask: torch.Tensor,
        epitope_mask: torch.Tensor,
        surface_prior: torch.Tensor,
) -> None:
    for view in ab_views:
        view['paratope_mask'] = paratope_mask.clone()
    for view in ag_views:
        view['epitope_mask'] = epitope_mask.clone()
        view['surface_prior'] = surface_prior.clone()


def bce_from_aux(aux: dict, key: str) -> torch.Tensor:
    logits = aux.get(f'{key}_logits')
    targets = aux.get(f'{key}_target')
    if logits is None or targets is None or logits.numel() == 0:
        if logits is not None:
            return logits.new_tensor(0.0)
        return torch.tensor(0.0, device=aux.get('node_embeddings', torch.tensor(0.0)).device)
    targets = targets.float().to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


def _summarize_transforms(transform_cfg):
    """Return a pair of (pretty_strings, canonical_tokens) for comparing configs."""
    if not transform_cfg:
        return [], []

    readable, canonical = [], []
    for item in transform_cfg:
        if isinstance(item, dict):
            name = item.get('type', str(item))
            extras = {k: v for k, v in item.items() if k != 'type'}
            if extras:
                extras_str = ', '.join(f"{k}={extras[k]}" for k in sorted(extras))
                readable.append(f"{name}({extras_str})")
            else:
                readable.append(str(name))
            canonical.append((name, tuple(sorted(extras.items()))))
        else:
            readable.append(str(item))
            canonical.append((str(item), ()))
    return readable, canonical


def _build_idx_by_slot(chain_nb, res_nb):
    """
       chain_nb: [N] 槽位号 1/2/3/4
       res_nb  : [N] 链内 1-based 编号（可能有空洞）
       返回: {slot: [0-based 紧凑索引]}
       """
    import torch
    # to python list
    if isinstance(chain_nb, torch.Tensor): chain_nb = chain_nb.long().tolist()
    if isinstance(res_nb, torch.Tensor):  res_nb = res_nb.long().tolist()

    # 先按槽位收集“原始 res_nb”
    raw = {}
    for i, s in enumerate(chain_nb):
        if s > 0:
            raw.setdefault(int(s), []).append(int(res_nb[i]))

    # 对每个槽位做致密排名：按 res_nb 升序 → 0..len-1
    slot2idx = {}
    for s, vals in raw.items():
        # 去重并按数值排序，建立 rank 映射
        uniq_sorted = sorted(set(vals))
        rank = {v: j for j, v in enumerate(uniq_sorted)}
        slot2idx[s] = [rank[v] for v in vals]
    return slot2idx


def _make_md5_by_chain_for_sample(ab_seq_md5, ag_seq_md5):
    """
    ab_seq_md5: [H_md5, L_md5_or_None]
    ag_seq_md5: [AG1_md5, AG2_md5, ...]
    返回 {1:H, 2:L_or_AG1, 3:AG2, 4:AG3, ...}
    """
    out = {}
    H = ab_seq_md5[0] if len(ab_seq_md5) >= 1 else None
    L = ab_seq_md5[1] if len(ab_seq_md5) >= 2 else None
    ag_list = list(ag_seq_md5 or [])
    if H: out[1] = H
    if L:  # 有轻链 → 2 是 L，抗原从 3 开始
        out[2] = L
        slot = 3
        for md5 in ag_list:
            out[slot] = md5
            slot += 1
    else:  # 无轻链 → 2 是第一条 AG
        if ag_list:
            out[2] = ag_list[0]
            slot = 3
            for md5 in ag_list[1:]:
                out[slot] = md5
                slot += 1
    return out


@torch.no_grad()
def add_cached_feats_minimal_batch(ab_data, ag_data, feat_store):
    """
    期望以下字段都是“长度=B”的列表（每个元素是一条样本的 1D tensor/list）：
      ab_data['chain_nb'], ab_data['res_nb'], ab_data['seq_md5']        # seq_md5[i] = [H, L_or_None]
      ag_data['chain_nb'], ag_data['res_nb'], ag_data['seq_md5']        # seq_md5[i] = [AG1, AG2, ...]
    产出：
      ab_data['lang_feat'] : torch [∑N_ab_nodes, 256]
      ag_data['lang_feat'] : torch [∑N_ag_nodes, 256]
      ab_data['node_local_idx_batch'] : List[List[int]]
      ag_data['node_local_idx_batch'] : List[List[int]]
      ab_data['heavy_node_counts']    : List[int]
    """
    ab_feats, ag_feats = [], []
    ab_node_idx_batch, ag_node_idx_batch, heavy_counts = [], [], []

    B = len(ab_data['seq_md5'])
    for i in range(B):
        # ---- per-sample 的 md5_by_chain ----
        md5_by_chain = _make_md5_by_chain_for_sample(ab_data['seq_md5'][i], ag_data['seq_md5'][i])

        # ---- AB：slot=1 必是 H；slot=2 若有 L 则为 L（AntiBERTy）----
        ab_slots = _build_idx_by_slot(ab_data['chain_nb'][i], ab_data['res_nb'][i])
        has_light = (len(ab_slots.get(2, [])) > 0) and bool(md5_by_chain.get(2))
        # H
        idx_H = ab_slots.get(1, [])
        key_H = make_key("antiberty_base", "last", "last", "none", "antiberty-v1", md5_by_chain.get(1, ""))
        E_H = feat_store.get(key_H)
        assert E_H is not None, "missing AntiBERTy(H)"
        feat_H = gather_indices(E_H, idx_H) if idx_H else np.zeros((0, E_H.shape[1]), E_H.dtype)
        ab_idx = list(idx_H)
        # L (可选)
        if has_light:
            idx_L = ab_slots.get(2, [])
            key_L = make_key("antiberty_base", "last", "last", "none", "antiberty-v1", md5_by_chain.get(2, ""))
            E_L = feat_store.get(key_L)
            assert E_L is not None, "missing AntiBERTy(L)"
            feat_L = gather_indices(E_L, idx_L) if idx_L else np.zeros((0, E_L.shape[1]), E_L.dtype)
            ab_feats.append(np.concatenate([feat_H, feat_L], axis=0))
            ab_idx.extend(idx_L)
        else:
            ab_feats.append(feat_H)
        ab_node_idx_batch.append(ab_idx)
        heavy_counts.append(len(idx_H))

        # ---- AG：有 L→从 slot>=3；无 L→从 slot>=2（ESM）----
        ag_slots = _build_idx_by_slot(ag_data['chain_nb'][i], ag_data['res_nb'][i])
        start_slot = 3 if has_light else 2
        feat_list, ag_idx = [], []
        for s in sorted(k for k in ag_slots.keys() if k >= start_slot):
            md5 = md5_by_chain.get(s, "")
            if not md5: continue
            key = make_key("esm2_t33_650M", "33", "last", "none", "ur50d", md5)
            E = feat_store.get(key)
            if E is None: continue
            ids = ag_slots[s]
            feat_list.append(gather_indices(E, ids))
            ag_idx.extend(ids)
        ag_feats.append(np.concatenate(feat_list, axis=0) if feat_list else np.zeros((0, 256), np.float32))
        ag_node_idx_batch.append(ag_idx)

    def pad_feat_list(feat_list, length=None, device=None, pad_value=0.0):
        if len(feat_list) == 0:
            # 无法推断 D；保持原行为
            return torch.empty(0), torch.empty(0, dtype=torch.bool)

            # 统一 device / dtype
        tensors = []
        for x in feat_list:
            if device is not None:
                x = x.to(device)
            tensors.append(x)

        D = tensors[0].shape[-1]

        if length is None:
            # 沿用原逻辑：pad 到 maxN
            padded = pad_sequence(tensors, batch_first=True, padding_value=pad_value)  # [B, maxN, D]

            return padded

        # 指定长度：逐个截断/补齐后 stack
        L = int(length)
        out_list = []
        masks = []

        for t in tensors:
            n = t.size(0)
            if n >= L:
                out_list.append(t[:L])
                masks.append(torch.ones(L, dtype=torch.bool, device=t.device))
            else:
                pad = t.new_full((L - n, D), pad_value)
                out_list.append(torch.cat([t, pad], dim=0))
                mask = torch.zeros(L, dtype=torch.bool, device=t.device)
                mask[:n] = True
                masks.append(mask)

        padded = torch.stack(out_list, dim=0)  # [B, L, D]
        return padded

    ab_data['lang_feat'] = pad_feat_list([torch.from_numpy(x).float() for x in ab_feats], ab_data['aa'][0].size(0),
                                         device=args.device)
    ag_data['lang_feat'] = pad_feat_list([torch.from_numpy(x).float() for x in ag_feats], ag_data['aa'][0].size(0),
                                         device=args.device)


@torch.no_grad()
def evaluate_retrieval(model, val_loader, device):
    """
    极简评测 + 同模态相似度探针：
      - 交叉模态：R@1/5/10、MRR、margin（对称）
      - 同模态：Saa_off_mean / Sgg_off_mean、Saa_offmax_mean / Sgg_offmax_mean
      - 日志：diag / offmax / diag>offmax / top1_mis
    """
    model.eval()

    # ---- 可选全局设置（若外部未提供则给默认） -------------------------------
    g = globals()
    use_amp = bool(g.get("use_amp", True))
    autocast_dtype = g.get("autocast_dtype", torch.bfloat16)
    feat_store = g.get("feat_store", None)
    add_cached = g.get("add_cached_feats_minimal_batch", None)
    recursive_to = g.get("recursive_to", lambda x, *_: x)

    # ---- 1) 提取嵌入 -------------------------------------------------------
    zs_ab, zs_ag = [], []
    for batch in val_loader:
        ab_dict = batch['antibody']
        ag_dict = batch['antigen']
        if callable(add_cached):
            add_cached(ab_dict, ag_dict, feat_store)

        ab = recursive_to(ab_dict, device)
        ag = recursive_to(ag_dict, device)

        with autocast('cuda', dtype=autocast_dtype, enabled=use_amp):
            z_ab, _, _ = model(ab, True, True)  # [B,D]
            z_ag, _, _ = model(ag, False, True)  # [B,D]

        zs_ab.append(z_ab)
        zs_ag.append(z_ag)

    z_ab = F.normalize(torch.cat(zs_ab, 0), dim=1)  # [N,D]
    z_ag = F.normalize(torch.cat(zs_ag, 0), dim=1)  # [N,D]
    N = z_ab.size(0)
    assert 0 < N == z_ag.size(0), f"z_ab={z_ab.shape}, z_ag={z_ag.shape}"

    # ---- 2) 结合/互补打分矩阵 S（优先用你的互补核） --------------------------
    def compat_scores(left, right):
        if hasattr(model, "transform_right_for_retrieval"):
            right_t = model.transform_right_for_retrieval(right)  # [N,D]
            S = left @ right_t.t()
        elif hasattr(model, "compat_kernel") and hasattr(model.compat_kernel, "score"):
            S = model.compat_kernel.score(left, right)
        else:
            S = left @ right.t()

        if hasattr(model, "logit_scale"):
            LOGIT_MIN, LOGIT_MAX = -4, 4
            scale = torch.exp(model.logit_scale.clamp(LOGIT_MIN, LOGIT_MAX))
            S = S * scale
        return S

    S = compat_scores(z_ab, z_ag)  # [N,N] 交叉模态
    Saa = z_ab @ z_ab.t()  # [N,N] 抗体-抗体（L2N → 余弦）
    Sgg = z_ag @ z_ag.t()  # [N,N] 抗原-抗原
    eye = torch.eye(N, device=S.device, dtype=torch.bool)

    # ---- 3) 检索指标（对称） ------------------------------------------------
    gt = torch.arange(N, device=S.device)

    # ab -> ag
    rank_ag = torch.argsort(S, dim=1, descending=True)
    pos_rank_ab = (rank_ag == gt[:, None]).nonzero(as_tuple=False)[:, 1]
    r1_ab = (pos_rank_ab == 0).float().mean().item()
    r5_ab = (pos_rank_ab < 5).float().mean().item()
    r10_ab = (pos_rank_ab < 10).float().mean().item()
    mrr_ab = (1.0 / (pos_rank_ab.float() + 1)).mean().item()

    # ag -> ab
    rank_ab = torch.argsort(S.t(), dim=1, descending=True)
    pos_rank_ag = (rank_ab == gt[:, None]).nonzero(as_tuple=False)[:, 1]
    r1_ag = (pos_rank_ag == 0).float().mean().item()
    r5_ag = (pos_rank_ag < 5).float().mean().item()
    r10_ag = (pos_rank_ag < 10).float().mean().item()
    mrr_ag = (1.0 / (pos_rank_ag.float() + 1)).mean().item()

    # margin（对称）
    best_off_ab = S.masked_fill(eye, float('-inf')).max(dim=1).values
    best_off_ag = S.t().masked_fill(eye, float('-inf')).max(dim=1).values
    margin = 0.5 * ((S.diag() - best_off_ab).mean().item() + (S.diag() - best_off_ag).mean().item())

    # ---- 4) 同模态相似度探针 -------------------------------------------------
    Saa_off = Saa.masked_fill(eye, float('-inf'))
    Sgg_off = Sgg.masked_fill(eye, float('-inf'))
    Saa_off_mean = Saa_off[Saa_off > float('-inf')].mean().item()
    Sgg_off_mean = Sgg_off[Sgg_off > float('-inf')].mean().item()
    Saa_offmax_mean = Saa_off.max(dim=1).values.mean().item()
    Sgg_offmax_mean = Sgg_off.max(dim=1).values.mean().item()

    # ---- 5) 极简日志 ---------------------------------------------------------
    diag_mean = S.diag().mean().item()
    offmax_mean = S.masked_fill(eye, float('-inf')).max(dim=1).values.mean().item()
    diag_better = (S.diag() > S.masked_fill(eye, float('-inf')).max(dim=1).values).float().mean().item()
    top1 = torch.topk(S, k=1, dim=1).indices.squeeze(-1)
    top1_mismatch = (top1 != gt).float().mean().item()
    print(
        f"[Eval] diag={diag_mean:.3f} | offmax={offmax_mean:.3f} | diag>offmax={diag_better:.3f} | top1_mis={top1_mismatch:.3f}")
    print(f"[Probe] Saa_off_mean={Saa_off_mean:.3f} | Sgg_off_mean={Sgg_off_mean:.3f} | "
          f"Saa_offmax_mean={Saa_offmax_mean:.3f} | Sgg_offmax_mean={Sgg_offmax_mean:.3f}")
    wandb.log({
        "eval/diag_mean": diag_mean,
        "eval/offmax_mean": offmax_mean,
        "eval/diag>offmax": diag_better,
        "eval/top1_mismatch": top1_mismatch,
    })
    # ---- 6) 汇总 -------------------------------------------------------------
    return {
        "R1": 0.5 * (r1_ab + r1_ag),
        "R5": 0.5 * (r5_ab + r5_ag),
        "R10": 0.5 * (r10_ab + r10_ag),
        "MRR": 0.5 * (mrr_ab + mrr_ag),
        "margin": margin,
        # 同模态探针
        "Saa_off_mean": Saa_off_mean,
        "Sgg_off_mean": Sgg_off_mean,
        "Saa_offmax_mean": Saa_offmax_mean,
        "Sgg_offmax_mean": Sgg_offmax_mean,
        "N": N,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--is_train', type=int, default=0, help='train or save embeddings')
    parser.add_argument('--save_dir', type=str, default='./trained_models/retrieval')
    parser.add_argument('--chain_lmdb', type=str, default="./data/processed_com/chains_emb.lmdb", )
    wandb.init(project="AI4Sci-lps", name=f"with_egnn_500")
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)
    train_tfm_readable, train_tfm_canonical = _summarize_transforms(getattr(config.dataset.train, 'transform', None))
    val_tfm_readable, val_tfm_canonical = _summarize_transforms(getattr(config.dataset.val, 'transform', None))
    if train_tfm_readable:
        logger.info('Train transforms: %s', ' -> '.join(train_tfm_readable))
    else:
        logger.info('Train transforms: <none>')
    if val_tfm_readable:
        logger.info('Val transforms: %s', ' -> '.join(val_tfm_readable))
    else:
        logger.info('Val transforms: <none>')
    if train_tfm_canonical != val_tfm_canonical:
        logger.warning('Train/Val transforms differ. Retrieval quality can collapse if distributions mismatch.')
    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_loader = DataLoader(
        train_dataset,
        # batch_sampler=BalancedBatchSampler(
        #     labels=train_dataset.cluster_id.tolist(),
        #     n_clusters=16, n_per_cluster=4, drop_last=True
        # ),
        batch_size=250,
        shuffle=True,
        collate_fn=SplitPaddingCollate(),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    valid_loader = DataLoader(val_dataset, batch_size=250, collate_fn=SplitPaddingCollate(),
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=2,
                              )
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    it_first = 1

    # 实例化对比学习模型，定义损失函数与优化器
    cl_model = ContrastiveLearningModel(device=args.device).to(args.device)
    # 如果你需要 DDP 包装（训练用；纯推理可以不包）
    # 注意：仅在需要反向传播/训练时使用
    # 氨基酸之间关注度损失
    NIloss = NodeInteractionBilinearLoss(hidden_size=512, device=args.device)
    # diffuser = Diffuser(input_edge_dim=37, num_node_attr=25, device=device).to(device).to(device)
    # optimizer = optim.AdamW(cl_model.parameters(), lr=config.train.optimizer.lr, weight_decay=0.0)
    optimizer = optim.AdamW([
        {"params": cl_model.parameters(), "lr": config.train.optimizer.lr, "weight_decay": 0.0},
        {"params": NIloss.parameters(), "lr": 1e-3},
    ])
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_steps = args.max_epoch * steps_per_epoch
    warmup_steps = max(1, int(getattr(args, "warmup_ratio", 0.05) * total_steps))
    scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps,
                             total_steps=total_steps, min_lr=getattr(args, "min_lr", 5e-6))

    max_grad_norm = getattr(args, "max_grad_norm", 5.0)
    # -------- 早停器 --------
    es = EarlyStopper(patience=getattr(args, "early_stop_patience", 20),
                      min_delta=getattr(args, "early_stop_delta", 1e-4))
    best_loss = float('inf')

    use_amp = True
    use_bf16 = torch.cuda.get_device_capability()[0] >= 8  # A100/H100
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=(use_amp and not use_bf16))
    use_xbm = getattr(args, "use_xbm", False)  # 开启/关闭
    xbm_capacity = getattr(args, "xbm_capacity", 8192)
    xbm_warmup_ep = getattr(args, "xbm_warmup_ep", 5)  # 队列参与损失前的预热 epoch
    xbm = None  # 延迟用实际维度初始化
    best_score = 0
    best_tie = (0, 0, 0)
    temperature = getattr(cl_model, 'temperature', 0.5)
    supcon_weight = 0.1
    paratope_weight = 0.0
    epitope_weight = 0.0
    warmup_epochs = 10
    bias_warmup_ep = 2
    feat_store = ChainStore(args.chain_lmdb, readonly=True)
    # 开始训练
    if args.is_train == 0:
        for epoch in range(args.max_epoch):

            cl_model.train()  # 设置模型为训练模式
            # metrics = evaluate_retrieval(cl_model, train_loader, args.device)
            total_loss, total_seen = 0.0, 0
            cur_supcon = supcon_weight if epoch >= warmup_epochs else 0.0
            cur_paratope_w = paratope_weight if epoch >= warmup_epochs else 0.0
            cur_epitope_w = epitope_weight if epoch >= warmup_epochs else 0.0
            # tqdm 直接包装 train_loader
            loop = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch [{epoch}/{args.max_epoch}]",
                leave=False
            )
            optimizer.zero_grad(set_to_none=True)

            for step_idx, batch in enumerate(loop, start=0):
                # continue
                ag_data, ab_data = batch['antigen'], batch['antibody']
                y_ab = torch.tensor([train_dataset.cluster_name_to_int.get(name, -1) for name in ab_data['cluster']],
                                    dtype=torch.long, device=args.device)
                # 加入esm特征
                add_cached_feats_minimal_batch(ab_data, ag_data, feat_store)

                # 可以做数据增强
                ag_data = recursive_to(ag_data, args.device)
                ab_data = recursive_to(ab_data, args.device)
                # 先不用管
                paratope_mask, epitope_mask = compute_interface_masks(ab_data, ag_data)
                surface_prior = normalize_surface_prior(ag_data.get('sasa'), _get_valid_mask(ag_data))
                # 不用管
                (ab_v1, ag_v1), (ab_v2, ag_v2) = build_two_views_pose_invariant(
                    ab_data, ag_data,
                    atom_drop_p=0.05, edge_drop_p=0.00, jitter_std=0.02
                )
                assign_weak_labels((ab_v1, ab_v2), (ag_v1, ag_v2), paratope_mask, epitope_mask, surface_prior)

                with autocast('cuda', dtype=autocast_dtype, enabled=use_amp):

                    z_ag_1, _, aux_ag_1 = cl_model(ag_data, False, True)
                    z_ab_1, _, aux_ab_1 = cl_model(ab_data, True, True)

                    if use_xbm and (xbm is None):
                        D = z_ab_1.size(-1)
                        xbm = XBM(dim=D, capacity=xbm_capacity, device=z_ab_1.device)
                    logit_scale = unwrap(cl_model).logit_scale

                    # 取队列负样本（预热期内不使用）
                    mem_ab = mem_ag = mem_y = None
                    if use_xbm and (xbm is not None) and (epoch >= xbm_warmup_ep):
                        mem_ab, mem_ag, mem_y = xbm.get()
                    loss_main, logs = compat_infonce_loss(cl_model, z_ab_1, z_ag_1,
                                                          anti_margin=0.1, anti_weight=0.1)
                    loss_reg = kernel_regularizer(cl_model, l2_w=1e-4, l1_diag=0.0)
                    loss_1 = loss_main + loss_reg


                def variance_loss(z, eps=1e-4):
                    std = z.float().std(dim=0) + eps
                    return torch.relu(1.0 - std).mean()


                def cov_loss(z):
                    zc = z - z.mean(dim=0, keepdim=True)
                    C = (zc.T @ zc) / (z.size(0) - 1)
                    off = C - torch.diag(torch.diag(C))
                    return (off ** 2).mean()


                loss_2 = (NIloss(aux_ab_1, aux_ag_1, ab_data["pos_heavyatom"], ag_data["pos_heavyatom"],
                                 ab_data['mask'], ag_data['mask'], ab_data['mask_heavyatom'], ag_data['mask_heavyatom'],
                                 threshold=4.5) +
                          NIloss(aux_ag_1, aux_ab_1, ag_data["pos_heavyatom"], ab_data["pos_heavyatom"],
                                 ag_data['mask'], ab_data['mask'], ag_data['mask_heavyatom'], ab_data['mask_heavyatom'],
                                 threshold=4.5)) * 0.5
                loss_raw = loss_1
                # 梯度累计
                loss = loss_raw / grad_accum_steps

                scaler.scale(loss).backward()

                do_update = ((step_idx + 1) % grad_accum_steps == 0) or (step_idx + 1 == len(train_loader))

                if do_update:
                    scaler.unscale_(optimizer)
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(unwrap(cl_model).parameters(), max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if use_xbm and (xbm is not None):
                        with torch.no_grad():
                            z_ab_cur = z_ab_1.detach()
                            z_ag_cur = z_ag_1.detach()
                            y_ab_cur = y_ab.detach()
                            xbm.enqueue(z_ab_cur, z_ag_cur, y_ab_cur)
                    # 约束温度
                    with torch.no_grad():
                        unwrap(cl_model).logit_scale.clamp_(LOGIT_MIN, LOGIT_MAX)

                bs = z_ab_1.size(0)
                total_loss += float(loss_raw.detach().item()) * bs
                total_seen += bs

                # print(f"[{step_idx}] "
                #       f"load={t_load - t0:.3f}s feat={t_feat - t_load:.3f}s "
                #       f"mask={t_mask - t_gpu_before_mask:.3f}s aug={t_aug - t_mask:.3f}s "
                #       f"fwd={t_fwd - t_aug:.3f}s loss={t_loss - t_fwd:.3f}s "
                #       f"bwd={t_bwd - t_loss:.3f}s upd={t_upd - t_bwd:.3f}s")
            avg_loss = total_loss / max(total_seen, 1)

            print("Epoch %d train loss %.6f" % (epoch + 1, avg_loss))
            wandb.log({
                "train/loss": avg_loss,
            })
            #######验证########

            best_path = os.path.join(args.save_dir, "best_cl_model.pth")
            # 训练循环里：
            metrics = evaluate_retrieval(cl_model, valid_loader, args.device)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics["loss"])  # 或 -metrics["R1"]
            else:
                scheduler.step()
            score = metrics["R1"]  # 主指标
            tie = (metrics["R5"], metrics["MRR"])  # 辅指标（示例）
            print(
                f"Epoch {epoch + 1} valid R@1 {metrics['R1']:.4f} | R@5 {metrics['R5']:.4f} | R@10 {metrics['R10']:.4f} | MRR {metrics['MRR']:.4f} | margin {metrics['margin']:.4f}")
            wandb.log({
                "valid/R1": metrics["R1"],
                "valid/R5": metrics["R5"],
                "valid/R10": metrics["R10"],
                "valid/MRR": metrics["MRR"],
                "valid/margin": metrics["margin"],
            })

            if (score > best_score) or (score == best_score and tie > best_tie):
                best_score, best_tie = score, tie
                torch.save(unwrap(cl_model).state_dict(), best_path)  # 原权重
                print(f"模型已保存至: {best_path}")

else:

    save_dataset = get_dataset(config.dataset.save_emb)

    save_loader = DataLoader(
        save_dataset,
        batch_size=config.train.batch_size,
        collate_fn=SplitPaddingCollate(),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    best_path = os.path.join(args.save_dir, "best_cl_model.pth")
    ckpt = torch.load(best_path, map_location=args.device, weights_only=True)
    cl_model.load_state_dict(ckpt)
    cl_model.eval()
    all_ab_embs, all_ag_embs = [], []
    ab_meta_local, ag_meta_local = [], []
    with torch.no_grad():
        for batch in tqdm(save_loader, desc="Saving embeddings"):
            ag_data, ab_data = batch['antigen'], batch['antibody']
            ag_data = recursive_to(ag_data, args.device)
            ab_data = recursive_to(ab_data, args.device)

            with autocast('cuda', dtype=(torch.bfloat16 if use_bf16 else torch.float16), enabled=use_amp):
                z_ab, _, _ = cl_model(ab_data)  # [B, D]
                z_ag, _, _ = cl_model(ag_data)  # [B, D]
            all_ab_embs.append(z_ab.cpu())
            all_ag_embs.append(z_ag.cpu())


            def _pick_meta_from_batched(data: dict, i: int):
                """从 batched 张量字典里抽取第 i 条样本的轻量 meta（不拷贝大矩阵）"""
                res_mask = data['mask'][i].bool()
                meta = {
                    'resseq': data['resseq'][i][res_mask],
                    'aa': data['aa'][i][res_mask],
                    'chain_nb': data['chain_nb'][i][res_mask],
                    'pos_heavyatom': data['pos_heavyatom'][i][res_mask],
                    'fragment_type': data['fragment_type'][i][res_mask],
                    'res_nb': data['res_nb'][i][res_mask],
                    'generate_flag': data['generate_flag'][i][res_mask],
                    'cdr_flag': data['cdr_flag'][i][res_mask],
                    'mask_heavyatom': data['mask_heavyatom'][i][res_mask],
                }
                return meta


            for i in range(ag_data['aa'].shape[0]):
                ab_meta_local.append(_pick_meta_from_batched(ab_data, i))
                ag_meta_local.append(_pick_meta_from_batched(ag_data, i))

    all_ab_embs = torch.cat(all_ab_embs, dim=0)
    all_ag_embs = torch.cat(all_ag_embs, dim=0)
    emb_ab = np.ascontiguousarray(all_ab_embs.detach().cpu().float().numpy(), dtype=np.float32)
    emb_ag = np.ascontiguousarray(all_ag_embs.detach().cpu().float().numpy(), dtype=np.float32)

    faiss.normalize_L2(emb_ab)
    faiss.normalize_L2(emb_ag)
    d = int(emb_ab.shape[1])
    index_ab_cpu = faiss.IndexFlatIP(d)
    index_ab_cpu.add(emb_ab)  # 必须是 np.float32 且 C-contiguous
    os.makedirs(args.save_dir, exist_ok=True)
    faiss.write_index(index_ab_cpu, os.path.join(args.save_dir, 'ab_index.faiss'))
    with open(os.path.join(args.save_dir, 'ab_meta.pkl'), 'wb') as f:
        pickle.dump(ab_meta_local, f)
    with open(os.path.join(args.save_dir, 'ag_meta.pkl'), 'wb') as f:
        pickle.dump(ag_meta_local, f)
    # embeddings 保存完
    writer.flush()
    writer.close()  # 若使用了 TensorBoard
    del save_loader
    import gc

    gc.collect()
    torch.cuda.empty_cache()
