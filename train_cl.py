import argparse
import pickle
import shutil
from typing import Tuple, Optional

import faiss
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.tensorboard
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cl_model.cl_premodel import ContrastiveLearningModel, LOGIT_MIN, LOGIT_MAX, info_nce_masked, supcon_samecluster
from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.antiberty_feature import (
    add_antibody_language_features_to_batch, AntibodyLanguageModelExtractor,
)
from diffab.utils.augment import build_two_views_pose_invariant
from diffab.utils.data import *
from diffab.utils.esm_feature import ESMFeatureExtractor, add_esm_features_to_batch
from diffab.utils.misc import *
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.utils.train import *
from tools import unwrap

mp.set_sharing_strategy('file_system')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

    @torch.no_grad()
    def enqueue(self, z_ab, z_ag):
        n = z_ab.size(0)
        end = self.ptr + n
        if end <= self.capacity:
            self.mem_ab[self.ptr:end].copy_(z_ab)
            self.mem_ag[self.ptr:end].copy_(z_ag)
        else:
            first = self.capacity - self.ptr
            self.mem_ab[self.ptr:].copy_(z_ab[:first])
            self.mem_ag[self.ptr:].copy_(z_ag[:first])
            rest = end - self.capacity
            self.mem_ab[:rest].copy_(z_ab[first:])
            self.mem_ag[:rest].copy_(z_ag[first:])
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    @torch.no_grad()
    def get(self):
        if self.size == 0:
            return None, None
        return self.mem_ab[:self.size], self.mem_ag[:self.size]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--is_train', type=int, default=0, help='train or save embeddings')
    parser.add_argument('--save_dir', type=str, default='./trained_models/retrieval')
    parser.add_argument('--antiberty_model', type=str, default='immune-repertoire/antiberty-base')
    parser.add_argument('--antiberty_batch_size', type=int, default=4)
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
        #     n_clusters=128, n_per_cluster=5, drop_last=True
        # ),
        batch_size=512,
        collate_fn=SplitPaddingCollate(),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    valid_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=SplitPaddingCollate(),
                              shuffle=False,
                              num_workers=args.num_workers
                              )
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    # 实例化对比学习模型，定义损失函数与优化器
    cl_model = ContrastiveLearningModel(device=args.device).to(args.device)
    # 如果你需要 DDP 包装（训练用；纯推理可以不包）
    # 注意：仅在需要反向传播/训练时使用

    # diffuser = Diffuser(input_edge_dim=37, num_node_attr=25, device=device).to(device).to(device)
    optimizer = optim.AdamW(cl_model.parameters(), lr=config.train.optimizer.lr, weight_decay=0.0)
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
    use_xbm = False
    xbm = None  # 训练开始后用实际维度初始化
    best_score = 0
    best_tie = (0, 0, 0)
    temperature = getattr(cl_model, 'temperature', 0.25)
    supcon_weight = 0.1
    paratope_weight = 0.2
    epitope_weight = 0.2
    warmup_epochs = 5
    esm_device = (
        args.device
        if isinstance(args.device, str) and args.device.startswith('cuda') and torch.cuda.is_available()
        else 'cpu'
    )
    antiberty_extractor = AntibodyLanguageModelExtractor(

        device=esm_device,
        max_batch_size=getattr(args, 'antiberty_batch_size', 4),
    )
    esm_extractor = ESMFeatureExtractor(device=esm_device, max_batch_size=getattr(args, 'esm_batch_size', 4))
    # 开始训练
    if args.is_train == 0:
        for epoch in range(args.max_epoch):

            cl_model.train()  # 设置模型为训练模式
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

                ag_data, ab_data = batch['antigen'], batch['antibody']
                y_ab = torch.tensor([train_dataset.cluster_name_to_int.get(name, -1) for name in ab_data['cluster']],
                                    dtype=torch.long, device=args.device)

                mask_ag_cpu = _get_valid_mask(ag_data)
                mask_ab_cpu = _get_valid_mask(ab_data)
                add_antibody_language_features_to_batch(ab_data, mask_ab_cpu, antiberty_extractor, ('antiberty',))
                add_esm_features_to_batch(ag_data, mask_ag_cpu, esm_extractor, ('esm2',))

                # 可以做数据增强
                ag_data = recursive_to(ag_data, args.device)
                ab_data = recursive_to(ab_data, args.device)
                (ab_v1, ag_v1), (ab_v2, ag_v2) = build_two_views_pose_invariant(
                    ab_data, ag_data,
                    atom_drop_p=0.00, edge_drop_p=0.00, jitter_std=0.02
                )
                paratope_mask, epitope_mask = compute_interface_masks(ab_data, ag_data)
                surface_prior = normalize_surface_prior(ag_data.get('sasa'), _get_valid_mask(ag_data))
                assign_weak_labels((ab_v1, ab_v2), (ag_v1, ag_v2), paratope_mask, epitope_mask, surface_prior)
                with autocast('cuda', dtype=autocast_dtype, enabled=use_amp):
                    z_ab_1, _, aux_ab_1 = cl_model(ab_v1, True)
                    z_ag_1, _, aux_ag_1 = cl_model(ag_v1, False)
                    z_ab_2, _, aux_ab_2 = cl_model(ab_v2, True)
                    z_ag_2, _, aux_ag_2 = cl_model(ag_v2, False)

                    loss_1 = info_nce_masked(z_ab_1, z_ag_1, temp=temperature, y_ab=y_ab)
                    loss_2 = info_nce_masked(z_ab_2, z_ag_2, temp=temperature, y_ab=y_ab)
                    if cur_supcon > 0.0:
                        loss_1 = loss_1 + cur_supcon * supcon_samecluster(z_ab_1, y_ab, temp=temperature)
                        loss_2 = loss_2 + cur_supcon * supcon_samecluster(z_ab_2, y_ab, temp=temperature)

                    paratope_loss = 0.5 * (bce_from_aux(aux_ab_1, 'paratope') + bce_from_aux(aux_ab_2, 'paratope'))
                    epitope_loss = 0.5 * (bce_from_aux(aux_ag_1, 'epitope') + bce_from_aux(aux_ag_2, 'epitope'))

                    loss_raw = 0.5 * (loss_1 + loss_2) + cur_paratope_w * paratope_loss + cur_epitope_w * epitope_loss
                    for n, p in unwrap(cl_model).named_parameters():
                        if "logit_scale" in n:
                            p.requires_grad_(False)
                    # 梯度累计
                    loss = loss_raw / grad_accum_steps

                scaler.scale(loss).backward()

                do_update = (step_idx % grad_accum_steps == 0) or (step_idx == len(train_loader))
                if do_update:
                    scaler.unscale_(optimizer)
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(unwrap(cl_model).parameters(), max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # 约束温度
                    with torch.no_grad():
                        unwrap(cl_model).logit_scale.clamp_(LOGIT_MIN, LOGIT_MAX)

                bs = z_ab_1.size(0)
                total_loss += float(loss_raw.detach().item()) * bs
                total_seen += bs
            avg_loss = total_loss / max(total_seen, 1)
            print("Epoch %d train loss %.6f" % (epoch + 1, avg_loss))


            #######验证########
            @torch.no_grad()
            def evaluate_retrieval(model, val_loader, device):
                model.eval()
                zs_ab, zs_ag = [], []
                y_chunks = []

                for batch in val_loader:
                    ab_dict = batch['antibody']
                    ag_dict = batch['antigen']
                    add_antibody_language_features_to_batch(ab_dict, _get_valid_mask(ab_dict), antiberty_extractor,
                                                            ('antiberty',))
                    add_esm_features_to_batch(ag_dict, _get_valid_mask(ag_dict), esm_extractor,
                                              ('esm2',))
                    ab = recursive_to(ab_dict, device)
                    ag = recursive_to(ag_dict, device)

                    # 批内对齐检查（局部索引）
                    assert torch.equal(ab['batch_indices'], ag['batch_indices'])

                    # 记录簇标签（顺序与 z_ab 对齐）
                    y_chunks.append(ab['cluster'])

                    z_ab, _, _ = model(ab, True)
                    z_ag, _, _ = model(ag, False)

                    zs_ab.append(z_ab)
                    zs_ag.append(z_ag)

                z_ab = F.normalize(torch.cat(zs_ab, 0), dim=1)  # [N,D]
                z_ag = F.normalize(torch.cat(zs_ag, 0), dim=1)  # [N,D]

                # 展平标签并映射到整型
                flat_names = sum((list(x) for x in y_chunks), [])
                y_ab = torch.tensor(
                    [train_dataset.cluster_name_to_int.get(n, -1) for n in flat_names],
                    dtype=torch.long, device=device
                )
                with torch.no_grad():
                    # 余弦相似
                    Sab = z_ab @ z_ag.t()
                    Saa = z_ab @ z_ab.t()
                    Sgg = z_ag @ z_ag.t()

                    N = Sab.size(0)
                    eye = torch.eye(N, device=Sab.device, dtype=torch.bool)

                    # 统计正样本 vs 最强负样本
                    diag = Sab.diag()
                    offmax = Sab.masked_fill(eye, -1e9).max(dim=1).values
                    print(
                        f"[check] ab-ag diag mean={diag.mean():.3f} | offmax mean={offmax.mean():.3f} | ratio={(diag > offmax).float().mean():.3f}")

                    # 看“各自空间是否分得开”（如同类粘在一起但跨模态对不上）
                    offmax_aa = Saa.masked_fill(eye, -1e9).max(dim=1).values
                    offmax_gg = Sgg.masked_fill(eye, -1e9).max(dim=1).values
                    print(
                        f"[check] ab-ab self offmax mean={offmax_aa.mean():.3f} | ag-ag self offmax mean={offmax_gg.mean():.3f}")

                    # 2) 最近邻是否总是落在同簇（说明簇标签比配对还“强”）
                    # y_ab: [N] 的 cluster id
                    nn_idx = Sab.argmax(dim=1)  # 每个 ab 最近的 ag
                    hit_same_cluster = (y_ab == y_ab[nn_idx]).float().mean().item()
                    print(f"[check] nearest ag shares cluster with ab: {hit_same_cluster:.3f}")
                # 相似度矩阵（对角即正样本）
                S = z_ab @ z_ag.t()
                N = S.size(0)
                gt = torch.arange(N, device=device)

                # 指标
                rank_ag = torch.argsort(S, dim=1, descending=True)
                pos_rank_ab = (rank_ag == gt[:, None]).nonzero(as_tuple=False)[:, 1]
                r1_ab = (pos_rank_ab == 0).float().mean().item()
                r5_ab = (pos_rank_ab < 5).float().mean().item()
                r10_ab = (pos_rank_ab < 10).float().mean().item()
                mrr_ab = (1.0 / (pos_rank_ab.float() + 1)).mean().item()

                rank_ab = torch.argsort(S.t(), dim=1, descending=True)
                pos_rank_ag = (rank_ab == gt[:, None]).nonzero(as_tuple=False)[:, 1]
                r1_ag = (pos_rank_ag == 0).float().mean().item()
                r5_ag = (pos_rank_ag < 5).float().mean().item()
                r10_ag = (pos_rank_ag < 10).float().mean().item()
                mrr_ag = (1.0 / (pos_rank_ag.float() + 1)).mean().item()

                # margin（对称）
                eye = torch.eye(N, device=device, dtype=torch.bool)
                best_off_ab = S.masked_fill(eye, -1e9).max(dim=1).values
                m_ab = (S.diag() - best_off_ab).mean().item()
                best_off_ag = S.t().masked_fill(eye, -1e9).max(dim=1).values
                m_ag = (S.t().diag() - best_off_ag).mean().item()

                # 可选：评估期监控损失（不反传）
                mask = (y_ab >= 0)
                if mask.any():
                    val_loss = info_nce_masked(z_ab[mask], z_ag[mask], temp=temperature, y_ab=y_ab[mask])
                    if supcon_weight > 0.0:
                        val_loss = val_loss + supcon_weight * supcon_samecluster(z_ab[mask], y_ab[mask],
                                                                                 temp=temperature)
                else:
                    val_loss = torch.tensor(float('nan'), device=device)
                N = S.size(0)
                assert S.size(0) == S.size(1) and N > 0, f"S shape={S.shape}"

                eye = torch.eye(N, device=S.device, dtype=torch.bool)
                diag = S.diag()
                offmax_row = S.masked_fill(eye, -1e9).max(dim=1).values

                from tqdm import tqdm
                tqdm.write(
                    f"[Eval] diag mean={diag.mean().item():.3f}, "
                    f"offmax mean={offmax_row.mean().item():.3f}, "
                    f"diag>offmax ratio={(diag > offmax_row).float().mean().item():.3f}"
                )

                top1 = torch.topk(S, k=1, dim=1).indices.squeeze(-1)
                gt = torch.arange(N, device=S.device)
                mismatch = (top1 != gt).float().mean().item()
                tqdm.write(f"[Eval] Top1 mismatch ratio={mismatch:.3f}")
                return {
                    "R1": 0.5 * (r1_ab + r1_ag),
                    "R5": 0.5 * (r5_ab + r5_ag),
                    "R10": 0.5 * (r10_ab + r10_ag),
                    "MRR": 0.5 * (mrr_ab + mrr_ag),
                    "margin": 0.5 * (m_ab + m_ag),
                    "loss": val_loss,
                    "N": N,
                }


            best_path = os.path.join(args.save_dir, "best_cl_model.pth")
            # 训练循环里：
            metrics = evaluate_retrieval(cl_model, valid_loader, args.device)
            score = metrics["R1"]  # 主指标
            tie = (metrics["R5"], metrics["MRR"], metrics["loss"])  # 辅指标（示例）
            print(
                f"Epoch {epoch + 1} valid R@1 {metrics['R1']:.4f} | R@5 {metrics['R5']:.4f} | R@10 {metrics['R10']:.4f} | MRR {metrics['MRR']:.4f} | margin {metrics['margin']:.4f} | loss {metrics['loss']:.4f}")
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
