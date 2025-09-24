import argparse
import pickle
import shutil
import torch.nn.functional as F
import faiss
import torch.utils.tensorboard
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cl_model.cl_model import ContrastiveLearningModel, LOGIT_MIN, LOGIT_MAX
from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.augment import build_two_views_pose_invariant
from diffab.utils.data import *
from diffab.utils.misc import *
from diffab.utils.train import *
from tools import unwrap

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
    parser.add_argument('--max_epoch', type=int, default=70)
    parser.add_argument('--is_train', type=int, default=0, help='train or save embeddings')
    parser.add_argument('--save_dir', type=str, default='./trained_models/retrieval')
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

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=SplitPaddingCollate(),
        shuffle=True,
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
    cl_model = ContrastiveLearningModel(input_node_dim=26, input_edge_dim=65, num_node_attr=6, device=args.device).to(
        args.device)
    # 如果你需要 DDP 包装（训练用；纯推理可以不包）
    # 注意：仅在需要反向传播/训练时使用

    # diffuser = Diffuser(input_edge_dim=37, num_node_attr=25, device=device).to(device).to(device)
    optimizer = optim.AdamW(cl_model.parameters(), lr=config.train.optimizer.lr)
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
    # 开始训练
    if args.is_train == 0:
        for epoch in range(args.max_epoch):

            cl_model.train()  # 设置模型为训练模式
            total_loss, total_seen = 0.0, 0
            # tqdm 直接包装 train_loader
            loop = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch [{epoch}/{args.max_epoch}]",
                leave=False,  # False：epoch 结束后会清掉这行
            )
            optimizer.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(loop, start=1):

                ag_data, ab_data = batch['antigen'], batch['antibody']

                # 如果 collate 对 antibody/antigen 分别做了排序/打包，可能需要从它们内部字段取各自的 id：
                pid_ab = ab_data['batch_indices']
                pid_ag = ag_data['batch_indices']

                assert torch.equal(pid_ab, pid_ag), \
                    f"Pair IDs misaligned after collate: ab={pid_ab[:8]} ag={pid_ag[:8]}"

                # 可以做数据增强
                ag_data = recursive_to(ag_data, args.device)
                ab_data = recursive_to(ab_data, args.device)
                (ab_v1, ag_v1), (ab_v2, ag_v2) = build_two_views_pose_invariant(
                    ab_data, ag_data,
                    atom_drop_p=0.00, edge_drop_p=0.00, jitter_std=0.02
                )

                with autocast('cuda', dtype=autocast_dtype, enabled=use_amp):
                    z_ab_1, _ = cl_model(ab_v1, True)
                    z_ag_1, _ = cl_model(ag_v1, False)
                    z_ab_2, _ = cl_model(ab_v2, True)
                    z_ag_2, _ = cl_model(ag_v2, False)

                    # 初始化 XBM
                    if use_xbm and (xbm is None):
                        xbm = XBM(dim=z_ab_1.size(1), capacity=max(8000, 8 * z_ab_1.size(0)), device=args.device)

                    # 拼接队列负样本（可显著提升小batch对比学习）
                    if use_xbm and xbm.size > 0:

                        mem_ab, mem_ag = xbm.get()  # [M, D]

                    else:
                        mem_ab, mem_ag = None, None


                    def _contrastive_with_mem(z_ab, z_ag, logit_scale=None):
                        z_ab = torch.nn.functional.normalize(z_ab, dim=1)
                        z_ag = torch.nn.functional.normalize(z_ag, dim=1)
                        if logit_scale is None:
                            # 固定温度（先把温度学习关掉，稳定再打开）
                            logit_scale = torch.tensor(math.log(1 / 0.07), device=z_ab.device)
                        s = z_ab @ z_ag.t()  # [B, B]
                        ls = torch.exp(logit_scale).clamp(1e-3, 1e3)
                        logits_ab = ls * s
                        logits_ag = ls * s.t()
                        labels = torch.arange(z_ab.size(0), device=z_ab.device)
                        loss = 0.5 * (
                                torch.nn.functional.cross_entropy(logits_ab, labels) +
                                torch.nn.functional.cross_entropy(logits_ag, labels)
                        )
                        return loss


                    loss_1 = _contrastive_with_mem(z_ab_1, z_ag_1)
                    loss_2 = _contrastive_with_mem(z_ab_2, z_ag_2)
                    loss_raw = 0.5 * (loss_1 + loss_2)
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

                    # 更新 XBM（放在一步完成后）
                    if use_xbm:
                        with torch.no_grad():
                            xbm.enqueue(
                                torch.nn.functional.normalize(z_ab_1.detach(), dim=1),
                                torch.nn.functional.normalize(z_ag_1.detach(), dim=1)
                            )

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
                ids_ab, ids_ag = [], []

                for batch in val_loader:
                    ab = recursive_to(batch['antibody'], device)
                    ag = recursive_to(batch['antigen'], device)
                    z_ab, _ = model(ab, True)
                    z_ag, _ = model(ag, False)
                    zs_ab.append(z_ab)
                    zs_ag.append(z_ag)
                    ids_ab.append(ab['batch_indices'].to(device))
                    ids_ag.append(ag['batch_indices'].to(device))

                z_ab = F.normalize(torch.cat(zs_ab, 0), dim=1)
                z_ag = F.normalize(torch.cat(zs_ag, 0), dim=1)
                id_ab = torch.cat(ids_ab, 0)
                id_ag = torch.cat(ids_ag, 0)

                # 关键：基于 batch_indices 做“全局对齐”
                common = torch.tensor(sorted(set(id_ab.tolist()) & set(id_ag.tolist())), device=device)
                order_ab = torch.argsort(id_ab)
                id_ab_sorted = id_ab[order_ab]
                order_ag = torch.argsort(id_ag)
                id_ag_sorted = id_ag[order_ag]
                pos_ab = torch.searchsorted(id_ab_sorted, common)
                pos_ag = torch.searchsorted(id_ag_sorted, common)
                sel_ab = order_ab[pos_ab]
                sel_ag = order_ag[pos_ag]

                z_ab = z_ab[sel_ab]
                z_ag = z_ag[sel_ag]

                # 相似度矩阵（cosine）
                S = z_ab @ z_ag.t()  # [Na, Ng]

                logits = S

                B = logits.size(0)
                labels = torch.arange(B, device=logits.device)
                loss = 0.5 * (
                        torch.nn.functional.cross_entropy(logits, labels) +
                        torch.nn.functional.cross_entropy(logits.t(), labels)
                )

                def _retrieval_metrics(S):
                    Na, Ng = S.shape
                    # 假设验证集构造为“一一配对、顺序对齐”
                    assert Na == Ng, "验证集需要一一配对对齐"
                    gt = torch.arange(Na, device=S.device)

                    # AB->AG
                    rank_ag = torch.argsort(S, dim=1, descending=True)
                    pos_rank_ab = (rank_ag == gt[:, None]).nonzero(as_tuple=False)[:, 1]
                    r1_ab = (pos_rank_ab == 0).float().mean().item()
                    r5_ab = (pos_rank_ab < 5).float().mean().item()
                    r10_ab = (pos_rank_ab < 10).float().mean().item()
                    mrr_ab = (1.0 / (pos_rank_ab.float() + 1)).mean().item()

                    # AG->AB
                    rank_ab = torch.argsort(S.t(), dim=1, descending=True)
                    pos_rank_ag = (rank_ab == gt[:, None]).nonzero(as_tuple=False)[:, 1]
                    r1_ag = (pos_rank_ag == 0).float().mean().item()
                    r5_ag = (pos_rank_ag < 5).float().mean().item()
                    r10_ag = (pos_rank_ag < 10).float().mean().item()
                    mrr_ag = (1.0 / (pos_rank_ag.float() + 1)).mean().item()

                    # margin（对称）
                    best_off_ab = S.clone()
                    eye = torch.eye(S.size(0), device=S.device, dtype=torch.bool)
                    best_off_ab[eye] = -1e9
                    m_ab = (S.diag() - best_off_ab.max(dim=1).values).mean().item()

                    best_off_ag = S.t().clone()
                    best_off_ag[eye] = -1e9
                    m_ag = (S.t().diag() - best_off_ag.max(dim=1).values).mean().item()

                    metrics = {
                        "R1": 0.5 * (r1_ab + r1_ag),
                        "R5": 0.5 * (r5_ab + r5_ag),
                        "R10": 0.5 * (r10_ab + r10_ag),
                        "MRR": 0.5 * (mrr_ab + mrr_ag),
                        "margin": 0.5 * (m_ab + m_ag),
                        "loss": loss
                    }
                    return metrics

                # 1) 统计对角线分布 vs 最大非对角
                diag = S.diag()
                offmax_row = (S - torch.eye(S.size(0), device=S.device) * 1e9).max(dim=1).values
                print(f"[Eval] diag mean={diag.mean():.3f}, offmax mean={offmax_row.mean():.3f}, "
                      f"diag>offmax ratio={(diag > offmax_row).float().mean().item():.3f}")

                # 2) 抽样看前5名排名
                topk = torch.topk(S, k=5, dim=1).indices
                mismatch = (topk[:, 0] != torch.arange(S.size(0), device=S.device)).float().mean().item()
                print(f"[Eval] Top1 mismatch ratio={mismatch:.3f}")
                return _retrieval_metrics(S)


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

            # # ===================== 验证 =====================
            # cl_model.eval()
            # val_total, val_seen = 0.0, 0
            # with torch.no_grad():
            #     vloop = tqdm(valid_loader, total=len(valid_loader), desc=f"Valid [{epoch + 1}]", leave=False)
            #     for vbatch in vloop:
            #         ag_v, ab_v = vbatch['antigen'], vbatch['antibody']
            #         ag_v = recursive_to(ag_v, args.device)
            #         ab_v = recursive_to(ab_v, args.device)
            #
            #         with autocast('cuda', dtype=(torch.bfloat16 if use_bf16 else torch.float16), enabled=use_amp):
            #             z_ab_v, _ = cl_model(ab_v)
            #             z_ag_v, _ = cl_model(ag_v)
            #             vloss = contrastive_loss(z_ab_v, z_ag_v, unwrap(cl_model).logit_scale)
            #         bsv = z_ab_v.size(0)
            #         val_total += float(vloss.detach().item()) * bsv
            #         val_seen += bsv
            #
            # val_avg = val_total / max(val_seen, 1.0)
            # print("Epoch %d valid loss %.6f" % (epoch + 1, val_avg))
            # # ===================== 保存/早停（同步） =====================
            #
            # best_path = os.path.join(args.save_dir, "best_cl_model.pth")
            # improved, should_stop = es.step(val_avg)
            # if improved:
            #     torch.save(unwrap(cl_model).state_dict(), best_path)
            #     print(f"模型已保存至: {best_path}")
            # else:
            #     if should_stop:
            #         print(f"早停触发，停止于 epoch {epoch + 1}（patience={es.patience}）")
            #         break

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
                    z_ab, _ = cl_model(ab_data)  # [B, D]
                    z_ag, _ = cl_model(ag_data)  # [B, D]
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
