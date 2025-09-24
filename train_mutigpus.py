import argparse
import shutil

import torch.distributed as dist
import torch.utils.tensorboard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.data import *
from diffab.utils.misc import *
from diffab.utils.train import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def ddp_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


@torch.no_grad()
def reduce_tensor(t, op="mean"):
    """All-reduce a scalar tensor across ranks."""
    if not is_dist_avail_and_initialized():
        return t
    t = t.clone()
    dist.all_reduce(t)
    if op == "mean":
        t /= get_world_size()
    return t


@torch.no_grad()
def reduce_loss_dict(loss_dict, op="mean"):
    """All-reduce each tensor in a dict (in-place safe copy)."""
    out = {}
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            out[k] = reduce_tensor(v.detach(), op=op)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default=None, help="Ignored in DDP; use --device for single-GPU only.")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--find_unused_params', action='store_true', default=False)
    args = parser.parse_args()

    # -----------------------------
    # DDP init (torchrun friendly)
    # -----------------------------
    # torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = local_rank != -1 and world_size > 1

    if use_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = f"cuda:{local_rank}"
    else:
        # Fallback single GPU / CPU
        device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load configs & seed
    # -----------------------------
    config, config_name = load_config(args.config)
    seed_all(config.train.seed + (rank if use_ddp else 0))  # 每个 rank 做偏移，提升数据打乱多样性

    # -----------------------------
    # Logging (rank0 only)
    # -----------------------------
    if args.debug:
        logger = get_logger('train', None if get_rank() == 0 else None)
        writer = BlackHole()
    else:
        if get_rank() == 0:
            if args.resume:
                log_dir = os.path.dirname(os.path.dirname(args.resume))
            else:
                log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
            ckpt_dir = os.path.join(log_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            logger = get_logger('train', log_dir)
            writer = torch.utils.tensorboard.SummaryWriter(log_dir)
            tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
            cfg_dst = os.path.join(log_dir, os.path.basename(args.config))
            if not os.path.exists(cfg_dst):
                shutil.copyfile(args.config, cfg_dst)
        else:
            # 非 rank0 禁用文件日志与 TB
            logger = get_logger('train', None)
            writer = BlackHole()
            log_dir = None
            ckpt_dir = None

    if get_rank() == 0:
        logger.info(args)
        logger.info(config)

    # -----------------------------
    # Data
    # -----------------------------
    if get_rank() == 0:
        logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    train_iterator = inf_iterator(train_loader)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    if get_rank() == 0:
        logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # -----------------------------
    # Model
    # -----------------------------
    if get_rank() == 0:
        logger.info('Building model...')
    model = get_model(config.model).to(device)
    if get_rank() == 0:
        logger.info('Number of parameters: %d' % count_parameters(model))

    # -----------------------------
    # Optimizer & scheduler
    # -----------------------------
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # -----------------------------
    # Resume / Finetune
    # -----------------------------
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        if get_rank() == 0:
            logger.info('Resuming from checkpoint: %s' % ckpt_path)
        map_loc = {"cuda:%d" % 0: device}  # remap any saved device to current local device
        ckpt = torch.load(ckpt_path, map_location=device)
        it_first = ckpt.get('iteration', 1)
        model.load_state_dict(ckpt['model'], strict=True)
        # 仅在 resume（非 finetune）时恢复优化器/调度器
        if args.resume is not None:
            if get_rank() == 0:
                logger.info('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            if get_rank() == 0:
                logger.info('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])

    # -----------------------------
    # Wrap with DDP
    # -----------------------------
    if use_ddp:
        # broadcast_buffers=True 保持 BN/缓冲区一致；find_unused 视模型而定
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_params,
            broadcast_buffers=True
        )

    # -----------------------------
    # Train / Validate fns
    # -----------------------------
    def train_one_iter(it):
        time_start = current_milli_time()
        if use_ddp and isinstance(train_loader.sampler, DistributedSampler):
            # 让每个 rank 的 shuffle 随迭代更新
            train_loader.sampler.set_epoch(it)

        model.train()
        batch = recursive_to(next(train_iterator), device)

        # Forward
        loss_dict = model(batch) if not use_ddp else model.module(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        # 注意：DDP 下需要对 model.module.parameters() 做裁剪
        params = model.parameters() if not use_ddp else model.module.parameters()
        orig_grad_norm = clip_grad_norm_(params, config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # 减少到全局平均再记录（只在 rank0 写日志）
        loss_dict_reduced = reduce_loss_dict(loss_dict, op="mean")
        grad_norm_t = torch.tensor([float(orig_grad_norm)], device=device)
        grad_norm_avg = reduce_tensor(grad_norm_t, op="mean").item()

        if get_rank() == 0:
            log_losses(loss_dict_reduced, it, 'train', logger, writer, others={
                'grad': grad_norm_avg,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000.0,
                'time_backward': (time_backward_end - time_forward_end) / 1000.0,
            })

        if not torch.isfinite(loss):
            if get_rank() == 0:
                logger.error('NaN or Inf detected.')
                if log_dir is not None:
                    torch.save({
                        'config': config,
                        'model': (model.state_dict() if not use_ddp else model.module.state_dict()),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'batch': recursive_to(batch, 'cpu'),
                    }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            # 同步后再中断，避免僵尸进程
            ddp_barrier()
            raise KeyboardInterrupt()

    @torch.no_grad()
    def validate(it):
        # 分布式下，确保验证集切分稳定
        if use_ddp and isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(it)

        # 验证循环
        if use_ddp:
            model.module.eval()
        else:
            model.eval()

        loss_tape = ValidationLossTape()
        for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True) if get_rank() == 0 else val_loader):
            batch = recursive_to(batch, device)
            loss_dict = (model.module if use_ddp else model)(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss

            # 逐步 reduce 成全局平均（此处先累再一次性 reduce 也可）
            loss_dict_red = reduce_loss_dict(loss_dict, op="mean")
            loss_tape.update(loss_dict_red, 1)

        # 只有 rank0 负责触发调度与打印
        if get_rank() == 0:
            avg_loss = loss_tape.log(it, logger, writer, 'val')
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            return avg_loss
        else:
            # 非 rank0 也要同步 step 一下，避免 rank 间步长不一致（对非 plateau）
            if config.train.scheduler.type != 'plateau':
                scheduler.step()
            return None

    # -----------------------------
    # Loop
    # -----------------------------
    try:
        for it in range(it_first, config.train.max_iters + 1):
            train_one_iter(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                # 保存 ckpt 仅 rank0
                if (not args.debug) and get_rank() == 0:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': (model.state_dict() if not use_ddp else model.module.state_dict()),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
        ddp_barrier()
    except KeyboardInterrupt:
        if get_rank() == 0:
            logger.info('Terminating...')
    finally:
        if is_dist_avail_and_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
