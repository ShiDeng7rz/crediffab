import math
import random

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

DEFAULT_PAD_VALUES = {
    'aa': 21,
    'chain_id': ' ',
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    'origin',
    'edges',
    'edge_features'
}


class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=None, no_padding=None, eight=True):
        super().__init__()
        if no_padding is None:
            no_padding = DEFAULT_NO_PADDING
        if pad_values is None:
            pad_values = DEFAULT_PAD_VALUES
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)

        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


class SplitPaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=None, no_padding=None, eight=True):
        super().__init__()
        if no_padding is None:
            no_padding = DEFAULT_NO_PADDING
        if pad_values is None:
            pad_values = DEFAULT_PAD_VALUES
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def _collate_group(self, group_list):
        if not group_list:
            return None

        B = len(group_list)
        # 计算长度 & 8 对齐
        lengths = [int(d[self.length_ref_key].size(0)) for d in group_list]
        max_len = max(lengths)
        if self.eight:
            max_len = math.ceil(max_len / 8) * 8

        # 键集合（排除边）
        keys = self._get_common_keys(group_list)
        edge_keys = {"edges", "edge_index"}
        keys_core = [k for k in keys if k not in edge_keys]

        # === 收集边（保持 list，不进入 default_collate）===
        edges_list = [d.get('edges', d.get('edge_index')) for d in group_list]

        batch = {}  # 最终返回

        # === 为每个 core key 一次性分配输出，再逐样本拷贝 ===
        for k in keys_core:
            v0 = group_list[0][k]
            if isinstance(v0, torch.Tensor):
                # 目标形状：[B, max_len, *v0.shape[1:]]
                out_shape = (B, max_len, *v0.shape[1:])
                out = torch.full(
                    out_shape,
                    fill_value=self._get_pad_value(k),
                    dtype=v0.dtype,
                    device='cpu',  # collate 一律在 CPU
                )
                for i, (d, L) in enumerate(zip(group_list, lengths)):
                    vi = d[k]
                    if k in self.no_padding:
                        # 不 padding：直接堆成 list，留给下游自行处理
                        # （如果你确定这些 key 都能直接 stack，可改成收集后 torch.stack）
                        batch.setdefault(k, []).append(vi)
                    else:
                        out[i, :L] = vi
                if k not in self.no_padding:
                    batch[k] = out
            elif isinstance(v0, list):
                # 列表字段：按样本拼接（或保留列表）
                merged = []
                for d, L in zip(group_list, lengths):
                    vi = d[k]
                    if k in self.no_padding:
                        merged.append(vi)
                    else:
                        merged += vi  # 你原逻辑是 sum(list, [])，若需“对齐到 max_len”的列表也可补 pad
                batch[k] = merged
            else:
                # 其它类型（标量/字符串等），统一成 list
                batch[k] = [d[k] for d in group_list]

        # mask
        mask = torch.zeros((B, max_len), dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True
        batch['mask'] = mask

        # 边以列表形式附上（不会被 default_collate 触发 stack）
        batch['edges'] = edges_list

        return batch

    def __call__(self, data_list):
        """
        data_list: List[{'antibody': dict or None, 'antigen': dict or None}]
        返回：{'antibody': batch_ab 或 None, 'antigen': batch_ag 或 None}
        另外附加 'batch_indices' 记录该子 batch 对应的原始样本下标，便于对齐（可选）
        """
        # 抽取两个分组的条目与原始索引
        ab_items, ab_indices = [], []
        ag_items, ag_indices = [], []
        complex_items = []
        for i, item in enumerate(data_list):
            ab = item.get('antibody')
            ag = item.get('antigen')
            complex = item.get('complex')
            if ab is not None:
                ab_items.append(ab)
                ab_indices.append(i)
            if ag is not None:
                ag_items.append(ag)
                ag_indices.append(i)
            if complex is not None:
                complex_items.append(complex)

        ab_batch = self._collate_group(ab_items)
        ag_batch = self._collate_group(ag_items)

        # 可选：记录映射关系，方便回溯（若不需要可去掉）
        if ab_batch is not None:
            ab_batch['batch_indices'] = torch.tensor(ab_indices, dtype=torch.long)
        if ag_batch is not None:
            ag_batch['batch_indices'] = torch.tensor(ag_indices, dtype=torch.long)

        max_length = max([data[self.length_ref_key].size(0) for data in complex_items])
        keys = self._get_common_keys(complex_items)

        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        complex_items_padded = []
        for data in complex_items:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            complex_items_padded.append(data_padded)
        complex = default_collate(complex_items_padded)

        return {'antibody': ab_batch, 'antigen': ag_batch, 'complex': complex}


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    每个 batch 采 n_clusters 个簇，每簇 n_per 个样本。
    - 剔除 label=-1（无簇）样本
    - 不足时对簇内循环取（with replacement）
    - __len__ 返回批次数；__iter__ 精确产生这些批次后停止
    """
    def __init__(self, labels, n_clusters, n_per_cluster, drop_last=True, seed=None):
        super().__init__()
        self.labels = np.asarray(labels, dtype=np.int64)
        self.n_clusters = int(n_clusters)
        self.n_per = int(n_per_cluster)
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        keep = self.labels >= 0
        self.valid_idx = np.nonzero(keep)[0]
        self.valid_lab = self.labels[self.valid_idx]

        # 构建类到索引列表
        self.cls2idx = {}
        for idx, c in zip(self.valid_idx, self.valid_lab):
            self.cls2idx.setdefault(int(c), []).append(int(idx))

        # 没有任何有效簇，直接报错
        if not self.cls2idx:
            raise ValueError("No valid clusters (labels >= 0) in dataset.")

        # 预洗牌
        for c in self.cls2idx:
            self.rng.shuffle(self.cls2idx[c])

        # 估算批次数：按总可用样本粗略估算
        total = sum(len(v) for v in self.cls2idx.values())
        bs = self.n_clusters * self.n_per
        self.num_batches = total // bs if drop_last else math.ceil(total / bs)

        # 也可以改成：限定每簇的使用上限，算一个更保守的 num_batches

        # 为每簇建一个读取指针
        self.ptr = {c: 0 for c in self.cls2idx}

    def __len__(self):
        return self.num_batches

    def _take_from_class(self, c, k):
        """从簇 c 取 k 个样本，不足则循环洗牌后继续取。"""
        pool = self.cls2idx[c]
        p = self.ptr[c]
        out = []
        need = k
        while need > 0:
            remain = len(pool) - p
            if remain == 0:
                self.rng.shuffle(pool)
                p = 0
                remain = len(pool)
            t = min(need, remain)
            out.extend(pool[p:p+t])
            p += t
            need -= t
        self.ptr[c] = p
        return out

    def __iter__(self):
        classes = list(self.cls2idx.keys())
        made = 0
        while made < self.num_batches:
            # 保证可抽足 n_clusters 个类
            self.rng.shuffle(classes)
            if len(classes) < self.n_clusters:
                # 类的总数就这么多，直接循环补齐
                chosen = []
                while len(chosen) < self.n_clusters:
                    chosen.extend(classes)
                chosen = chosen[:self.n_clusters]
            else:
                chosen = classes[:self.n_clusters]

            batch = []
            for c in chosen:
                batch.extend(self._take_from_class(c, self.n_per))

            yield batch
            made += 1
        # 自然结束，StopIteration

