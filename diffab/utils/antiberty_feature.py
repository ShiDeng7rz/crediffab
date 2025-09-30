"""Utilities for extracting antibody language-model representations."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from diffab.utils.esm_feature import ChainRecord, tensor_to_sequence


def _chunk_iterable(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _normalize_runner_output(raw: Any, key: str) -> List[torch.Tensor]:
    if isinstance(raw, torch.Tensor):
        return [raw.detach().cpu()]
    if isinstance(raw, dict):
        for candidate in (
                key,
                f"{key}_embeddings",
                f"{key}_embedding",
                "embeddings",
                "residue_embeddings",
                "sequence_embeddings",
                "node_embeddings",
        ):
            if candidate in raw:
                return _normalize_runner_output(raw[candidate], key)
        values: List[torch.Tensor] = []
        for value in raw.values():
            values.extend(_normalize_runner_output(value, key))
        if values:
            return values
        raise RuntimeError(f"Unable to locate embeddings for key '{key}' in runner output keys {tuple(raw.keys())}")
    if isinstance(raw, (list, tuple)):
        outputs: List[torch.Tensor] = []
        for item in raw:
            outputs.extend(_normalize_runner_output(item, key))
        return outputs
    raise RuntimeError(f"Unsupported runner output type: {type(raw).__name__}")


class AntibodyLanguageModelExtractor:
    """Lazily load AntiBERTy models to embed antibody sequences."""

    SUPPORTED_KEYS = ("antiberty",)

    def __init__(
            self,
            device: str | torch.device = "cpu",
            dtype: torch.dtype = torch.float32,
            max_batch_size: int = 4,
            model_dir: str | None = None,
            chain_window_margin: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max(1, int(max_batch_size))
        self.model_dir = model_dir
        if chain_window_margin is None:
            self.chain_window_margin = 0
        else:
            self.chain_window_margin = max(0, int(chain_window_margin))
        self.runners: Dict[str, Any] = {}
        self.cache: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    def _ensure_runner(self, key: str) -> None:
        if key in self.runners:
            return
        if key not in self.SUPPORTED_KEYS:
            raise KeyError(f"Unknown antibody LM key: {key}")
        spec = importlib.util.find_spec("antiberty")
        if spec is None:
            raise RuntimeError(
                "The 'antiberty' package is required to compute AntiBERTy embeddings. "
                "Install it with `pip install antiberty`."
            )
        antiberty = importlib.import_module("antiberty")
        runner_cls = getattr(antiberty, "AntiBERTyRunner", None)
        if runner_cls is None:
            raise RuntimeError(
                "The 'antiberty' package does not provide an AntiBERTyRunner implementation."
            )
        runner_kwargs: Dict[str, Any] = {}
        signature = None
        try:
            signature = inspect.signature(runner_cls)
        except (TypeError, ValueError):
            signature = None
        parameters = signature.parameters if signature is not None else {}
        if self.model_dir is not None:
            if "model_dir" in parameters:
                runner_kwargs["model_dir"] = self.model_dir
            elif "weights_path" in parameters:
                runner_kwargs["weights_path"] = self.model_dir
        if "device" in parameters:
            runner_kwargs["device"] = str(self.device)
        elif "devices" in parameters:
            if self.device.type == "cuda":
                runner_kwargs["devices"] = [
                    self.device.index if self.device.index is not None else 0
                ]
            else:
                runner_kwargs["devices"] = [str(self.device)]
        if "dtype" in parameters:
            runner_kwargs["dtype"] = self.dtype
        try:
            runner = runner_cls(**runner_kwargs)
        except TypeError:
            runner = runner_cls()
        if hasattr(runner, "to"):
            try:
                runner = runner.to(self.device)
            except Exception:
                pass
        self.runners[key] = runner

    def _call_runner(self, key: str, sequences: Sequence[str]) -> List[torch.Tensor]:
        self._ensure_runner(key)
        runner = self.runners[key]
        method_candidates = (
            "embed",
            "get_antiberty_embeddings",
            "embed_antibodies",
            "infer",
        )
        for name in method_candidates:
            method = getattr(runner, name, None)
            if method is None:
                continue
            try:
                raw = method(sequences)
            except TypeError:
                continue
            embeddings = _normalize_runner_output(raw, key)
            if len(embeddings) == 1:
                first = embeddings[0]
                if first.dim() == 3 and first.size(0) == len(sequences):
                    embeddings = [tensor.detach().cpu() for tensor in first]
            if len(embeddings) != len(sequences):
                raise RuntimeError(
                    f"Expected {len(sequences)} embeddings from runner method '{name}', got {len(embeddings)}"
                )
            return [emb.float().detach().cpu() for emb in embeddings]
        raise RuntimeError(
            "AntiBERTy runner does not expose a known embedding method. Expected one of: "
            + ", ".join(method_candidates)
        )

    def embed_batch(self, key: str, sequences: Sequence[str]) -> List[torch.Tensor]:
        if not sequences:
            return []
        cache = self.cache[key]
        outputs: List[torch.Tensor | None] = [cache.get(seq) for seq in sequences]
        missing: List[Tuple[int, str]] = [
            (idx, seq) for idx, (seq, cached) in enumerate(zip(sequences, outputs)) if cached is None
        ]
        if missing:
            seqs = [seq for _, seq in missing]
            new_embeddings: List[torch.Tensor] = []
            for chunk in _chunk_iterable(seqs, self.max_batch_size):
                new_embeddings.extend(self._call_runner(key, list(chunk)))
            if len(new_embeddings) != len(seqs):
                raise RuntimeError(
                    f"Runner returned {len(new_embeddings)} embeddings for {len(seqs)} sequences"
                )
            for (idx, seq), emb in zip(missing, new_embeddings):
                cache[seq] = emb
                outputs[idx] = emb
        return [out.clone() for out in outputs if out is not None]

    def _resolve_window_margin(self, override: int | None) -> int:
        if override is None:
            return self.chain_window_margin
        return max(0, int(override))

    def embed_batch_from_tensor(
            self,
            aa: torch.Tensor,
            mask: torch.Tensor,
            key: str,
            chain_ids: Sequence[Sequence[Any]] | torch.Tensor | None = None,
            residue_index: Sequence[Sequence[Any]] | torch.Tensor | None = None,
            chain_window_margin: int | None = None,
    ) -> torch.Tensor:
        if aa.dim() != 2:
            raise ValueError(f"Expected amino acid tensor of shape [B, L], got {tuple(aa.shape)}")
        if mask.shape != aa.shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} must match amino acid tensor {tuple(aa.shape)}")
        mask_bool = mask.bool()
        aa_cpu = aa.detach().cpu()
        mask_cpu = mask_bool.detach().cpu()
        B, L = aa_cpu.shape
        chain_rows = _normalize_chain_annotations(chain_ids, aa_cpu.shape) if chain_ids is not None else None
        residue_rows = (
            _normalize_nested(residue_index, aa_cpu.shape, "residue_index")
            if residue_index is not None
            else None
        )
        if chain_rows is not None:
            margin = self._resolve_window_margin(chain_window_margin)
            records = _collect_chain_records(aa_cpu, mask_cpu, chain_rows, residue_rows, margin)
            if not records:
                return torch.zeros(B, L, 0, dtype=torch.float32)
            sequences: List[str] = []
            owners: List[ChainRecord] = []
            for record in records:
                seq_positions = record.window_positions if record.window_positions else record.positions
                seq_tensor = aa_cpu[record.batch_idx][seq_positions]
                seq = tensor_to_sequence(seq_tensor)
                if not seq:
                    continue
                sequences.append(seq)
                owners.append(record)
            if not owners:
                return torch.zeros(B, L, 0, dtype=torch.float32)
            embeddings = self.embed_batch(key, sequences)
            if not embeddings:
                return torch.zeros(B, L, 0, dtype=torch.float32)
            dim = embeddings[0].size(-1)
            outputs = torch.zeros(B, L, dim, dtype=torch.float32)
            for record, emb in zip(owners, embeddings):
                seq_positions = record.window_positions if record.window_positions else record.positions
                expected_len = len(seq_positions)
                if emb.size(0) != expected_len:
                    raise RuntimeError(
                        f"Embedding length {emb.size(0)} does not match expected residues {expected_len}"
                    )
                gather_idx = record.gather_indices if record.gather_indices else list(range(expected_len))
                gathered = emb[gather_idx]
                if gathered.size(0) != len(record.positions):
                    raise RuntimeError(
                        "Mismatch between gathered embedding length "
                        f"{gathered.size(0)} and target positions {len(record.positions)}"
                    )
                outputs[record.batch_idx, record.positions] = gathered.to(torch.float32)
            return outputs

        sequences: List[str] = []
        owners: List[Tuple[int, torch.Tensor]] = []
        for b in range(B):
            valid = mask_cpu[b]
            if not torch.any(valid):
                continue
            seq = tensor_to_sequence(aa_cpu[b][valid])
            if not seq:
                continue
            sequences.append(seq)
            owners.append((b, valid))
        if not sequences:
            return torch.zeros(B, L, 0, dtype=torch.float32)
        embeddings = self.embed_batch(key, sequences)
        if not embeddings:
            return torch.zeros(B, L, 0, dtype=torch.float32)
        dim = embeddings[0].size(-1)
        outputs = torch.zeros(B, L, dim, dtype=torch.float32)
        for (b, valid), emb in zip(owners, embeddings):
            if emb.size(0) != int(valid.sum().item()):
                raise RuntimeError(
                    f"Embedding length {emb.size(0)} does not match valid residue count {int(valid.sum().item())}"
                )
            outputs[b, valid] = emb.to(torch.float32)
        return outputs


def add_antibody_language_features_to_batch(
        data: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        extractor: AntibodyLanguageModelExtractor | None,
        model_keys: Iterable[str] = ("antiberty",),
        chain_window_margin: int | None = None,
) -> None:
    if extractor is None:
        return
    aa = data.get("aa")
    if aa is None or aa.numel() == 0:
        return
    mask_bool = mask.bool()
    chain_ids = data.get("chain_nb") if data.get("chain_nb") is not None else data.get("chain_id")
    residue_index = data.get("res_nb") if data.get("res_nb") is not None else data.get("resseq")
    for key in model_keys:
        if key in data and isinstance(data[key], torch.Tensor):
            continue
        embeddings = extractor.embed_batch_from_tensor(
            aa,
            mask_bool,
            key,
            chain_ids=chain_ids,
            residue_index=residue_index,
            chain_window_margin=chain_window_margin,
        )
        if embeddings.numel() == 0:
            continue
        data[key] = embeddings


def _normalize_nested(
        value: Sequence[Sequence[Any]] | torch.Tensor,
        shape: Tuple[int, int],
        name: str,
) -> List[List[Any]]:
    B, L = shape
    if isinstance(value, torch.Tensor):
        if value.dim() == 1:
            if value.numel() != B * L:
                raise ValueError(f"{name} tensor has incompatible shape {tuple(value.shape)} for {shape}")
            value = value.view(B, L)
        if value.dim() != 2 or value.size(0) != B:
            raise ValueError(f"{name} tensor must have shape [B, L]; got {tuple(value.shape)}")
        rows = value.cpu().tolist()
        return [row[:L] for row in rows]

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Unsupported type for {name}: {type(value)!r}")
    if len(value) != B:
        if len(value) == B * L:
            reshaped = [list(value[i * L:(i + 1) * L]) for i in range(B)]
            return reshaped
        raise ValueError(f"{name} list length {len(value)} does not match batch size {B}")

    rows: List[List[Any]] = []
    for row in value:
        if isinstance(row, torch.Tensor):
            row_list = row.cpu().tolist()
        elif isinstance(row, (list, tuple)):
            row_list = list(row)
        elif row is None:
            row_list = [None] * L
        elif isinstance(row, str):
            row_list = list(row)
        else:
            row_list = list(row)
        if len(row_list) < L:
            row_list = row_list + [None] * (L - len(row_list))
        rows.append(row_list[:L])
    return rows


def _normalize_chain_annotations(
        chain_ids: Sequence[Sequence[Any]] | torch.Tensor,
        shape: Tuple[int, int],
) -> List[List[Any]]:
    rows = _normalize_nested(chain_ids, shape, "chain_ids")
    normalized: List[List[Any]] = []
    for row in rows:
        row_norm: List[Any] = []
        for item in row:
            if torch.is_tensor(item):  # type: ignore[truthy-bool]
                item = item.item() if item.numel() == 1 else item.tolist()
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            if isinstance(item, str):
                trimmed = item.strip()
                row_norm.append(trimmed if trimmed else None)
            else:
                row_norm.append(item)
        normalized.append(row_norm)
    return normalized


def _collect_chain_records(
        aa: torch.Tensor,
        mask: torch.Tensor,
        chain_rows: List[List[Any]],
        residue_rows: List[List[Any]] | None,
        window_margin: int,
) -> List[ChainRecord]:
    records: List[ChainRecord] = []
    B, _ = aa.shape
    for b in range(B):
        valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        chain_row = chain_rows[b] if b < len(chain_rows) else None
        if chain_row is None:
            continue
        residue_row = residue_rows[b] if residue_rows is not None and b < len(residue_rows) else None
        groups: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
        full_entries: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
        row_length = len(chain_row)
        for pos in range(row_length):
            label_all = chain_row[pos]
            if label_all is None:
                continue
            if isinstance(label_all, torch.Tensor):
                label_all = label_all.item() if label_all.numel() == 1 else tuple(label_all.tolist())
            res_full = residue_row[pos] if residue_row is not None and pos < len(residue_row) else pos
            if isinstance(res_full, torch.Tensor):
                res_full = res_full.item() if res_full.numel() == 1 else tuple(res_full.tolist())
            try:
                res_full_int = int(res_full)
            except (TypeError, ValueError):
                res_full_int = pos
            full_entries[label_all].append((res_full_int, pos))
        for pos in valid_idx.tolist():
            if pos >= row_length:
                continue
            label = chain_row[pos]
            if label is None:
                continue
            if isinstance(label, torch.Tensor):
                label = label.item() if label.numel() == 1 else tuple(label.tolist())
            if isinstance(label, str) and not label:
                continue
            res_idx = residue_row[pos] if residue_row is not None and pos < len(residue_row) else pos
            if isinstance(res_idx, torch.Tensor):
                res_idx = res_idx.item() if res_idx.numel() == 1 else tuple(res_idx.tolist())
            try:
                res_idx_int = int(res_idx)
            except (TypeError, ValueError):
                res_idx_int = pos
            groups[label].append((res_idx_int, pos))
        for label, entries in groups.items():
            if not entries:
                continue
            entries.sort(key=lambda item: (item[0], item[1]))
            positions = [pos for _, pos in entries]
            residue_indices = [res for res, _ in entries]
            window_positions = positions.copy()
            window_residue_indices = residue_indices.copy()
            gather_indices = list(range(len(positions)))
            if label in full_entries and full_entries[label]:
                full_list = sorted(full_entries[label], key=lambda item: (item[0], item[1]))
                selected = full_list
                if window_margin > 0 and residue_indices:
                    min_res = min(residue_indices)
                    max_res = max(residue_indices)
                    lower = min_res - window_margin
                    upper = max_res + window_margin
                    filtered = [
                        (res, pos_full)
                        for res, pos_full in full_list
                        if lower <= res <= upper
                    ]
                    if filtered:
                        selected = filtered
                merged: Dict[int, int] = {pos_full: res_full for res_full, pos_full in selected}
                for res_orig, pos_orig in entries:
                    merged[pos_orig] = res_orig
                combined = sorted(merged.items(), key=lambda item: (item[1], item[0]))
                window_positions = [pos for pos, _ in combined]
                window_residue_indices = [res for _, res in combined]
                pos_to_idx = {pos: idx for idx, pos in enumerate(window_positions)}
                gather_indices = [pos_to_idx[pos] for pos in positions]
            records.append(
                ChainRecord(
                    batch_idx=b,
                    chain_label=label,
                    positions=positions,
                    residue_indices=residue_indices,
                    window_positions=window_positions,
                    window_residue_indices=window_residue_indices,
                    gather_indices=gather_indices,
                )
            )
    return records
