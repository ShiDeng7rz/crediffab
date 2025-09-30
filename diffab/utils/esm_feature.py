import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from diffab.datasets.AminoAcidVocab import ORDERED_SYMBOLS

LOGGER = logging.getLogger(__name__)


def _build_aa_lookup() -> Dict[int, str]:
    table: Dict[int, str] = {}
    for idx, symbol in enumerate(ORDERED_SYMBOLS):
        # Ensure uppercase residues for ESM input; fallback to 'X' for unknowns.
        table[idx] = symbol.upper()
    return table


AA_INDEX_TO_LETTER = _build_aa_lookup()
DEFAULT_RESIDUE = "X"


def tensor_to_sequence(residue_ids: torch.Tensor) -> str:
    """Convert a 1D tensor of amino-acid indices to an ESM-compatible string."""
    if residue_ids.dim() != 1:
        raise ValueError(f"Expected 1D tensor of residue ids, got shape {tuple(residue_ids.shape)}")
    letters: List[str] = []
    for idx in residue_ids.tolist():
        letters.append(AA_INDEX_TO_LETTER.get(int(idx), DEFAULT_RESIDUE))
    return "".join(letters)


@dataclass
class ChainRecord:
    batch_idx: int
    chain_label: str | int
    positions: List[int]
    residue_indices: List[int]
    window_positions: List[int]
    window_residue_indices: List[int]
    gather_indices: List[int]


class ESMFeatureExtractor:
    """Utility to lazily load ESM models and cache per-sequence embeddings."""

    MODEL_SPECS: Dict[str, str] = {
        "esm_if1": "esm_if1_gvp4_t16_142M_UR50",
        "esm2": "esm2_t33_650M_UR50D",
        "esm": "esm1_t34_670M_UR50D",
    }

    def __init__(
            self,
            device: str | torch.device = "cpu",
            dtype: torch.dtype = torch.float32,
            max_batch_size: int = 4,
            chain_window_margin: int | None = 64,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max(1, int(max_batch_size))
        if chain_window_margin is None:
            self.chain_window_margin = 0
        else:
            self.chain_window_margin = max(0, int(chain_window_margin))
        self.models: Dict[str, torch.nn.Module] = {}
        self.alphabets = {}
        self.batch_converters = {}
        self.coord_batch_converters: Dict[str, Any] = {}
        self.model_layers: Dict[str, int] = {}
        self.model_dims: Dict[str, int] = {}
        self.cache: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    def _find_model_attribute(
            self,
            model: torch.nn.Module,
            attribute_names: Sequence[str],
            module_attributes: Sequence[str] = ("encoder", "esm", "model", "backbone", "trunk"),
    ) -> Any | None:
        """Search ``model`` and common nested modules for a given attribute.

        Some ESM wrappers (notably ``esm_if1``) expose metadata such as ``num_layers``
        or ``embed_dim`` on a nested module instead of the top-level instance.  This
        helper walks a small set of known attribute names to locate the information
        without hard-coding per-model logic.
        """

        queue: deque[Any] = deque([model])
        visited: set[int] = set()
        while queue:
            current = queue.popleft()
            identifier = id(current)
            if identifier in visited:
                continue
            visited.add(identifier)
            for attr in attribute_names:
                value = getattr(current, attr, None)
                if value is not None:
                    return value
                # Some models expose hyper-parameters through an ``args`` or ``config``
                # attribute (e.g., ``model.args.embed_dim``).  Check both mapping-like
                # and attribute-style containers as part of the search.
                for container_name in ("args", "config"):
                    container = getattr(current, container_name, None)
                    if container is None:
                        continue
                    if isinstance(container, dict):
                        if attr in container and container[attr] is not None:
                            return container[attr]
                    else:
                        value = getattr(container, attr, None)
                        if value is not None:
                            return value
            for child_name in module_attributes:
                if hasattr(current, child_name):
                    queue.append(getattr(current, child_name))
        return None

    def _ensure_model(self, key: str) -> None:
        if key in self.models:
            return
        if key not in self.MODEL_SPECS:
            raise KeyError(f"Unknown ESM model key: {key}")
        try:
            import esm
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "The 'esm' package is required to compute ESM features. Install it via `pip install fair-esm`."
            ) from exc

        loader_name = self.MODEL_SPECS[key]
        if not hasattr(esm.pretrained, loader_name):
            raise RuntimeError(f"ESM pretrained loader '{loader_name}' not found in esm.pretrained")

        loader = getattr(esm.pretrained, loader_name)
        model, alphabet = loader()
        model.eval()
        model = model.to(self.device)
        if self.dtype == torch.float16:
            model = model.half()
        elif self.dtype == torch.bfloat16:
            model = model.bfloat16()
        else:
            model = model.float()
        for param in model.parameters():
            param.requires_grad_(False)

        layer = self._find_model_attribute(model, ("num_layers",))
        if layer is None:
            raise RuntimeError(f"Unable to infer number of layers for ESM model '{key}'")
        dim = self._find_model_attribute(model, ("embed_dim", "embedding_dim", "hidden_dim", "encoder_embed_dim"))
        if dim is None:
            raise RuntimeError(f"Unable to infer embedding dimension for ESM model '{key}'")

        self.models[key] = model
        self.alphabets[key] = alphabet
        batch_converter = alphabet.get_batch_converter()
        self.batch_converters[key] = batch_converter

        coord_converter: Any | None = None
        if key == "esm_if1":
            try:  # pragma: no cover - optional dependency path
                from esm.inverse_folding.util import CoordBatchConverter  # type: ignore
            except (ImportError, AttributeError):
                LOGGER.warning(
                    "ESM-IF1 coordinate batch converter unavailable; falling back to sequence-only batches.",
                )
            else:
                constructor_attempts = (
                    lambda: CoordBatchConverter(alphabet),
                    lambda: CoordBatchConverter(alphabet, device=self.device),
                )
                for build in constructor_attempts:
                    try:
                        coord_converter = build()
                        break
                    except TypeError:
                        continue
                if coord_converter is None:
                    LOGGER.warning(
                        "Unable to instantiate ESM-IF1 CoordBatchConverter; structure features will be skipped.",
                    )
        self.coord_batch_converters[key] = coord_converter
        self.model_layers[key] = int(layer)
        self.model_dims[key] = int(dim)

    def get_model_dim(self, key: str) -> int | None:
        return self.model_dims.get(key)

    def embed_batch(self, key: str, sequences: Sequence[str]) -> List[torch.Tensor]:
        """Embed a batch of sequences with the specified model key."""
        if not sequences:
            return []
        self._ensure_model(key)
        outputs: List[torch.Tensor | None] = [self.cache[key].get(seq) for seq in sequences]
        missing: List[Tuple[int, str]] = [
            (idx, seq) for idx, (seq, cached) in enumerate(zip(sequences, outputs)) if cached is None
        ]
        if missing:
            converter = self.batch_converters[key]
            layer = self.model_layers[key]
            model = self.models[key]
            # Process in mini-batches to control memory usage.
            for start in range(0, len(missing), self.max_batch_size):
                chunk = missing[start:start + self.max_batch_size]
                if not chunk:
                    continue
                labels = [f"seq_{i}" for i, _ in chunk]
                chunk_seqs = [seq for _, seq in chunk]
                batch_inputs = list(zip(labels, chunk_seqs))
                _, _, tokens = converter(batch_inputs)
                tokens = tokens.to(self.device)
                with torch.no_grad():
                    result = model(tokens, repr_layers=[layer], return_contacts=False)
                reps = result["representations"][layer].to(torch.float32).cpu()
                for row, (out_idx, seq) in enumerate(chunk):
                    length = len(seq)
                    emb = reps[row, 1:length + 1].contiguous()
                    self.cache[key][seq] = emb
                    outputs[out_idx] = emb
        return [tensor.contiguous() for tensor in outputs]  # type: ignore[arg-type]

    def embed_batch_from_tensor(
            self,
            aa: torch.Tensor,
            mask: torch.Tensor,
            key: str,
            coords: torch.Tensor | None = None,
            coord_mask: torch.Tensor | None = None,
            chain_ids: Sequence[Sequence[Any]] | torch.Tensor | None = None,
            residue_index: Sequence[Sequence[Any]] | torch.Tensor | None = None,
            chain_window_margin: int | None = None,
    ) -> torch.Tensor:
        """Compute per-residue embeddings for a batch of padded tensors.

        Args:
            aa: Amino-acid indices shaped ``[B, L]``.
            mask: Boolean mask with the same shape as ``aa``.
            key: Which pretrained ESM variant to query.
            coords: Optional heavy-atom coordinates ``[B, L, A, 3]``.
            coord_mask: Optional heavy-atom mask ``[B, L, A]`` aligned with ``coords``.
            chain_ids: Optional per-residue chain identifiers used to split multi-chain
                complexes before running ESM. Accepts either a tensor of shape
                ``[B, L]`` or a nested sequence mirroring the batch layout.
            residue_index: Optional per-residue numbering used to order residues
                within each chain. Falls back to the tensor index when omitted.
        """
        if aa.dim() != 2:
            raise ValueError(f"Expected aa tensor of shape [B, L], got {tuple(aa.shape)}")
        if mask.shape != aa.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} must match amino acid tensor shape {tuple(aa.shape)}"
            )
        if not mask.any():
            LOGGER.debug("All residues masked for key %s; skipping ESM extraction.", key)
            # Return an empty tensor to signal that no embeddings were produced.
            return torch.zeros(0, dtype=torch.float32)

        aa_cpu = aa.detach().cpu()
        mask_cpu = mask.detach().cpu().bool()
        coords_cpu = coords.detach().cpu() if coords is not None else None
        coord_mask_cpu = coord_mask.detach().cpu().bool() if coord_mask is not None else None
        chain_rows = self._normalize_chain_annotations(chain_ids, aa_cpu.shape) if chain_ids is not None else None
        residue_rows = (
            self._normalize_nested(residue_index, aa_cpu.shape, "residue_index")
            if residue_index is not None
            else None
        )
        self._ensure_model(key)
        dim = self.model_dims[key]
        B, L = aa_cpu.shape
        outputs = torch.zeros(B, L, dim, dtype=torch.float32)
        margin = self._resolve_window_margin(chain_window_margin)

        if chain_rows is not None:
            return self._embed_with_chain_split(
                key,
                aa_cpu,
                mask_cpu,
                outputs,
                chain_rows,
                residue_rows,
                coords_cpu,
                coord_mask_cpu,
                margin,
            )

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
            return outputs

        embeddings = self.embed_batch(key, sequences)
        for (b, valid), emb in zip(owners, embeddings):
            if emb.size(0) != int(valid.sum().item()):
                raise RuntimeError(
                    f"Mismatch between embedding length {emb.size(0)} and valid residues {int(valid.sum().item())}"
                )
            outputs[b, valid] = emb
        return outputs

    def _normalize_nested(
            self,
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
                # Attempt to reshape a flattened list.
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
            self,
            chain_ids: Sequence[Sequence[Any]] | torch.Tensor,
            shape: Tuple[int, int],
    ) -> List[List[Any]]:
        rows = self._normalize_nested(chain_ids, shape, "chain_ids")
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

    def _resolve_window_margin(self, override: int | None) -> int:
        if override is None:
            return self.chain_window_margin
        return max(0, int(override))

    def _collect_chain_records(
            self,
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

    def _embed_sequences_for_records(
            self,
            key: str,
            records: List[ChainRecord],
            aa: torch.Tensor,
            outputs: torch.Tensor,
    ) -> torch.Tensor:
        if not records:
            return outputs
        sequences: List[str] = []
        owners: List[ChainRecord] = []
        for record in records:
            seq_positions = record.window_positions if record.window_positions else record.positions
            seq_tensor = aa[record.batch_idx][seq_positions]
            seq = tensor_to_sequence(seq_tensor)
            if not seq:
                continue
            sequences.append(seq)
            owners.append(record)
        if not owners:
            return outputs
        embeddings = self.embed_batch(key, sequences)
        for record, emb in zip(owners, embeddings):
            seq_positions = record.window_positions if record.window_positions else record.positions
            expected_len = len(seq_positions)
            if emb.size(0) != expected_len:
                raise RuntimeError(
                    "Mismatch between chain embedding length "
                    f"{emb.size(0)} and residue count {expected_len}",
                )
            gather_idx = record.gather_indices if record.gather_indices else list(range(expected_len))
            gathered = emb[gather_idx]
            if gathered.size(0) != len(record.positions):
                raise RuntimeError(
                    "Mismatch between gathered embedding length "
                    f"{gathered.size(0)} and target positions {len(record.positions)}",
                )
            outputs[record.batch_idx, record.positions] = gathered
        return outputs

    def _embed_with_chain_split(
            self,
            key: str,
            aa: torch.Tensor,
            mask: torch.Tensor,
            outputs: torch.Tensor,
            chain_rows: List[List[Any]],
            residue_rows: List[List[Any]] | None,
            coords: torch.Tensor | None,
            coord_mask: torch.Tensor | None,
            window_margin: int,
    ) -> torch.Tensor:
        records = self._collect_chain_records(aa, mask, chain_rows, residue_rows, window_margin)
        if not records:
            return outputs

        return self._embed_sequences_for_records(key, records, aa, outputs)


def add_esm_features_to_batch(
        data: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        extractor: ESMFeatureExtractor,
        model_keys: Iterable[str],
) -> None:
    """Populate the given data dictionary with ESM features for each requested key."""
    if extractor is None:
        return
    aa = data.get("aa")
    if aa is None or aa.numel() == 0:
        return
    if mask is None:
        raise ValueError("Mask tensor is required to compute ESM features")
    mask_bool = mask.bool()
    coords = data.get("pos_heavyatom")
    coord_mask = data.get("mask_heavyatom")
    chain_ids = data.get("chain_nb") if data.get("chain_nb") is not None else data.get("chain_id")
    residue_index = data.get("res_nb") if data.get("res_nb") is not None else data.get("resseq")
    for key in model_keys:
        if key in data and isinstance(data[key], torch.Tensor):
            # Already populated; skip.
            continue
        try:
            embeddings = extractor.embed_batch_from_tensor(
                aa,
                mask_bool,
                key,
                coords=coords,
                coord_mask=coord_mask,
                chain_ids=chain_ids,
                residue_index=residue_index,
            )
        except RuntimeError as exc:
            LOGGER.warning("Failed to compute %s features: %s", key, exc)
            continue
        if embeddings.numel() == 0:
            continue
        data[key] = embeddings
