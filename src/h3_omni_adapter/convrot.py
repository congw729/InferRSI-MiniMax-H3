"""Bounded CPU recovery of explicitly described INT8 ConvRot projections.

This is a model-preserving checkpoint adapter candidate, not an INT8 GEMM.
It restores quantized weights to BF16; original BF16 parity is not implied.

The regular B4 rotation and scale order were audited against Comfy Kitchen
revision e40e76d9c544b3f74c3b9ea923f7bc3d45a96372:
  comfy_kitchen/tensor/int8_utils.py:24-55
  comfy_kitchen/backends/eager/quantization.py:916-939
The implementation below independently builds the mathematical matrix by
recursive block filling. It does not import or copy a third-party kernel.

Workspace accounting covers explicitly allocated CPU tensor storage only.
Caller-owned inputs and the separately budgeted BF16 result are excluded.
BLAS/C++ internals and allocator overhead require an external RSS monitor.
A later native H3 loader's QKV reorder copy and model target are also excluded.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import torch


_REGULAR_B4 = (
    (1, 1, 1, -1),
    (1, 1, -1, 1),
    (1, -1, 1, 1),
    (-1, 1, 1, 1),
)
_MAX_ROWS_PER_CHUNK = 32
_METADATA_KEYS = frozenset(("format", "convrot", "convrot_groupsize"))


@dataclass(frozen=True)
class ConvRotSpec:
    format: str
    convrot: bool
    convrot_groupsize: int


class ConvRotCancelled(RuntimeError):
    """Raised when the caller cancels before a complete result is returned."""


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate metadata JSON key")
        result[key] = value
    return result


def _validate_fields(fields: dict[str, object]) -> ConvRotSpec:
    if set(fields) != _METADATA_KEYS:
        raise ValueError("metadata must contain exactly the three ConvRot fields")
    if type(fields["format"]) is not str or fields["format"] != "int8_tensorwise":
        raise ValueError("only explicit int8_tensorwise metadata is supported")
    if fields["convrot"] is not True:
        raise ValueError("convrot must be true")
    group_size = fields["convrot_groupsize"]
    if type(group_size) is not int or group_size not in (64, 256):
        raise ValueError("convrot_groupsize must be 64 or 256")
    return ConvRotSpec("int8_tensorwise", True, group_size)


def parse_comfy_quant(
    metadata: bytes | str | dict,
    *,
    max_metadata_bytes: int,
) -> ConvRotSpec:
    """Parse strict metadata without guessing a format or rotation group.

    For dictionary input the byte budget applies to its compact UTF-8 JSON
    representation. No tensor metadata input or alternate field is accepted.
    """
    _positive_integer(max_metadata_bytes, "max_metadata_bytes")
    if type(metadata) is dict:
        spec = _validate_fields(metadata)
        raw = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(raw) > max_metadata_bytes:
            raise ValueError("metadata exceeds explicit byte budget")
        return spec
    if type(metadata) is bytes:
        if len(metadata) > max_metadata_bytes:
            raise ValueError("metadata exceeds explicit byte budget")
        text = metadata.decode("utf-8")
    elif type(metadata) is str:
        # Check character length before allocating its UTF-8 representation.
        if len(metadata) > max_metadata_bytes or len(metadata.encode("utf-8")) > max_metadata_bytes:
            raise ValueError("metadata exceeds explicit byte budget")
        text = metadata
    else:
        raise TypeError("metadata must be bytes, str, or dict")
    fields = json.loads(text, object_pairs_hook=_unique_object)
    if type(fields) is not dict:
        raise ValueError("metadata JSON must be an object")
    return _validate_fields(fields)


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ConvRotCancelled("ConvRot recovery cancelled")


@torch.no_grad()
def recover_bf16_projection(
    weight_i8: torch.Tensor,
    weight_scale_f32: torch.Tensor,
    spec: ConvRotSpec,
    *,
    max_output_bytes: int,
    max_workspace_bytes: int,
    row_layout: str = "plain",
    qkv_heads: int | None = None,
    qkv_head_dim: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Recover a projection using bounded FP32 group tiles and a BF16 result.

    grouped_qkv means source rows [Q/K/V, head, head_dim]. Returned rows
    are [head, Q/K/V, head_dim], as required by H3's native loader.
    Scales are always selected using the original source row.

    Only successfully completed results escape. Every owned tensor local,
    including matrix construction views, is cleared in finally so retaining
    an exception traceback does not retain an incomplete output or workspace.
    """
    output = matrix = tile_input = tile_rotated = base = block = None
    try:
        _positive_integer(max_output_bytes, "max_output_bytes")
        _positive_integer(max_workspace_bytes, "max_workspace_bytes")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        _check_cancelled(cancelled)
        if type(spec) is not ConvRotSpec:
            raise TypeError("spec must be a parsed ConvRotSpec")
        _validate_fields(
            {
                "format": spec.format,
                "convrot": spec.convrot,
                "convrot_groupsize": spec.convrot_groupsize,
            }
        )
        if not isinstance(weight_i8, torch.Tensor) or not isinstance(weight_scale_f32, torch.Tensor):
            raise TypeError("source and scale must be tensors")
        if (
            weight_i8.device.type != "cpu"
            or weight_i8.layout != torch.strided
            or weight_i8.dtype != torch.int8
            or weight_i8.ndim != 2
            or not weight_i8.is_contiguous()
        ):
            raise ValueError("source must be a contiguous two-dimensional CPU INT8 tensor")
        rows, columns = weight_i8.shape
        group_size = spec.convrot_groupsize
        if rows <= 0 or columns <= 0 or columns % group_size:
            raise ValueError("positive source dimensions and whole rotation groups are required")
        if (
            weight_scale_f32.device.type != "cpu"
            or weight_scale_f32.layout != torch.strided
            or weight_scale_f32.dtype != torch.float32
            or tuple(weight_scale_f32.shape) != (rows, 1)
            or not weight_scale_f32.is_contiguous()
        ):
            raise ValueError("scale must be contiguous CPU F32 with shape [rows, 1]")
        if type(row_layout) is not str or row_layout not in ("plain", "grouped_qkv"):
            raise ValueError("unsupported source row layout")
        if row_layout == "grouped_qkv":
            _positive_integer(qkv_heads, "qkv_heads")
            _positive_integer(qkv_head_dim, "qkv_head_dim")
            if rows != 3 * qkv_heads * qkv_head_dim:
                raise ValueError("grouped QKV rows do not match heads and head dimension")
        elif qkv_heads is not None or qkv_head_dim is not None:
            raise ValueError("QKV dimensions are only accepted with grouped_qkv")

        output_bytes = rows * columns * 2
        if output_bytes > max_output_bytes:
            raise ValueError("projection exceeds explicit output budget")
        matrix_bytes = group_size * group_size * 4
        bytes_per_tile_row = group_size * 4
        affordable_rows = (max_workspace_bytes - matrix_bytes) // (2 * bytes_per_tile_row)
        if affordable_rows < 1:
            raise ValueError("workspace cannot hold the rotation matrix and one row of both tiles")
        chunk_rows = min(rows, _MAX_ROWS_PER_CHUNK, affordable_rows)
        tile_bytes = chunk_rows * bytes_per_tile_row
        workspace_peak_bytes = matrix_bytes + 2 * tile_bytes

        # Conservative domain: the largest absolute rotated dot product is at
        # most 128 * scale * sqrt(group_size). A 2x margin protects FP32
        # products/sums and the final BF16 conversion without another tensor.
        # Over-cap scales are rejected even for an all-zero source row.
        max_safe_scale = float(torch.finfo(torch.bfloat16).max) / (2.0 * 128.0 * math.sqrt(group_size))
        # Scalar reads avoid a full-size bool tensor or a float copy for validation.
        # These select views still share the caller's storage.
        for row in range(rows):
            if row % _MAX_ROWS_PER_CHUNK == 0:
                _check_cancelled(cancelled)
            scale_value = float(weight_scale_f32[row, 0])
            if not math.isfinite(scale_value) or not 0 <= scale_value <= max_safe_scale:
                raise ValueError("every scale must be finite, nonnegative and within the conservative BF16 recovery domain")

        _check_cancelled(cancelled)
        output = torch.empty((rows, columns), dtype=torch.bfloat16, device="cpu")
        matrix = torch.empty((group_size, group_size), dtype=torch.float32, device="cpu")
        matrix[:1, :1].fill_(1.0)
        size = 1
        while size < group_size:
            base = matrix[:size, :size]
            for block_row in range(4):
                for block_column in range(4):
                    if block_row == 0 and block_column == 0:
                        continue
                    block = matrix[
                        block_row * size : (block_row + 1) * size,
                        block_column * size : (block_column + 1) * size,
                    ]
                    block.copy_(base)
                    if _REGULAR_B4[block_row][block_column] < 0:
                        block.neg_()
            base = block = None
            size *= 4
        matrix.mul_(1.0 / math.sqrt(group_size))
        tile_input = torch.empty((chunk_rows, group_size), dtype=torch.float32, device="cpu")
        tile_rotated = torch.empty((chunk_rows, group_size), dtype=torch.float32, device="cpu")
        chunks_processed = 0
        for column in range(0, columns, group_size):
            for row in range(0, rows, chunk_rows):
                _check_cancelled(cancelled)
                count = min(chunk_rows, rows - row)
                tile_input[:count].copy_(weight_i8[row : row + count, column : column + group_size])
                tile_input[:count].mul_(weight_scale_f32[row : row + count])
                torch.mm(tile_input[:count], matrix.t(), out=tile_rotated[:count])
                if row_layout == "plain":
                    output[row : row + count, column : column + group_size].copy_(tile_rotated[:count])
                else:
                    for local_row in range(count):
                        source_row = row + local_row
                        projection, remainder = divmod(source_row, qkv_heads * qkv_head_dim)
                        head, dimension = divmod(remainder, qkv_head_dim)
                        native_row = (3 * head + projection) * qkv_head_dim + dimension
                        output[native_row, column : column + group_size].copy_(tile_rotated[local_row])
                chunks_processed += 1
        _check_cancelled(cancelled)
        stats = {
            "output_bytes": output_bytes,
            "workspace_peak_bytes": workspace_peak_bytes,
            "matrix_bytes": matrix_bytes,
            "tile_input_bytes": tile_bytes,
            "tile_rotated_bytes": tile_bytes,
            "max_output_bytes": max_output_bytes,
            "max_workspace_bytes": max_workspace_bytes,
            "rows_per_chunk": chunk_rows,
            "chunks_processed": chunks_processed,
            "group_size": group_size,
            "source_row_layout": row_layout,
            "workspace_scope": (
                "explicit owned CPU tensor storage; excludes caller inputs, separately budgeted BF16 output, "
                "BLAS/C++/allocator internals, and later native loader QKV copies/model targets"
            ),
            "original_bf16_parity_required": False,
            "max_safe_scale": max_safe_scale,
            "scale_domain": "BF16max/(2*128*sqrt(group)); over-cap zero rows are also rejected",
        }
        return output, stats
    finally:
        # Do not keep tensor aliases in exception frames, including construction views.
        base = block = tile_input = tile_rotated = matrix = output = None
