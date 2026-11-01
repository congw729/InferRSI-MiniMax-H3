"""Original MiniMax-H3 TP1 native dense checkpoint mmap prototype.

Preserves the upstream native grouped-QKV and gate/up weight loaders. No
FastH3 quantization, PDD, VSA gates, format fallback, or generated defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .mmap_loader import (
    Binding,
    H3MmapLoader,
    H3MmapPlan,
    Source,
    _DTYPES,
    _header,
    _scoped,
    _state,
    _unique,
)


class H3NativeMmapPlan(H3MmapPlan):
    @classmethod
    def build(cls, model, root, *, tensor_parallel_size=1, max_total_file_bytes):
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
            MiniMaxH3DiTModel,
        )

        if not isinstance(model, MiniMaxH3DiTModel):
            raise ValueError("native plan requires upstream MiniMaxH3DiTModel")
        if tensor_parallel_size != 1 or model.parallel_config.tensor_parallel_size != 1:
            raise ValueError("native mmap prototype supports only TP1")
        if model._diffusers_weights or model._serialized_weight_map or model.vsa_gates_enabled:
            raise ValueError("original native dense format required")
        if model.arch.pdd_steps is not None or getattr(model, "quant_config", None) is not None:
            raise ValueError("quantized or PDD models are outside native dense scope")
        if type(max_total_file_bytes) is not int or max_total_file_bytes <= 0:
            raise ValueError("explicit positive checkpoint file budget required")
        root = Path(root).resolve()
        index_path = _scoped(root, "model.safetensors.index.json")
        if index_path.stat().st_size > 4 * 1024**2:
            raise ValueError("index exceeds metadata budget")
        index = json.loads(index_path.read_text(), object_pairs_hook=_unique).get("weight_map")
        if not isinstance(index, dict) or not index:
            raise ValueError("nonempty native index required")
        selected = {name: _scoped(root, file) for name, file in index.items()}
        paths = set(selected.values())
        if sum(path.stat().st_size for path in paths) > max_total_file_bytes:
            raise ValueError("checkpoint exceeds explicit file budget")
        headers, files = {}, {}
        for path in sorted(paths):
            headers[path], digest = _header(path)
            stat = path.stat()
            files[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, digest)
            if {key for key, value in selected.items() if value == path} != set(headers[path]):
                raise ValueError("native index and shard contents disagree")
        targets = _state(model)
        if set(selected) != set(targets):
            raise ValueError("native checkpoint must exactly cover parameters and buffers")
        sources, bindings = {}, {}
        for name, target in targets.items():
            path = selected[name]
            info = headers[path][name]
            if info["shape"] != tuple(target.shape):
                raise ValueError("native checkpoint/runtime shape mismatch: " + name)
            # Original native checkpoint already stores the required mixed precision.
            # Reject implicit downcasts until a separately frozen protocol covers them.
            if info["dtype"] not in {"BF16", "F32"} or _DTYPES[info["dtype"]] != target.dtype:
                raise ValueError("native checkpoint/runtime dtype mismatch: " + name)
            sources[name] = Source(name, path, info["shape"], info["dtype"], info["bytes"])
            bindings[name] = Binding((name,), tuple(target.shape), target.dtype)
        return cls(sources, bindings, files, model)


class H3NativeMmapLoader(H3MmapLoader):
    def __init__(self, model, plan, *, device, max_group_bytes):
        if not isinstance(plan, H3NativeMmapPlan):
            raise ValueError("original native plan required")
        super().__init__(model, plan, device=device, max_group_bytes=max_group_bytes)

    def _expected_loaded(self, names):
        # Native RoPE is checkpoint-backed and loaded only in the resident group.
        return set(names)
