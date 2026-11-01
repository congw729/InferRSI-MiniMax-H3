"""TP1 H3 checkpoint mmap prototype with real upstream weight-loader semantics.

This module deliberately has no ordinary-loader fallback. The file budget is a
local experiment guard, not a claim that full-model deployment is qualified.
It streams one block at a time; there is no prefetch or concurrent request support.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import weakref
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

_WIDTH = {"U8": 1, "BF16": 2, "F32": 4}
_DTYPES = {"U8": torch.uint8, "BF16": torch.bfloat16, "F32": torch.float32}

def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result

def _scoped(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid checkpoint filename")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint path escapes selected root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("checkpoint file missing or outside root")
    return path

def _header(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(4 * 1024**2, size - 8):
            raise ValueError("invalid safetensors header size")
        raw = stream.read(length)
    metadata = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(metadata, dict):
        raise TypeError("header must be an object")
    values, intervals = {}, []
    for key, info in metadata.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict) or info.get("dtype") not in _WIDTH:
            raise ValueError("unsupported tensor metadata or dtype")
        shape, offsets = info.get("shape"), info.get("data_offsets")
        if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("invalid tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError("invalid tensor offsets")
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - length or end - start != math.prod(shape) * _WIDTH[info["dtype"]]:
            raise ValueError("tensor offsets or byte count mismatch")
        values[key] = {"dtype": info["dtype"], "shape": tuple(shape), "bytes": end - start}
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("overlapping or noncontiguous tensor offsets")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("unclaimed payload bytes")
    return values, hashlib.sha256(prefix + raw).hexdigest()

def _state(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))

def _group(name):
    parts = name.split(".")
    if parts[0] == "blocks":
        return ".".join(parts[:2])
    if parts[:2] == ["token_refiner", "blocks"]:
        return ".".join(parts[:3])
    return "resident"

@dataclass(frozen=True)
class Source:
    key: str
    path: Path
    shape: tuple[int, ...]
    dtype: str
    bytes: int

@dataclass(frozen=True)
class Binding:
    sources: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    default: str | None = None

class H3MmapPlan:
    def __init__(self, sources, bindings, files, model):
        self.model_ref = weakref.ref(model)
        self.sources = sources
        self.bindings = bindings
        self.files = files
        self.groups = {}
        for name in bindings:
            self.groups.setdefault(_group(name), []).append(name)

    @classmethod
    def build(cls, model, root, *, sidecars, tensor_parallel_size=1, max_total_file_bytes=64 * 1024**2):
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import _Projection
        from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        from vllm_omni.diffusion.models.minimax_h3.pdd import MiniMaxH3PDDOutput
        if tensor_parallel_size != 1 or model.parallel_config.tensor_parallel_size != 1:
            raise ValueError("this loading plan supports only TP1")
        if not isinstance(model, MiniMaxH3DiTModel) or not model._diffusers_weights:
            raise ValueError("only upstream Diffusers-format H3 is supported")
        if not model._serialized_weight_map or not model.vsa_gates_enabled:
            raise ValueError("NVFP4 map and VSA gates must exist before planning")
        if type(max_total_file_bytes) is not int or max_total_file_bytes <= 0:
            raise ValueError("positive explicit file budget required")
        root = Path(root).resolve()
        index_path = _scoped(root, "diffusion_pytorch_model.safetensors.index.json")
        if index_path.stat().st_size > 4 * 1024**2:
            raise ValueError("index exceeds metadata budget")
        index = json.loads(index_path.read_text(), object_pairs_hook=_unique).get("weight_map")
        if not isinstance(index, dict) or not index:
            raise ValueError("nonempty explicit index required")
        selected = {key: _scoped(root, filename) for key, filename in index.items()}
        sidecar_paths = [_scoped(root, filename) for filename in sidecars]
        paths = set(selected.values()) | set(sidecar_paths)
        if sum(p.stat().st_size for p in paths) > max_total_file_bytes:
            raise ValueError("selected checkpoint exceeds experiment file budget")
        headers, files = {}, {}
        for path in sorted(paths):
            values, digest = _header(path)
            stat = path.stat()
            headers[path] = values
            files[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, digest)
        for path in sidecar_paths:
            for key in headers[path]:
                if key in selected and selected[key] != path:
                    raise ValueError("duplicate source across selected files")
                selected[key] = path
        for path in paths:
            if {k for k, p in selected.items() if p == path} != set(headers[path]):
                raise ValueError("selected file contains unindexed or missing entries")
        sources, target_parts = {}, defaultdict(list)
        serialized = model._serialized_weight_map
        targets = _state(model)
        layouts = {}
        for key, path in selected.items():
            info = headers[path].get(key)
            if info is None:
                raise ValueError("selected tensor not in header")
            sources[key] = Source(key, path, info["shape"], info["dtype"], info["bytes"])
            if key in serialized:
                target, layout = serialized[key], "packed"
            else:
                module, _, field = key.rpartition(".")
                native = _resolve_native_target(module)
                if native is None or field not in {"weight", "bias"}:
                    raise ValueError("unknown dense source tensor")
                target, layout = native[0] + "." + field, native[1]
            if target not in targets:
                raise ValueError("source maps to absent runtime tensor")
            target_parts[target].append(key)
            layouts[key] = layout
        bindings = {}
        for name, target in targets.items():
            parent_path, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent_path)
            keys = target_parts.get(name, [])
            default = None
            if not keys:
                if name == "rope.inv_freq":
                    default = "rope"
                elif isinstance(owner, _Projection) and leaf == "input_scale" and "input_scale" not in owner.spec.tensors:
                    default = "one"
                elif isinstance(owner, MiniMaxH3PDDOutput) and leaf == "coefficients":
                    if name not in {"final_layer.video_out.coefficients", "final_layer.audio_out.coefficients"}:
                        raise ValueError("unknown PDD coefficient owner")
                    default = "pdd_video" if ".video_out." in name else "pdd_audio"
                else:
                    raise ValueError("runtime tensor has no checkpoint source or known default: " + name)
            else:
                parts = [sources[k] for k in keys]
                layout_set = {layouts[k] for k in keys}
                shape = parts[0].shape
                if layout_set == {"q", "k", "v"}:
                    if len(parts) != 3 or len({p.shape[1:] for p in parts}) != 1:
                        raise ValueError("invalid QKV source set")
                    shape = (sum(p.shape[0] for p in parts), *shape[1:])
                elif len(parts) != 1:
                    raise ValueError("unsupported many-to-one checkpoint binding")
                if isinstance(owner, _Projection) and leaf not in {"weight", "scales"}:
                    if math.prod(shape) != 1:
                        raise ValueError("packed scalar must have one element")
                    shape = (1,)
                if shape != tuple(target.shape):
                    raise ValueError("checkpoint/runtime shape mismatch: " + name)
                for part in parts:
                    src_dtype = _DTYPES[part.dtype]
                    if src_dtype != target.dtype and not (src_dtype in {torch.bfloat16, torch.float32} and target.dtype in {torch.bfloat16, torch.float32}):
                        raise ValueError("unsupported dtype conversion: " + name)
                if isinstance(owner, _Projection) and leaf in {"weight", "scales"} and parts[0].dtype != "U8":
                    raise ValueError("packed tensor must be uint8")
                if ".qkv_proj." in name and not isinstance(owner, _Projection) and layout_set != {"q", "k", "v"}:
                    raise ValueError("incomplete dense QKV")
            bindings[name] = Binding(tuple(keys), tuple(target.shape), target.dtype, default)
        return cls(sources, bindings, files, model)

    def check_files_unchanged(self):
        for path, expected in self.files.items():
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity != expected[:4] or _header(path)[1] != expected[4]:
                raise ValueError("checkpoint changed after preflight")
        # Full payload hashes remain a separate model-integrity gate.

class H3MmapLoader:
    def __init__(self, model, plan, *, device, max_group_bytes):
        self.model, self.plan = model, plan
        self.device = torch.device(device)
        if type(max_group_bytes) is not int or max_group_bytes <= 0:
            raise ValueError("positive integer allocation budget required")
        self.max_group_bytes = max_group_bytes
        self.templates = _state(model)
        if set(self.templates) != set(plan.bindings) or not all(t.is_meta for t in self.templates.values()):
            raise ValueError("loader requires the exact preflighted all-meta model")
        if plan.model_ref() is not model or any(
            tuple(t.shape) != plan.bindings[n].shape or t.dtype != plan.bindings[n].dtype
            for n, t in self.templates.items()
        ):
            raise ValueError("model changed after preflight")
        self.parameter_names = set(dict(model.named_parameters()))
        self.active = set()
        self.handles = []
        self.closed = False
        self.thread = threading.get_ident()
        self.stats = {"groups_loaded": 0, "max_active_streamed_groups": 0,
                      "max_group_runtime_bytes": 0, "max_source_tensor_bytes": 0,
                      "source_tensors_read": 0, "mmap_observed": False,
                      "retained_checkpoint_tensors": 0,
                      "scope": "logical group bytes, not process RSS or full working-set bound"}

    def _check(self):
        if self.closed or threading.get_ident() != self.thread:
            raise RuntimeError("closed loader or unsupported concurrent caller")

    def _replace(self, name, tensor):
        parent_path, _, leaf = name.rpartition(".")
        owner = self.model.get_submodule(parent_path)
        template = self.templates[name]
        if name in self.parameter_names:
            value = torch.nn.Parameter.__new__(type(template), tensor, template.requires_grad)
            value.__dict__.update(template.__dict__)
            owner._parameters[leaf] = value
        else:
            owner._buffers[leaf] = tensor

    def _default(self, binding):
        if binding.default == "one":
            return torch.ones(binding.shape, dtype=binding.dtype, device=self.device)
        if binding.default == "rope":
            n = self.model.arch.rope_inv_freq_len
            return (1 / (self.model._rope_theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32, device="cpu") / (2 * n)))).to(self.device)
        if binding.default in {"pdd_video", "pdd_audio"}:
            from vllm_omni.diffusion.models.minimax_h3.pdd import pdd_head_weights
            shift = 12.0 if binding.default == "pdd_video" else 3.0
            with torch.device("cpu"):
                coefficients = torch.stack([pdd_head_weights(shift, i) for i in range(8)])
            return coefficients.to(self.device)
        return torch.empty(binding.shape, dtype=binding.dtype, device=self.device)

    @torch.no_grad()
    def load_group(self, group):
        self._check()
        if group not in self.plan.groups:
            raise ValueError("unknown tensor group")
        if group in self.active:
            raise RuntimeError("group already resident")
        if group != "resident" and "resident" not in self.active:
            raise RuntimeError("resident buffers must be initialized before blocks")
        if group != "resident" and any(g != "resident" for g in self.active):
            raise RuntimeError("one streamed group at a time")
        names = self.plan.groups[group]
        runtime_bytes = sum(self.templates[n].numel() * self.templates[n].element_size() for n in names)
        if runtime_bytes > self.max_group_bytes:
            raise ValueError("group exceeds explicit allocation budget")
        self.plan.check_files_unchanged()
        keys = sorted({key for n in names for key in self.plan.bindings[n].sources})
        self.active.add(group)
        try:
            for name in names:
                self._replace(name, self._default(self.plan.bindings[name]))
            def weights():
                with ExitStack() as stack:
                    readers = {}
                    for key in keys:
                        source = self.plan.sources[key]
                        if source.path not in readers:
                            readers[source.path] = stack.enter_context(safe_open(source.path, framework="pt", device="cpu"))
                        tensor = readers[source.path].get_tensor(key)
                        if tuple(tensor.shape) != source.shape or tensor.dtype != _DTYPES[source.dtype]:
                            raise ValueError("mapped tensor metadata differs from plan")
                        self.stats["mmap_observed"] |= any(str(source.path) in line for line in Path("/proc/self/maps").read_text().splitlines())
                        self.stats["source_tensors_read"] += 1
                        self.stats["max_source_tensor_bytes"] = max(self.stats["max_source_tensor_bytes"], source.bytes)
                        yield key, tensor
                        del tensor
            iterator = weights()
            try:
                loaded = self._load_native_weights(iterator)
            finally:
                iterator.close()
            expected = self._expected_loaded(names)
            if loaded != expected:
                raise ValueError("native loader did not consume exactly the planned group")
            self.stats["groups_loaded"] += 1
            self.stats["max_active_streamed_groups"] = max(self.stats["max_active_streamed_groups"], sum(g != "resident" for g in self.active))
            self.stats["max_group_runtime_bytes"] = max(self.stats["max_group_runtime_bytes"], runtime_bytes)
        except BaseException:
            self.release_group(group)
            raise

    def _load_native_weights(self, weights):
        return self.model.load_weights(weights)

    def _expected_loaded(self, names):
        return {n for n in names if self.plan.bindings[n].sources} | {"rope.inv_freq"}

    def release_group(self, group):
        self._check()
        if group not in self.active:
            return
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        for name in self.plan.groups[group]:
            parent, _, leaf = name.rpartition(".")
            owner = self.model.get_submodule(parent)
            if name in self.parameter_names:
                owner._parameters[leaf] = self.templates[name]
            else:
                owner._buffers[leaf] = self.templates[name]
        self.active.remove(group)

    def install_block_hooks(self):
        self._check()
        if self.handles:
            raise RuntimeError("hooks already installed")
        for group in self.plan.groups:
            if group == "resident":
                continue
            module = self.model.get_submodule(group)
            self.handles.append(module.register_forward_pre_hook(lambda module, args, group=group: self.load_group(group)))
            self.handles.append(module.register_forward_hook(lambda module, args, output, group=group: self.release_group(group), always_call=True))

    def close(self):
        if self.closed:
            return
        self._check()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for group in sorted(self.active, key=lambda value: value == "resident"):
            self.release_group(group)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

