"""Original H3 mixed INT8 ConvRot checkpoint streaming through native loaders.

This opt-in TP1 adapter recovers one CPU BF16 projection at a time. It is
not an INT8 compute path. Explicit tensor budgets exclude mmap pages,
allocator/BLAS internals and the native loader's separate QKV reorder copy.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from .convrot import ConvRotCancelled, parse_comfy_quant, recover_bf16_projection
from .mmap_loader import Binding, H3MmapLoader, H3MmapPlan, Source, _group, _scoped, _state, _unique

_WIDTHS = {"I8": 1, "U8": 1, "BF16": 2, "F32": 4}
_TYPES = {"I8": torch.int8, "U8": torch.uint8, "BF16": torch.bfloat16, "F32": torch.float32}
_QUANT_SUFFIXES = (
    "adaln_proj.linear.weight", "attn.out_proj.weight", "attn.qkv_proj.weight",
    "mlp.fc1.weight", "mlp.fc2.weight",
)
_HEADER_LIMIT = 4 * 1024**2


class ConvRotMmapLoadError(RuntimeError):
    """A failed group was discarded; the original tensor-owning frames are gone."""


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _read_header(path):
    """Separate I8-capable reader; leaves existing dense/FastH3 parser unchanged."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(_HEADER_LIMIT, size - 8):
            raise ValueError("invalid safetensors header length")
        raw = stream.read(length)
    header = json.loads(raw, object_pairs_hook=_unique)
    if type(header) is not dict:
        raise ValueError("safetensors header must be an object")
    entries, intervals = {}, []
    for key, value in header.items():
        if key == "__metadata__":
            if type(value) is not dict or any(type(k) is not str or type(v) is not str for k, v in value.items()):
                raise ValueError("invalid safetensors file metadata")
            continue
        if type(key) is not str or not key or type(value) is not dict or value.get("dtype") not in _WIDTHS:
            raise ValueError("unsupported tensor header")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if type(shape) is not list or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("invalid tensor shape")
        if type(offsets) is not list or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError("invalid tensor offsets")
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - length or end - start != math.prod(shape) * _WIDTHS[value["dtype"]]:
            raise ValueError("tensor offsets disagree with shape/dtype")
        entries[key] = {"shape": tuple(shape), "dtype": value["dtype"], "bytes": end - start,
                        "offset": 8 + length + start}
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("overlapping or unclaimed tensor payload")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("unclaimed trailing payload")
    return entries, hashlib.sha256(prefix + raw).hexdigest()


def _shape(value):
    if not isinstance(value, (list, tuple)) or any(type(d) is not int or d <= 0 for d in value):
        raise ValueError("shape must contain positive integer dimensions")
    return tuple(value)


def _is_qkv(name):
    return name.endswith("attn.qkv_proj.weight") or name.endswith("attn.qkv_proj.bias")


def _describe_error(error):
    """Preserve only primitives before leaving an except block."""
    kind, message = type(error).__name__, str(error)
    mode = "cancel" if isinstance(error, ConvRotCancelled) else (
        "interrupt" if isinstance(error, KeyboardInterrupt) else (
            "exit" if isinstance(error, SystemExit) else "error"))
    exit_code = None
    if isinstance(error, SystemExit):
        # Preserve ordinary process exit semantics without retaining a tensor or
        # arbitrary caller-owned object through SystemExit.code.
        exit_code = error.code if type(error.code) in (type(None), int, str, bool, float) else message
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(e for e in (current.__context__, current.__cause__) if e is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = current.__context__ = current.__cause__ = None
    return mode, kind + ": " + message, exit_code


def _raise_safe(failure):
    mode, message, exit_code = failure
    if mode == "cancel":
        raise ConvRotCancelled(message) from None
    if mode == "interrupt":
        raise KeyboardInterrupt(message) from None
    if mode == "exit":
        raise SystemExit(exit_code) from None
    raise ConvRotMmapLoadError(message) from None


class H3NativeConvRotMmapPlan(H3MmapPlan):
    @staticmethod
    def audit_headers(headers, specs, targets, *, qkv_heads, qkv_head_dim):
        """Pure metadata audit; specs are prefix -> parsed JSON dictionaries."""
        _positive(qkv_heads, "qkv_heads")
        _positive(qkv_head_dim, "qkv_head_dim")
        if any(type(value) is not dict or not value for value in (headers, targets)) or type(specs) is not dict:
            raise ValueError("nonempty header/target maps and a spec map are required")
        clean = {}
        for key, value in headers.items():
            if type(key) is not str or not key or type(value) is not dict or value.get("dtype") not in _WIDTHS:
                raise ValueError("invalid source descriptor")
            shape = _shape(value.get("shape"))
            size = value.get("bytes")
            if type(size) is not int or size != math.prod(shape) * _WIDTHS[value["dtype"]]:
                raise ValueError("source byte count mismatch")
            clean[key] = (shape, value["dtype"], size)
        bindings, groups, group_bytes, consumed, used_specs = {}, {}, {}, set(), set()
        quantized = max_projection = max_qkv = 0
        for name, target in targets.items():
            if type(name) is not str or not name or type(target) is not dict or target.get("dtype") not in ("BF16", "F32"):
                raise ValueError("invalid target descriptor")
            shape, dtype = _shape(target.get("shape")), target["dtype"]
            if name not in clean or clean[name][0] != shape:
                raise ValueError("missing source or shape mismatch: " + name)
            source_dtype = clean[name][1]
            layout = "grouped_qkv" if _is_qkv(name) else "plain"
            if layout == "grouped_qkv" and (not shape or shape[0] != 3 * qkv_heads * qkv_head_dim):
                raise ValueError("QKV dimensions do not match architecture: " + name)
            keys, kind, group_size = [name], "dense", None
            size = math.prod(shape) * _WIDTHS[dtype]
            if source_dtype == "I8":
                if not name.startswith("blocks.") or not name.endswith(_QUANT_SUFFIXES) or len(shape) != 2 or dtype != "BF16":
                    raise ValueError("unsupported quantized target: " + name)
                prefix = name[:-7]
                if prefix not in specs:
                    raise ValueError("quantized weight has no explicit metadata")
                spec = parse_comfy_quant(specs[prefix], max_metadata_bytes=_HEADER_LIMIT)
                scale_key, meta_key = prefix + ".weight_scale", prefix + ".comfy_quant"
                if scale_key not in clean or clean[scale_key][:2] != ((shape[0], 1), "F32"):
                    raise ValueError("quantized scale must be F32 [rows, 1]")
                if meta_key not in clean or clean[meta_key][1] != "U8" or len(clean[meta_key][0]) != 1:
                    raise ValueError("quantized metadata must be one-dimensional U8")
                if shape[1] % spec.convrot_groupsize:
                    raise ValueError("projection columns must contain whole rotation groups")
                keys += [scale_key, meta_key]
                used_specs.add(prefix)
                quantized += 1
                kind, group_size = "convrot", spec.convrot_groupsize
                max_projection = max(max_projection, size)
            elif source_dtype != dtype:
                raise ValueError("dense source dtype differs from target: " + name)
            if consumed.intersection(keys):
                raise ValueError("source reused across target bindings")
            consumed.update(keys)
            if layout == "grouped_qkv":
                max_projection = max(max_projection, size)
                max_qkv = max(max_qkv, size)
            bindings[name] = {"sources": keys, "shape": list(shape), "dtype": dtype,
                              "kind": kind, "row_layout": layout, "group_size": group_size}
            group = _group(name)
            groups.setdefault(group, []).append(name)
            group_bytes[group] = group_bytes.get(group, 0) + size
        if consumed != set(headers) or used_specs != set(specs):
            raise ValueError("unconsumed/unknown checkpoint sources or ConvRot metadata")
        return {"target_count": len(targets), "source_count": len(headers),
                "quantized_count": quantized, "dense_count": len(targets) - quantized,
                "auxiliary_count": 2 * quantized, "bindings": bindings, "groups": groups,
                "group_runtime_bytes": group_bytes, "max_projection_bytes": max_projection,
                "max_native_qkv_copy_bytes": max_qkv,
                "budget_scope": "logical target/output/QKV-copy sizes; excludes mmap residency, allocator and BLAS"}

    @classmethod
    def build(cls, model, root, *, checkpoint, sidecars=(), max_total_file_bytes,
              max_metadata_bytes=4096, tensor_parallel_size=1):
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        if not isinstance(model, MiniMaxH3DiTModel) or tensor_parallel_size != 1 or model.parallel_config.tensor_parallel_size != 1:
            raise ValueError("original MiniMaxH3DiTModel TP1 required")
        if (model._diffusers_weights or model._serialized_weight_map or model.vsa_gates_enabled
                or model.arch.pdd_steps is not None or getattr(model, "quant_config", None) is not None):
            raise ValueError("original unquantized runtime architecture required")
        _positive(max_total_file_bytes, "max_total_file_bytes")
        _positive(max_metadata_bytes, "max_metadata_bytes")
        if isinstance(sidecars, (str, bytes)) or not isinstance(sidecars, (tuple, list)):
            raise ValueError("sidecars must be an explicit sequence of filenames")
        targets = _state(model)
        if not all(t.is_meta for t in targets.values()):
            raise ValueError("preflight requires an all-meta model")
        root = Path(root).resolve()
        checkpoint_path = _scoped(root, checkpoint)
        selections, index_files = {}, {}
        if checkpoint.endswith(".index.json"):
            before = _identity(checkpoint_path)
            if before[2] > _HEADER_LIMIT:
                raise ValueError("checkpoint index exceeds metadata budget")
            raw = checkpoint_path.read_bytes()
            index = json.loads(raw, object_pairs_hook=_unique)
            if type(index) is not dict or type(index.get("weight_map")) is not dict or not index["weight_map"]:
                raise ValueError("nonempty checkpoint weight_map required")
            selections = {key: _scoped(root, file) for key, file in index["weight_map"].items()}
            if any(type(key) is not str or not key for key in selections):
                raise ValueError("invalid checkpoint index tensor key")
            if _identity(checkpoint_path) != before:
                raise ValueError("checkpoint index changed while reading")
            index_files[checkpoint_path] = (*before, hashlib.sha256(raw).hexdigest())
        elif checkpoint.endswith(".safetensors"):
            selections = None
        else:
            raise ValueError("checkpoint must explicitly name a safetensors file or index")
        extra_paths = [_scoped(root, name) for name in sidecars]
        main_paths = {checkpoint_path} if selections is None else set(selections.values())
        if len(set(extra_paths)) != len(extra_paths) or set(extra_paths).intersection(main_paths):
            raise ValueError("duplicate selected checkpoint file")
        paths = main_paths | set(extra_paths)
        if sum(path.stat().st_size for path in paths | set(index_files)) > max_total_file_bytes:
            raise ValueError("selected checkpoint files exceed explicit budget")
        headers, files = {}, {}
        for path in sorted(paths):
            before = _identity(path)
            entries, digest = _read_header(path)
            if _identity(path) != before:
                raise ValueError("checkpoint changed while reading header")
            headers[path], files[path] = entries, (*before, digest)
        if selections is None:
            selections = {key: checkpoint_path for key in headers[checkpoint_path]}
        for path in extra_paths:
            for key in headers[path]:
                if key in selections:
                    raise ValueError("duplicate tensor across selected files")
                selections[key] = path
        for path in paths:
            if {key for key, selected in selections.items() if selected == path} != set(headers[path]):
                raise ValueError("index and selected shard contents disagree")
        flat = {key: headers[path][key] for key, path in selections.items()}
        specs = {}
        for key, info in flat.items():
            if not key.endswith(".comfy_quant"):
                continue
            if info["dtype"] != "U8" or len(info["shape"]) != 1 or info["bytes"] > max_metadata_bytes:
                raise ValueError("comfy_quant entry exceeds metadata format/budget")
            path = selections[key]
            with path.open("rb") as stream:
                stream.seek(info["offset"])
                raw = stream.read(info["bytes"])
            if len(raw) != info["bytes"] or _identity(path) != files[path][:4]:
                raise ValueError("checkpoint changed while reading metadata")
            spec = parse_comfy_quant(raw, max_metadata_bytes=max_metadata_bytes)
            specs[key[:-12]] = {"format": spec.format, "convrot": spec.convrot,
                               "convrot_groupsize": spec.convrot_groupsize}
        descriptors = {name: {"shape": tuple(t.shape), "dtype": {torch.bfloat16: "BF16", torch.float32: "F32"}.get(t.dtype)}
                       for name, t in targets.items()}
        audit = cls.audit_headers(flat, specs, descriptors, qkv_heads=model.arch.num_attention_heads,
                                  qkv_head_dim=model.arch.attention_head_dim)
        sources = {key: Source(key, path, flat[key]["shape"], flat[key]["dtype"], flat[key]["bytes"])
                   for key, path in selections.items()}
        bindings = {name: Binding(tuple(info["sources"]), tuple(info["shape"]), _TYPES[info["dtype"]])
                    for name, info in audit["bindings"].items()}
        plan = cls(sources, bindings, files, model)
        plan.audit, plan.specs, plan.index_files = audit, specs, index_files
        plan.qkv_heads, plan.qkv_head_dim = model.arch.num_attention_heads, model.arch.attention_head_dim
        plan.check_files_unchanged()
        return plan

    def check_files_unchanged(self):
        for path, expected in self.files.items():
            if _identity(path) != expected[:4] or _read_header(path)[1] != expected[4]:
                raise ValueError("checkpoint changed after preflight")
        for path, expected in self.index_files.items():
            if _identity(path) != expected[:4] or hashlib.sha256(path.read_bytes()).hexdigest() != expected[4]:
                raise ValueError("checkpoint index changed after preflight")
        # This detects identity/header changes, not same-identity payload tampering.


class H3NativeConvRotMmapLoader(H3MmapLoader):
    def __init__(self, model, plan, *, device, max_group_bytes, max_projection_bytes,
                 max_workspace_bytes, cancelled=None):
        if not isinstance(plan, H3NativeConvRotMmapPlan):
            raise ValueError("a native ConvRot plan is required")
        _positive(max_projection_bytes, "max_projection_bytes")
        _positive(max_workspace_bytes, "max_workspace_bytes")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("only CPU/CUDA targets are supported")
        self.poisoned = False
        super().__init__(model, plan, device=device, max_group_bytes=max_group_bytes)
        self.max_projection_bytes, self.max_workspace_bytes = max_projection_bytes, max_workspace_bytes
        self.cancelled = cancelled
        self.stats.update(projections_recovered=0, max_recovery_output_bytes=0,
                          max_recovery_workspace_bytes=0, max_dense_qkv_copy_bytes=0,
                          max_native_qkv_copy_bytes=0,
                          resident_runtime_bytes=plan.audit["group_runtime_bytes"].get("resident", 0),
                          scope="separate group/output/workspace counters; excludes mmap pages, native QKV copy, allocator/BLAS/RSS")

    def _check(self):
        super()._check()
        if self.poisoned:
            raise RuntimeError("loader is poisoned after cleanup failure; only close is permitted")

    def _cancel(self):
        if self.cancelled is not None and self.cancelled():
            raise ConvRotCancelled("ConvRot group loading cancelled")

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _restore_meta(self, group):
        for name in self.plan.groups[group]:
            parent, _, leaf = name.rpartition(".")
            owner = self.model.get_submodule(parent)
            if name in self.parameter_names:
                owner._parameters[leaf] = self.templates[name]
            else:
                owner._buffers[leaf] = self.templates[name]
        self.active.discard(group)

    def _weights(self, names):
        tensor = scale = output = reader = stack = None
        readers = {}
        try:
            with ExitStack() as stack:
                for name in names:
                    self._cancel()
                    info = self.plan.audit["bindings"][name]
                    def get(key):
                        source = self.plan.sources[key]
                        if source.path not in readers:
                            readers[source.path] = stack.enter_context(safe_open(source.path, framework="pt", device="cpu"))
                        value = readers[source.path].get_tensor(key)
                        if tuple(value.shape) != source.shape or value.dtype != _TYPES[source.dtype]:
                            raise ValueError("mapped tensor differs from preflight metadata")
                        self.stats["source_tensors_read"] += 1
                        self.stats["max_source_tensor_bytes"] = max(self.stats["max_source_tensor_bytes"], source.bytes)
                        maps = Path("/proc/self/maps")
                        if maps.is_file():
                            self.stats["mmap_observed"] |= str(source.path) in maps.read_text()
                        return value
                    tensor = get(name)
                    if info["kind"] == "convrot":
                        prefix = name[:-7]
                        scale = get(prefix + ".weight_scale")
                        spec = parse_comfy_quant(self.plan.specs[prefix], max_metadata_bytes=_HEADER_LIMIT)
                        qkv = info["row_layout"] == "grouped_qkv"
                        output, stats = recover_bf16_projection(
                            tensor, scale, spec, max_output_bytes=self.max_projection_bytes,
                            max_workspace_bytes=self.max_workspace_bytes, row_layout=info["row_layout"],
                            qkv_heads=self.plan.qkv_heads if qkv else None,
                            qkv_head_dim=self.plan.qkv_head_dim if qkv else None, cancelled=self.cancelled)
                        self.stats["projections_recovered"] += 1
                        self.stats["max_recovery_output_bytes"] = max(self.stats["max_recovery_output_bytes"], stats["output_bytes"])
                        self.stats["max_recovery_workspace_bytes"] = max(self.stats["max_recovery_workspace_bytes"], stats["workspace_peak_bytes"])
                    elif info["row_layout"] == "grouped_qkv":
                        shape = tensor.shape
                        output = tensor.reshape(3, self.plan.qkv_heads, self.plan.qkv_head_dim, *shape[1:]).permute(
                            1, 0, 2, *range(3, tensor.ndim + 2)).reshape(shape).contiguous()
                        self.stats["max_dense_qkv_copy_bytes"] = max(self.stats["max_dense_qkv_copy_bytes"], output.numel() * output.element_size())
                    else:
                        output = tensor
                    if info["row_layout"] == "grouped_qkv":
                        self.stats["max_native_qkv_copy_bytes"] = max(self.stats["max_native_qkv_copy_bytes"], output.numel() * output.element_size())
                    yield name, output
                    output = tensor = scale = None
        finally:
            tensor = scale = output = reader = stack = None
            readers.clear()

    def _load_native_weights(self, weights):
        # The upstream for-loop retains loaded_weight, its QKV reorder result,
        # and FC1 split views until its next iteration. End that native frame
        # after each whole native tensor before asking for the next recovery.
        # This intentionally incurs a native params-map traversal per tensor.
        loaded = set()
        item = None
        try:
            while True:
                try:
                    item = next(weights)
                except StopIteration:
                    break
                accepted = self.model.load_weights(iter((item,)))
                if accepted != {item[0]}:
                    raise ValueError("native loader did not consume exactly one planned tensor")
                loaded.update(accepted)
                item = None
            return loaded
        finally:
            item = None

    @torch.no_grad()
    def load_group(self, group):
        self._check()
        self._cancel()
        if group not in self.plan.groups:
            raise ValueError("unknown tensor group")
        if group in self.active:
            raise RuntimeError("group already resident")
        if group != "resident" and ("resident" not in self.active or any(g != "resident" for g in self.active)):
            raise RuntimeError("resident must precede a single streamed group")
        names = self.plan.groups[group]
        runtime_bytes = self.plan.audit["group_runtime_bytes"][group]
        if runtime_bytes > self.max_group_bytes:
            raise ValueError("group exceeds explicit allocation budget")
        for name in names:
            info = self.plan.audit["bindings"][name]
            if info["kind"] == "convrot" or info["row_layout"] == "grouped_qkv":
                if math.prod(info["shape"]) * _WIDTHS[info["dtype"]] > self.max_projection_bytes:
                    raise ValueError("projection exceeds explicit output budget")
            if info["kind"] == "convrot":
                g = info["group_size"]
                if 4 * g * g + 8 * g > self.max_workspace_bytes:
                    raise ValueError("workspace cannot hold rotation and one row of tiles")
        self.plan.check_files_unchanged()
        failure = None
        iterator = None
        self.active.add(group)
        try:
            for name in names:
                self._replace(name, self._default(self.plan.bindings[name]))
            iterator = self._weights(names)
            try:
                loaded = self._load_native_weights(iterator)
            finally:
                iterator.close()
                iterator = None
            if loaded != set(names):
                raise ValueError("native loader did not consume exactly the planned group")
            self.plan.check_files_unchanged()
            self._cancel()
            self.stats["groups_loaded"] += 1
            self.stats["max_active_streamed_groups"] = max(self.stats["max_active_streamed_groups"], sum(g != "resident" for g in self.active))
            self.stats["max_group_runtime_bytes"] = max(self.stats["max_group_runtime_bytes"], runtime_bytes)
        except BaseException as error:
            failure = _describe_error(error)
            try:
                self.release_group(group)
            except BaseException as cleanup:
                cleanup_failure = _describe_error(cleanup)
                failure = (failure[0], failure[1] + "; cleanup failure: " + cleanup_failure[1], failure[2])
        finally:
            iterator = None
        if failure is not None:
            _raise_safe(failure)

    def release_group(self, group):
        self._check()
        if group not in self.active:
            return
        failure = None
        try:
            self._sync_device()
        except BaseException as error:
            failure = _describe_error(error)
            self.poisoned = True
        finally:
            try:
                self._restore_meta(group)
            except BaseException as error:
                self.poisoned = True
                extra = _describe_error(error)
                failure = extra if failure is None else (failure[0], failure[1] + "; restore failure: " + extra[1], failure[2])
        if failure is not None:
            _raise_safe(failure)

    def install_block_hooks(self):
        self._check()
        if self.handles:
            raise RuntimeError("hooks already installed")
        for group in self.plan.groups:
            if group == "resident":
                continue
            module = self.model.get_submodule(group)
            def before(module, args, group=group):
                self.load_group(group)
            def after(module, args, output, group=group):
                try:
                    self.release_group(group)
                finally:
                    error = sys.exception()
                    if error is not None:
                        # Remove completed forward-frame locals, preserving traceback
                        # locations/type/message. This intentionally drops debug locals.
                        traceback.clear_frames(error.__traceback__)
            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after, always_call=True))

    def close(self):
        if self.closed:
            return
        H3MmapLoader._check(self)
        failures = []
        remaining_handles = []
        for handle in self.handles:
            try:
                handle.remove()
            except BaseException as error:
                failures.append(_describe_error(error))
                remaining_handles.append(handle)
        self.handles = remaining_handles
        for group in sorted(self.active, key=lambda name: name == "resident"):
            try:
                self._sync_device()
            except BaseException as error:
                failures.append(_describe_error(error))
            finally:
                try:
                    self._restore_meta(group)
                except BaseException as error:
                    failures.append(_describe_error(error))
        # A failed restore/remove must remain reachable by a later close call.
        # Ordinary loading stays prohibited after any cleanup failure.
        self.closed = not self.active and not self.handles
        if failures:
            self.poisoned = True
            # Complete the remaining owned cleanup first, then preserve process
            # control flow rather than converting an interrupt into a load error.
            # Every retained descriptor contains primitives, never exception frames.
            for failure in failures:
                if failure[0] in ("interrupt", "exit"):
                    _raise_safe(failure)
            raise ConvRotMmapLoadError("close cleanup failed: " + "; ".join(failure[1] for failure in failures)) from None
