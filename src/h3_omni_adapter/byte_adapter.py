"""Experimental TP1 H3 NVFP4 byte-loading adapter; no torch, CUDA or runtime imports.

This is an experiment-local contract prototype, not a drop-in vLLM-Omni loader.
Scale index equations are derived from the pinned checkpoint_quantization.py
reshape/permute contract. Source attribution is in plan/provenance.json.
"""
import json
import math
import mmap
import re
import struct
from pathlib import Path

WIDTH = {"U8": 1, "BF16": 2, "F32": 4}
FIELD = {"_nvfp4_weight": "weight", "_nvfp4_weight_scale": "scales",
         "_nvfp4_alpha": "alpha", "_weight_global_sf": "weight_global_scale",
         "_nvfp4_input_global_sf": "input_scale"}
MAX_FIXTURE_FILE = 8 * 1024 * 1024  # Scope guard for this local tiny prototype.
MAX_HEADER = 1024 * 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_header(path):
    path = Path(path)
    size = path.stat().st_size
    if not 8 <= size <= MAX_FIXTURE_FILE:
        raise ValueError("truncated or out-of-scope synthetic file")
    with path.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= count <= min(MAX_HEADER, size - 8):
            raise ValueError("invalid header length")
        metadata = json.loads(stream.read(count), object_pairs_hook=unique_object)
    if not isinstance(metadata, dict):
        raise ValueError("header must be an object")
    base = 8 + count
    entries = {}
    intervals = []
    for name, info in metadata.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError("tensor metadata must be an object")
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in WIDTH or not isinstance(shape, list):
            raise ValueError("unsupported dtype or shape")
        if any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("invalid dimension")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)):
            raise ValueError("invalid offsets")
        start, end = offsets
        length = math.prod(shape) * WIDTH[dtype]
        if not 0 <= start <= end <= size - base or end - start != length:
            raise ValueError("out-of-range offset or tensor byte-count mismatch")
        entries[name] = {"dtype": dtype, "shape": tuple(shape), "offset": base + start, "length": length}
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("overlapping or noncontiguous offsets")
        cursor = end
    if cursor != size - base:
        raise ValueError("unclaimed payload bytes")
    return entries


def scoped_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid selected filename")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source path escapes checkpoint root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("source path escapes checkpoint root")
    if not path.is_file():
        raise ValueError("selected checkpoint file is missing")
    return path


def native_target(source):
    match = re.fullmatch(r"transformer_blocks\.(\d+)\.(attn\.to_[a-z_]+|ff\.fc_(?:in|out))", source)
    if not match:
        raise ValueError("unsupported prototype projection name")
    number, suffix = match.groups()
    mapping = {
        "attn.to_q": ("attn.qkv_proj", "q"), "attn.to_k": ("attn.qkv_proj", "k"),
        "attn.to_v": ("attn.qkv_proj", "v"), "attn.to_out": ("attn.out_proj", "plain"),
        "attn.to_gate_compress": ("attn.to_gate_compress", "plain"),
        "ff.fc_in": ("mlp.fc1", "swap_halves"), "ff.fc_out": ("mlp.fc2", "plain"),
    }
    if suffix not in mapping:
        raise ValueError("unsupported projection suffix")
    target, layout = mapping[suffix]
    return f"blocks.{number}.{target}", layout


def plan_packed(header):
    grouped = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        source, separator, field = key.partition("::")
        if not separator or field not in FIELD:
            raise ValueError("unknown serialized packed field")
        grouped.setdefault(source, {})[FIELD[field]] = (key, info)
    projections = {}
    for source, tensors in grouped.items():
        if not {"weight", "scales", "alpha", "weight_global_scale"} <= tensors.keys():
            raise ValueError("incomplete projection")
        weight = tensors["weight"][1]
        shape = weight["shape"]
        if weight["dtype"] != "U8" or len(shape) != 2:
            raise ValueError("packed weight must be UINT8 matrix")
        rows, packed_columns = shape
        columns = packed_columns * 2
        if rows <= 0 or columns <= 0 or rows % 128 or columns % 64:
            raise ValueError("unsupported packed geometry")
        for field, (_, info) in tensors.items():
            expected = (rows, columns // (2 if field == "weight" else 16))
            if field in ("weight", "scales"):
                if info["dtype"] != "U8" or tuple(info["shape"]) != expected:
                    raise ValueError("packed dtype/shape mismatch")
            elif info["dtype"] not in {"BF16", "F32"} or math.prod(info["shape"]) != 1:
                raise ValueError("expected a BF16/F32 scalar")
        target, layout = native_target(source)
        projections.setdefault(target, []).append((source, layout, rows, columns, tensors))
    bindings = {}
    for target, members in projections.items():
        if target.endswith(".qkv_proj"):
            by_role = {m[1]: m for m in members}
            if len(members) != 3 or set(by_role) != {"q", "k", "v"}:
                raise ValueError("QKV must contain exactly Q, K and V")
            members = [by_role[part] for part in ("q", "k", "v")]
        elif len(members) != 1:
            raise ValueError("conflicting projection targets")
        for slot, (source, layout, rows, columns, tensors) in enumerate(members):
            prefix = f"{target}.packed.{slot}."
            for field, (key, info) in tensors.items():
                bindings[prefix + field] = {
                    "key": key, "field": field, "dtype": info["dtype"], "shape": tuple(info["shape"]),
                    "rows": rows, "columns": columns, "swap_halves": layout == "swap_halves",
                    "default": None,
                }
            if "input_scale" not in tensors:
                bindings[prefix + "input_scale"] = {
                    "key": None, "field": "input_scale", "dtype": "F32", "shape": (1,),
                    "rows": rows, "columns": columns, "swap_halves": False, "default": 1.0, "length": 4,
                }
    return bindings


def build_plan(root, *, sidecars, tp=1):
    if tp != 1:
        raise ValueError("only TP1 is qualified by this prototype")
    root = Path(root)
    indexed = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"),
                         object_pairs_hook=unique_object)["weight_map"]
    headers = {}
    selected = {}

    def header_for(path):
        if path not in headers:
            headers[path] = read_header(path)
        return headers[path]

    def add(key, path):
        info = header_for(path).get(key)
        if info is None:
            raise ValueError("index names a missing tensor")
        identity = (path, key)
        if key in selected and selected[key][0] != identity:
            raise ValueError("conflicting duplicate source tensor")
        selected[key] = (identity, info)

    for key, filename in indexed.items():
        add(key, scoped_file(root, filename))
    for filename in sidecars:
        path = scoped_file(root, filename)
        for key in header_for(path):
            add(key, path)

    packed = {key: info for key, (_, info) in selected.items() if "::" in key}
    bindings = plan_packed(packed)
    dense_names = {"proj_in.bias": "video_patch_proj.bias",
                   "time_embedder.linear_1.bias": "time_embedder.proj_in.bias"}
    for key, (_, info) in selected.items():
        if "::" not in key:
            if key not in dense_names:
                raise ValueError("dense name outside this prototype contract")
            name = dense_names[key]
            if name in bindings:
                raise ValueError("duplicate runtime binding")
            bindings[name] = {"key": key, "field": "dense", "dtype": info["dtype"], "shape": info["shape"],
                              "swap_halves": False, "default": None}
    for binding in bindings.values():
        if binding["key"] is not None:
            (path, _), info = selected[binding["key"]]
            binding.update(file=path, offset=info["offset"], length=info["length"])
    return bindings


def scale_address(row, group, groups):
    # [tile_row, tile_col, lane32, quadrant4, lane4] physical byte order.
    return (((row // 128 * (groups // 4) + group // 4) * 32 + row % 32) * 4
            + (row % 128) // 32) * 4 + group % 4


def linear_scales(data, rows, columns):
    groups = columns // 16
    if rows % 128 or columns % 64 or len(data) != rows * groups:
        raise ValueError("invalid scale layout geometry")
    return bytes(data[scale_address(r, g, groups)] for r in range(rows) for g in range(groups))


def swizzled_scales(data, rows, columns):
    groups = columns // 16
    if rows % 128 or columns % 64 or len(data) != rows * groups:
        raise ValueError("invalid scale layout geometry")
    output = bytearray(len(data))
    for row in range(rows):
        for group in range(groups):
            output[scale_address(row, group, groups)] = data[row * groups + group]
    return bytes(output)


def transform(binding, data):
    field = binding["field"]
    if field == "dense":
        return binding["dtype"], binding["shape"], data
    if field in {"weight", "scales"}:
        if field == "scales":
            data = linear_scales(data, binding["rows"], binding["columns"])
        if binding["swap_halves"]:
            mid = len(data) // 2
            data = data[mid:] + data[:mid]
        if field == "scales":
            data = swizzled_scales(data, binding["rows"], binding["columns"])
        return "U8", binding["shape"], data
    if binding["dtype"] == "BF16":
        value = struct.unpack("<f", b"\0\0" + data)[0]
    elif binding["dtype"] == "F32":
        value = struct.unpack("<f", data)[0]
    else:
        raise ValueError("unsupported scalar dtype")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("global scale must be finite and positive")
    return "F32", (1,), struct.pack("<f", value)


class MappedStager:
    """Two reusable CPU payload slots. Synchronous, unpinned; no GPU prefetch."""

    def __init__(self, plan):
        self.maps, self.handles, self.closed_maps = {}, [], []
        self.closed = False
        self.loads = self.bytes_staged = 0
        capacity = max(b["length"] for b in plan.values())
        if capacity > MAX_FIXTURE_FILE:
            raise ValueError("staging size outside prototype scope")
        self.slots = [bytearray(capacity), bytearray(capacity)]
        try:
            for path in sorted({b["file"] for b in plan.values() if b.get("default") is None}):
                handle = path.open("rb")
                self.handles.append(handle)
                self.maps[path] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

    def load(self, binding):
        if self.closed:
            raise ValueError("stager is closed")
        if binding.get("default") is not None:
            return "F32", (1,), struct.pack("<f", binding["default"])
        length, offset = binding["length"], binding["offset"]
        slot = self.slots[self.loads % 2]
        mapping = self.maps[binding["file"]]
        if length > len(slot) or offset < 0 or offset + length > len(mapping):
            raise ValueError("binding exceeds staging or file capacity")
        with memoryview(mapping) as view:
            part = view[offset:offset + length]
            try:
                slot[:length] = part
            finally:
                part.release()
        data = bytes(memoryview(slot)[:length])
        self.loads += 1
        self.bytes_staged += length
        return transform(binding, data)

    def stats(self):
        return {"slot_count": len(self.slots), "slot_capacity_bytes": len(self.slots[0]),
                "allocated_slot_bytes": sum(len(s) for s in self.slots),
                "tensor_loads": self.loads, "bytes_staged": self.bytes_staged,
                "mapped_files": len(self.maps)}

    def close(self):
        if self.closed:
            return
        self.closed_maps = list(self.maps.values())
        for mapping in self.closed_maps:
            mapping.close()
        for handle in self.handles:
            handle.close()
        self.maps.clear()
        self.handles.clear()
        self.slots.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
