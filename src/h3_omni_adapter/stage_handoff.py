"""Bounded, digest-checked JSON+safetensors handoff; never pickle objects."""
import hashlib
import json
import math
from pathlib import Path

_LIMIT = 64 * 1024**2

def write_payload(destination, payload):
    import torch
    from safetensors.torch import save_file
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("handoff destination already exists")
    tensors = {}
    total = 0
    def encode(value, depth=0):
        nonlocal total
        if depth > 24:
            raise ValueError("handoff nesting exceeds budget")
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
            if total > _LIMIT:
                raise ValueError("handoff tensors exceed budget")
            key = f"tensor_{len(tensors):05d}"
            tensors[key] = value.detach().cpu().contiguous().clone()
            return {"type": "tensor", "key": key}
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError("handoff mapping keys must be strings")
            return {"type": "dict", "items": {k: encode(v, depth + 1) for k, v in value.items()}}
        if isinstance(value, (list, tuple)):
            return {"type": "list" if isinstance(value, list) else "tuple",
                    "items": [encode(v, depth + 1) for v in value]}
        if value is None or type(value) in (str, int, bool) or type(value) is float and math.isfinite(value):
            return {"type": "scalar", "value": value}
        raise TypeError("unsupported handoff value")
    tree = encode(payload)
    metadata = json.dumps(tree)
    if len(metadata.encode()) > 1024**2:
        raise ValueError("handoff metadata exceeds budget")
    destination.mkdir(parents=True)
    tensor_path = destination / "tensors.safetensors"
    save_file(tensors, tensor_path)
    manifest = {"schema": "h3-local-handoff/v1", "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
                "tree": tree, "logical_tensor_bytes": total}
    temporary = destination / "manifest.pending"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")

def read_payload(source):
    from safetensors.torch import load_file
    source = Path(source).resolve()
    path = source / "tensors.safetensors"
    manifest_path = source / "manifest.json"
    if path.resolve().parent != source or manifest_path.resolve().parent != source:
        raise ValueError("handoff files escape selected directory")
    if path.stat().st_size > _LIMIT + 1024**2 or manifest_path.stat().st_size > 2 * 1024**2:
        raise ValueError("handoff file budget exceeded")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "h3-local-handoff/v1":
        raise ValueError("unsupported handoff schema")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("tensor_sha256"):
        raise ValueError("handoff tensor digest mismatch")
    tensors = load_file(path, device="cpu")
    if sum(t.numel() * t.element_size() for t in tensors.values()) != manifest.get("logical_tensor_bytes"):
        raise ValueError("handoff byte count mismatch")
    used = set()
    def decode(node, depth=0):
        if depth > 24 or not isinstance(node, dict):
            raise ValueError("invalid handoff node")
        kind = node.get("type")
        if kind == "tensor":
            key = node["key"]
            if key not in tensors:
                raise ValueError("handoff tensor missing")
            used.add(key)
            return tensors[key]
        if kind == "dict":
            return {k: decode(v, depth + 1) for k, v in node["items"].items()}
        if kind in {"tuple", "list"}:
            values = [decode(v, depth + 1) for v in node["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "scalar":
            value = node["value"]
            if value is None or type(value) in (str, int, bool) or type(value) is float and math.isfinite(value):
                return value
        raise ValueError("unsupported handoff node")
    result = decode(manifest["tree"])
    if used != set(tensors):
        raise ValueError("unreferenced handoff tensor")
    return result
