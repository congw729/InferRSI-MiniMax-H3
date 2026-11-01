"""Bounded H3 runtime-layout cache for the native DLO transport.

This experimental bridge normalizes one CPU group at a time using the native
weight loader. It does not replace the generic pipeline constructor or CLI.
The cache is additional disk storage; a live RSS monitor is still required.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
import weakref
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .mmap_loader import H3MmapLoader, H3MmapPlan, _state


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _nonpersistent(model) -> set[str]:
    names = set()
    for name, _ in model.named_buffers():
        parent, _, leaf = name.rpartition(".")
        if leaf in model.get_submodule(parent)._non_persistent_buffers_set:
            names.add(name)
    return names


class H3RuntimeCache:
    """A single-model, single-bind cache receipt; not a general cache loader."""

    def __init__(self, model, directory, manifest, stats):
        self._model = weakref.ref(model)
        self.directory = directory
        self.manifest = manifest
        self.stats = stats
        self._manifest_text = _encoded(manifest)
        self._bound = False

    def bind(self, model, *, component_path: str):
        """Validate the completed cache, restore PDD buffers and emit native bindings."""
        from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan, TensorBinding

        if self._bound or self._model() is not model:
            raise ValueError("cache belongs to a different model or was already bound")
        if not isinstance(component_path, str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", component_path):
            raise ValueError("valid component module path required")
        tensors = _state(model)
        if not all(value.is_meta for value in tensors.values()):
            raise ValueError("cache binding requires the original all-meta model")
        if (self.directory / "manifest.json").stat().st_size > 4 * 1024**2:
            raise ValueError("cache manifest exceeds metadata budget")
        persisted = json.loads((self.directory / "manifest.json").read_text())
        if _encoded(persisted) != self._manifest_text or _encoded(self.manifest) != self._manifest_text:
            raise ValueError("cache manifest changed")
        if set(self.manifest["tensors"]) != set(tensors):
            raise ValueError("cache/model target names changed")
        nonpersistent = _nonpersistent(model)
        if nonpersistent != set(self.manifest["nonpersistent"]):
            raise ValueError("cache/model buffer persistence changed")
        # Verify every byte before mutating the model. This is deliberately
        # separate from DLO's header-only source compatibility checks.
        for filename, info in self.manifest["files"].items():
            path = self.directory / filename
            if path.stat().st_size != info["bytes"] or _sha(path) != info["sha256"]:
                raise ValueError("runtime cache file changed: " + filename)
        bindings = {}
        restored = {}
        for filename in self.manifest["files"]:
            with safe_open(self.directory / filename, framework="pt", device="cpu") as reader:
                expected_names = {name for name, info in self.manifest["tensors"].items()
                                  if info["file"] == filename}
                if set(reader.keys()) != expected_names:
                    raise ValueError("cache tensor key set changed")
                for name in sorted(expected_names):
                    value = reader.get_tensor(name)
                    target = tensors[name]
                    if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
                        raise ValueError("runtime cache metadata mismatch: " + name)
                    if name in nonpersistent:
                        restored[name] = value.clone()
                    else:
                        bindings[component_path + "." + name] = TensorBinding(name, str(self.directory / filename))
                    del value
        for name, value in restored.items():
            parent, _, leaf = name.rpartition(".")
            model.get_submodule(parent)._buffers[leaf] = value
        self._bound = True
        return HostWeightPlan("checkpoint_mmap", bindings, frozenset({component_path + "."}))


def export_runtime_cache(
    model,
    source_plan: H3MmapPlan,
    destination,
    *,
    max_group_bytes: int,
    max_live_bytes: int,
    max_output_bytes: int,
    min_free_bytes: int,
) -> H3RuntimeCache:
    """Normalize one resident group plus at most one streamed group on CPU.

    The byte limits below bound logical tensors and disk output, not process RSS.
    No existing cache is overwritten. An incomplete directory is retained with a
    failure receipt, and has no committed manifest.
    """
    for limit in (max_group_bytes, max_live_bytes, max_output_bytes, min_free_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("positive integer resource budgets required")
    tensors = _state(model)
    if source_plan.model_ref() is not model or not all(value.is_meta for value in tensors.values()):
        raise ValueError("exact preflighted all-meta model required")
    if set(tensors) != set(source_plan.bindings):
        raise ValueError("model target inventory changed")
    names = ["resident", *sorted(name for name in source_plan.groups if name != "resident")]
    group_sizes = {
        group: sum(math.prod(source_plan.bindings[name].shape) *
                   torch.empty((), dtype=source_plan.bindings[name].dtype, device="meta").element_size()
                   for name in source_plan.groups[group])
        for group in names
    }
    live_bytes = group_sizes["resident"] + max(
        (size for group, size in group_sizes.items() if group != "resident"), default=0)
    if max(group_sizes.values()) > max_group_bytes:
        raise ValueError("runtime group exceeds budget")
    if live_bytes > max_live_bytes:
        raise ValueError("resident plus streamed group exceeds live tensor budget")
    # Conservative JSON header allowance avoids writing past the output budget.
    upper_bound = 4 * 1024**2 + sum(group_sizes.values()) + sum(
        1024 + sum(256 + len(name.encode("utf-8")) for name in source_plan.groups[group])
        for group in names
    )
    if upper_bound > max_output_bytes:
        raise ValueError("normalized cache exceeds output budget")
    nonpersistent = _nonpersistent(model)
    if nonpersistent != {"final_layer.video_out.coefficients", "final_layer.audio_out.coefficients"}:
        raise ValueError("unsupported nonpersistent buffer contract")
    directory = Path(destination).resolve()
    if directory.exists():
        raise FileExistsError("cache destination already exists")
    if not directory.parent.is_dir():
        raise ValueError("cache parent must already exist")
    if shutil.disk_usage(directory.parent).free < min_free_bytes + upper_bound:
        raise ValueError("insufficient cache disk headroom")
    source_plan.check_files_unchanged()
    manifest = {
        "schema": "h3-native-dlo-runtime-cache/v1", "complete": True,
        "scope": "final runtime layout; no native checkpoint transforms may be reapplied",
        "nonpersistent": sorted(nonpersistent), "files": {}, "tensors": {},
        "source_headers": [{"file": path.name, "bytes": info[2], "header_sha256": info[4]}
                           for path, info in sorted(source_plan.files.items())],
    }
    loader = H3MmapLoader(model, source_plan, device="cpu", max_group_bytes=max_group_bytes)
    directory.mkdir()
    stats = {}
    try:
        loader.load_group("resident")
        for index, group in enumerate(names):
            if group != "resident":
                loader.load_group(group)
            values = {name: _state(model)[name].detach().contiguous() for name in source_plan.groups[group]}
            filename = f"group-{index:04d}.safetensors"
            path = directory / filename
            save_file(values, path)
            manifest["files"][filename] = {
                "group": group, "bytes": path.stat().st_size, "sha256": _sha(path),
            }
            for name, value in values.items():
                manifest["tensors"][name] = {
                    "file": filename, "shape": list(value.shape), "dtype": str(value.dtype),
                }
            del value, values
            if group != "resident":
                loader.release_group(group)
            gc.collect()
        if sum(info["bytes"] for info in manifest["files"].values()) > max_output_bytes:
            raise RuntimeError("serialized cache exceeded preflighted output budget")
        stats = dict(loader.stats)
        stats["logical_live_runtime_bytes_bound"] = live_bytes
        stats["logical_output_bytes"] = sum(group_sizes.values())
        stats["scope"] = "logical resident + one group; RSS and serializer scratch require external monitor"
        loader.close()
        manifest_text = _encoded(manifest) + "\n"
        manifest_bytes = len(manifest_text.encode("utf-8"))
        files_bytes = sum(info["bytes"] for info in manifest["files"].values())
        if manifest_bytes > 4 * 1024**2 or files_bytes + manifest_bytes > max_output_bytes:
            raise RuntimeError("complete cache including manifest exceeds output budget")
        stats["complete_cache_bytes"] = files_bytes + manifest_bytes
        temporary = directory / "manifest.partial.json"
        temporary.write_text(manifest_text, encoding="utf-8")
        temporary.replace(directory / "manifest.json")
    except BaseException as exc:
        try:
            (directory / "failure.json").write_text(
                _encoded({"complete": False, "error_type": type(exc).__name__}) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise
    finally:
        loader.close()
        gc.collect()
    return H3RuntimeCache(model, directory, manifest, stats)
