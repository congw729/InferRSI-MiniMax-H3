"""Experimental compact FastH3 VAE construction; no full native VAE allocation.

Uses the unmodified pinned native ViT and upstream vLLM-Omni packed modules.
The caller supplies an already reviewed, locally available native source package.
The loader is single-use, synchronous and bounded; it does not download files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .mmap_loader import _DTYPES, _header, _state


def build_compact_decoder_meta(config, header):
    from video_vae.vae_vit import ViT3DDecoder
    from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import (
        FastH3NVFP4Decoder,
        PackedNVFP4Linear,
        _QKVProjection,
        _SwiGLU,
    )
    metadata = header.get("__metadata__", {})
    if metadata.get("format") != "fastvideo_h3_decoder_nvfp4_packed_v1":
        raise ValueError("unsupported decoder format")
    recipe = json.loads(metadata.get("nvfp4_metadata", "{}"))
    expected = {"block_size": 16, "act_scale": "unit", "rotation_group": None, "skip_blocks": []}
    if any(recipe.get(k, object()) != v for k, v in expected.items()):
        raise ValueError("unsupported decoder quantization metadata")
    depth = recipe.get("student_layers")
    native_kwargs = dict(config["vit_decoder_kwargs"])
    if type(depth) is not int or not 1 <= depth <= native_kwargs["num_layers"]:
        raise ValueError("invalid compact decoder depth")
    if recipe.get("num_linears") != depth * 6:
        raise ValueError("decoder must contain six packed projections per block")
    native_kwargs["num_layers"] = depth
    native_kwargs.update(patch_size=math.prod(config["space_down"]),
                         patch_size_t=math.prod(config["time_down"]),
                         in_channels=config["z_channels"], out_channels=config["out_ch"],
                         t_causal=config["causal_decoder"])
    with torch.device("meta"):
        original = ViT3DDecoder(**native_kwargs)
        decoder = FastH3NVFP4Decoder(original, depth)
        post = nn.Conv3d(config["embed_dim"], config["z_channels"], 1)
    components = nn.ModuleDict({"decoder": decoder, "post_quant_conv": post})
    remap = {}
    fields = ("packed_weight", "weight_inv_scale", "weight_global_sf", "bias", "input_amax")
    def linear(source, target, rows, width):
        packed = []
        for field in fields:
            key = source + "." + field
            if key not in header:
                raise ValueError("missing packed tensor: " + key)
            info = header[key]
            if info["dtype"] not in _DTYPES:
                raise ValueError("unsupported packed tensor dtype")
            packed.append(torch.empty(info["shape"], dtype=_DTYPES[info["dtype"]], device="meta"))
            remap[key] = target + "." + field
        layer = PackedNVFP4Linear(*packed)
        if (layer.out_features, layer.in_features) != (rows, width):
            raise ValueError("packed projection does not match native architecture")
        return layer
    for i, block in enumerate(decoder.transformer_blocks):
        prefix = f"decoder.transformer_blocks.{i}"
        qkv = block.attn.to_qkv
        block.attn.to_qkv = _QKVProjection(
            *(linear(f"{prefix}.attn.to_{part}", f"{prefix}.attn.to_qkv.{part}",
                     qkv.out_features // 3, qkv.in_features) for part in ("q", "k", "v")),
            head_dim=block.attn.dim_head)
        out = block.attn.to_out
        block.attn.to_out = linear(prefix + ".attn.to_out.0", prefix + ".attn.to_out",
                                   out.out_features, out.in_features)
        w1, w2 = block.ff.w1, block.ff.w2
        block.ff = nn.Sequential(
            linear(prefix + ".ff.net.0.proj", prefix + ".ff.0", w1.out_features, w1.in_features),
            _SwiGLU(),
            linear(prefix + ".ff.net.2", prefix + ".ff.2", w2.out_features, w2.in_features))
    state = components.state_dict()
    mapping = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        target = remap.get(key, key)
        if target not in state or target in mapping.values():
            raise ValueError("unexpected or duplicate decoder tensor: " + key)
        value = state[target]
        if list(value.shape) != list(info["shape"]) or value.dtype != _DTYPES.get(info["dtype"]):
            raise ValueError("decoder target shape or dtype mismatch: " + key)
        mapping[key] = target
    if set(mapping.values()) != set(state):
        raise ValueError("incomplete compact decoder checkpoint")
    defaults = {k: "reciprocal stored weight_global_sf" if k.endswith(".alpha")
                else "ones_like stored weight_global_sf" if k.endswith(".input_global_scale")
                else "native RotaryEmbeddingND constructor formula" if k == "decoder.rope.inv_freq"
                else "UNKNOWN"
                for k in _state(components).keys() - state.keys()}
    if "UNKNOWN" in defaults.values():
        raise ValueError("unaccounted decoder buffer")
    return components, mapping, defaults


def load_compact_decoder_mmap(config, path, *, device, max_file_bytes=64 * 1024**2):
    """Copy one source tensor at a time into final tensors; rebuild nonpersistent buffers."""
    from video_vae.base_module import RotaryEmbeddingND
    from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import PackedNVFP4Linear

    path = Path(path).resolve(strict=True)
    if type(max_file_bytes) is not int or max_file_bytes <= 0 or path.stat().st_size > max_file_bytes:
        raise ValueError("decoder checkpoint exceeds explicit experiment budget")
    info, digest = _header(path)
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        header = {**info, "__metadata__": checkpoint.metadata() or {}}
        components, mapping, defaults = build_compact_decoder_meta(config, header)
        if _header(path)[1] != digest:
            raise ValueError("decoder header changed during preflight")
        after = path.stat()
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("decoder file changed during preflight")
        for source, target in mapping.items():
            value = checkpoint.get_tensor(source)
            # Scalar positivity is required before derived alpha or CUDA execution.
            if source.endswith(".weight_global_sf") and (not torch.isfinite(value).all() or not (value > 0).all()):
                raise ValueError("invalid decoder weight_global_sf")
            parent_name, _, leaf = target.rpartition(".")
            parent = components.get_submodule(parent_name)
            tensor = value.to(device=device, copy=True)
            if leaf in parent._parameters:
                parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
            else:
                parent._buffers[leaf] = tensor
            del tensor, value
    for module in components.modules():
        if isinstance(module, PackedNVFP4Linear):
            module.input_global_scale = torch.ones_like(module.weight_global_sf)
            module.alpha = module.weight_global_sf.reciprocal()
    opts = config["vit_decoder_kwargs"]
    rope_dim = int(opts["dim_head"] * opts["rope_dim_ratio"])
    with torch.device("cpu"):
        rope = RotaryEmbeddingND(rope_dim, opts["rope_theta"], n_dim=3, use_angle=True)
    components["decoder"].rope.inv_freq = rope.inv_freq.to(device)
    if any(t.device.type == "meta" for t in _state(components).values()):
        raise RuntimeError("compact decoder has unmaterialized tensors")
    return components.eval(), {"targets": len(_state(components)), "source_keys": len(mapping),
                               "defaults": defaults, "header_sha256": digest,
                               "file_bytes": path.stat().st_size,
                               "full_native_vae_constructed": False}
