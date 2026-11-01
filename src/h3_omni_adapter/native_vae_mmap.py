"""Original F32 MiniMax-H3 video VAE weights, resident plus one decoder block.

The reviewed native source package is supplied by the caller. This module only
constructs the decoder and post-quant convolution; no remote code or weights are
downloaded. Native state_dict copy semantics are used without quantization.
Decoding callers must use inference mode; the loader is synchronous and TP1.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

import torch
from torch import nn

from .mmap_loader import Binding, H3MmapLoader, H3MmapPlan, Source, _header, _state


def build_native_decoder_meta(config):
    """Build only the native decoder and post_quant_conv, explicitly F32/meta."""
    from video_vae.vae_vit import ViT3DDecoder

    config = deepcopy(dict(config))
    if config.get("use_3d_conv") is not True or config.get("use_vit_decoder") is not True:
        raise ValueError("original 3D ViT VAE configuration required")
    for field in ("embed_dim", "z_channels", "out_ch"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError("positive native channel dimensions required")
    for field in ("space_down", "time_down"):
        values = config.get(field)
        if not isinstance(values, (list, tuple)) or not values or any(
            type(value) is not int or value <= 0 for value in values
        ):
            raise ValueError("explicit native spatial/temporal factors required")
    # Preserve AutoencoderKLLegacy's kwargs precedence and defaults exactly.
    options = {
        "patch_size": math.prod(config["space_down"]),
        "in_channels": config["z_channels"],
        "out_channels": config["out_ch"],
        **dict(config.get("vit_decoder_kwargs") or {}),
    }
    options.setdefault("patch_size_t", math.prod(config["time_down"]))
    options.setdefault("t_causal", config.get("causal_decoder", True))
    if type(options.get("num_layers")) is not int or options["num_layers"] <= 0:
        raise ValueError("explicit positive decoder layer count required")
    with torch.device("meta"):
        components = nn.ModuleDict({
            "decoder": ViT3DDecoder(**options),
            "post_quant_conv": nn.Conv3d(config["embed_dim"], config["z_channels"], 1),
        })
        # The native checkpoint is F32 even when the ambient default is BF16.
        components.to(dtype=torch.float32)
    components._native_vae_config = config
    return components.eval().requires_grad_(False)


def _excluded_inventory(config):
    """Describe actual omitted native modules on meta, rather than allow prefixes."""
    from video_vae.vae_cnn import EncoderFCN3D

    options = {
        "double_z": True,
        "z_channels": config["z_channels"],
        "zq_ch": config.get("zq_ch_encoder"),
        "in_channels": config.get("in_channels", 3),
        "ch": config.get("ch", 128),
        "num_res_blocks": config.get("num_res_blocks", 2),
        "ch_mult": config.get("ch_mult", [1, 2, 2, 4, 4, 8]),
        "space_down": config["space_down"],
        "time_down": config["time_down"],
        "padding_mode": config.get("padding_mode", "zeros"),
        "padding_mode_t": config.get("padding_mode_t"),
        "causal": config.get("causal_encoder", True),
        "use_t_isolated_gn": config.get("use_t_isolated_gn", False),
    }
    with torch.device("meta"):
        excluded = nn.ModuleDict({
            "encoder": EncoderFCN3D(**options),
            "quant_conv": nn.Conv3d(config["z_channels"] * 2, config["embed_dim"] * 2, 1),
        }).to(dtype=torch.float32)
    return {name: (tuple(value.shape), value.dtype)
            for name, value in excluded.state_dict().items()}


class H3NativeVAEMmapPlan(H3MmapPlan):
    def __init__(self, sources, bindings, files, model, excluded):
        super().__init__(sources, bindings, files, model)
        self.excluded = excluded
        self.groups = {}
        for name in bindings:
            match = re.match(r"^(decoder\.transformer_blocks\.\d+)\.", name)
            group = match[1] if match else "resident"
            self.groups.setdefault(group, []).append(name)

    @staticmethod
    def audit_headers(model, headers):
        """Return exact selected bindings and validated decode-only exclusions."""
        from video_vae.vae_vit import ViT3DDecoder

        if not isinstance(model, nn.ModuleDict) or set(model.keys()) != {"decoder", "post_quant_conv"}:
            raise TypeError("native decoder-only components required")
        if type(model["decoder"]) is not ViT3DDecoder or type(model["post_quant_conv"]) is not nn.Conv3d:
            raise TypeError("unmodified native decoder and post_quant_conv required")
        config = getattr(model, "_native_vae_config", None)
        if not isinstance(config, dict):
            raise ValueError("native source configuration missing")
        if model["decoder"].spatial_parallel or model["decoder"].gradient_checkpointing:
            raise ValueError("only non-checkpointed TP1 decoder supported")
        targets = _state(model)
        if not targets or any(not value.is_meta or value.dtype != torch.float32
                              for value in targets.values()):
            raise ValueError("coverage audit requires all-meta F32 targets")
        persistent = set(model.state_dict())
        default_name = "decoder.pos_embed.inv_freq"
        if set(targets) - persistent != {default_name}:
            raise ValueError("unknown native nonpersistent tensor")
        omitted = _excluded_inventory(config)
        excluded = {}
        for key, info in headers.items():
            if not isinstance(key, str) or not isinstance(info, dict):
                raise TypeError("invalid native tensor metadata")
            shape = info.get("shape")
            if not isinstance(shape, (list, tuple)) or any(
                type(dim) is not int or dim <= 0 for dim in shape
            ):
                raise ValueError("invalid native tensor shape")
            if info.get("dtype") != "F32" or type(info.get("bytes")) is not int:
                raise ValueError("native VAE checkpoint must be F32")
            if info["bytes"] != math.prod(shape) * 4:
                raise ValueError("native tensor shape and bytes disagree")
            if key in persistent:
                expected_shape, expected_dtype = tuple(targets[key].shape), targets[key].dtype
            elif key in omitted:
                expected_shape, expected_dtype = omitted[key]
                excluded[key] = "native decode-only route omits encoder/quant_conv"
            else:
                raise ValueError("unknown original VAE source tensor: " + key)
            if tuple(shape) != expected_shape or expected_dtype != torch.float32:
                raise ValueError("native VAE shape/dtype mismatch: " + key)
        missing = persistent - headers.keys()
        if missing:
            raise ValueError("incomplete original VAE checkpoint: " + sorted(missing)[0])
        bindings = {
            name: Binding(
                (name,) if name in persistent else (),
                tuple(value.shape),
                value.dtype,
                None if name in persistent else "native_vae_rope",
            )
            for name, value in targets.items()
        }
        return bindings, excluded

    @classmethod
    def build(cls, model, path, *, max_total_file_bytes):
        """Plan a single local safetensors file with an explicit whole-file budget."""
        if type(max_total_file_bytes) is not int or max_total_file_bytes <= 0:
            raise ValueError("positive explicit whole-file budget required")
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("native VAE checkpoint must be a file")
        before = path.stat()
        if before.st_size > max_total_file_bytes:
            raise ValueError("native VAE checkpoint exceeds file budget")
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        headers, digest = _header(path)
        bindings, excluded = cls.audit_headers(model, headers)
        after = path.stat()
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("native VAE checkpoint changed during preflight")
        if _header(path)[1] != digest:
            raise ValueError("native VAE checkpoint header changed during preflight")
        sources = {
            key: Source(key, path, tuple(info["shape"]), info["dtype"], info["bytes"])
            for key, info in headers.items() if key not in excluded
        }
        return cls(sources, bindings, {path: (*identity, digest)}, model, excluded)


class H3NativeVAEMmapLoader(H3MmapLoader):
    def __init__(self, model, plan, *, device, max_group_bytes):
        if not isinstance(plan, H3NativeVAEMmapPlan):
            raise TypeError("original native VAE plan required")
        super().__init__(model, plan, device=device, max_group_bytes=max_group_bytes)

    def _default(self, binding):
        if binding.default == "native_vae_rope":
            from video_vae.base_module import RotaryEmbeddingND

            config = self.model["decoder"].config
            dim = int(config.dim_head * config.rope_dim_ratio)
            with torch.device("cpu"):
                rope = RotaryEmbeddingND(dim, config.rope_theta, n_dim=3, use_angle=True)
            value = rope.inv_freq
            if tuple(value.shape) != binding.shape or value.dtype != binding.dtype:
                raise ValueError("native rotary default disagrees with meta target")
            return value.to(self.device)
        return super()._default(binding)

    def _load_native_weights(self, weights):
        loaded = set()
        targets = _state(self.model)
        for name, value in weights:
            if name in loaded or name not in self.plan.sources:
                raise ValueError("duplicate or undeclared native VAE source")
            binding = self.plan.bindings.get(name)
            if binding is None or binding.sources != (name,):
                raise ValueError("original VAE requires direct one-to-one state_dict binding")
            target = targets[name]
            if target.is_meta or target.dtype != torch.float32:
                raise ValueError("native VAE target must be materialized F32")
            parent, _, leaf = name.rpartition(".")
            owner = self.model.get_submodule(parent)
            # This invokes the native module's ordinary copy path, preserving
            # parameter identity/attributes; exact group coverage is checked below.
            incompatible = owner.load_state_dict({leaf: value}, strict=False, assign=False)
            if incompatible.unexpected_keys or leaf in incompatible.missing_keys:
                raise ValueError("native state_dict did not consume the planned tensor")
            loaded.add(name)
        return loaded

    def _expected_loaded(self, names):
        return {name for name in names if self.plan.bindings[name].sources}
