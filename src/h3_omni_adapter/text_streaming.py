"""Strict TP1 pure-text H3 NVFP4 streaming using the native text model and loaders.

Vision is excluded explicitly; this is a model-object prototype, not the generic
vLLM-Omni loader or pipeline entry. Only inference without concurrent requests.
"""
from __future__ import annotations

import weakref
from collections import defaultdict
from pathlib import Path

import torch

from .mmap_loader import (
    _DTYPES,
    Binding,
    H3MmapLoader,
    H3MmapPlan,
    Source,
    _header,
    _scoped,
    _state,
)


class H3TextOnly(torch.nn.Module):
    def __init__(self, config, encoder_group, quant_config, *, max_tokens):
        super().__init__()
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            MiniMaxH3Qwen3VLTextModel,
            _tp_range,
        )
        if _tp_range(encoder_group) != (0, 1):
            raise ValueError("text-only streaming supports TP1")
        if not isinstance(quant_config, FastH3NVFP4Config) or quant_config.rank != 0:
            raise ValueError("explicit native FastH3 NVFP4 TP1 configuration required")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("explicit positive token budget required")
        self.config = config
        self.encoder_group = encoder_group
        self.quant_config = quant_config
        self.max_tokens = max_tokens
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self.spatial_merge_size = int(config.vision_config.spatial_merge_size)
        with torch.device("meta"):
            self.text_model = MiniMaxH3Qwen3VLTextModel(
                encoder_group, config.text_config, MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
                dtype=torch.bfloat16, quant_config=quant_config,
            )
        self.eval()

    @torch.inference_mode()
    def encode_ids(self, input_ids, **multimodal):
        from vllm_omni.diffusion.models.minimax_h3.encoder import _get_rope_index
        if multimodal:
            raise ValueError("text-only entry does not accept pixel/grid inputs")
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
            raise TypeError("input IDs must be a torch.long tensor")
        if input_ids.is_meta or input_ids.ndim != 1 or not 0 < input_ids.numel() <= self.max_tokens:
            raise ValueError("nonempty 1D input IDs within explicit token budget required")
        if bool(((input_ids < 0) | (input_ids >= self.config.text_config.vocab_size)).any()):
            raise ValueError("input ID outside vocabulary")
        if bool(((input_ids == self.image_token_id) | (input_ids == self.video_token_id)).any()):
            raise ValueError("multimodal placeholder tokens are unsupported")
        embedding = self.text_model.embed_tokens.weight
        if embedding.is_meta:
            raise RuntimeError("text embedding is not resident")
        ids = input_ids.to(device=embedding.device)[None]
        positions, _ = _get_rope_index(
            ids, mm_token_type_ids=torch.zeros_like(ids, dtype=torch.int32),
            image_grid_thw=None, video_grid_thw=None,
            attention_mask=torch.ones_like(ids), spatial_merge_size=self.spatial_merge_size,
        )
        hidden = self.text_model(self.text_model.embed_tokens(ids), positions)[0]
        if tuple(hidden.shape) != (input_ids.numel(), self.config.text_config.hidden_size):
            raise ValueError("unexpected native text output shape")
        # Native H3 consumes the retained layer output without a final norm.
        return hidden.to(device="cpu", dtype=torch.bfloat16)


class H3TextMmapPlan(H3MmapPlan):
    def __init__(self, sources, bindings, files, model, routes, excluded):
        self.model_ref = weakref.ref(model)
        self.sources, self.bindings, self.files = sources, bindings, files
        self.routes, self.excluded = routes, excluded
        self.groups = {}
        for name in bindings:
            group = ".".join(name.split(".")[:3]) if name.startswith("text_model.layers.") else "resident"
            self.groups.setdefault(group, []).append(name)

    @classmethod
    def build(cls, model, root, *, files, tensor_parallel_size=1, max_total_file_bytes=64*1024**2):
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import _Projection
        from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder, _tp_range
        if not isinstance(model, H3TextOnly):
            raise TypeError("actual H3TextOnly model required")
        if tensor_parallel_size != 1 or _tp_range(model.encoder_group) != (0, 1):
            raise ValueError("TP1 only")
        if type(max_total_file_bytes) is not int or max_total_file_bytes <= 0:
            raise ValueError("positive explicit file budget required")
        if not isinstance(files, (list, tuple)) or not files or len(set(files)) != len(files):
            raise ValueError("explicit nonempty unique checkpoint file list required")
        root = Path(root).resolve()
        paths = [_scoped(root, name) for name in files]
        if len(set(paths)) != len(paths):
            raise ValueError("aliased checkpoint files")
        if sum(path.stat().st_size for path in paths) > max_total_file_bytes:
            raise ValueError("checkpoint exceeds explicit file budget")
        targets = _state(model)
        if not all(t.is_meta for t in targets.values()):
            raise ValueError("preflight requires all-meta text model")
        sources, identities, routes, excluded = {}, {}, {}, {}
        parts = defaultdict(list)
        seen = set()
        for path in paths:
            header, digest = _header(path)
            stat = path.stat()
            identities[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, digest)
            for key, info in header.items():
                if key in seen:
                    raise ValueError("duplicate checkpoint source")
                seen.add(key)
                if key.startswith("model.visual."):
                    excluded[key] = "pure-text route does not construct vision"
                    continue
                if key in {"model.language_model.norm.weight", "lm_head.weight"}:
                    excluded[key] = "native retained hidden-state route omits final norm and LM head"
                    continue
                if key in model.quant_config.weight_map:
                    target, shard = model.quant_config.weight_map[key], None
                else:
                    mapped = MiniMaxH3Qwen3VLEncoder._map_weight_name(model, key)
                    if mapped is None:
                        raise ValueError("unknown or unselected text checkpoint source")
                    target, shard = mapped
                if target not in targets or not target.startswith("text_model."):
                    raise ValueError("checkpoint source has no selected text target")
                # Released NVFP4 text checkpoints store each projection separately.
                # Dense fused QKV/gate-up is a different route requiring its own plan.
                if shard is not None:
                    raise ValueError("dense fused projections are outside this NVFP4 plan")
                sources[key] = Source(key, path, info["shape"], info["dtype"], info["bytes"])
                routes[key] = (target, shard)
                parts[target].append(key)
        bindings = {}
        for name, target in targets.items():
            parent, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent)
            keys = parts.get(name, [])
            default = None
            if not keys:
                if isinstance(owner, _Projection) and leaf == "input_scale" and leaf not in owner.spec.tensors:
                    default = "one"
                else:
                    raise ValueError("missing text parameter or buffer: " + name)
            else:
                if len(keys) != 1:
                    raise ValueError("unexpected many-to-one NVFP4 binding")
                src = sources[keys[0]]
                shape = src.shape
                if isinstance(owner, _Projection) and leaf not in {"weight", "scales"}:
                    if src.bytes != 4 or _DTYPES[src.dtype] != torch.float32:
                        raise ValueError("packed global scale must be one FP32 element")
                    shape = (1,)
                if shape != tuple(target.shape) or _DTYPES[src.dtype] != target.dtype:
                    raise ValueError("text checkpoint shape or dtype mismatch: " + name)
            bindings[name] = Binding(tuple(keys), tuple(target.shape), target.dtype, default)
        return cls(sources, bindings, identities, model, routes, excluded)


class H3TextMmapLoader(H3MmapLoader):
    def _load_native_weights(self, weights):
        from vllm_omni.diffusion.models.minimax_h3.encoder import _default_weight_loader
        params = dict(self.model.named_parameters())
        loaded = set()
        seen = set()
        for key, tensor in weights:
            if key in seen or key not in self.plan.routes:
                raise ValueError("duplicate or undeclared text source")
            seen.add(key)
            name, shard = self.plan.routes[key]
            if name in loaded:
                raise ValueError("text target loaded twice")
            parameter = params[name]
            loader = getattr(parameter, "weight_loader", _default_weight_loader)
            loader(parameter, tensor, shard)
            loaded.add(name)
        return loaded

    def _expected_loaded(self, names):
        return {n for n in names if self.plan.bindings[n].sources}
