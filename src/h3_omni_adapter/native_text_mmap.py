"""Original MiniMax-H3 BF16 pure-text mmap loading, TP1 and one layer at a time.

The native text model, source-name mapping and parameter weight loaders remain
the computation authority. This prototype is separate from generic Omni startup.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

from .mmap_loader import Binding, H3MmapLoader, Source, _header, _scoped, _state, _unique
from .text_streaming import H3TextMmapPlan, H3TextOnly


class H3NativeTextOnly(torch.nn.Module):
    def __init__(self, config, encoder_group, *, max_tokens):
        super().__init__()
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            MiniMaxH3Qwen3VLTextModel,
            _tp_range,
        )

        if _tp_range(encoder_group) != (0, 1):
            raise ValueError("original text mmap supports TP1 only")
        if getattr(config, "quantization_config", None):
            raise ValueError("original dense text configuration required")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("explicit positive token budget required")
        if int(config.text_config.num_hidden_layers) <= 0:
            raise ValueError("at least one text layer required")
        self.config = config
        self.encoder_group = encoder_group
        self.quant_config = None
        self.max_tokens = max_tokens
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self.spatial_merge_size = int(config.vision_config.spatial_merge_size)
        with torch.device("meta"):
            self.text_model = MiniMaxH3Qwen3VLTextModel(
                encoder_group, config.text_config, MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
                dtype=torch.bfloat16, quant_config=None,
            )
            self.text_model.to(dtype=torch.bfloat16)
        self.eval()

    @torch.inference_mode()
    def encode_ids(self, input_ids, **multimodal):
        # Match the upstream public encoder's temporary SDPA setting. The shared
        # pure-text body uses the native position builder and text-model forward.
        previous = torch.backends.cuda.cudnn_sdp_enabled()
        torch.backends.cuda.enable_cudnn_sdp(True)
        try:
            return H3TextOnly.encode_ids(self, input_ids, **multimodal)
        finally:
            torch.backends.cuda.enable_cudnn_sdp(previous)


class H3NativeTextMmapPlan(H3TextMmapPlan):
    @staticmethod
    def audit_headers(model, headers):
        """Return exact bindings/routes/exclusions without opening payloads."""
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            MiniMaxH3Qwen3VLEncoder,
            MiniMaxH3Qwen3VLMergedColumnParallelLinear,
            MiniMaxH3Qwen3VLQKVParallelLinear,
            _fused_source_shards,
            _tp_range,
        )

        if not isinstance(model, H3NativeTextOnly) or model.quant_config is not None:
            raise TypeError("original H3NativeTextOnly required")
        if _tp_range(model.encoder_group) != (0, 1):
            raise ValueError("TP1 only")
        targets = _state(model)
        if not targets or not all(t.is_meta for t in targets.values()):
            raise ValueError("coverage audit requires an all-meta model")
        parts, routes, excluded = defaultdict(list), {}, {}
        retained = len(model.text_model.layers)
        total_layers = int(model.config.text_config.num_hidden_layers)
        suffixes = {
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        }
        for key, info in headers.items():
            if info["dtype"] != "BF16":
                raise ValueError("original text checkpoint must be BF16")
            if any(type(dim) is not int or dim <= 0 for dim in info["shape"]):
                raise ValueError("invalid source shape")
            if info["bytes"] != math.prod(info["shape"]) * 2:
                raise ValueError("source shape and bytes disagree")
            if key.startswith("model.visual."):
                excluded[key] = "explicit pure-text route omits vision"
                continue
            if key in {"lm_head.weight", "model.language_model.norm.weight"}:
                excluded[key] = "native unnormalized layer-50 output omits head/final norm"
                continue
            layer = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", key)
            if layer:
                index = int(layer[1])
                if index >= total_layers or layer[2] not in suffixes:
                    raise ValueError("unknown text layer/source")
                if index >= retained:
                    excluded[key] = "beyond native selected layer"
                    continue
            mapped = MiniMaxH3Qwen3VLEncoder._map_weight_name(model, key)
            if mapped is None or mapped[0] not in targets:
                raise ValueError("unknown or missing native text target: " + key)
            routes[key] = mapped
            parts[mapped[0]].append((key, mapped[1]))
        bindings = {}
        for name, target in targets.items():
            parent, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent)
            expected = _fused_source_shards(owner) if leaf == "weight" else None
            selected = parts.get(name, [])
            if expected:
                shard_ids = [part[1] for part in selected]
                if len(shard_ids) != len(expected) or set(shard_ids) != {s for _, s in expected}:
                    raise ValueError("incomplete/duplicate fused source set: " + name)
                shapes = {}
                if isinstance(owner, MiniMaxH3Qwen3VLQKVParallelLinear):
                    q = owner.local_num_heads * owner.head_dim
                    kv = owner.local_num_kv_heads * owner.head_dim
                    shapes = {s: (rows, owner.hidden_size) for s, rows in (("q", q), ("k", kv), ("v", kv))}
                elif isinstance(owner, MiniMaxH3Qwen3VLMergedColumnParallelLinear):
                    shapes = {s: (owner.intermediate_size_per_partition, target.shape[1]) for _, s in expected}
                else:
                    raise ValueError("unsupported fused parameter owner")
                if sum(shape[0] for shape in shapes.values()) != target.shape[0]:
                    raise ValueError("fused target rows disagree")
            else:
                if len(selected) != 1 or selected[0][1] is not None:
                    raise ValueError("missing/duplicate non-fused source: " + name)
                shapes = {None: tuple(target.shape)}
            for key, shard in selected:
                if tuple(headers[key]["shape"]) != shapes[shard] or target.dtype != torch.bfloat16:
                    raise ValueError("native text shape/dtype mismatch: " + key)
            bindings[name] = Binding(tuple(k for k, _ in selected), tuple(target.shape), target.dtype)
        return bindings, routes, excluded

    @classmethod
    def build(cls, model, root, *, max_total_file_bytes, tensor_parallel_size=1):
        if tensor_parallel_size != 1:
            raise ValueError("TP1 only")
        if type(max_total_file_bytes) is not int or max_total_file_bytes <= 0:
            raise ValueError("explicit positive file budget required")
        root = Path(root).resolve()
        index_path = _scoped(root, "model.safetensors.index.json")
        if index_path.stat().st_size > 4 * 1024**2:
            raise ValueError("index metadata budget exceeded")
        document = json.loads(index_path.read_text(), object_pairs_hook=_unique)
        index = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(index, dict) or not index:
            raise ValueError("nonempty native checkpoint index required")
        selected = {k: _scoped(root, name) for k, name in index.items()}
        paths = set(selected.values())
        if sum(path.stat().st_size for path in paths) > max_total_file_bytes:
            raise ValueError("checkpoint file budget exceeded")
        headers, identities = {}, {}
        for path in sorted(paths):
            header, digest = _header(path)
            if set(header) != {k for k, p in selected.items() if p == path}:
                raise ValueError("index does not exactly cover shard")
            if headers.keys() & header.keys():
                raise ValueError("duplicate source across shards")
            headers.update(header)
            stat = path.stat()
            identities[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, digest)
        bindings, routes, excluded = cls.audit_headers(model, headers)
        sources = {key: Source(key, selected[key], headers[key]["shape"], headers[key]["dtype"],
                               headers[key]["bytes"]) for key in routes}
        return cls(sources, bindings, identities, model, routes, excluded)


class H3NativeTextMmapLoader(H3MmapLoader):
    def __init__(self, model, plan, *, device, max_group_bytes):
        if not isinstance(plan, H3NativeTextMmapPlan):
            raise TypeError("original native text plan required")
        super().__init__(model, plan, device=device, max_group_bytes=max_group_bytes)

    def _load_native_weights(self, weights):
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            _default_weight_loader, _fused_source_shards, _verify_fused_shards_complete,
        )

        params = dict(self.model.named_parameters())
        loaded, seen, expected, shards = set(), set(), {}, {}
        for key, tensor in weights:
            if key in seen or key not in self.plan.routes:
                raise ValueError("duplicate or undeclared native text source")
            seen.add(key)
            name, shard = self.plan.routes[key]
            parameter = params[name]
            owner = self.model.get_submodule(name.rsplit(".", 1)[0])
            fused = _fused_source_shards(owner)
            if fused:
                expected[name] = fused
                done = shards.setdefault(name, set())
                if shard in done:
                    raise ValueError("duplicate fused shard")
                done.add(shard)
            elif name in loaded:
                raise ValueError("duplicate non-fused target")
            loader = getattr(parameter, "weight_loader", _default_weight_loader)
            loader(parameter, tensor, shard)
            loaded.add(name)
        _verify_fused_shards_complete(expected, shards)
        if seen != {key for name in loaded for key in self.plan.bindings[name].sources}:
            raise ValueError("native loader did not consume complete planned sources")
        # Entire missing targets are rejected by H3MmapLoader's exact group check.
        return loaded

    def _expected_loaded(self, names):
        return set(names)
