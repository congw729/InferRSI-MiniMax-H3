"""Frozen v1: full retained H3 Qwen3-VL encoder metadata coverage, no weights."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import traceback
import torch
from validate_meta_coverage_v2 import setup_dist

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "payload_read": False, "model_forward": False,
              "scope": "real full encoder wrapper meta constructor with explicit header-derived NVFP4 config"}
    code = 1
    try:
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config, NVFP4Projection, _Projection
        from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
        headers, provenance = {}, []
        for path in sorted((a.research / "memory-budget").glob("FastVideo__FastH3-Omni-NVFP4-QAD__text_encoder__*.header.json")):
            saved = json.loads(path.read_text())
            if hashlib.sha256(path.with_suffix(".raw").read_bytes()).hexdigest() != saved["header_sha256"]:
                raise ValueError("saved header digest mismatch")
            values = {k: v for k, v in saved["header"].items() if k != "__metadata__"}
            if headers.keys() & values.keys():
                raise ValueError("duplicate encoder key")
            headers.update(values)
            provenance.append({"file": path.name, "raw_sha256": saved["header_sha256"]})
        projections = defaultdict(list)
        for key, info in headers.items():
            if not key.endswith(".weight_packed"):
                continue
            source = key.removesuffix(".weight_packed")
            target = source.replace("model.language_model.", "text_model.")
            for part in ("q", "k", "v"):
                target = target.replace(f".self_attn.{part}_proj", ".self_attn.qkv_proj")
            for part in ("gate", "up"):
                target = target.replace(f".mlp.{part}_proj", ".mlp.gate_up_proj")
            tensors = {"weight": key, "scales": source + ".weight_scale", "alpha": source + ".weight_global_scale"}
            if not all(k in headers for k in tensors.values()):
                raise ValueError("incomplete encoder packed group")
            rows, half_k = info["shape"]
            projections[target].append(NVFP4Projection(source, rows, half_k * 2, tensors, reciprocal_scale=True))
        order = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "gate_proj": 0, "up_proj": 1}
        for target, specs in projections.items():
            if len(specs) > 1:
                specs.sort(key=lambda s: order[s.source.rsplit(".", 1)[1]])
            expected = 3 if target.endswith("qkv_proj") else 2 if target.endswith("gate_up_proj") else 1
            if len(specs) != expected:
                raise ValueError("incomplete encoder fused projection")
        quant = FastH3NVFP4Config(dict(projections), rank=0)
        config_path = a.research / "config-text_encoder__config.json"
        config = json.loads(config_path.read_text())["content"]
        original_quant = config.pop("quantization_config")
        result["config_route"] = {"architecture": "unchanged saved config",
            "quantization": "same explicit FastH3NVFP4Config generated from saved header shapes",
            "omitted_auto_detection_field": "quantization_config (avoids opening absent payload files)",
            "original_quantization_config": original_quant}
        config_root = a.output.parent / "metadata-config"
        config_root.mkdir()
        (config_root / "config.json").write_text(json.dumps(config, indent=2))
        setup_dist(a.output.parent)
        with torch.device("meta"):
            model = MiniMaxH3Qwen3VLEncoder(str(config_root), device=torch.device("cuda"),
                                           load_model=True, quant_config=quant)
        state = dict(list(model.named_parameters()) + list(model.named_buffers()))
        parts, ignored = defaultdict(list), []
        for key, info in headers.items():
            if key in quant.weight_map:
                target, shard = quant.weight_map[key], None
            else:
                mapped = model._map_weight_name(key)
                if mapped is None:
                    ignored.append(key)
                    continue
                target, shard = mapped
            parts[target].append({"source": key, "shard": shard, **info})
        errors, inventory = [], []
        for name, tensor in state.items():
            owner_path, _, leaf = name.rpartition(".")
            owner = model.get_submodule(owner_path)
            sources = parts.get(name, [])
            row = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "device": str(tensor.device), "sources": sources,
                   "kind": "parameter" if name in dict(model.named_parameters()) else "buffer",
                   "logical_bytes": tensor.numel() * tensor.element_size()}
            if sources:
                if len(sources) != 1:
                    errors.append({"target": name, "error": "unexpected dense fused source"})
                shape = list(sources[0]["shape"])
                if isinstance(owner, _Projection) and leaf == "alpha":
                    if tensor.numel() != 1:
                        errors.append({"target": name, "error": "alpha not scalar"})
                    shape = [1]
                    row["transform"] = "reciprocal FP32 global scale"
                if shape != list(tensor.shape):
                    errors.append({"target": name, "error": "shape mismatch"})
                source_dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}[sources[0]["dtype"]]
                if source_dtype != tensor.dtype:
                    errors.append({"target": name, "error": "dtype mismatch"})
            elif isinstance(owner, _Projection) and leaf == "input_scale" and leaf not in owner.spec.tensors:
                row["default"] = "constructor ones, FP32 [1]"
                if list(tensor.shape) != [1] or tensor.dtype != torch.float32:
                    errors.append({"target": name, "error": "input scale default metadata"})
            else:
                errors.append({"target": name, "error": "missing source/default"})
            if not tensor.is_meta:
                errors.append({"target": name, "error": "non-meta tensor"})
            inventory.append(row)
        for extra in sorted(parts.keys() - state.keys()):
            errors.append({"target": extra, "error": "source has no runtime target"})
        result.update(headers=provenance, inventory=inventory, source_count=len(headers),
                      target_count=len(state), parameter_count=len(dict(model.named_parameters())),
                      buffer_count=len(dict(model.named_buffers())), ignored_source_keys=ignored,
                      logical_bytes=sum(x["logical_bytes"] for x in inventory),
                      errors=errors, complete_encoder_meta_coverage=not errors)
        code = int(bool(errors))
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    text = json.dumps(result, indent=2)
    for source, label in ((str(a.research), "<RESEARCH_ROOT>"), (str(a.output.parent), "<RUN_ROOT>"),
                          (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    a.output.write_text(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"inventory", "headers", "traceback"}}))
    return code
if __name__ == "__main__":
    raise SystemExit(main())
