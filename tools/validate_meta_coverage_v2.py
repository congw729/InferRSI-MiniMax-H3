"""Frozen v2: actual full DiT meta inventory against saved headers, no payload."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import traceback

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
    os.environ[key] = "1"
os.environ["MAX_JOBS"] = "2"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_inputs(research):
    prefix = "FastVideo__FastH3-Omni-NVFP4-QAD__transformer_ref__"
    dense, packed, provenance = {}, {}, []
    for path in sorted((research / "memory-budget").glob(prefix + "*.header.json")):
        wrapper = json.loads(path.read_text())
        if digest(path.with_suffix(".raw")) != wrapper["header_sha256"]:
            raise ValueError("saved header digest mismatch")
        values = {k: v for k, v in wrapper["header"].items() if k != "__metadata__"}
        dest = packed if "nvfp4_weights" in path.name else dense
        if dest.keys() & values.keys():
            raise ValueError("duplicate source key")
        dest.update(values)
        provenance.append({"file": path.name, "sha256": digest(path), "raw_sha256": wrapper["header_sha256"]})
    config_path = research / "config-transformer_ref__config.json"
    config = json.loads(config_path.read_text())["content"]
    index_path = research / "config-transformer_ref__diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())["content"]["weight_map"]
    if set(index) != set(dense):
        raise ValueError("saved index/header key sets disagree")
    return config, dense, packed, provenance

def quant_from_headers(packed):
    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config, NVFP4Projection
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
    fields = {"_nvfp4_weight": "weight", "_nvfp4_weight_scale": "scales",
              "_nvfp4_alpha": "alpha", "_weight_global_sf": "weight_global_scale",
              "_nvfp4_input_global_sf": "input_scale"}
    groups = defaultdict(dict)
    for key in packed:
        source, sep, field = key.partition("::")
        if not sep or field not in fields:
            raise ValueError("unknown packed field")
        groups[source][fields[field]] = key
    projections = defaultdict(list)
    for source, tensors in groups.items():
        if not {"weight", "scales", "alpha", "weight_global_scale"} <= tensors.keys():
            raise ValueError("incomplete packed projection")
        native = source.replace(".ff.fc_in", ".ff.net.0.proj").replace(".ff.fc_out", ".ff.net.2")
        if native.endswith(".attn.to_out"):
            native += ".0"
        target = _resolve_native_target(native)
        if target is None:
            raise ValueError("unresolved packed projection")
        rows, half_k = packed[tensors["weight"]]["shape"]
        projections[target[0]].append(NVFP4Projection(source, rows, half_k * 2, tensors, target[1] == "swap_halves"))
    for name, specs in projections.items():
        if name.endswith(".qkv_proj"):
            specs.sort(key=lambda spec: {"q": 0, "k": 1, "v": 2}[spec.source[-1]])
            if [s.source[-1] for s in specs] != ["q", "k", "v"]:
                raise ValueError("incomplete QKV")
    return FastH3NVFP4Config(dict(projections), rank=0)

def setup_dist(work):
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    rendezvous = work / "rendezvous"
    if rendezvous.exists():
        raise FileExistsError("use a new run directory")
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method=rendezvous.as_uri(), backend="gloo")
    initialize_model_parallel(backend="gloo")

def make_model(config, quant, device="meta"):
    import torch
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    od = OmniDiffusionConfig(model=None, model_class_name="MiniMaxH3Pipeline",
        tf_model_config=TransformerConfig.from_dict(config),
        cache_config={"minimax_h3_adaln_cache": False, "minimax_h3_adaln_offload": False})
    with set_current_diffusion_config(od), torch.device(device):
        model = MiniMaxH3DiTModel(od, quant, diffusers_weights=True)
        model.enable_vsa_gates()
    return model

def audit(model, dense, packed, quant):
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
    expected = defaultdict(list)
    errors = []
    for key, info in dense.items():
        module, _, field = key.rpartition(".")
        target = _resolve_native_target(module)
        if target is None or field not in {"weight", "bias", "weight_scale"}:
            errors.append({"source": key, "error": "unresolved source"})
            continue
        expected[target[0] + "." + field].append({"source": key, **info, "layout": target[1]})
    for key, info in packed.items():
        target = quant.weight_map[key]
        expected[target].append({"source": key, **info, "layout": "upstream NVFP4 weight_loader"})
    defaults = {}
    for name, specs in quant.projections.items():
        for i, spec in enumerate(specs):
            if "input_scale" not in spec.tensors:
                defaults[f"{name}.packed.{i}.input_scale"] = {"kind": "ones", "shape": [1], "dtype": "torch.float32"}
    defaults["rope.inv_freq"] = {"kind": "load_weights regenerates rope_theta frequencies",
                                 "shape": [model.arch.rope_inv_freq_len], "dtype": "torch.float32"}
    for head in ("video_out", "audio_out"):
        defaults[f"final_layer.{head}.coefficients"] = {
            "kind": "nonpersistent upstream PDD analytic head weights",
            "shape": [8, 4], "dtype": "torch.float32"}
    state = {name: (value, "parameter") for name, value in model.named_parameters()}
    for name, value in model.named_buffers():
        if name in state:
            raise ValueError("parameter/buffer name collision")
        state[name] = (value, "buffer")
    rows = []
    dtype_names = {"U8": "torch.uint8", "BF16": "torch.bfloat16", "F32": "torch.float32"}
    for name, (value, kind) in sorted(state.items()):
        row = {"name": name, "kind": kind, "shape": list(value.shape), "dtype": str(value.dtype),
               "persistent": kind == "parameter" or name.rsplit(".", 1)[1] not in model.get_submodule(name.rsplit(".", 1)[0])._non_persistent_buffers_set,
               "device": str(value.device), "logical_bytes": value.numel() * value.element_size(),
               "loader": getattr(getattr(value, "weight_loader", None), "__qualname__", "default_weight_loader"),
               "missing_param_init": getattr(value, "missing_param_init", None)}
        parts = expected.get(name, [])
        row["sources"] = parts
        if parts:
            layouts = {p["layout"] for p in parts}
            source_shape = list(parts[0]["shape"])
            if layouts == {"q", "k", "v"}:
                if len(parts) != 3 or len({tuple(p["shape"][1:]) for p in parts}) != 1:
                    errors.append({"target": name, "error": "invalid dense QKV set"})
                source_shape[0] = sum(p["shape"][0] for p in parts)
                row["transform"] = "dense Q/K/V concatenation"
            elif len(parts) != 1:
                errors.append({"target": name, "error": "unexpected many-to-one mapping"})
            if ".packed." in name and name.rsplit(".", 1)[1] not in {"weight", "scales"}:
                import math
                if math.prod(source_shape) != 1:
                    errors.append({"target": name, "error": "packed scalar shape"})
                source_shape = [1]
                row["transform"] = "scalar reshape + FP32 cast"
            if list(value.shape) != source_shape:
                errors.append({"target": name, "error": "shape mismatch",
                               "source_shape": source_shape, "runtime_shape": list(value.shape)})
            src_types = {dtype_names.get(p["dtype"], p["dtype"]) for p in parts}
            if src_types != {str(value.dtype)}:
                allowed = src_types <= {"torch.bfloat16", "torch.float32"} and str(value.dtype) in {"torch.bfloat16", "torch.float32"}
                if not allowed:
                    errors.append({"target": name, "error": "unsupported dtype conversion"})
                row["dtype_conversion"] = {"from": sorted(src_types), "to": str(value.dtype)}
        elif name in defaults:
            row["default"] = defaults[name]
            if list(value.shape) != defaults[name]["shape"] or str(value.dtype) != defaults[name]["dtype"]:
                errors.append({"target": name, "error": "default shape/dtype mismatch"})
        else:
            errors.append({"target": name, "error": "missing source and default"})
        if value.device.type != "meta":
            errors.append({"target": name, "error": "non-meta model tensor"})
        rows.append(row)
    for extra in sorted((expected.keys() | defaults.keys()) - state.keys()):
        errors.append({"target": extra, "error": "source/default target absent from model"})
    return rows, errors, defaults

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "payload_read": False, "model_forward": False,
              "scope": "full 50-block DiT meta inventory only; encoder/VAE not yet audited"}
    code = 1
    try:
        config, dense, packed, provenance = read_inputs(args.research)
        quant = quant_from_headers(packed)
        setup_dist(args.output.parent)
        model = make_model(config, quant)
        rows, errors, defaults = audit(model, dense, packed, quant)
        result.update(architecture=asdict(model.arch), headers=provenance, inventory=rows,
                      parameter_count=sum(r["kind"] == "parameter" for r in rows),
                      buffer_count=sum(r["kind"] == "buffer" for r in rows),
                      source_tensor_count=len(dense) + len(packed),
                      target_tensor_count=len(rows), default_count=len(defaults),
                      logical_bytes=sum(r["logical_bytes"] for r in rows),
                      errors=errors, complete_dit_meta_coverage=not errors)
        code = int(bool(errors))
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            import torch.distributed as dist
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    serialized = json.dumps(result, indent=2)
    for source, label in ((str(args.research), "<RESEARCH_ROOT>"), (str(args.output.parent), "<RUN_ROOT>"),
                          (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        serialized = serialized.replace(source, label)
    args.output.write_text(serialized + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"inventory", "headers", "traceback"}}))
    return code
if __name__ == "__main__":
    raise SystemExit(main())

