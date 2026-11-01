"""Frozen v2: complete tiny H3 native-loader parity and mmap lifecycle."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import traceback
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from validate_meta_coverage_v2 import read_inputs, quant_from_headers, make_model, setup_dist

def state(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))

def fixture(research, root):
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
    config, dense, packed, _ = read_inputs(research)
    config.update(num_layers=2, num_refiner_layers=1, hidden_size=128, num_attention_heads=1,
                  attention_head_dim=128, ffn_dim=128, text_dim=128, freq_dim=64,
                  time_embed_hidden_dim=128, time_embed_dim=64,
                  adaln_out_features=18 * 128, final_adaln_out_features=2 * 128,
                  in_channels=4, audio_in_channels=4)
    small = {}
    for key, info in packed.items():
        index = int(key.split(".")[1])
        if index >= config["num_layers"]:
            continue
        info = dict(info)
        rows = 256 if ".ff.fc_in::" in key else 128
        if key.endswith("::_nvfp4_weight"):
            info.update(shape=[rows, 64])
        elif key.endswith("::_nvfp4_weight_scale"):
            info.update(shape=[rows, 8])
        small[key] = info
    quant = quant_from_headers(small)
    meta = make_model(config, quant)
    tensors = state(meta)
    dense_values = {}
    for key, info in dense.items():
        module, _, field = key.rpartition(".")
        target, layout, _ = _resolve_native_target(module)
        target += "." + field
        if target not in tensors:
            continue
        shape = list(tensors[target].shape)
        if layout in {"q", "k", "v"}:
            shape[0] //= 3
        count = 1
        for dim in shape:
            count *= dim
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16) % 13
        value = ((torch.arange(count, dtype=torch.float32) + seed).remainder(17) - 8) / 64
        if "norm" in key:
            value += 1
        dense_values[key] = value.reshape(shape).to({"BF16": torch.bfloat16, "F32": torch.float32}[info["dtype"]])
    packed_values = {}
    for key, info in small.items():
        shape = info["shape"]
        if info["dtype"] == "U8":
            count = 1
            for dim in shape:
                count *= dim
            if key.endswith("::_nvfp4_weight_scale"):
                value = torch.full(shape, 56, dtype=torch.uint8)
            else:
                seed = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16) % 7
                value = ((torch.arange(count) + seed).remainder(7) + 17).to(torch.uint8).reshape(shape)
        else:
            value = torch.full(shape, 0.125 if key.endswith("::_nvfp4_alpha") else 1.0,
                               dtype={"BF16": torch.bfloat16, "F32": torch.float32}[info["dtype"]])
        packed_values[key] = value
    root.mkdir(parents=True)
    save_file(dense_values, root / "dense.safetensors")
    save_file(packed_values, root / "nvfp4_weights.safetensors")
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "dense.safetensors" for key in dense_values}}))
    (root / "config.json").write_text(json.dumps(config))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    (root / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2))
    return config, quant, dense_values, packed_values

def native_load(model, root):
    def weights():
        for file in ("dense.safetensors", "nvfp4_weights.safetensors"):
            with safe_open(root / file, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    yield key, reader.get_tensor(key)
    model.load_weights(weights())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "model_quality_verified": False,
              "scope": "all parameters and buffers of real reduced-dimension upstream H3",
              "checks": [], "device": a.device}
    def check(name, ok, **details):
        result["checks"].append({"name": name, "ok": bool(ok), **details})
    code = 1
    try:
        setup_dist(a.output.parent)
        config, _, dense, packed = fixture(a.research, a.output.parent / "fixture")
        root = a.output.parent / "fixture"
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from h3_omni_adapter.mmap_loader import H3MmapPlan, H3MmapLoader
        quant = FastH3NVFP4Config.from_dit(root / "nvfp4_weights.safetensors", rank=0)
        ordinary = make_model(config, quant, device="cpu")
        native_load(ordinary, root)
        reference = {k: v.detach().clone() for k, v in state(ordinary).items()}
        candidate = make_model(config, quant)
        plan = H3MmapPlan.build(candidate, root, sidecars=["nvfp4_weights.safetensors"],
                                max_total_file_bytes=64 * 1024**2)
        check("all_tensors_planned", set(plan.bindings) == set(reference))
        check("index_plus_selected_sidecar", len(plan.sources) == len(dense) + len(packed))
        check("preflight_keeps_all_model_tensors_meta", all(v.is_meta for v in state(candidate).values()))
        loader = H3MmapLoader(candidate, plan, device=a.device, max_group_bytes=32 * 1024**2)
        seen = set()
        loader.load_group("resident")
        for group in plan.groups:
            if group != "resident":
                loader.load_group(group)
            names = plan.groups[group]
            current = state(candidate)
            mismatched = [name for name in names if current[name].is_meta or
                          not torch.equal(current[name].detach().cpu(), reference[name])]
            check("group_exact_" + group, not mismatched, tensor_count=len(names), mismatched=mismatched)
            check("parameter_classes_" + group, all(type(current[n]) is type(state(ordinary)[n]) for n in names))
            seen.update(names)
            if group != "resident":
                loader.release_group(group)
                check("group_released_" + group, all(state(candidate)[name].is_meta for name in names))
        check("all_parameters_and_buffers_compared", seen == set(reference), count=len(seen))
        for _ in range(3):
            loader.load_group("blocks.0")
            check("repeated_native_loader_exact", all(torch.equal(state(candidate)[n].detach().cpu(), reference[n])
                                                     for n in plan.groups["blocks.0"]))
            loader.release_group("blocks.0")
        if a.device == "cuda":
            ordinary.blocks[0].attn.qkv_proj.to("cuda")
            loader.load_group("blocks.0")
            x = (torch.arange(16 * 128, device="cuda", dtype=torch.float32).reshape(16, 128).remainder(5) / 16).bfloat16()
            with torch.inference_mode():
                expected = ordinary.blocks[0].attn.qkv_proj(x)[0]
                actual = candidate.blocks[0].attn.qkv_proj(x)[0]
            torch.cuda.synchronize()
            check("original_qkv_nvfp4_forward_exact", torch.equal(actual, expected) and torch.isfinite(actual).all())
            loader.release_group("blocks.0")
        loader.install_block_hooks()
        try:
            candidate.blocks[0]()
        except TypeError:
            check("forward_exception_releases_group", all(state(candidate)[n].is_meta for n in plan.groups["blocks.0"]))
        else:
            check("forward_exception_releases_group", False)
        result["loader_stats"] = dict(loader.stats)
        loader.close()
        gc.collect()
        check("close_releases_all_groups", all(v.is_meta for v in state(candidate).values()))
        check("close_releases_file_maps", not any(str(root) in line for line in Path("/proc/self/maps").read_text().splitlines()))
        for name, kwargs in (("missing_sidecar_rejected", {"sidecars": []}),
                             ("unsupported_tp_rejected", {"sidecars": ["nvfp4_weights.safetensors"], "tensor_parallel_size": 2}),
                             ("file_budget_rejected", {"sidecars": ["nvfp4_weights.safetensors"], "max_total_file_bytes": 1})):
            try:
                H3MmapPlan.build(candidate, root, **kwargs)
            except ValueError:
                check(name, True)
            else:
                check(name, False)
        bad = a.output.parent / "bad-scalar"
        bad.mkdir()
        bad_values = dict(packed)
        scalar_key = next(k for k in bad_values if k.endswith("::_nvfp4_alpha"))
        bad_values[scalar_key] = torch.full_like(bad_values[scalar_key], -1)
        save_file(dense, bad / "dense.safetensors")
        save_file(bad_values, bad / "nvfp4_weights.safetensors")
        (bad / "diffusion_pytorch_model.safetensors.index.json").write_bytes((root / "diffusion_pytorch_model.safetensors.index.json").read_bytes())
        bad_plan = H3MmapPlan.build(candidate, bad, sidecars=["nvfp4_weights.safetensors"])
        broken = H3MmapLoader(candidate, bad_plan, device=a.device, max_group_bytes=32 * 1024**2)
        broken.load_group("resident")
        try:
            broken.load_group("blocks.0")
        except ValueError:
            check("invalid_scalar_rejected_by_real_upstream_loader", True)
            check("failed_load_releases_partial_group", all(state(candidate)[n].is_meta for n in bad_plan.groups["blocks.0"]))
        else:
            check("invalid_scalar_rejected_by_real_upstream_loader", False)
        finally:
            broken.close()
        result["checks_total"] = len(result["checks"])
        result["checks_ok"] = sum(c["ok"] for c in result["checks"])
        code = int(result["checks_total"] != result["checks_ok"])
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
    print(json.dumps({k: v for k, v in result.items() if k not in {"checks", "traceback"}}))
    return code
if __name__ == "__main__":
    raise SystemExit(main())

