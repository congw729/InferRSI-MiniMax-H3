"""Frozen v1: real-shape synthetic H3 text embedding plus one native layer."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import struct
import time
import traceback

import torch

from capacity_worker_v1 import WIDTH, element_bytes
from validate_meta_coverage_v2 import setup_dist

GIB = 1024**3


def constant(key, dtype):
    if dtype == "U8":
        return 56 if key.endswith(".weight_scale") else 17
    if key.endswith(".weight_global_scale"):
        return 8192.0
    return 0.0 if key.endswith(".bias") else 1.0 if "norm" in key else 1 / 8192


def read_sources(research):
    selected, provenance = {}, []
    for path in sorted((research / "memory-budget").glob("FastVideo__FastH3-Omni-NVFP4-QAD__text_encoder__*.header.json")):
        saved = json.loads(path.read_text())
        digest = hashlib.sha256(path.with_suffix(".raw").read_bytes()).hexdigest()
        if digest != saved["header_sha256"]:
            raise ValueError("saved source header changed")
        provenance.append({"file": path.name, "raw_sha256": digest})
        for key, info in saved["header"].items():
            if key == "model.language_model.embed_tokens.weight" or key.startswith("model.language_model.layers.0."):
                if key in selected:
                    raise ValueError("duplicate source key")
                selected[key] = {"shape": info["shape"], "dtype": info["dtype"]}
    config_path = research / "config-text_encoder__config.json"
    saved = json.loads(config_path.read_text())
    config = saved["content"]
    provenance.append({"file": config_path.name, "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                       "source_url": saved["source_url"], "revision": saved["revision"]})
    original_layers = config["text_config"]["num_hidden_layers"]
    config["text_config"]["num_hidden_layers"] = 1
    config["num_hidden_layers_override"] = 1
    return config, selected, provenance, original_layers


def write_fixture(path, infos):
    header, offset = {}, 0
    for key, info in sorted(infos.items()):
        size = math.prod(info["shape"]) * WIDTH[info["dtype"]]
        header[key] = {**info, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    if not 0 < offset + len(encoded) + 8 < 3 * GIB:
        raise ValueError("synthetic file exceeds budget")
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        stream.write(prefix)
        digest.update(prefix)
        for key, info in header.items():
            element = element_bytes(constant(key, info["dtype"]), info["dtype"])
            chunk = element * ((4 * 1024**2) // len(element))
            remaining = info["data_offsets"][1] - info["data_offsets"][0]
            while remaining:
                value = chunk[:min(len(chunk), remaining)]
                stream.write(value)
                digest.update(value)
                remaining -= len(value)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "synthetic": True, "keys": len(header), "payload_bytes": offset}


def verify_fixture(path, manifest, infos):
    if path.stat().st_size != manifest["bytes"] or path.stat().st_size >= 3 * GIB:
        raise ValueError("fixture size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
        if digest.hexdigest() != manifest["sha256"]:
            raise ValueError("fixture payload SHA mismatch")
        stream.seek(0)
        header_len = struct.unpack("<Q", stream.read(8))[0]
        if header_len > 2 * 1024**2:
            raise ValueError("header exceeds limit")
        header = json.loads(stream.read(header_len))
        if set(header) != set(infos):
            raise ValueError("fixture source inventory mismatch")
        for key, info in header.items():
            if any(info[field] != infos[key][field] for field in ("shape", "dtype")):
                raise ValueError("fixture metadata mismatch")
            element = element_bytes(constant(key, info["dtype"]), info["dtype"])
            lo, hi = info["data_offsets"]
            if hi - lo != math.prod(info["shape"]) * WIDTH[info["dtype"]]:
                raise ValueError("fixture tensor byte length mismatch")
            for offset in (lo, hi - len(element)):
                stream.seek(8 + header_len + offset)
                if stream.read(len(element)) != element:
                    raise ValueError("fixture first/last element bytes mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "real_weights": False, "checks": [],
              "scope": "real vocabulary and hidden width; one native text layer; synthetic 64-token capacity"}
    loader, code = None, 1
    started = time.monotonic()

    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)

    def mark(name):
        torch.cuda.synchronize()
        row = {"phase": name, "elapsed_seconds": time.monotonic() - started,
               "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
               "cuda_max_allocated": torch.cuda.max_memory_allocated(),
               "cuda_max_reserved": torch.cuda.max_memory_reserved(),
               "worker_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        with (args.output.parent / "phase-events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        result.setdefault("phases", []).append(row)
        if max(row["cuda_allocated"], row["cuda_max_allocated"]) > 4 * GIB:
            raise RuntimeError("CUDA phase high-water protection exceeded")

    try:
        free_disk = shutil.disk_usage(args.output.parent).free
        result["initial_output_disk_free_bytes"] = free_disk
        if free_disk < 8 * GIB:
            raise RuntimeError("insufficient output disk headroom")
        config, sources, provenance, original_layers = read_sources(args.research)
        result.update(source_header_provenance=provenance, original_config_layers=original_layers,
                      original_retained_layers=50, architecture=config)
        payload_bytes = sum(math.prod(info["shape"]) * WIDTH[info["dtype"]] for info in sources.values())
        check("selected_payload_within_3GiB", 0 < payload_bytes < 3 * GIB)
        fixture = args.fixture or args.output.parent / "fixture"
        if args.fixture is None:
            fixture.mkdir()
            manifest = write_fixture(fixture / "model.safetensors", sources)
            (fixture / "config.json").write_text(json.dumps(config, indent=2))
            (fixture / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {key: "model.safetensors" for key in sources}}))
            (fixture / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2))
        else:
            manifest = json.loads((fixture / "fixture-manifest.json").read_text())
            if json.loads((fixture / "config.json").read_text()) != config:
                raise ValueError("reused synthetic config differs")
        verify_fixture(fixture / "model.safetensors", manifest, sources)
        result.update(fixture=manifest, fixture_reused=args.fixture is not None)
        check("source_metadata_full_SHA_and_edge_bytes_verified_before_GPU_load", True)
        setup_dist(args.output.parent)
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        result["cuda_initial"] = {"free_bytes": free_cuda, "total_bytes": total_cuda, "allocator_fraction": 0.40}
        if free_cuda < 5 * GIB:
            raise RuntimeError("insufficient CUDA headroom")
        torch.cuda.set_per_process_memory_fraction(0.40)
        from transformers import Qwen3VLConfig
        from vllm.distributed import get_tp_group
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from h3_omni_adapter.mmap_loader import _state
        from h3_omni_adapter.text_streaming import H3TextOnly, H3TextMmapPlan, H3TextMmapLoader
        quant = FastH3NVFP4Config.from_encoder(fixture, rank=0)
        check("native_fused_projection_order",
              [s.source.rsplit(".", 1)[1] for s in quant.projections["text_model.layers.0.self_attn.qkv_proj"]]
              == ["q_proj", "k_proj", "v_proj"]
              and [s.source.rsplit(".", 1)[1] for s in quant.projections["text_model.layers.0.mlp.gate_up_proj"]]
              == ["gate_proj", "up_proj"])
        model = H3TextOnly(Qwen3VLConfig(**config), get_tp_group(), quant, max_tokens=64)
        check("complete_default_meta_constructor", all(t.is_meta for t in _state(model).values()))
        check("real_architecture_and_one_retained_layer",
              tuple(model.text_model.embed_tokens.weight.shape) == (151936, 5120)
              and len(model.text_model.layers) == 1
              and config["text_config"]["intermediate_size"] == 25600)
        plan = H3TextMmapPlan.build(model, fixture, files=["model.safetensors"], max_total_file_bytes=3 * GIB)
        check("strict_all_targets_no_vision_or_final_norm", set(plan.bindings) == set(_state(model))
              and set(plan.sources) == set(sources) and not plan.excluded)
        result["group_logical_bytes"] = {group: sum(math.prod(plan.bindings[n].shape) *
            torch.empty((), dtype=plan.bindings[n].dtype).element_size() for n in names)
            for group, names in plan.groups.items()}
        result["targets"] = len(plan.bindings)
        result["source_keys"] = len(plan.sources)
        loader = H3TextMmapLoader(model, plan, device="cuda", max_group_bytes=2 * GIB)
        mark("verified_fixture_and_meta_plan")
        loader.load_group("resident")
        mark("embedding_resident_loaded")

        def verify_group(group):
            current = _state(model)
            for name in plan.groups[group]:
                binding = plan.bindings[name]
                if binding.sources:
                    key = binding.sources[0]
                    expected = constant(key, plan.sources[key].dtype)
                    if key.endswith(".weight_global_scale"):
                        expected = 1 / expected
                else:
                    if binding.default != "one":
                        raise ValueError("unknown explicit default")
                    expected = 1
                low, high = torch.aminmax(current[name])
                if low.item() != expected or high.item() != expected:
                    raise AssertionError("native loaded value mismatch: " + name)
            return True

        check("embedding_full_constant_exact", verify_group("resident"))
        loader.load_group("text_model.layers.0")
        mark("single_layer_loaded_for_value_check")
        check("single_layer_constants_reciprocal_and_defaults_exact", verify_group("text_model.layers.0"))
        loader.release_group("text_model.layers.0")
        gc.collect()
        check("single_layer_released_to_meta", all(_state(model)[n].is_meta for n in plan.groups["text_model.layers.0"]))
        loader.install_block_hooks()
        ids = torch.arange(64, dtype=torch.long)
        mark("before_64_token_native_forward")
        output = model.encode_ids(ids)
        check("native_hidden_shape_cpu_bf16", tuple(output.shape) == (64, 5120)
              and output.device.type == "cpu" and output.dtype == torch.bfloat16)
        check("native_hidden_finite", torch.isfinite(output).all().item())
        check("resident_only_after_forward", loader.active == {"resident"})
        result["output_shape"] = list(output.shape)
        result["output_min_max"] = [output.min().item(), output.max().item()]
        result["output_sha256"] = hashlib.sha256(output.view(torch.uint8).numpy().tobytes()).hexdigest()
        mark("after_64_token_native_forward")
        del output, ids
        loader.close()
        gc.collect()
        mark("all_groups_closed")
        check("all_targets_meta_after_close", all(t.is_meta for t in _state(model).values()))
        check("fixture_mmap_closed", str(fixture) not in Path("/proc/self/maps").read_text())
        check("one_streamed_layer_and_expected_load_count",
              loader.stats["max_active_streamed_groups"] == 1 and loader.stats["groups_loaded"] == 3)
        result["loader_stats"] = dict(loader.stats)
        code = 0
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        if loader is not None:
            loader.close()
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["checks_ok"] = sum(item["ok"] for item in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["exit_code"] = code
    text = json.dumps(result, indent=2)
    for source, label in ((str(args.output.parent), "<RUN_ROOT>"), (str(args.research), "<RESEARCH_ROOT>"),
            (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    if args.fixture:
        text = text.replace(str(args.fixture), "<REUSED_FIXTURE>")
    args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
