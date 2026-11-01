"""Frozen v1 worker: bounded full-width synthetic H3 block loading and short forward."""
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

from validate_meta_coverage_v2 import make_model, quant_from_headers, read_inputs, setup_dist
from h3_omni_adapter.mmap_loader import H3MmapLoader, H3MmapPlan, _state


GIB = 1024**3
WIDTH = {"U8": 1, "BF16": 2, "F32": 4}


def source_constant(key, dtype):
    if dtype == "U8":
        return 56 if key.endswith("::_nvfp4_weight_scale") else 17
    if "::" in key:
        return 0.125 if key.endswith("::_nvfp4_alpha") else 1.0
    return 0.0 if key.endswith(".bias") else 1.0 if "norm" in key else 1 / 8192


def element_bytes(value, dtype):
    if dtype == "U8":
        return bytes([int(value)])
    packed = struct.pack("<f", value)
    return packed[2:] if dtype == "BF16" else packed


def write_synthetic(path, infos):
    header = {}
    offset = 0
    for key in sorted(infos):
        info = infos[key]
        count = math.prod(info["shape"]) * WIDTH[info["dtype"]]
        header[key] = {"dtype": info["dtype"], "shape": info["shape"],
                       "data_offsets": [offset, offset + count]}
        offset += count
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    if offset + len(encoded) + 8 > 3 * GIB:
        raise ValueError("individual file exceeds experiment budget")
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        stream.write(prefix)
        digest.update(prefix)
        for key, info in header.items():
            element = element_bytes(source_constant(key, info["dtype"]), info["dtype"])
            chunk = element * ((4 * 1024**2) // len(element))
            remaining = info["data_offsets"][1] - info["data_offsets"][0]
            while remaining:
                value = chunk[:min(len(chunk), remaining)]
                stream.write(value)
                digest.update(value)
                remaining -= len(value)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "synthetic": True, "keys": len(header)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "synthetic": True, "checks": [],
              "scope": "real block widths, one DiT plus one refiner; loading and 8-token forward only"}
    loader = None
    code = 1
    started = time.monotonic()
    def mark(name):
        torch.cuda.synchronize()
        row = {"phase": name, "elapsed_seconds": time.monotonic() - started,
               "cuda_allocated": torch.cuda.memory_allocated(),
               "cuda_reserved": torch.cuda.memory_reserved(),
               "cuda_max_allocated": torch.cuda.max_memory_allocated(),
               "cuda_max_reserved": torch.cuda.max_memory_reserved(),
               "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        with (a.output.parent / "phase-events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        result.setdefault("phases", []).append(row)
        if row["cuda_allocated"] > 3 * GIB:
            raise RuntimeError("worker CUDA allocation protection exceeded")
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)
    try:
        if shutil.disk_usage(a.output.parent).free < 8 * GIB:
            raise RuntimeError("insufficient free output disk")
        free, total = torch.cuda.mem_get_info()
        result["cuda_initial_free_bytes"] = free
        result["cuda_total_bytes"] = total
        if free < 4 * GIB:
            raise RuntimeError("insufficient initial CUDA headroom")
        mark("imports_and_cuda_context")
        setup_dist(a.output.parent)
        config, dense, packed, provenance = read_inputs(a.research)
        result["original_layer_counts"] = {k: config[k] for k in ("num_layers", "num_refiner_layers")}
        config.update(num_layers=1, num_refiner_layers=1)
        packed = {k: v for k, v in packed.items() if int(k.split(".")[1]) == 0}
        quant = quant_from_headers(packed)
        model = make_model(config, quant).eval()
        targets = _state(model)
        from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
        chosen = {}
        for key, info in dense.items():
            module, _, field = key.rpartition(".")
            resolved = _resolve_native_target(module)
            if resolved is not None and resolved[0] + "." + field in targets:
                chosen[key] = info
        total_bytes = sum(math.prod(v["shape"]) * WIDTH[v["dtype"]] for v in [*chosen.values(), *packed.values()])
        check("synthetic_payload_within_3GiB_limit", total_bytes < 3 * GIB)
        check("all_model_tensors_meta_before_loading", all(t.is_meta for t in targets.values()))
        result["architecture"] = config
        result["source_header_provenance"] = provenance
        result["synthetic_payload_bytes"] = total_bytes
        del targets
        fixture = a.output.parent / "fixture"
        fixture.mkdir()
        mark("meta_constructed_before_file_write")
        manifest = [write_synthetic(fixture / "dense.safetensors", chosen),
                    write_synthetic(fixture / "nvfp4_weights.safetensors", packed)]
        (fixture / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "dense.safetensors" for key in chosen}}))
        (fixture / "config.json").write_text(json.dumps(config, indent=2))
        (fixture / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2))
        result["fixture"] = manifest
        mark("synthetic_files_written")
        plan = H3MmapPlan.build(model, fixture, sidecars=["nvfp4_weights.safetensors"], max_total_file_bytes=3 * GIB)
        result["group_logical_bytes"] = {g: sum(math.prod(plan.bindings[n].shape) * torch.empty(
            (), dtype=plan.bindings[n].dtype).element_size() for n in names) for g, names in plan.groups.items()}
        loader = H3MmapLoader(model, plan, device="cuda", max_group_bytes=GIB)
        mark("before_resident_load")
        loader.load_group("resident")
        mark("resident_loaded")
        def verify(group):
            current = _state(model)
            for name in plan.groups[group]:
                binding = plan.bindings[name]
                value = current[name]
                if binding.sources:
                    expected = {source_constant(k, plan.sources[k].dtype) for k in binding.sources}
                    if len(expected) != 1:
                        raise ValueError("fixture verification requires same constant per fused target")
                    expected = next(iter(expected))
                    low, high = torch.aminmax(value)
                    if low.item() != expected or high.item() != expected:
                        raise AssertionError("loaded full-width tensor content differs: " + name)
                elif binding.default == "one" and not torch.all(value == 1).item():
                    raise AssertionError("default input scale differs")
            return True
        check("resident_constant_payload_exact", verify("resident"))
        for group in ("token_refiner.blocks.0", "blocks.0"):
            loader.load_group(group)
            mark(group + "_loaded")
            check(group + "_constant_payload_exact", verify(group))
            loader.release_group(group)
            gc.collect()
            check(group + "_released_to_meta", all(_state(model)[n].is_meta for n in plan.groups[group]))
            mark(group + "_released")
        loader.install_block_hooks()
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
        from vllm_omni.diffusion.forward_context import set_forward_context
        od = OmniDiffusionConfig(model=None, model_class_name="MiniMaxH3Pipeline",
            tf_model_config=TransformerConfig.from_dict(config),
            cache_config={"minimax_h3_adaln_cache": False, "minimax_h3_adaln_offload": False})
        def ids(values):
            return {"position_ids": torch.tensor(values, device="cuda")}
        kwargs = {
            "x": torch.full((1, 8, model.video_patch_proj.input_size), 0.01, device="cuda"),
            "audio_x": torch.full((1, 8, model.audio_patch_proj.input_size), 0.01, device="cuda"),
            "img_position_ids": torch.zeros(1, 8, 3, device="cuda"),
            "unique_timesteps": torch.tensor([0.5], device="cuda"),
            "inverse_indices": torch.zeros(8, dtype=torch.long, device="cuda"),
            "update_mask": torch.tensor([1, 1, 1, 0], device="cuda"),
            "token_tags": torch.tensor([0, 0, 1, 1, 1, 1, 2, 2], device="cuda"),
            "prompt_embeds": torch.full((2, config["text_dim"]), 0.01, dtype=torch.bfloat16, device="cuda"),
            "img_pos_info": ids([2, 3, 4, 5]), "audio_pos_info": ids([6, 7]),
            "text_pos_info": ids([0, 1]), "img_pos_for_infer_output_info": ids([2, 3, 4, 5]),
            "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
                                  "max_seqlen_q": 8, "num_requests": 1},
            "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
                                          "max_seqlen_q": 2, "num_requests": 1},
        }
        result["attention_backend"] = model.blocks[0].attn.attention.attn_backend.get_name()
        mark("before_short_full_width_forward")
        with torch.inference_mode(), set_forward_context(omni_diffusion_config=od):
            output = model(pdd_step=0, **kwargs)
        check("full_width_short_forward_finite", all(torch.isfinite(t).all().item() for t in output))
        check("full_width_condition_mask", torch.count_nonzero(output[0][-1]).item() == 0)
        check("streamed_groups_released_after_forward", loader.active == {"resident"})
        result["output_shapes"] = [list(t.shape) for t in output]
        mark("after_short_full_width_forward")
        del output, kwargs
        loader.close()
        gc.collect()
        mark("all_model_groups_closed")
        check("all_model_tensors_meta_after_close", all(t.is_meta for t in _state(model).values()))
        check("checkpoint_unmapped_after_close", str(fixture) not in Path("/proc/self/maps").read_text())
        result["loader_stats"] = dict(loader.stats)
        code = 0
    except Exception as exc:
        result["error"] = str(exc)
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
    result["checks_passed"] = sum(c["ok"] for c in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["exit_code"] = code
    text = json.dumps(result, indent=2)
    for path, label in ((str(a.output.parent), "<RUN_ROOT>"), (str(a.research), "<RESEARCH_ROOT>"),
                        (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(path, label)
    a.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
