"""Frozen v1: real-width H3 single-block forward at native long-sequence geometry."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import time
import traceback

import torch

from h3_omni_adapter.mmap_loader import H3MmapLoader, H3MmapPlan, _state
from validate_meta_coverage_v2 import make_model, setup_dist
from validate_request_geometry_v2 import native_geometry

GIB = 1024**3


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--seconds", type=int, choices=(3, 4), required=True)
    p.add_argument("--size", type=int, choices=(128, 256), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "real_weights": False,
              "scope": "one real-width DiT and refiner; one PDD-step forward, synthetic payload",
              "checks": [], "phases": []}
    loader = None
    code = 1
    started = time.monotonic()
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)
    def mark(name):
        torch.cuda.synchronize()
        row = {"phase": name, "elapsed_seconds": time.monotonic()-started,
               "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
               "cuda_max_allocated": torch.cuda.max_memory_allocated(), "cuda_max_reserved": torch.cuda.max_memory_reserved(),
               "worker_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        result["phases"].append(row)
        with (a.output.parent / "phase-events.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        if row["cuda_allocated"] > 4*GIB or row["cuda_max_allocated"] > 4*GIB:
            raise RuntimeError("4GiB torch allocation high-water guard exceeded")
    try:
        if shutil.disk_usage(a.output.parent).free < 4*GIB:
            raise RuntimeError("insufficient output disk headroom")
        free, total = torch.cuda.mem_get_info()
        if free < 5*GIB:
            raise RuntimeError("insufficient CUDA initial free headroom")
        torch.cuda.set_per_process_memory_fraction(0.40)
        result["cuda_initial"] = {"free_bytes": free, "total_bytes": total, "allocator_fraction": 0.40}
        geometry_document = json.loads(a.geometry.read_text())
        item = next(c for c in geometry_document["cases"]
                    if c["requested_seconds"] == a.seconds and c["height"] == a.size)
        geometry = native_geometry(a.seconds, a.size, 64)
        check("geometry_recomputed_exact", all(item[k] == v for k,v in geometry.items()))
        result["geometry"] = item
        result["geometry_sha256"] = sha(a.geometry)
        check("request_route_classification", item["request_accepted"] == (a.seconds == 4))
        files = json.loads((a.fixture / "fixture-manifest.json").read_text())
        check("bounded_reused_fixture", sum(row["bytes"] for row in files) < 3*GIB)
        check("fixture_hashes_match_original", all(sha(a.fixture/row["file"]) == row["sha256"] for row in files))
        result["reused_fixture_files"] = files
        mark("initial_context_and_fixture_integrity")
        setup_dist(a.output.parent)
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        config = json.loads((a.fixture / "config.json").read_text())
        check("one_full_width_dit_and_refiner", config["num_layers"] == 1 and config["num_refiner_layers"] == 1
              and config["hidden_size"] > 128 and config["text_dim"] == 5120)
        quant = FastH3NVFP4Config.from_dit(a.fixture/"nvfp4_weights.safetensors", rank=0)
        model = make_model(config, quant).eval()
        check("all_meta_before_load", all(t.is_meta for t in _state(model).values()))
        plan = H3MmapPlan.build(model, a.fixture, sidecars=["nvfp4_weights.safetensors"], max_total_file_bytes=3*GIB)
        result["group_logical_bytes"] = {g:sum(math.prod(plan.bindings[n].shape)*torch.empty(
            (),dtype=plan.bindings[n].dtype).element_size() for n in names) for g,names in plan.groups.items()}
        loader = H3MmapLoader(model, plan, device="cuda", max_group_bytes=GIB)
        loader.load_group("resident")
        loader.install_block_hooks()
        mark("resident_loaded")
        from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
        from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
        from vllm_omni.diffusion.models.minimax_h3.pdd import pdd_sigmas
        packed = minimax_h3_packed_sequence(text_len=64, latent_t=item["latent_t"], latent_h=item["latent_h"],
            latent_w=item["latent_w"], audio_t=item["audio_t"], include_keyframe_cond=False)
        check("packed_shapes_match_frozen_geometry", int(packed["seq_len"]) == item["padded_tokens"]
              and packed["img_pos"].numel() == item["video_tokens"] and packed["audio_pos"].numel() == item["audio_tokens"])
        hidden = torch.full((64, config["text_dim"]), 0.01, dtype=torch.bfloat16)
        branch = MiniMaxH3DenoiseBranch(packed=packed, text_embeddings=hidden,
            token_tags=packed["token_tags"], device=torch.device("cuda"))
        sv, sa = pdd_sigmas(12.0)[0], pdd_sigmas(3.0)[0]
        video_rows = torch.randn(item["video_tokens"],96,generator=torch.Generator().manual_seed(17)).cuda()*sv
        audio_rows = torch.randn(item["audio_tokens"],32,generator=torch.Generator().manual_seed(17)).cuda()*sa
        kwargs = branch.forward_kwargs(video_rows=video_rows,audio_rows=audio_rows,t_video=1-sv,t_audio=1-sa,
                                       imgvid_cond_timestep=1-sv,audio_ref_cond_timestep=1-sa)
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
        from vllm_omni.diffusion.forward_context import set_forward_context
        od = OmniDiffusionConfig(model=None, model_class_name="MiniMaxH3Pipeline",
            tf_model_config=TransformerConfig.from_dict(config),
            cache_config={"minimax_h3_adaln_cache": False, "minimax_h3_adaln_offload": False})
        result["attention_backend"] = model.blocks[0].attn.attention.attn_backend.get_name()
        mark("before_native_long_forward")
        with torch.inference_mode(), set_forward_context(omni_diffusion_config=od):
            output = model(pdd_step=0, **kwargs)
        mark("after_native_long_forward")
        check("expected_output_shapes", [list(t.shape) for t in output] ==
              [[item["video_tokens"],96],[item["audio_tokens"],32]])
        check("all_outputs_finite", all(torch.isfinite(t).all().item() for t in output))
        check("all_groups_streamed_once", loader.stats["groups_loaded"] == 3 and
              loader.stats["max_active_streamed_groups"] == 1)
        check("only_resident_after_forward", loader.active == {"resident"})
        check("all_streamed_tensors_meta", all(_state(model)[n].is_meta for g,names in plan.groups.items()
                                             if g != "resident" for n in names))
        result["output_shapes"] = [list(t.shape) for t in output]
        result["loader_stats"] = dict(loader.stats)
        del output, kwargs, branch, hidden, video_rows, audio_rows, packed
        loader.close()
        gc.collect()
        mark("closed_and_inputs_released")
        check("all_model_tensors_meta_after_close", all(t.is_meta for t in _state(model).values()))
        check("checkpoint_mmaps_closed", str(a.fixture) not in Path("/proc/self/maps").read_text())
        code = 0
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        if loader is not None:
            loader.close()
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            if torch.distributed.is_initialized():
                destroy_model_parallel()
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result["checks_total"] = len(result["checks"])
    result["checks_ok"] = sum(c["ok"] for c in result["checks"])
    result["exit_code"] = code
    text = json.dumps(result, indent=2)
    for source,label in ((str(a.output.parent),"<RUN_ROOT>"),(str(a.fixture),"<REUSED_FIXTURE>"),
                         (str(a.geometry),"<FROZEN_GEOMETRY>"),(str(Path.home()),"<LINUX_HOME>"),
                         (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
        text = text.replace(source,label)
    a.output.write_text(text+"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
