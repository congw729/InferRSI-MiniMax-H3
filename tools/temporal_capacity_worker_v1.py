"""Frozen v1: complete 107-frame uint8 temporal decode with real 8-layer synthetic weights."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import weakref

import torch
from decoder_capacity_worker_v1 import verify_fixture
from validate_temporal_decoder_v2 import temporal_mode

GIB = 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "real_weights": False,
        "scope": "complete 8-layer synthetic decoder, six native temporal clips, 107x256x256 uint8 output", "checks": []}
    started, code = time.monotonic(), 1

    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)

    def mark(name):
        torch.cuda.synchronize()
        row = {"phase": name, "elapsed_seconds": time.monotonic() - started,
            "cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
            "cuda_max_allocated": torch.cuda.max_memory_allocated(), "cuda_max_reserved": torch.cuda.max_memory_reserved(),
            "worker_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        with (args.output.parent / "phase-events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        result.setdefault("phases", []).append(row)
        if max(row["cuda_allocated"], row["cuda_max_allocated"]) > 4 * GIB:
            raise RuntimeError("CUDA phase high-water protection exceeded")

    try:
        if shutil.disk_usage(args.output.parent).free < 8 * GIB:
            raise RuntimeError("insufficient output disk")
        path = args.research / "memory-budget/FastVideo__FastH3-NVFP4-VAE-Decoder__fast-8L__decoder-nvfp4.safetensors.header.json"
        saved = json.loads(path.read_text())
        check("source_header_SHA_intact", hashlib.sha256(path.with_suffix(".raw").read_bytes()).hexdigest() == saved["header_sha256"])
        sources = {key: value for key, value in saved["header"].items() if key != "__metadata__"}
        manifest = json.loads((args.fixture / "fixture-manifest.json").read_text())
        checkpoint = args.fixture / "decoder-synthetic.safetensors"
        verify_fixture(checkpoint, manifest, sources)
        check("reused_fixture_SHA_metadata_bytes_intact", True)
        result["reused_fixture"] = manifest
        sys.path.insert(0, str(args.native_source / "Ref2VA"))
        from h3_omni_adapter.temporal_decoder import H3TemporalVideoDecoder
        from h3_omni_adapter.mmap_loader import _state
        config = json.loads((args.native_source / "Ref2VA/video_vae/source/config.json").read_text())
        outer = json.loads((args.native_source / "Ref2VA/video_vae/config.json").read_text())
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        if free_cuda < 5 * GIB:
            raise RuntimeError("insufficient CUDA headroom")
        torch.cuda.set_per_process_memory_fraction(0.40)
        result["cuda_initial"] = {"free_bytes": free_cuda, "total_bytes": total_cuda, "allocator_fraction": 0.40}
        mark("verified_reused_fixture")
        with temporal_mode(True):
            candidate = H3TemporalVideoDecoder(config, outer, checkpoint, device="cuda",
                max_latent_tokens=32, max_height=256, max_width=256, max_output_frames=107, output_mode="uint8")
            check("complete_real_8layer_host_without_encoder", len(candidate.native.decoder.transformer_blocks) == 8
                  and not hasattr(candidate.native, "encoder") and len(_state(candidate.native)) == 378)
            check("native_uint8_patch_installed", "_decode_temporal_streaming" in vars(candidate.native))
            check("all_parameters_and_buffers_materialized", all(not value.is_meta for value in _state(candidate.native).values()))
            check("mmap_closed_after_loading", str(checkpoint) not in Path("/proc/self/maps").read_text())
            mark("temporal_decoder_loaded")
            clips = []
            def count_clip(module, inputs):
                clips.append(list(inputs[0].shape))
            hook = candidate.native.decoder.register_forward_pre_hook(count_clip)
            torch.manual_seed(212)
            latent = torch.randn(1, 24, 32, 16, 16, device="cuda")
            mark("before_complete_107frame_decode")
            output = candidate.decode_latent(latent)
            check("complete_output_geometry", tuple(output.shape) == (1, 3, 107, 256, 256))
            check("uint8_output", output.dtype == torch.uint8)
            check("six_native_clips_of_seven_latents", clips == [[1, 24, 7, 16, 16]] * 6)
            check("native_output_frame_plan", candidate.last_plan == {
                "chunks": 6, "pad_tokens": 0, "total_frames": 107, "pad_frames": 0, "output_frames": 107})
            result["frame_plan"] = candidate.last_plan
            result["clip_shapes"] = clips
            result["output_shape"] = list(output.shape)
            result["output_dtype"] = str(output.dtype)
            result["output_min_max"] = [output.min().item(), output.max().item()]
            result["output_sha256"] = hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()
            result["output_bytes"] = output.numel()
            mark("after_complete_107frame_decode")
            hook.remove()
            del hook
            refs = [weakref.ref(value) for value in _state(candidate.native).values()]
            allocated = torch.cuda.memory_allocated()
            candidate.close()
            del candidate, latent, output
            gc.collect()
            mark("decoder_and_output_released")
            check("all_model_tensor_objects_released", all(ref() is None for ref in refs))
            check("CUDA_allocation_drops_after_release", torch.cuda.memory_allocated() < allocated)
            check("mmap_closed_after_release", str(checkpoint) not in Path("/proc/self/maps").read_text())
        code = 0
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["checks_ok"] = sum(item["ok"] for item in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["exit_code"] = code
    text = json.dumps(result, indent=2)
    for source, label in ((str(args.output.parent), "<RUN_ROOT>"), (str(args.research), "<RESEARCH_ROOT>"),
            (str(args.native_source), "<NATIVE_SOURCE>"), (str(args.fixture), "<REUSED_FIXTURE>"),
            (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
