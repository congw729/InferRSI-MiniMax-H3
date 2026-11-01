"""Frozen v1: complete real-width 8-layer compact decoder, one native temporal clip."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import struct
import sys
import time
import traceback
from types import MethodType, SimpleNamespace
import weakref

import torch
from capacity_worker_v1 import WIDTH, element_bytes

GIB = 1024**3
FILE_LIMIT = 512 * 1024**2


def constant(key, dtype):
    if dtype == "U8":
        return 56 if key.endswith(".weight_inv_scale") else 18
    if key.endswith(".weight_global_sf"):
        return 8192.0
    if key.endswith(".input_amax") or (key.endswith(".weight") and "norm" in key):
        return 1.0
    if key.endswith((".scale1", ".scale2")):
        return 0.125
    return 0.0 if key.endswith(".bias") else 1 / 8192


def write_fixture(path, sources, metadata):
    header, offset = {"__metadata__": {**metadata, "synthetic": "true"}}, 0
    for key, info in sorted(sources.items()):
        size = math.prod(info["shape"]) * WIDTH[info["dtype"]]
        header[key] = {"shape": info["shape"], "dtype": info["dtype"], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    if not 0 < offset + len(encoded) + 8 < FILE_LIMIT:
        raise ValueError("fixture exceeds 512MiB")
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        stream.write(prefix)
        digest.update(prefix)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            element = element_bytes(constant(key, info["dtype"]), info["dtype"])
            chunk = element * ((4 * 1024**2) // len(element))
            remaining = info["data_offsets"][1] - info["data_offsets"][0]
            while remaining:
                value = chunk[:min(len(chunk), remaining)]
                stream.write(value)
                digest.update(value)
                remaining -= len(value)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "payload_bytes": offset, "synthetic": True, "keys": len(sources)}


def verify_fixture(path, manifest, sources):
    if path.stat().st_size != manifest["bytes"] or path.stat().st_size >= FILE_LIMIT:
        raise ValueError("fixture size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
        if digest.hexdigest() != manifest["sha256"]:
            raise ValueError("fixture SHA mismatch")
        stream.seek(0)
        header_len = struct.unpack("<Q", stream.read(8))[0]
        if header_len > 2 * 1024**2:
            raise ValueError("header too large")
        header = json.loads(stream.read(header_len))
        if set(header) - {"__metadata__"} != set(sources):
            raise ValueError("fixture key mismatch")
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if any(info[field] != sources[key][field] for field in ("shape", "dtype")):
                raise ValueError("fixture dtype or shape mismatch")
            element = element_bytes(constant(key, info["dtype"]), info["dtype"])
            lo, hi = info["data_offsets"]
            if hi - lo != math.prod(info["shape"]) * WIDTH[info["dtype"]]:
                raise ValueError("tensor byte length mismatch")
            for offset in (lo, hi - len(element)):
                stream.seek(8 + header_len + offset)
                if stream.read(len(element)) != element:
                    raise ValueError("edge bytes mismatch")


def native_frame_plan(outer_config, config):
    from video_vae.klvae import AutoencoderKL
    clip = int(outer_config["vae_clip_length"])
    ratio = int(config["vae_ratio_t"])
    drop = int(outer_config["vae_token_drop"])
    chunk = math.ceil(clip / ratio)
    pre = (-clip) % ratio
    overlap = (-drop) % chunk
    state = SimpleNamespace(clip_length=clip, vae_ratio_t=ratio, token_drop=drop,
        tokens_chunk_size=chunk, frame_pre_padding=pre, token_overlap=overlap)
    state._decode_temporal_pad_frames = MethodType(AutoencoderKL._decode_temporal_pad_frames, state)
    cases = []
    for latent_t in (22, 32):
        pseudo = latent_t + drop
        pad = (-pseudo) % chunk
        num_chunks = (pseudo + pad) // chunk - int(drop > 0)
        shape_only = SimpleNamespace(shape=(1, 24, latent_t + pad, 1, 1))
        frames = AutoencoderKL._decode_temporal_output_frame_plan(
            state, shape_only, None, None, num_chunks, pad)
        cases.append({"latent_t": latent_t, "chunks": num_chunks, "pad_tokens": pad,
                      "total_frames": frames[0], "pad_frames": frames[1], "output_frames": frames[2]})
    return {"scope": "actual native frame-plan methods with minimal scalar/shape state; not full temporal decoding",
            "clip_length": clip, "ratio_t": ratio, "token_drop": drop, "chunk_tokens": chunk,
            "token_overlap": overlap, "frame_pre_padding": pre, "frame_overlap": max(overlap * ratio - pre, 0),
            "decoder_clip_latents": chunk + overlap, "decoder_clip_raw_frames": (chunk + overlap) * ratio,
            "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--size", type=int, choices=(128, 256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "real_weights": False, "checks": [],
              "scope": "complete real-shape 8-layer compact decoder; one 7-latent clip, raw 28-frame output"}
    code = 1
    started = time.monotonic()

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
        sys.path.insert(0, str(args.native_source / "Ref2VA"))
        from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
        from video_vae.base_module import RotaryEmbeddingND
        from h3_omni_adapter.compact_decoder import build_compact_decoder_meta, load_compact_decoder_mmap
        from h3_omni_adapter.mmap_loader import _state
        _ensure_vae_parallel_state()
        native_root = args.native_source / "Ref2VA/video_vae"
        config = json.loads((native_root / "source/config.json").read_text())
        outer_config = json.loads((native_root / "config.json").read_text())
        header_path = args.research / "memory-budget/FastVideo__FastH3-NVFP4-VAE-Decoder__fast-8L__decoder-nvfp4.safetensors.header.json"
        saved = json.loads(header_path.read_text())
        check("saved_header_SHA_intact",
              hashlib.sha256(header_path.with_suffix(".raw").read_bytes()).hexdigest() == saved["header_sha256"])
        result["source_provenance"] = {
            "saved_header_sha256": saved["header_sha256"],
            "native_files": {str(p.relative_to(native_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (native_root / "klvae.py", native_root / "config.json", native_root / "source/config.json")}}
        geometry = native_frame_plan(outer_config, config)
        result["geometry"] = geometry
        check("native_frame_plans_73_and_107", [c["output_frames"] for c in geometry["cases"]] == [73, 107]
              and [c["chunks"] for c in geometry["cases"]] == [4, 6])
        check("native_clip_seven_latents_28_raw_with_five_overlap",
              geometry["decoder_clip_latents"] == 7 and geometry["decoder_clip_raw_frames"] == 28
              and geometry["frame_overlap"] == 5 and geometry["frame_pre_padding"] == 3)
        header = saved["header"]
        sources = {key: value for key, value in header.items() if key != "__metadata__"}
        meta, mapping, defaults = build_compact_decoder_meta(config, header)
        check("all_378_real_targets_covered_on_meta", len(_state(meta)) == 378
              and all(t.is_meta for t in _state(meta).values())
              and set(mapping.values()) | set(defaults) == set(_state(meta)))
        check("eight_real_width_layers_without_encoder", len(meta["decoder"].transformer_blocks) == 8
              and config["vit_decoder_kwargs"]["heads"] * config["vit_decoder_kwargs"]["dim_head"] == 2048
              and set(meta.keys()) == {"decoder", "post_quant_conv"})
        result["logical_model_bytes"] = sum(t.numel() * t.element_size() for t in _state(meta).values())
        del meta
        fixture = args.fixture or args.output.parent / "fixture"
        if args.fixture is None:
            fixture.mkdir()
            manifest = write_fixture(fixture / "decoder-synthetic.safetensors", sources, header["__metadata__"])
            (fixture / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2))
            (fixture / "config.json").write_text(json.dumps(config, indent=2))
        else:
            manifest = json.loads((fixture / "fixture-manifest.json").read_text())
            if json.loads((fixture / "config.json").read_text()) != config:
                raise ValueError("reused synthetic config differs")
        path = fixture / "decoder-synthetic.safetensors"
        verify_fixture(path, manifest, sources)
        result.update(fixture=manifest, fixture_reused=args.fixture is not None)
        check("full_fixture_SHA_metadata_edge_bytes_before_GPU", True)
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        result["cuda_initial"] = {"free_bytes": free_cuda, "total_bytes": total_cuda, "allocator_fraction": 0.40}
        if free_cuda < 5 * GIB:
            raise RuntimeError("insufficient CUDA headroom")
        torch.cuda.set_per_process_memory_fraction(0.40)
        mark("verified_fixture_and_native_geometry")
        model, stats = load_compact_decoder_mmap(config, path, device="cuda", max_file_bytes=FILE_LIMIT)
        result["loader"] = stats
        mark("full_8layer_decoder_loaded")

        def verify_values(candidate):
            current = _state(candidate)
            for source, target in mapping.items():
                low, high = torch.aminmax(current[target])
                expected = constant(source, sources[source]["dtype"])
                if low.item() != expected or high.item() != expected:
                    raise AssertionError("loaded value mismatch: " + target)
            opts = config["vit_decoder_kwargs"]
            rope = RotaryEmbeddingND(int(opts["dim_head"] * opts["rope_dim_ratio"]),
                opts["rope_theta"], n_dim=3, use_angle=True)
            for name in defaults:
                value = current[name]
                if name == "decoder.rope.inv_freq":
                    if not torch.equal(value.cpu(), rope.inv_freq):
                        raise AssertionError("native rope formula mismatch")
                else:
                    expected = 1 / 8192 if name.endswith(".alpha") else 1
                    low, high = torch.aminmax(value)
                    if low.item() != expected or high.item() != expected:
                        raise AssertionError("derived buffer mismatch: " + name)
            return True

        check("all_loaded_tensors_and_derived_buffers_exact", verify_values(model))
        check("mmap_closed_after_loading", str(path) not in Path("/proc/self/maps").read_text())
        latent = torch.full((1, 24, 7, args.size // 16, args.size // 16), 0.01, device="cuda")
        mark("before_real_clip_forward")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model["decoder"](model["post_quant_conv"](latent))
        check("raw_clip_geometry", tuple(output.shape) == (1, 3, 28, args.size, args.size))
        check("raw_clip_finite", torch.isfinite(output).all().item())
        result["output_shape"] = list(output.shape)
        result["output_dtype"] = str(output.dtype)
        result["output_min_max"] = [output.min().item(), output.max().item()]
        mark("after_real_clip_forward")
        owned = [weakref.ref(value) for value in _state(model).values()]
        allocated_before = torch.cuda.memory_allocated()
        del model, output, latent
        gc.collect()
        mark("model_and_raw_output_released")
        check("all_model_tensor_objects_released", all(ref() is None for ref in owned))
        check("CUDA_allocation_drops_after_stage", torch.cuda.memory_allocated() < allocated_before)
        check("mmap_closed_after_release", str(path) not in Path("/proc/self/maps").read_text())
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
            (str(args.native_source), "<NATIVE_SOURCE>"), (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"),
            (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    if args.fixture:
        text = text.replace(str(args.fixture), "<REUSED_FIXTURE>")
    args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
