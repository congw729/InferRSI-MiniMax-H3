"""Frozen v1: tiny native-constructor temporal reference vs compact decoder-only host."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import traceback
import weakref
from contextlib import contextmanager

import torch


@contextmanager
def temporal_mode(stream):
    keys = {"MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT": str(int(stream)),
            "MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE": "keep",
            "VLLM_OMNI_VAE_LEGACY_TEMPORAL": "0"}
    old = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def native_kwargs(outer):
    return {key: int(outer["vae_" + key]) for key in ("clip_length", "token_drop",
        "encoder_tiling", "decoder_tiling", "parallel_tiling", "tile_size", "tile_overlap_min",
        "encoder_parallel", "decoder_parallel", "chunk_dim")}


def make_reference(config, outer, fixture):
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.vae_vit import ViT3DDecoder
    from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import install_nvfp4_video_decoder
    with torch.device("meta"):
        model = AutoencoderKLLegacy.from_config(copy.deepcopy(config), **native_kwargs(outer))
    if any(not value.is_meta for value in model.parameters()):
        raise AssertionError("reference constructor allocated non-meta parameters")
    model.encoder = None
    model.quant_conv = None
    options = dict(config["vit_decoder_kwargs"])
    options.update(patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, t_causal=False)
    model.decoder = ViT3DDecoder(**options)
    model.post_quant_conv = torch.nn.Conv3d(24, 24, 1)
    install_nvfp4_video_decoder(model, fixture)
    return model.eval().to("cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--tiny-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "real_weights": False, "checks": [],
              "scope": "tiny nonuniform native temporal and FP32 pixel equality"}
    code = 1

    def check(name, ok, **details):
        result["checks"].append({"name": name, "ok": bool(ok), **details})
        if not ok:
            raise AssertionError(name)

    def rejects(name, call):
        try:
            call()
        except (ValueError, RuntimeError, TypeError):
            check(name, True)
        else:
            check(name, False)

    try:
        torch.cuda.set_per_process_memory_fraction(0.40)
        sys.path.insert(0, str(args.native_source / "Ref2VA"))
        from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
        from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import released_decoder_pixels
        from h3_omni_adapter.mmap_loader import _state
        from h3_omni_adapter.temporal_decoder import H3TemporalVideoDecoder
        _ensure_vae_parallel_state()
        config = json.loads((args.native_source / "Ref2VA/video_vae/source/config.json").read_text())
        outer = json.loads((args.native_source / "Ref2VA/video_vae/config.json").read_text())
        config["vit_decoder_kwargs"].update(heads=2, num_layers=1)
        result["fixture_sha256"] = hashlib.sha256(args.tiny_fixture.read_bytes()).hexdigest()
        reference = make_reference(config, outer, args.tiny_fixture)
        raw = H3TemporalVideoDecoder(config, outer, args.tiny_fixture, device="cuda",
            max_latent_tokens=32, max_height=64, max_width=64, max_output_frames=107, output_mode="raw")
        pixels = H3TemporalVideoDecoder(config, outer, args.tiny_fixture, device="cuda",
            max_latent_tokens=32, max_height=64, max_width=64, max_output_frames=107, output_mode="uint8")
        check("candidate_constructs_no_encoder_or_quant_conv", not hasattr(raw.native, "encoder")
              and not hasattr(raw.native, "quant_conv") and not hasattr(pixels.native, "encoder"))
        ref_state = _state(reference)
        check("all_candidate_tensor_names_exact", set(ref_state) == set(_state(raw.native)) == set(_state(pixels.native)))
        check("all_candidate_tensor_bytes_exact_native_installer", all(
            torch.equal(value, _state(raw.native)[name]) and torch.equal(value, _state(pixels.native)[name])
            for name, value in ref_state.items()))
        del ref_state
        counts = {"reference": 0, "raw": 0, "uint8": 0}
        def counter(name):
            def prehook(module, inputs):
                counts[name] += 1
            return prehook
        handles = [model.decoder.register_forward_pre_hook(counter(name)) for name, model in
                   (("reference", reference), ("raw", raw.native), ("uint8", pixels.native))]
        cases = []
        for latent_t, frame_count, clips in ((7, 22, 1), (13, 43, 3), (22, 73, 4), (32, 107, 6)):
            torch.manual_seed(1900 + latent_t)
            latent = torch.randn(1, 24, latent_t, 2, 2, device="cuda")
            mean = torch.tensor(outer["latents_mean"], device="cuda", dtype=latent.dtype).reshape(1, 24, 1, 1, 1)
            std = torch.tensor(outer["latents_std"], device="cuda", dtype=latent.dtype).reshape(1, 24, 1, 1, 1)
            denormalized = latent * std + mean
            baseline = None
            for stream in (False, True):
                before = dict(counts)
                with temporal_mode(stream), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected_raw = reference.decode_base(denormalized)
                    expected_pixels = released_decoder_pixels(expected_raw).mul(255).round().to(torch.uint8)
                    actual_raw = raw.decode_latent(latent)
                    actual_pixels = pixels.decode_latent(latent)
                name = f"t{latent_t}_stream{int(stream)}"
                check(name + "_raw_exact", torch.equal(actual_raw, expected_raw))
                check(name + "_uint8_exact_FP32_reference", torch.equal(actual_pixels, expected_pixels))
                check(name + "_geometry", tuple(actual_pixels.shape) == (1, 3, frame_count, 32, 32))
                check(name + "_finite_nonuniform", torch.isfinite(actual_raw).all().item()
                      and actual_raw.min().item() != actual_raw.max().item()
                      and actual_pixels.min().item() != actual_pixels.max().item())
                check(name + "_clip_count", all(counts[k] - before[k] == clips for k in counts))
                if baseline is None:
                    baseline = actual_pixels.clone()
                else:
                    check(name + "_stream_vs_nonstream_exact", torch.equal(baseline, actual_pixels))
                cases.append({"latent_t": latent_t, "stream": stream, "frames": frame_count,
                              "raw_max_abs": (actual_raw.float() - expected_raw.float()).abs().max().item(),
                              "pixel_mismatches": torch.count_nonzero(actual_pixels != expected_pixels).item()})
                del actual_raw, actual_pixels, expected_raw, expected_pixels
                if torch.cuda.max_memory_allocated() > 4 * 1024**3:
                    raise RuntimeError("CUDA phase peak exceeds protection")
            del latent, mean, std, denormalized, baseline
        result["cases"] = cases
        for handle in handles:
            handle.remove()
        del handles
        valid = torch.zeros(1, 24, 7, 2, 2, device="cuda")
        rejects("wrong_channels_rejected", lambda: raw.decode_latent(valid[:, :23]))
        rejects("batch_gt_one_rejected", lambda: raw.decode_latent(valid.expand(2, -1, -1, -1, -1)))
        rejects("integer_dtype_rejected", lambda: raw.decode_latent(valid.to(torch.int32)))
        rejects("CPU_input_rejected", lambda: raw.decode_latent(valid.cpu()))
        rejects("meta_input_rejected", lambda: raw.decode_latent(torch.empty(1, 24, 7, 2, 2, device="meta")))
        rejects("time_budget_rejected", lambda: raw.decode_latent(torch.zeros(1, 24, 33, 2, 2, device="cuda")))
        rejects("spatial_budget_rejected", lambda: raw.decode_latent(torch.zeros(1, 24, 7, 5, 2, device="cuda")))
        rejects("too_short_temporal_rejected", lambda: raw.decode_latent(valid[:, :, :2]))
        refs = [weakref.ref(value) for model in (reference, raw.native, pixels.native) for value in _state(model).values()]
        raw.close()
        pixels.close()
        raw.close()
        rejects("closed_decoder_rejected", lambda: raw.decode_latent(valid))
        del reference, raw, pixels, valid
        gc.collect()
        torch.cuda.synchronize()
        check("all_model_tensor_objects_released", all(ref() is None for ref in refs))
        check("checkpoint_mmap_closed", str(args.tiny_fixture) not in Path("/proc/self/maps").read_text())
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["cuda_allocated_after_release"] = torch.cuda.memory_allocated()
        code = 0
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["checks_ok"] = sum(item["ok"] for item in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["exit_code"] = code
    text = json.dumps(result, indent=2)
    for source, label in ((str(args.output.parent), "<RUN_ROOT>"), (str(args.tiny_fixture), "<TINY_FIXTURE>"),
            (str(args.native_source), "<NATIVE_SOURCE>"), (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"),
            (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
