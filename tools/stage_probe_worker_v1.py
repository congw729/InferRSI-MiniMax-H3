"""Frozen synthetic stage worker using real H3 conditioning and NVFP4 code.

Only the denoise worker executes a real small projection. Encode/decode payloads
are synthetic protocol fixtures, not a text encoder or VAE output.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import traceback

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("encode", "denoise", "decode"), required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--fail", action="store_true")
    p.add_argument("--delay", type=float, default=0)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    report = {"stage": a.stage, "campaign_verdict": "INCONCLUSIVE", "model_video_generated": False}
    code = 1
    try:
        if a.delay:
            time.sleep(a.delay)
        if a.fail:
            raise RuntimeError("intentional worker failure")
        import torch
        from h3_omni_adapter.stage_handoff import write_payload, read_payload
        from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning
        if a.device == "cuda":
            report["initial_cuda_allocated"] = torch.cuda.memory_allocated()
            probe = torch.ones(16 * 1024**2, dtype=torch.uint8, device="cuda")
        else:
            probe = torch.ones(16 * 1024**2, dtype=torch.uint8)
        if a.stage == "encode":
            conditioning = MiniMaxH3EncoderConditioning(
                hidden_states=torch.arange(4 * 5120, dtype=torch.float32).reshape(4, 5120).remainder(7).bfloat16() / 16,
                token_tags=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                task="t2va", height=64, width=64, num_frames=5, latent_t=2, audio_t=4)
            payload = {"kind": "synthetic_encoder_conditioning", "conditioning": conditioning.to_omni_payload(),
                       "seed": 17, "fps": 24, "steps": 1,
                       "note": "five-frame protocol fixture; not the requested three-second baseline"}
            decoded = MiniMaxH3EncoderConditioning.from_omni_payload(payload["conditioning"])
            report["conditioning_roundtrip"] = bool(torch.equal(conditioning.hidden_states, decoded.hidden_states)
                                                    and torch.equal(conditioning.token_tags, decoded.token_tags))
        elif a.stage == "denoise":
            payload_in = read_payload(a.input)
            conditioning = MiniMaxH3EncoderConditioning.from_omni_payload(payload_in["conditioning"])
            from validate_meta_coverage_v2 import make_model, setup_dist
            from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
            from h3_omni_adapter.mmap_loader import H3MmapPlan, H3MmapLoader
            setup_dist(a.work)
            config = json.loads((a.checkpoint / "config.json").read_text())
            quant = FastH3NVFP4Config.from_dit(a.checkpoint / "nvfp4_weights.safetensors", rank=0)
            model = make_model(config, quant)
            plan = H3MmapPlan.build(model, a.checkpoint, sidecars=["nvfp4_weights.safetensors"])
            with H3MmapLoader(model, plan, device=a.device, max_group_bytes=32 * 1024**2) as loader:
                loader.load_group("resident")
                loader.load_group("blocks.0")
                if a.device == "cuda":
                    x = conditioning.hidden_states[:, :128].contiguous().to("cuda")
                    with torch.inference_mode():
                        value = model.blocks[0].attn.qkv_proj(x)[0]
                    scalar = value.detach().float().mean().cpu()
                    report["real_nvfp4_projection_executed"] = True
                    report["projection_finite"] = bool(torch.isfinite(value).all())
                else:
                    scalar = torch.tensor(0.125)
                    report["real_nvfp4_projection_executed"] = False
                report["loader_stats"] = dict(loader.stats)
            report["loader_all_meta_after_close"] = all(t.is_meta for t in model.parameters())
            payload = {"kind": "synthetic_decode_latents", "seed": payload_in["seed"], "fps": payload_in["fps"],
                       "steps": payload_in["steps"], "conditioning": payload_in["conditioning"],
                       "minimax_h3_decode": {
                           "video_latents": [torch.full((1, 24, 2, 4, 4), scalar.item(), dtype=torch.float32)],
                           "audio_latents": [torch.full((2, 32, 4), scalar.item(), dtype=torch.float32)],
                           "height": conditioning.height, "width": conditioning.width,
                           "preencode_mp4": False, "preencode_batch_frames": 3, "video_codec_options": {}}}
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        else:
            payload_in = read_payload(a.input)
            conditioning = MiniMaxH3EncoderConditioning.from_omni_payload(payload_in["conditioning"])
            decode = payload_in["minimax_h3_decode"]
            assert len(decode["video_latents"]) == len(decode["audio_latents"]) == 1
            assert tuple(decode["video_latents"][0].shape) == (1, 24, 2, 4, 4)
            assert tuple(decode["audio_latents"][0].shape) == (2, 32, 4)
            assert conditioning.height == decode["height"] == 64
            assert payload_in["seed"] == 17 and payload_in["fps"] == 24 and payload_in["steps"] == 1
            report["text_audio_video_metadata_preserved"] = True
            payload = {"kind": "synthetic_decode_contract_checked", "video_generated": False,
                       "received_video_latent": decode["video_latents"][0],
                       "received_audio_latent": decode["audio_latents"][0], "seed": payload_in["seed"]}
        write_payload(a.work / "handoff", payload)
        del probe
        if a.device == "cuda":
            torch.cuda.synchronize()
            report["cuda_peak_allocated"] = torch.cuda.max_memory_allocated()
        report["completed"] = True
        code = 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    text = json.dumps(report, indent=2)
    for source, label in ((str(a.work), "<STAGE_WORK>"), (str(a.checkpoint), "<SYNTHETIC_CHECKPOINT>"),
                          (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    (a.work / "worker-result.json").write_text(text + "\n")
    print(json.dumps({"stage": a.stage, "completed": report.get("completed", False), "exit_code": code}))
    return code
if __name__ == "__main__":
    raise SystemExit(main())
