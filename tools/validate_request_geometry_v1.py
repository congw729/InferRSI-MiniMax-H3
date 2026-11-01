"""Frozen v1: execute real request constraints and native token geometry without weights."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def native_geometry(seconds, size, text_tokens):
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER as planner
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    frames = planner.align_frame_count(round(seconds * 24))
    lt = planner.video_latent_t(frames)
    at = planner.audio_latent_t(frames / 24)
    packed = minimax_h3_packed_sequence(text_len=text_tokens, latent_t=lt, latent_h=size//16,
        latent_w=size//16, audio_t=at, include_keyframe_cond=False)
    return {"requested_seconds": seconds, "requested_frames": round(seconds*24),
        "frames": frames, "fps": 24, "aligned_seconds": frames/24, "height": size, "width": size,
        "latent_t": lt, "latent_h": size//16, "latent_w": size//16, "audio_t": at,
        "text_tokens": text_tokens, "video_tokens": packed["img_pos"].numel(),
        "audio_tokens": packed["audio_pos"].numel(), "used_tokens": int(packed["cu_seqlens"][1]),
        "padded_tokens": int(packed["seq_len"]), "patch_size": [1, 2, 2]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "payload_read": False, "checks": [], "cases": []}
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import resolve_minimax_h3_shape
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.errors import OmniClientError
    for seconds, size in ((3, 128), (3, 256), (4, 128), (4, 256)):
        sample = SimpleNamespace(fps=24, extra_args={"duration_seconds": seconds}, height=size,
                                 width=size, num_frames=round(seconds*24))
        item = native_geometry(seconds, size, 64)
        try:
            shape = resolve_minimax_h3_shape("t2va", sample, None)
        except OmniClientError as exc:
            item.update(request_accepted=False, request_error=str(exc))
            check(f"request_{seconds}s_{size}_rejected_as_expected", seconds == 3 and "duration" in str(exc))
        else:
            item.update(request_accepted=True, resolved_shape=list(shape))
            check(f"request_{seconds}s_{size}_accepted_as_expected", seconds == 4)
            check(f"request_shape_matches_planner_{size}", list(shape) ==
                  [size,size,item["frames"],item["latent_t"],item["audio_t"]])
        check(f"all_modalities_counted_{seconds}_{size}", item["used_tokens"] ==
              item["text_tokens"]+item["video_tokens"]+item["audio_tokens"])
        check(f"native_64_alignment_{seconds}_{size}", item["padded_tokens"]%64 == 0 and
              item["used_tokens"] <= item["padded_tokens"] < item["used_tokens"]+64)
        result["cases"].append(item)
    result["source_provenance"] = {}
    for obj in (resolve_minimax_h3_shape, type(MINIMAX_H3_SHAPE_PLANNER), minimax_h3_packed_sequence):
        source = Path(inspect.getfile(obj))
        result["source_provenance"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["checks_total"] = len(result["checks"])
    result["checks_ok"] = sum(c["ok"] for c in result["checks"])
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "checks"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
