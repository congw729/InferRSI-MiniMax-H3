"""Frozen v2: real native video/audio VAE constructors on meta, header coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import traceback

import torch
import yaml


def audit(model, header, defaults):
    state = dict(list(model.named_parameters()) + list(model.named_buffers()))
    errors = []
    inventory = []
    for key, value in state.items():
        item = {"name": key, "shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device), "bytes": value.numel() * value.element_size()}
        if key in header:
            info = header[key]
            if info["shape"] != list(value.shape):
                errors.append({"shape": key, "actual": list(value.shape), "source": info["shape"]})
            if info["dtype"] != "F32" or value.dtype != torch.float32:
                errors.append({"dtype": key, "actual": str(value.dtype), "source": info["dtype"]})
            item["source"] = "checkpoint"
        elif key in defaults:
            item["source"] = defaults[key]
        else:
            errors.append({"missing": key})
        if value.device.type != "meta":
            errors.append({"non_meta": key})
        inventory.append(item)
    errors.extend({"extra": key} for key in header.keys() - state.keys())
    return {"parameters": len(list(model.named_parameters())), "buffers": len(list(model.named_buffers())),
            "targets": len(state), "source_keys": len(header), "logical_bytes": sum(t["bytes"] for t in inventory),
            "errors": errors, "inventory": inventory}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "payload_read": False, "model_forward": False,
              "scope": "unmodified pinned native video and audio constructors on meta"}
    code = 1
    try:
        sys.path.insert(0, str(a.native_source / "Ref2VA"))
        from video_vae.klvae import AutoencoderKLLegacy
        from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
        from audio_vae.dac_audio_vae import DacAudioVAE
        _ensure_vae_parallel_state()
        def header(name):
            path = a.research / "memory-budget" / name
            obj = json.loads(path.read_text())
            if hashlib.sha256(path.with_suffix(".raw").read_bytes()).hexdigest() != obj["header_sha256"]:
                raise ValueError("header digest mismatch")
            return {k: v for k, v in obj["header"].items() if k != "__metadata__"}
        config = json.loads((a.native_source / "Ref2VA/video_vae/source/config.json").read_text())
        with torch.device("meta"):
            video = AutoencoderKLLegacy.from_config(config, clip_length=17, token_drop=3,
                encoder_tiling=True, decoder_tiling=True, parallel_tiling=True,
                tile_size=256, tile_overlap_min=64, encoder_parallel=False, decoder_parallel=False, chunk_dim=-1)
        result["video"] = audit(video,
            header("MiniMaxAI__MiniMax-H3__Ref2VA__video_vae__source__model.safetensors.header.json"),
            {"decoder.pos_embed.inv_freq": "RotaryEmbeddingND exact constructor formula, nonpersistent"})
        del video
        metadata = json.loads((a.native_source / "Ref2VA/audio_vae/metadata.json").read_text())["metadata"]["kwargs"]
        config = yaml.safe_load((a.native_source / "Ref2VA/audio_vae/config.yaml").read_text())["model_config"]
        with torch.device("meta"):
            audio = DacAudioVAE(encoder_rates=metadata["encoder_rates"], decoder_rates=metadata["decoder_rates"],
                attn_proj=metadata["attn_proj"], decoder_type=metadata["decoder_type"],
                decoder_dim=config["decoder_dim"], vae_latent_channels=config["vae_latent_channels"],
                sample_rate=metadata.get("sample_rate", config["sr"]))
        result["audio"] = audit(audio,
            header("MiniMaxAI__MiniMax-H3__Ref2VA__audio_vae__model.safetensors.header.json"), {})
        code = int(any(result[k]["errors"] for k in ("video", "audio")))
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["exit_code"] = code
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: ({x: y for x, y in v.items() if x != "inventory"} if isinstance(v, dict) else v)
                      for k, v in result.items() if k != "traceback"}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
