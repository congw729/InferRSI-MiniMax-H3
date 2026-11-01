"""Shared native geometry for the frozen small three-stage integration probe."""
import gc
import hashlib
import json
import shutil
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from validate_meta_coverage_v2 import make_model, quant_from_headers, read_inputs
from validate_mmap_v2 import native_load, state


def prepare(research, text_fixture, decoder_fixture, native_source, destination):
    destination.mkdir()
    if (text_fixture / "model.safetensors").stat().st_size > 16*1024**2 or decoder_fixture.stat().st_size > 16*1024**2:
        raise ValueError("small fixture budget")
    text = destination / "text"
    text.mkdir()
    config = json.loads((text_fixture / "config.json").read_text())
    config["quantization_config"] = json.loads((research / "config-text_encoder__config.json").read_text())["content"]["quantization_config"]
    (text / "config.json").write_text(json.dumps(config))
    shutil.copyfile(text_fixture / "model.safetensors", text / "model.safetensors")
    with safe_open(text / "model.safetensors", framework="pt", device="cpu") as reader:
        index = {key: "model.safetensors" for key in reader.keys()}
    (text / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    dit_fixture(research, destination / "dit")
    shutil.copyfile(decoder_fixture, destination / "decoder.safetensors")
    config = json.loads((native_source / "Ref2VA/video_vae/source/config.json").read_text())
    config["vit_decoder_kwargs"].update(heads=2, num_layers=1)
    (destination / "decoder-config.json").write_text(json.dumps(config))
    shutil.copyfile(native_source / "Ref2VA/video_vae/config.json", destination / "vae-normalization.json")
    files = {p.relative_to(destination).as_posix(): {"bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in destination.rglob("*") if p.is_file()}
    if sum(v["bytes"] for v in files.values()) > 32*1024**2:
        raise ValueError("total small fixture budget")
    (destination / "fixture-manifest.json").write_text(json.dumps(files, indent=2))


def text_model(root, streamed):
    from transformers import Qwen3VLConfig
    from vllm.distributed import get_tp_group
    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
    quant = FastH3NVFP4Config.from_encoder(root, rank=0)
    if streamed:
        from h3_omni_adapter.text_streaming import H3TextMmapLoader, H3TextMmapPlan, H3TextOnly
        config = Qwen3VLConfig.from_dict(json.loads((root / "config.json").read_text()))
        model = H3TextOnly(config, get_tp_group(), quant, max_tokens=64)
        plan = H3TextMmapPlan.build(model, root, files=["model.safetensors"], max_total_file_bytes=16*1024**2)
        loader = H3TextMmapLoader(model, plan, device="cuda", max_group_bytes=8*1024**2)
        loader.load_group("resident")
        loader.install_block_hooks()
        return model, loader
    model = MiniMaxH3Qwen3VLEncoder(str(root), device=torch.device("cuda"), load_model=True,
                                  encoder_group=get_tp_group(), quant_config=quant)
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as reader:
        model.load_weights((key, reader.get_tensor(key)) for key in reader.keys())
    model.text_model.to("cuda")
    return model, None


def text_hidden(model, streamed):
    ids = torch.tensor([1, 8], dtype=torch.long)
    if streamed:
        return model.encode_ids(ids)
    from validate_text_streaming_v3 import reference_forward
    return reference_forward(model, ids)


def dit_model(root, streamed):
    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
    config = json.loads((root / "config.json").read_text())
    quant = FastH3NVFP4Config.from_dit(root / "nvfp4_weights.safetensors", rank=0)
    model = make_model(config, quant, device="meta" if streamed else "cpu").eval()
    if streamed:
        from h3_omni_adapter.mmap_loader import H3MmapLoader, H3MmapPlan
        plan = H3MmapPlan.build(model, root, sidecars=["nvfp4_weights.safetensors"])
        loader = H3MmapLoader(model, plan, device="cuda", max_group_bytes=8*1024**2)
        loader.load_group("resident")
        loader.install_block_hooks()
    else:
        native_load(model, root)
        model.to("cuda")
        loader = None
    return model, loader, config


def denoise(model, config, hidden):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )
    from vllm_omni.diffusion.models.minimax_h3.pdd import pdd_sigmas
    packed = minimax_h3_packed_sequence(text_len=2, latent_t=2, latent_h=2, latent_w=2,
                                       audio_t=1, include_keyframe_cond=False)
    branch = MiniMaxH3DenoiseBranch(packed=packed, text_embeddings=hidden,
        token_tags=packed["token_tags"], device=torch.device("cuda"))
    initial = torch.randn(1, 24, 2, 2, 2, generator=torch.Generator().manual_seed(17))
    rows = minimax_h3_patchify_video_latent(initial, patch_size=(1, 2, 2))
    roundtrip = minimax_h3_unpatchify_video_tokens(rows, latent_shape=(2, 1, 1, 24), patch_size=(1, 2, 2))
    if not torch.equal(initial, roundtrip):
        raise AssertionError("native patch inverse")
    sigmas_video, sigmas_audio = pdd_sigmas(12.0), pdd_sigmas(3.0)
    audio = torch.randn(2, 32, generator=torch.Generator().manual_seed(17))
    rows *= sigmas_video[0]
    audio *= sigmas_audio[0]
    steps = []
    def on_step(step, video_rows, audio_rows):
        steps.append({"step": step, "video_rows": video_rows.detach().cpu().clone(),
                      "audio_rows": audio_rows.detach().cpu().clone()})
    od = OmniDiffusionConfig(model=None, model_class_name="MiniMaxH3Pipeline",
        tf_model_config=TransformerConfig.from_dict(config),
        cache_config={"minimax_h3_adaln_cache": False, "minimax_h3_adaln_offload": False})
    with torch.inference_mode(), set_forward_context(omni_diffusion_config=od):
        vr, ar = minimax_h3_denoise_loop(model=model, positive=branch, initial_video_rows=rows,
            initial_audio_rows=audio, keyframe_cond_rows=None, sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio, device=torch.device("cuda"), on_step=on_step)
    latent = minimax_h3_unpatchify_video_tokens(vr, latent_shape=(2, 1, 1, 24), patch_size=(1, 2, 2))
    return {"kind": "native_sampled_latents", "video_latent": latent.cpu(), "audio_rows": ar.cpu(),
            "steps": steps, "sampler": "native Euler with native 8-step PDD sigmas",
            "patch_roundtrip": True, "used_tokens": int(packed["cu_seqlens"][1]),
            "padded_tokens": int(packed["seq_len"]), "not_raw_velocity": True}


def decoder_model(checkpoint, streamed):
    from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
    from video_vae.vae_vit import ViT3DDecoder
    from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import install_nvfp4_video_decoder

    from h3_omni_adapter.compact_decoder import load_compact_decoder_mmap
    _ensure_vae_parallel_state()
    config = json.loads((checkpoint / "decoder-config.json").read_text())
    if streamed:
        model, _ = load_compact_decoder_mmap(config, checkpoint / "decoder.safetensors", device="cuda")
        return model
    opts = dict(config["vit_decoder_kwargs"])
    opts.update(patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, t_causal=False)
    model = torch.nn.Module()
    model.decoder = ViT3DDecoder(**opts)
    model.post_quant_conv = torch.nn.Conv3d(24, 24, 1)
    install_nvfp4_video_decoder(model, checkpoint / "decoder.safetensors")
    return model.eval().to("cuda")


def decode(model, checkpoint, payload, streamed):
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    if payload.get("kind") != "native_sampled_latents" or not payload.get("not_raw_velocity"):
        raise ValueError("decoder requires sampled latent payload")
    latent = payload["video_latent"]
    if tuple(latent.shape) != (1, 24, 2, 2, 2) or not torch.isfinite(latent).all():
        raise ValueError("invalid small video latent")
    normalization = SimpleNamespace(config_dict=json.loads((checkpoint / "vae-normalization.json").read_text()))
    denormalized = MiniMaxH3VideoVAE._denormalize_latent(normalization, latent.cuda())
    decoder = model["decoder"] if streamed else model.decoder
    post_quant = model["post_quant_conv"] if streamed else model.post_quant_conv
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = decoder(post_quant(denormalized))
    return {"kind": "synthetic_native_raw_decoder_output", "raw_frames": output.cpu(),
            "model_video_generated": False, "latent_denormalization": "native per-channel mean/std",
            "output_scope": "raw decoder tensor; no pixel processor, trim, audio decode, MP4 or quality claim"}


def close_loader(model, loader):
    if loader is None:
        return None
    stats = dict(loader.stats)
    loader.close()
    if not all(t.is_meta for t in state(model).values()):
        raise AssertionError("streamed model retained real tensors")
    gc.collect()
    return stats


def dit_fixture(research, root):
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
    config, dense, packed, _ = read_inputs(research)
    config.update(num_layers=2, num_refiner_layers=1, hidden_size=128, num_attention_heads=1,
                  attention_head_dim=128, ffn_dim=128, text_dim=128, freq_dim=64,
                  time_embed_hidden_dim=128, time_embed_dim=64,
                  adaln_out_features=18 * 128, final_adaln_out_features=2 * 128,
                  in_channels=24, audio_in_channels=32)
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
