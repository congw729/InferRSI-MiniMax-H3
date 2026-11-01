"""Original MiniMax-H3 decoder-only mmap facade; TP1, one inference at a time.

Keeps the checkpoint's spatial/temporal decode methods and Omni pixel math.
The caller supplies reviewed local source imports and explicit resource bounds.
"""
from __future__ import annotations

import copy
import math
import threading
from types import SimpleNamespace

import torch
from torch import nn

from .native_vae_mmap import (
    H3NativeVAEMmapLoader,
    H3NativeVAEMmapPlan,
    build_native_decoder_meta,
)


def _native_host(config, outer, components):
    from diffusers.models import ModelMixin
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
    from video_vae.normalize import get_denormalize_transform, get_normalize_transform
    from video_vae.parallel import get_parallel_state
    from video_vae.utils import apply_spatial_parallel

    _ensure_vae_parallel_state()
    if int(get_parallel_state().get("sp_size", 1)) != 1:
        raise ValueError("native mmap decoder requires a single spatial rank")

    class DecoderOnly(AutoencoderKLLegacy):
        def __init__(self):
            # All decoding methods remain inherited from the original class.
            ModelMixin.__init__(self)
            self.register_to_config(**config)
            self.transform = get_normalize_transform(config["pixel_norm_type"])
            self.transform_rev = get_denormalize_transform(config["pixel_norm_type"])
            self.use_3d_conv = bool(config["use_3d_conv"])
            self.use_vit_decoder = bool(config["use_vit_decoder"])
            self.causal_encoder = bool(config["causal_encoder"])
            self.causal_decoder = bool(config["causal_decoder"])
            self.slidedec = self.causal_encoder and not self.causal_decoder
            self.vae_ratio = math.prod(config["space_down"])
            self.vae_ratio_t = math.prod(config["time_down"]) if config["time_down"] else 1
            self.register_to_config(vae_ratio=self.vae_ratio, vae_ratio_t=self.vae_ratio_t)
            options = {key: int(outer["vae_" + key]) for key in (
                "clip_length", "token_drop", "encoder_tiling", "decoder_tiling",
                "parallel_tiling", "tile_size", "tile_overlap_min",
                "encoder_parallel", "decoder_parallel", "chunk_dim")}
            if options["encoder_parallel"] or options["decoder_parallel"]:
                raise ValueError("tensor-parallel VAE is outside this prototype")
            self.setup_forward(**options)
            self.setup_training()
            self.post_quant_conv = components["post_quant_conv"]
            self.decoder = components["decoder"]
            apply_spatial_parallel(self.decoder, False, self.chunk_dim)
            self.gradient_checkpointing = False
            self.eval()
    return DecoderOnly()


class H3NativeTemporalVideoDecoder(nn.Module):
    """Bounded original F32 VAE weights with one materialized transformer block."""

    def __init__(self, config, outer, path, *, device, max_latent_tokens,
                 max_height, max_width, max_output_frames, max_file_bytes,
                 max_group_bytes, output_mode="uint8"):
        super().__init__()
        self.config_dict = copy.deepcopy(outer)
        config = copy.deepcopy(config)
        bounds = (max_latent_tokens, max_height, max_width, max_output_frames)
        if any(type(x) is not int or x <= 0 for x in bounds):
            raise ValueError("positive explicit geometry budgets required")
        if output_mode not in {"raw", "uint8"}:
            raise ValueError("output mode must be raw or uint8")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("native temporal decode currently requires CUDA")
        if not config["use_3d_conv"] or not config["use_vit_decoder"]:
            raise ValueError("only native 3D ViT decoder is supported")
        if config["embed_dim"] != outer["latent_channels"]:
            raise ValueError("latent channel metadata differs from native decoder")
        channels = outer["latent_channels"]
        for key in ("latents_mean", "latents_std"):
            values = outer[key]
            if len(values) != channels or any(not math.isfinite(x) for x in values):
                raise ValueError("invalid latent normalization metadata")
        if any(x <= 0 for x in outer["latents_std"]):
            raise ValueError("latent standard deviations must be positive")
        components = build_native_decoder_meta(config)
        plan = H3NativeVAEMmapPlan.build(components, path, max_total_file_bytes=max_file_bytes)
        self.loader = H3NativeVAEMmapLoader(
            components, plan, device=device, max_group_bytes=max_group_bytes)
        self.native = None
        self._closed = False
        self._busy = False
        self._owner = threading.get_ident()
        self.output_mode = output_mode
        self.max_latent_tokens, self.max_height, self.max_width, self.max_output_frames = bounds
        try:
            self.loader.load_group("resident")
            self.native = _native_host(config, outer, components)
            self.device = self.native.post_quant_conv.weight.device
            self.loader.install_block_hooks()
            if output_mode == "uint8":
                from vllm_omni.diffusion.models.minimax_h3.vae_temporal import install_temporal_stream_patches
                install_temporal_stream_patches(self.native, pixel_transform=None)
            self.eval()
        except BaseException:
            self.loader.close()
            self.native = None
            self._closed = True
            raise

    def _check_input(self, latent):
        if self._closed or self._owner != threading.get_ident() or self._busy:
            raise RuntimeError("closed, concurrent or reentrant decode")
        if self.training or self.native.training:
            raise RuntimeError("decoder is inference-only")
        if not isinstance(latent, torch.Tensor) or latent.dtype not in {torch.float32, torch.float16}:
            raise TypeError("native latent must be FP32 or FP16")
        if latent.device != self.device:
            raise ValueError("latent must be on the selected CUDA device")
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != self.config_dict["latent_channels"]:
            raise ValueError("expected one batch of native latent channels")
        _, channels, length, height, width = latent.shape
        if min(length, height, width) <= 0 or length > self.max_latent_tokens:
            raise ValueError("latent time budget exceeded")
        if height * self.native.vae_ratio > self.max_height or width * self.native.vae_ratio > self.max_width:
            raise ValueError("decoded spatial budget exceeded")
        native = self.native
        if native.isolated_first_frame or native.isolated_last_frame:
            raise ValueError("isolated endpoint frames are outside this prototype")
        pad = (-(length + native.token_drop)) % native.tokens_chunk_size
        chunks = (length + native.token_drop + pad) // native.tokens_chunk_size - int(native.token_drop > 0)
        if chunks <= 0:
            raise ValueError("input is shorter than one native temporal clip")
        shape_only = SimpleNamespace(shape=(1, channels, length + pad, height, width))
        total, padded, frames = native._decode_temporal_output_frame_plan(
            shape_only, None, None, chunks, pad)
        if not 0 < frames <= self.max_output_frames:
            raise ValueError("decoded frame budget exceeded")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("nonfinite latent")
        if self.loader.active != {"resident"}:
            raise RuntimeError("decoder weights are not in a ready state")
        return {"clips": chunks, "pad_tokens": pad, "total_frames": total,
                "pad_frames": padded, "output_frames": frames}

    @torch.inference_mode()
    def decode_latent(self, latent):
        from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
        plan = self._check_input(latent)
        self._busy = True
        try:
            denormalized = MiniMaxH3VideoVAE._denormalize_latent(self, latent)
            # Original pipeline contract. No global autocast/cache flag mutation.
            with torch.autocast("cuda", dtype=torch.float16):
                decoded = self.native.decode_base(denormalized)
            expected = (1, 3, plan["output_frames"], latent.shape[-2] * self.native.vae_ratio,
                        latent.shape[-1] * self.native.vae_ratio)
            if tuple(decoded.shape) != expected:
                raise RuntimeError("native temporal output differs from planned geometry")
            if self.output_mode == "uint8" and decoded.dtype != torch.uint8:
                proxy = SimpleNamespace(model=self.native, _pixel_transform=None)
                decoded = MiniMaxH3VideoVAE._revert_decoded_inplace(proxy, decoded)
                decoded = MiniMaxH3VideoVAE._normalize_decoded_frames(decoded)
                decoded = decoded.mul(255).round().to(torch.uint8)
            self.last_frame_plan = plan
            return decoded
        finally:
            self._busy = False

    def close(self):
        if self._closed:
            return
        if self._owner != threading.get_ident() or self._busy:
            raise RuntimeError("close must run on the idle owner thread")
        self.loader.close()
        self.native = None
        self._closed = True
