"""Bounded TP1 temporal decoding with native H3 loops and a compact decoder.

This adapter constructs only the two components owned by the released packed
decoder. Native temporal/spatial blending and pixel operations stay upstream.
The caller provides a reviewed local native package; no downloads occur here.
"""
from __future__ import annotations

import copy
import math
import threading
from types import SimpleNamespace

import torch
from torch import nn

from .compact_decoder import load_compact_decoder_mmap


def _native_host(config, outer, components):
    from diffusers.models import ModelMixin
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
    from video_vae.normalize import get_denormalize_transform, get_normalize_transform
    from video_vae.parallel import get_parallel_state

    _ensure_vae_parallel_state()
    if get_parallel_state()["sp_size"] != 1:
        raise ValueError("compact temporal prototype requires TP1 spatial state")

    class DecoderOnly(AutoencoderKLLegacy):
        def __init__(self):
            ModelMixin.__init__(self)
            self.register_to_config(**config)
            self.transform = get_normalize_transform(config["pixel_norm_type"])
            self.transform_rev = get_denormalize_transform(config["pixel_norm_type"])
            self.use_3d_conv = True
            self.use_vit_decoder = True
            self.causal_encoder = bool(config["causal_encoder"])
            self.causal_decoder = bool(config["causal_decoder"])
            self.slidedec = self.causal_encoder and not self.causal_decoder
            self.vae_ratio = math.prod(config["space_down"])
            self.vae_ratio_t = math.prod(config["time_down"])
            options = {key: int(outer["vae_" + key]) for key in (
                "clip_length", "token_drop", "encoder_tiling", "decoder_tiling",
                "parallel_tiling", "tile_size", "tile_overlap_min",
                "encoder_parallel", "decoder_parallel", "chunk_dim",
            )}
            if options["encoder_parallel"] or options["decoder_parallel"]:
                raise ValueError("parallel encoder/decoder is unsupported by this prototype")
            self.setup_forward(**options)
            self.setup_training()
            self.post_quant_conv = components["post_quant_conv"]
            self.decoder = components["decoder"]
            self.eval()

    return DecoderOnly()


class H3TemporalVideoDecoder(nn.Module):
    """Single-owner inference adapter returning raw CUDA tensors or uint8 pixels."""

    def __init__(
        self, source_config, outer_config, checkpoint, *, device, max_latent_tokens,
        max_height, max_width, max_output_frames, output_mode="uint8",
        max_file_bytes=512 * 1024**2,
    ):
        super().__init__()
        if output_mode not in {"raw", "uint8"}:
            raise ValueError("output_mode must be raw or uint8")
        budgets = (max_latent_tokens, max_height, max_width, max_output_frames)
        if any(type(value) is not int or value <= 0 for value in budgets):
            raise ValueError("positive integer time, spatial and output budgets required")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("temporal prototype requires CUDA")
        config = copy.deepcopy(source_config)
        outer = copy.deepcopy(outer_config)
        if not config["use_3d_conv"] or not config["use_vit_decoder"]:
            raise ValueError("native 3D ViT decoder configuration required")
        if int(outer["latent_channels"]) != int(config["embed_dim"]):
            raise ValueError("latent channel configuration mismatch")
        components, self.loader_stats = load_compact_decoder_mmap(
            config, checkpoint, device=target, max_file_bytes=max_file_bytes,
        )
        self.native = _native_host(config, outer, components)
        self.config_dict = outer
        self.device = next(self.native.parameters()).device
        self.output_mode = output_mode
        self.max_latent_tokens, self.max_height, self.max_width, self.max_output_frames = budgets
        self._owner_thread = threading.get_ident()
        self._closed = False
        self._busy = False
        self.last_plan = None
        if output_mode == "uint8":
            from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import released_decoder_pixels
            from vllm_omni.diffusion.models.minimax_h3.vae_temporal import install_temporal_stream_patches
            install_temporal_stream_patches(self.native, pixel_transform=released_decoder_pixels)
        self.eval()

    def _check_input(self, latent):
        if self._closed or self.native is None:
            raise RuntimeError("decoder is closed")
        if self._owner_thread != threading.get_ident() or self._busy:
            raise RuntimeError("concurrent or foreign-thread decode is unsupported")
        if self.training or self.native.training:
            raise RuntimeError("decoder is inference-only")
        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent must be a tensor")
        if latent.is_meta or latent.device != self.device:
            raise ValueError("latent must be on the decoder CUDA device")
        if latent.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise TypeError("floating latent dtype required")
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != int(self.config_dict["latent_channels"]):
            raise ValueError("latent must have shape [1, channels, time, height, width]")
        _, _, length, height, width = latent.shape
        native = self.native
        if not 0 < length <= self.max_latent_tokens or min(height, width) <= 0:
            raise ValueError("latent time/space exceeds explicit budget")
        if height * native.vae_ratio > self.max_height or width * native.vae_ratio > self.max_width:
            raise ValueError("decoded spatial size exceeds explicit budget")
        pseudo = length + native.token_drop
        pad = (-pseudo) % native.tokens_chunk_size
        chunks = (pseudo + pad) // native.tokens_chunk_size - int(native.token_drop > 0)
        if chunks <= 0 or native.isolated_first_frame or native.isolated_last_frame:
            raise ValueError("unsupported short or isolated-frame temporal plan")
        shape_only = SimpleNamespace(shape=(1, latent.shape[1], length + pad, height, width))
        total, padded, frames = native._decode_temporal_output_frame_plan(shape_only, None, None, chunks, pad)
        if not 0 < frames <= self.max_output_frames:
            raise ValueError("decoded temporal size exceeds explicit budget")
        if not torch.isfinite(latent).all().item():
            raise ValueError("latent contains nonfinite values")
        return {"chunks": chunks, "pad_tokens": pad, "total_frames": total,
                "pad_frames": padded, "output_frames": frames}

    @torch.inference_mode()
    def decode_latent(self, latent):
        plan = self._check_input(latent)
        self._busy = True
        try:
            from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
            from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import released_decoder_pixels
            denormalized = MiniMaxH3VideoVAE._denormalize_latent(self, latent)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = self.native.decode_base(denormalized)
            expected = (1, 3, plan["output_frames"],
                        latent.shape[3] * self.native.vae_ratio, latent.shape[4] * self.native.vae_ratio)
            if tuple(decoded.shape) != expected:
                raise RuntimeError("native temporal output shape differs from frame plan")
            if self.output_mode == "uint8" and decoded.dtype != torch.uint8:
                decoded = released_decoder_pixels(decoded).mul_(255).round_().to(torch.uint8)
            self.last_plan = plan
            return decoded
        finally:
            self._busy = False

    def close(self):
        if self._closed:
            return
        if self._owner_thread != threading.get_ident() or self._busy:
            raise RuntimeError("cannot close a busy or foreign-thread decoder")
        torch.cuda.current_stream(self.device).synchronize()
        self.native = None
        self._closed = True
