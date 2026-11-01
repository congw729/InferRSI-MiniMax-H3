"""Frozen shared native geometry and sampling for DLO/temporal chain evidence."""
import json
from types import SimpleNamespace

import torch


def geometry(scenario):
    if scenario=="tiny":
        return dict(scenario=scenario,latent_t=7,latent_h=2,latent_w=2,audio_t=1,text_tokens=8,
                    height=32,width=32,frames=22,fps=24,crop_frames=12,native_request=False)
    from validate_request_geometry_v2 import native_geometry
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import resolve_minimax_h3_shape
    g=native_geometry(4,256,64)
    shape=resolve_minimax_h3_shape("t2va",SimpleNamespace(fps=24,height=256,width=256,num_frames=96,
        extra_args={"duration_seconds":4,"aspect_ratio":"1:1"}),None)
    if list(shape)!=[256,256,107,32,178]:
        raise AssertionError("native request geometry changed")
    g.update(scenario=scenario,crop_frames=72,native_request=True)
    return g


def sample(model,config,hidden,g):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch,minimax_h3_denoise_loop
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent,minimax_h3_unpatchify_video_tokens
    from vllm_omni.diffusion.models.minimax_h3.pdd import pdd_sigmas
    lt,lh,lw=g["latent_t"],g["latent_h"],g["latent_w"]
    packed=minimax_h3_packed_sequence(text_len=g["text_tokens"],latent_t=lt,latent_h=lh,latent_w=lw,
                                     audio_t=g["audio_t"],include_keyframe_cond=False)
    branch=MiniMaxH3DenoiseBranch(packed=packed,text_embeddings=hidden,token_tags=packed["token_tags"],device=torch.device("cuda"))
    initial=torch.randn(1,24,lt,lh,lw,generator=torch.Generator().manual_seed(301))
    rows=minimax_h3_patchify_video_latent(initial,patch_size=(1,2,2))
    roundtrip=minimax_h3_unpatchify_video_tokens(rows,latent_shape=(lt,lh//2,lw//2,24),patch_size=(1,2,2))
    if not torch.equal(initial,roundtrip):
        raise AssertionError("patch inverse")
    sv,sa=pdd_sigmas(12.0),pdd_sigmas(3.0)
    rows*=sv[0]
    audio=torch.randn(g["audio_t"]*2,32,generator=torch.Generator().manual_seed(302))*sa[0]
    steps=[]
    def on_step(step,video_rows,audio_rows):
        steps.append(dict(step=step,video_rows=video_rows.detach().cpu().clone(),audio_rows=audio_rows.detach().cpu().clone()))
    od=OmniDiffusionConfig(model=None,model_class_name="MiniMaxH3Pipeline",
        tf_model_config=TransformerConfig.from_dict(config),
        cache_config={"minimax_h3_adaln_cache":False,"minimax_h3_adaln_offload":False})
    with torch.inference_mode(),set_forward_context(omni_diffusion_config=od):
        vr,ar=minimax_h3_denoise_loop(model=model,positive=branch,initial_video_rows=rows,initial_audio_rows=audio,
            keyframe_cond_rows=None,sigmas_video=sv,sigmas_audio=sa,device=torch.device("cuda"),on_step=on_step)
    latent=minimax_h3_unpatchify_video_tokens(vr,latent_shape=(lt,lh//2,lw//2,24),patch_size=(1,2,2))
    return dict(kind="native_sampled_latents",geometry=g,video_latent=latent.cpu(),audio_rows=ar.cpu(),steps=steps,
                not_raw_velocity=True,patch_roundtrip=True,used_tokens=int(packed["cu_seqlens"][1]),
                padded_tokens=int(packed["seq_len"]))


def decode(model,checkpoint,payload,*,streamed):
    from validate_temporal_decoder_v2 import temporal_mode
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import released_decoder_pixels
    g=payload["geometry"]
    if payload["kind"]!="native_sampled_latents" or not payload["not_raw_velocity"]:
        raise ValueError("sampled latent required")
    latent=payload["video_latent"].cuda()
    expected=(1,24,g["latent_t"],g["latent_h"],g["latent_w"])
    if tuple(latent.shape)!=expected or not torch.isfinite(latent).all():
        raise ValueError("latent handoff shape or values")
    with temporal_mode(True),torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        if streamed:
            pixels=model.decode_latent(latent)
        else:
            outer=json.loads((checkpoint/"vae-normalization.json").read_text())
            normalized=MiniMaxH3VideoVAE._denormalize_latent(SimpleNamespace(config_dict=outer),latent)
            raw=model.decode_base(normalized)
            pixels=released_decoder_pixels(raw).mul(255).round().to(torch.uint8)
    if tuple(pixels.shape)!=(1,3,g["frames"],g["height"],g["width"]):
        raise ValueError("temporal frame geometry")
    if not 0<g["crop_frames"]<=pixels.shape[2]:
        raise ValueError("crop out of range")
    cropped=pixels[:,:,:g["crop_frames"]].contiguous()
    return dict(kind="synthetic_video_only_pixels",geometry=g,full_pixels=pixels.cpu(),crop_pixels=cropped.cpu(),
                crop_seconds=g["crop_frames"]/g["fps"],video_only=True,audio_decoded=False,
                real_video_generated=False)
