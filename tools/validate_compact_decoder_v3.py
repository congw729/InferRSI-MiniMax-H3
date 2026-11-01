"""Frozen v3: full decoder metadata and tiny CUDA native equivalence, no model weights."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import traceback
import weakref

import torch
from safetensors.torch import save_file

from h3_omni_adapter.compact_decoder import build_compact_decoder_meta, load_compact_decoder_mmap
from h3_omni_adapter.mmap_loader import _state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "real_payload_read": False,
              "scope": "full 8-layer metadata; tiny 1-layer synthetic CUDA decoder equivalence",
              "checks": []}
    code = 1
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)
    try:
        sys.path.insert(0, str(a.native_source / "Ref2VA"))
        from video_vae.vae_vit import ViT3DDecoder
        from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
        from vllm_omni.diffusion.models.minimax_h3.vae_nvfp4 import install_nvfp4_video_decoder
        _ensure_vae_parallel_state()
        path = a.research / "memory-budget/FastVideo__FastH3-NVFP4-VAE-Decoder__fast-8L__decoder-nvfp4.safetensors.header.json"
        saved = json.loads(path.read_text())
        check("saved header sha256", hashlib.sha256(path.with_suffix(".raw").read_bytes()).hexdigest() == saved["header_sha256"])
        header = saved["header"]
        config = json.loads((a.native_source / "Ref2VA/video_vae/source/config.json").read_text())
        model, mapping, defaults = build_compact_decoder_meta(config, header)
        all_state = _state(model)
        check("all 8-layer targets covered", set(mapping.values()) | set(defaults) == set(all_state))
        check("no dense payload allocation", all(v.device.type == "meta" for v in all_state.values()))
        check("no encoder constructed", set(model.keys()) == {"decoder", "post_quant_conv"})
        check("released depth retained", len(model["decoder"].transformer_blocks) == 8)
        result["full_decoder"] = {"parameters": len(list(model.named_parameters())), "buffers": len(list(model.named_buffers())),
            "source_keys": len(mapping), "targets": len(all_state), "defaults": defaults,
            "logical_bytes": sum(t.numel() * t.element_size() for t in all_state.values()),
            "inventory": [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in all_state.items()]}
        del model, all_state
        tiny = copy.deepcopy(config)
        tiny["vit_decoder_kwargs"]["heads"] = 2
        tiny["vit_decoder_kwargs"]["num_layers"] = 1
        tiny_header = {}
        for key, value in header.items():
            if key == "__metadata__":
                tiny_header[key] = copy.deepcopy(value)
                metadata = json.loads(value["nvfp4_metadata"])
                metadata.update(student_layers=1, num_linears=6)
                tiny_header[key]["nvfp4_metadata"] = json.dumps(metadata)
                continue
            if key.startswith("decoder.transformer_blocks.") and int(key.split(".")[2]) != 0:
                continue
            item = copy.deepcopy(value)
            item["shape"] = [{2048: 128, 8192: 512, 16384: 1024, 1024: 64, 4096: 256, 128: 8, 512: 32}.get(n, n)
                             for n in item["shape"]]
            tiny_header[key] = item
        # Projection out remains 3072 channels and post-quant latent width remains 24.
        template, _, _ = build_compact_decoder_meta(tiny, tiny_header)
        del template
        values = {}
        torch.manual_seed(71)
        for key, info in tiny_header.items():
            if key == "__metadata__":
                continue
            dtype = {"U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16}[info["dtype"]]
            if key.endswith(".packed_weight"):
                value = torch.full(info["shape"], 0x12, dtype=dtype)
            elif key.endswith(".weight_inv_scale"):
                value = torch.full(info["shape"], 0x38, dtype=dtype)
            elif key.endswith((".weight_global_sf", ".input_amax")):
                value = torch.ones(info["shape"], dtype=dtype)
            elif key.endswith((".norm1.weight", ".norm2.weight", ".norm_out.weight")):
                value = torch.ones(info["shape"], dtype=dtype)
            elif key.endswith((".scale1", ".scale2")):
                value = torch.full(info["shape"], 0.1, dtype=dtype)
            else:
                value = torch.randn(info["shape"], dtype=dtype) * 0.01
            values[key] = value.contiguous()
        fixture = a.output.parent / "decoder-synthetic.safetensors"
        save_file(values, fixture, metadata=tiny_header["__metadata__"])
        result["fixture"] = {"file": fixture.name, "bytes": fixture.stat().st_size,
                             "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
        opts = dict(tiny["vit_decoder_kwargs"])
        opts.update(patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, t_causal=False)
        reference = torch.nn.Module()
        reference.decoder = ViT3DDecoder(**opts)
        reference.post_quant_conv = torch.nn.Conv3d(24, 24, 1)
        install_nvfp4_video_decoder(reference, fixture)
        reference.eval().to("cuda")
        candidate, stats = load_compact_decoder_mmap(tiny, fixture, device="cuda")
        result["loader"] = stats
        ref = _state(reference)
        check("all persistent and nonpersistent names exact", set(ref) == set(_state(candidate)))
        check("all tensor bytes equal native installer", all(torch.equal(ref[k], v) for k, v in _state(candidate).items()))
        del ref
        x = torch.randn(1, 24, 2, 2, 2, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference.decoder(reference.post_quant_conv(x))
            actual = candidate["decoder"](candidate["post_quant_conv"](x))
        check("complete tiny decoder forward exact", torch.equal(expected, actual))
        check("complete tiny decoder finite", torch.isfinite(actual).all().item())
        check("pixel geometry exact", tuple(actual.shape) == (1, 3, 8, 32, 32))
        check("checkpoint unmapped after materialization", str(fixture) not in Path("/proc/self/maps").read_text())
        before = torch.cuda.memory_allocated()
        owned_refs = [weakref.ref(t) for m in (candidate, reference) for t in _state(m).values()]
        del candidate, actual, expected, reference, x
        gc.collect()
        torch.cuda.synchronize()
        check("GPU tensors released after stage", torch.cuda.memory_allocated() < before)
        check("all model tensor objects released", all(r() is None for r in owned_refs))
        from flashinfer import utils as fi_utils
        buffers = list(fi_utils._cache_buf.values()) + list(fi_utils._cache_buf_retired)
        storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                   for t in buffers if t.device.type == "cuda"}
        before_cublas_clear = torch.cuda.memory_allocated()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated()
        result["release_accounting"] = {"torch_cuda_allocated_before_cublas_clear": before_cublas_clear,
            "cublas_workspace_released_bytes": before_cublas_clear - retained,
            "torch_cuda_allocated_after": retained,
            "flashinfer_cache_storage_bytes": sum(storage.values()),
            "cache_names": [str(k[0]) for k, t in fi_utils._cache_buf.items() if t.device.type == "cuda"],
            "model_tensor_references_alive": sum(r() is not None for r in owned_refs)}
        check("remaining CUDA allocation exactly owned library cache", retained == sum(storage.values()))
        malformed = copy.deepcopy(header)
        malformed.pop("decoder.norm_out.bias")
        try:
            build_compact_decoder_meta(config, malformed)
        except ValueError:
            check("missing dense target rejected", True)
        else:
            check("missing dense target rejected", False)
        try:
            load_compact_decoder_mmap(tiny, fixture, device="cpu", max_file_bytes=1)
        except ValueError:
            check("experiment file budget enforced", True)
        else:
            check("experiment file budget enforced", False)
        values["decoder.transformer_blocks.0.attn.to_q.weight_global_sf"].fill_(-1)
        negative = a.output.parent / "negative-scale.safetensors"
        save_file(values, negative, metadata=tiny_header["__metadata__"])
        try:
            load_compact_decoder_mmap(tiny, negative, device="cpu")
        except ValueError as exc:
            check("invalid scalar rejected before forward", "weight_global_sf" in str(exc))
        else:
            check("invalid scalar rejected before forward", False)
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        code = 0
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["checks_passed"] = sum(c["ok"] for c in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["exit_code"] = code
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())


