"""Frozen v3: native complete-encoder load reference and text-only streamed inference."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import traceback
import weakref

import torch
from safetensors.torch import save_file
from validate_meta_coverage_v2 import setup_dist


def state(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def fixture(research, root):
    from vllm.distributed import get_tp_group
    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import (
        FastH3NVFP4Config, NVFP4Projection,
    )
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
    config = json.loads((research / "config-text_encoder__config.json").read_text())["content"]
    config.pop("quantization_config")
    config["text_config"].update(hidden_size=128, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=1, num_key_value_heads=1, head_dim=128, vocab_size=256)
    config["vision_config"].update(depth=1, hidden_size=128, intermediate_size=128,
        num_heads=1, out_hidden_size=128, deepstack_visual_indexes=[0],
        num_position_embeddings=16)
    config.update(image_token_id=254, video_token_id=255)
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    projections, packed = {}, {}
    for layer in range(2):
        for target, parts in (("self_attn.qkv_proj", ("q", "k", "v")),
                              ("self_attn.o_proj", ("o",)),
                              ("mlp.gate_up_proj", ("gate", "up")),
                              ("mlp.down_proj", ("down",))):
            specs = []
            for index, part in enumerate(parts):
                source = f"model.language_model.layers.{layer}.{target.split('.')[0]}.{part}_proj"
                keys = {"weight": source + ".weight_packed", "scales": source + ".weight_scale",
                        "alpha": source + ".weight_global_scale"}
                specs.append(NVFP4Projection(source, 128, 128, keys, reciprocal_scale=True))
                packed[keys["weight"]] = ((torch.arange(128 * 64).reshape(128, 64) + index + layer)
                    .remainder(7) + 17).to(torch.uint8)
                packed[keys["scales"]] = torch.full((128, 8), 56, dtype=torch.uint8)
                packed[keys["alpha"]] = torch.tensor([32.0 * (index + layer + 1)])
            projections[f"text_model.layers.{layer}.{target}"] = specs
    quant = FastH3NVFP4Config(projections, rank=0)
    ordinary = MiniMaxH3Qwen3VLEncoder(str(root), device=torch.device("cuda"), load_model=True,
                                      encoder_group=get_tp_group(), quant_config=quant)
    values = dict(packed)
    serialized_targets = set(quant.weight_map.values())
    for name, param in ordinary.named_parameters():
        if name in serialized_targets:
            continue
        source = name.replace("vision.", "model.visual.", 1) if name.startswith("vision.") else (
            name.replace("text_model.", "model.language_model.", 1))
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:4], 16) % 13
        value = ((torch.arange(param.numel()).float() + seed).remainder(17) - 8) / 128
        if "norm" in name:
            value += 1
        values[source] = value.reshape(param.shape).to(param.dtype)
    values["model.language_model.norm.weight"] = torch.ones(128, dtype=torch.bfloat16)
    save_file(values, root / "model.safetensors")
    loaded = ordinary.load_weights(iter(values.items()))
    if loaded != set(dict(ordinary.named_parameters())):
        raise AssertionError("native outer complete-parameter guard did not pass")
    return config, quant, ordinary, values


def reference_forward(ordinary, ids):
    from vllm_omni.diffusion.models.minimax_h3.encoder import _get_rope_index
    ids = ids.cuda()[None]
    positions, _ = _get_rope_index(ids, mm_token_type_ids=torch.zeros_like(ids, dtype=torch.int32),
        image_grid_thw=None, video_grid_thw=None, attention_mask=torch.ones_like(ids),
        spatial_merge_size=int(ordinary.vision.config.spatial_merge_size))
    with torch.inference_mode():
        return ordinary.text_model(ordinary.text_model.embed_tokens(ids), positions)[0].cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "two 128-wide native text layers, synthetic NVFP4",
              "checks": [], "real_weights": False, "quality_verified": False}
    def check(name, ok, **details):
        result["checks"].append({"name": name, "ok": bool(ok), **details})
    def rejects(name, call, types=(ValueError, RuntimeError, TypeError)):
        try:
            call()
        except types:
            check(name, True)
        else:
            check(name, False)
    code = 1
    loader = None
    try:
        setup_dist(a.output.parent)
        torch.cuda.set_per_process_memory_fraction(0.25)
        from transformers import Qwen3VLConfig
        from vllm.distributed import get_tp_group
        from h3_omni_adapter.text_streaming import H3TextOnly, H3TextMmapPlan, H3TextMmapLoader
        config, quant, ordinary, values = fixture(a.research, a.output.parent / "fixture")
        root = a.output.parent / "fixture"
        refs = {k: t.detach().clone() for k, t in state(ordinary).items() if k.startswith("text_model.")}
        candidate = H3TextOnly(Qwen3VLConfig.from_dict(config), get_tp_group(), quant, max_tokens=64)
        plan = H3TextMmapPlan.build(candidate, root, files=["model.safetensors"], max_total_file_bytes=16*1024**2)
        check("all_text_tensors_planned", set(plan.bindings) == set(refs))
        check("vision_not_constructed", not hasattr(candidate, "vision"))
        check("all_meta_preflight", all(t.is_meta for t in state(candidate).values()))
        check("vision_and_final_norm_explicitly_excluded", set(plan.excluded) ==
              {k for k in values if k.startswith("model.visual.") or k == "model.language_model.norm.weight"})
        loader = H3TextMmapLoader(candidate, plan, device="cuda", max_group_bytes=8*1024**2)
        loader.load_group("resident")
        for group, names in plan.groups.items():
            if group != "resident":
                loader.load_group(group)
            check("tensor_exact_" + group, all(torch.equal(state(candidate)[n].cpu(), refs[n]) for n in names),
                  count=len(names))
            check("native_loader_metadata_" + group,
                  all(hasattr(t, "weight_loader") == hasattr(dict(ordinary.named_parameters()).get(n), "weight_loader")
                      for n, t in candidate.named_parameters() if n in names))
            if group != "resident":
                loader.release_group(group)
                check("release_" + group, all(state(candidate)[n].is_meta for n in names))
        for _ in range(2):
            loader.load_group("text_model.layers.0")
            check("reload_all_values_exact", all(torch.equal(state(candidate)[n].cpu(), refs[n])
                  for n in plan.groups["text_model.layers.0"]))
            loader.release_group("text_model.layers.0")
        ordinary.text_model.to("cuda")
        loader.install_block_hooks()
        ids = torch.tensor([1, 8, 3, 9, 8, 4, 6, 2], dtype=torch.long)
        for length in (8, 17, 8):
            token_ids = ids.repeat(3)[:length]
            expected = reference_forward(ordinary, token_ids)
            actual = candidate.encode_ids(token_ids)
            check("complete_native_text_forward_exact_" + str(length),
                  torch.equal(actual, expected) and torch.isfinite(actual).all(),
                  shape=list(actual.shape), max_abs=float((actual.float()-expected.float()).abs().max()))
            check("layers_meta_after_forward_" + str(length),
                  all(t.is_meta for n, t in state(candidate).items() if ".layers." in n))
        rejects("multimodal_image_token_rejected", lambda: candidate.encode_ids(torch.tensor([254])))
        rejects("multimodal_video_token_rejected", lambda: candidate.encode_ids(torch.tensor([255])))
        rejects("pixel_input_rejected", lambda: candidate.encode_ids(ids, pixel_values=torch.zeros(1)))
        rejects("grid_input_rejected", lambda: candidate.encode_ids(ids, image_grid_thw=torch.zeros(1)))
        rejects("float_ids_rejected", lambda: candidate.encode_ids(ids.float()))
        rejects("empty_ids_rejected", lambda: candidate.encode_ids(ids[:0]))
        rejects("batched_ids_rejected", lambda: candidate.encode_ids(ids[None]))
        rejects("negative_ids_rejected", lambda: candidate.encode_ids(torch.tensor([-1])))
        rejects("out_of_range_ids_rejected", lambda: candidate.encode_ids(torch.tensor([256])))
        rejects("length_budget_rejected", lambda: candidate.encode_ids(torch.zeros(65, dtype=torch.long)))
        rejects("layer_forward_exception", lambda: candidate.text_model.layers[0]())
        check("exception_hook_releases_layer", all(state(candidate)[n].is_meta for n in plan.groups["text_model.layers.0"]))
        result["loader_stats"] = dict(loader.stats)
        loader.close()
        check("close_all_meta", all(t.is_meta for t in state(candidate).values()))
        gc.collect()
        check("no_live_checkpoint_mmaps", str(root) not in Path("/proc/self/maps").read_text())
        rejects("closed_encode_rejected", lambda: candidate.encode_ids(ids))
        rejects("tp2_rejected", lambda: H3TextMmapPlan.build(candidate, root, files=["model.safetensors"],
                                                              tensor_parallel_size=2))
        rejects("file_budget_rejected", lambda: H3TextMmapPlan.build(candidate, root, files=["model.safetensors"],
                                                                    max_total_file_bytes=1))
        rejects("group_budget_rejected", lambda: H3TextMmapLoader(candidate, plan, device="cuda", max_group_bytes=1)
                .load_group("resident"))
        for label, change in (("missing", "model.language_model.embed_tokens.weight"),
                              ("extra", "model.language_model.unexpected.weight"),
                              ("bad_scale", next(k for k in values if k.endswith("weight_global_scale")))):
            folder = a.output.parent / label
            folder.mkdir()
            bad = dict(values)
            if label == "missing":
                del bad[change]
            elif label == "extra":
                bad[change] = torch.ones(1)
            else:
                bad[change] = torch.zeros_like(bad[change])
            save_file(bad, folder / "model.safetensors")
            if label != "bad_scale":
                rejects(label + "_source_rejected", lambda folder=folder: H3TextMmapPlan.build(candidate, folder,
                                                                       files=["model.safetensors"]))
            else:
                bad_plan = H3TextMmapPlan.build(candidate, folder, files=["model.safetensors"])
                broken = H3TextMmapLoader(candidate, bad_plan, device="cuda", max_group_bytes=8*1024**2)
                broken.load_group("resident")
                rejects("invalid_global_scale_native_loader_rejected",
                        lambda: broken.load_group("text_model.layers.0"))
                check("failed_load_restores_layer_meta", all(state(candidate)[n].is_meta
                      for n in bad_plan.groups["text_model.layers.0"]))
                broken.close()
                gc.collect()
                check("failed_load_closes_mmaps", str(folder) not in Path("/proc/self/maps").read_text())
        candidate_ref = weakref.ref(candidate)
        del candidate, loader, broken, bad_plan, plan
        loader = None
        gc.collect()
        check("candidate_object_released", candidate_ref() is None)
        result["fixture_bytes"] = (root / "model.safetensors").stat().st_size
        result["cuda_peak_allocated"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved"] = torch.cuda.max_memory_reserved()
        result["checks_total"] = len(result["checks"])
        result["checks_ok"] = sum(c["ok"] for c in result["checks"])
        code = int(result["checks_total"] != result["checks_ok"])
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        if loader is not None:
            loader.close()
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    text = json.dumps(result, indent=2)
    for source, label in ((str(a.research), "<RESEARCH_ROOT>"), (str(a.output.parent), "<RUN_ROOT>"),
            (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    a.output.write_text(text + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","traceback"}}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
