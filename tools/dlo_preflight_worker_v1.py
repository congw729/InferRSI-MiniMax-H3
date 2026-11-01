"""Frozen v1: actual upstream DLO checkpoint planning, bounded synthetic fixtures."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import traceback

import torch
from safetensors.torch import save_file

from validate_meta_coverage_v2 import make_model, setup_dist
from validate_mmap_v2 import state

GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny-fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.parent
    root.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "native DLO plan contract only",
              "real_weights": False, "full_pipeline_constructed": False, "forward_executed": False,
              "checks": [], "probes": {}}
    code = 1

    def check(name, ok, **details):
        result["checks"].append({"name": name, "ok": bool(ok), **details})
        if not ok:
            raise AssertionError(name)

    try:
        torch.cuda.set_per_process_memory_fraction(0.40)
        check("output_disk_headroom", shutil.disk_usage(root).free >= 8*GIB)
        expected = json.loads((a.tiny_fixture/"fixture-manifest.json").read_text())
        check("fixture_hashes_match", all(sha(a.tiny_fixture/name) == digest for name, digest in expected.items()))
        check("fixture_file_budget", sum((a.tiny_fixture/name).stat().st_size for name in expected) <= 16*1024**2)
        result["fixture_hashes"] = expected
        config = json.loads((a.tiny_fixture/"config.json").read_text())
        check("fixed_tiny_shape", config["num_layers"] == 2 and config["hidden_size"] == 128)
        setup_dist(root)
        from vllm_omni.diffusion.model_loader import host_weight_plan as hp
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
        from vllm_omni.diffusion.model_loader.checkpoint_adapters import get_direct_mmap_adapter
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
        from h3_omni_adapter.mmap_loader import H3MmapPlan

        source_files = [Path(hp.__file__), Path(__import__(
            "vllm_omni.diffusion.model_loader.checkpoint_adapters.direct_mmap", fromlist=["x"]).__file__)]
        result["upstream_sources"] = {p.name: sha(p) for p in source_files}
        check("native_pipeline_has_no_key_remap", not callable(getattr(MiniMaxH3Pipeline, "_remap_ckpt_key", None))
              and not callable(getattr(MiniMaxH3Pipeline, "remap_checkpoint_key", None)))
        quant = FastH3NVFP4Config.from_dit(a.tiny_fixture/"nvfp4_weights.safetensors", rank=0)
        model = make_model(config, quant, device="meta").eval()
        initial_ids = {name: id(value) for name, value in state(model).items()}
        check("h3_constructed_all_meta", all(t.is_meta for t in state(model).values()))
        pipeline = torch.nn.Module()
        pipeline.transformer = model
        check("h3_adapter_detected", type(get_direct_mmap_adapter(pipeline)).__name__ == "_MiniMaxH3DirectMmapAdapter")
        ComponentSource = DiffusersPipelineLoader.ComponentSource

        def sources(directory):
            return [ComponentSource(model_or_path=str(directory.parent), subfolder=directory.name,
                    revision=None, prefix="transformer.", fall_back_to_pt=False),
                    ComponentSource(model_or_path=str(directory.parent), subfolder=directory.name,
                    revision=None, prefix="transformer.", fall_back_to_pt=False,
                    allow_patterns_overrides=["nvfp4_weights.safetensors"])]

        def plan_call(carrier, src, **overrides):
            args = dict(dit_modules=[("transformer", carrier.transformer)], sources=src,
                        model_path=None, tensor_parallel_size=1, use_hsdp=False,
                        online_quantization=False, has_distilled_lora=False)
            args.update(overrides)
            return hp.build_checkpoint_mmap_plan(carrier, **args)

        def probe(name, carrier, src, remap=None):
            required = hp._collect_required_dit_tensors([("transformer", carrier.transformer)])
            mapped = hp._build_source_map(sources=src, model_path=None, remap_fn=remap)
            answer = plan_call(carrier, src)
            missing = sorted(set(required)-set(mapped))
            result["probes"][name] = {"plan_available": answer.plan is not None,
                "fallback_reason": answer.fallback_reason, "required_names": sorted(required),
                "source_bindings": {k: {"checkpoint_key": v[0], "file_path": v[1]} for k, v in mapped.items()},
                "missing_names": missing}
            return answer, mapped, missing

        indexed_sources = sources(a.tiny_fixture)
        result["native_component_sources"] = [asdict(s) for s in indexed_sources]
        answer, indexed, missing = probe("indexed_plus_explicit_sidecar", pipeline, indexed_sources)
        dense = json.loads((a.tiny_fixture/"diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
        check("indexed_map_matches_dense_keys", set(indexed) == {"transformer."+key for key in dense})
        check("explicit_sidecar_not_in_indexed_map", not any("::" in key for key in indexed))
        check("stock_h3_plan_falls_back_on_missing_bindings", answer.plan is None and len(missing)>0
              and str(len(missing))+" required DiT tensors" in answer.fallback_reason)

        external = H3MmapPlan.build(model, a.tiny_fixture, sidecars=["nvfp4_weights.safetensors"],
                                    max_total_file_bytes=16*1024**2)
        check("external_plan_covers_all_params_and_buffers", set(external.bindings) == set(state(model)))
        result["external_plan"] = {"targets": len(external.bindings), "sources": len(external.sources),
            "defaults": {k:v.default for k,v in external.bindings.items() if v.default},
            "multi_source_targets": {k:list(v.sources) for k,v in external.bindings.items() if len(v.sources)>1}}
        check("external_plan_has_explicit_defaults", len(result["external_plan"]["defaults"]) > 0)
        check("external_plan_has_multi_source_targets", len(result["external_plan"]["multi_source_targets"]) > 0)

        flat = root/"no-index-fixture"
        flat.mkdir()
        for name in ("dense.safetensors","nvfp4_weights.safetensors"):
            shutil.copyfile(a.tiny_fixture/name, flat/name)
        check("no_index_copy_hashes_match", all(sha(flat/name)==expected[name]
              for name in ("dense.safetensors","nvfp4_weights.safetensors")))
        flat_sources = sources(flat)
        flat_answer, mapped, flat_missing = probe("no_index_source_union", pipeline, flat_sources)
        check("without_index_all_source_keys_visible", set(mapped) == {"transformer."+key for key in external.sources})
        check("without_index_still_missing_runtime_bindings", flat_answer.plan is None and len(flat_missing)>0
              and str(len(flat_missing))+" required DiT tensors" in flat_answer.fallback_reason)

        remap = {"transformer."+source: "transformer."+target
                 for target,binding in external.bindings.items() for source in binding.sources}

        class RemappedCarrier(torch.nn.Module):
            @staticmethod
            def _remap_ckpt_key(key):
                return remap.get(key)

        renamed = RemappedCarrier()
        renamed.transformer = model
        remapped_answer = plan_call(renamed, flat_sources)
        result["probes"]["name_remap_only"] = {"plan_available": remapped_answer.plan is not None,
            "fallback_reason": remapped_answer.fallback_reason}
        check("name_remap_alone_rejects_multi_source_target", remapped_answer.plan is None
              and "multiple checkpoint entries map" in remapped_answer.fallback_reason)

        control_root = root/"linear-control"
        control_root.mkdir()
        control = torch.nn.Module()
        control.transformer = torch.nn.Linear(2, 2, bias=False, device="meta")
        save_file({"weight": torch.arange(4,dtype=torch.float32).reshape(2,2)},control_root/"control.safetensors")
        control_sources = [ComponentSource(model_or_path=str(control_root),subfolder=None,
                            revision=None,prefix="transformer.",fall_back_to_pt=False)]
        positive = plan_call(control, control_sources)
        check("generic_api_positive_control", positive.plan is not None and positive.fallback_reason is None)
        check("positive_control_complete_bindings", set(positive.plan.bindings) == {"transformer.weight"})
        result["control_positive"] = {"bindings": list(positive.plan.bindings),
                                      "source_prefixes": sorted(positive.plan.planned_source_prefixes)}
        controls = {"tp2": {"tensor_parallel_size": 2}, "hsdp": {"use_hsdp":True},
                    "online_quant": {"online_quantization":True}, "distilled_lora":{"has_distilled_lora":True}}
        result["control_rejections"] = {}
        for name, options in controls.items():
            rejected = plan_call(control, control_sources, **options)
            result["control_rejections"][name] = rejected.fallback_reason
            check("generic_api_rejects_"+name, rejected.plan is None and bool(rejected.fallback_reason))
        control.transformer.register_buffer("unbacked", torch.ones(1,device="meta"))
        rejected = plan_call(control, control_sources)
        check("generic_api_rejects_missing_persistent_buffer", rejected.plan is None and "no checkpoint binding" in rejected.fallback_reason)
        del control.transformer.unbacked
        control.transformer.weight = torch.nn.Parameter(torch.empty(3,2,device="meta"))
        rejected = plan_call(control, control_sources)
        check("generic_api_rejects_wrong_shape", rejected.plan is None and "shape mismatch" in rejected.fallback_reason)
        check("h3_unchanged_and_all_meta", initial_ids == {n:id(v) for n,v in state(model).items()}
              and all(t.is_meta for t in state(model).values()))
        check("original_fixture_unchanged", all(sha(a.tiny_fixture/n)==v for n,v in expected.items()))
        check("cuda_peak_within_guard", torch.cuda.max_memory_allocated() <= 4*GIB)
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["copied_fixture_bytes"] = sum(p.stat().st_size for p in flat.iterdir())
        result["full_model_fit_proven"] = False
        code = 0
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
        gc.collect()
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result["checks_total"] = len(result["checks"])
    result["checks_ok"] = sum(c["ok"] for c in result["checks"])
    text = json.dumps(result, indent=2)
    for source,label in ((str(a.tiny_fixture),"<TINY_FIXTURE>"),(str(root),"<RUN_ROOT>"),
                         (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"),(str(Path.home()),"<LINUX_HOME>")):
        text = text.replace(source,label)
    a.output.write_text(text+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","traceback","probes"}}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
