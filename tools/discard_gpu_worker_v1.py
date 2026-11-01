"""Frozen tiny native DLO lifecycle, instrumented CPU restores."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
import weakref

GIB = 1024**3
REQUIRED = [
    "cache_manifest_identity", "cache_files_intact", "fixture_intact", "two_distinct_blocks",
    "all_targets_mapped", "no_aliases", "reference_finite_eight_steps", "reference_released",
    "native_dlo_two_hooks", "rank_local_mmap", "two_slot_sizes", "candidate_finite_eight_steps",
    "every_step_and_final_exact", "shutdown", "model_objects_released", "maps_released",
    "cache_unchanged", "no_new_cache", "cuda_budget", "restore_policy", "slot_objects_released",
    "repeated_shutdown", "reuse_policy",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--mode", choices=["native", "discard"], required=True)
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args()
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "tiny synthetic DLO lifecycle",
              "mode": a.mode, "checks": [], "real_weights": False}
    backend = None
    code = 1
    started = time.monotonic()

    def event(name, **info):
        rss = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "RssAnon", "RssFile", "RssShmem"}:
                rss[key] = int(value.split()[0]) * 1024
        with (a.work / "phases.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": name, "elapsed": time.monotonic() - started,
                                     "rss": rss, **info}) + "\n")
            stream.flush()

    def check(name, ok):
        if name not in REQUIRED or any(x["name"] == name for x in result["checks"]):
            raise AssertionError("bad check")
        result["checks"].append({"name": name, "ok": bool(ok)})
        event(name)
        if not ok:
            raise AssertionError(name)

    try:
        import torch
        from safetensors import safe_open
        from h3_omni_adapter.mmap_loader import _state
        from h3_omni_adapter.stage_dlo import StageDLOBackend
        from validate_meta_coverage_v2 import make_model, setup_dist
        from dlo_temporal_common_v1 import geometry, sample
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan, TensorBinding
        from vllm_omni.diffusion.offloader.base import OffloadConfig
        from vllm_omni.diffusion.offloader.config import OffloadStrategy
        import vllm_omni.diffusion.offloader.distributed_layerwise_backend as native
        torch.cuda.set_per_process_memory_fraction(0.40)
        if torch.cuda.mem_get_info()[0] < 5 * GIB:
            raise RuntimeError("initial CUDA headroom")
        manifest_sha = sha(a.cache / "manifest.json")
        manifest = json.loads((a.cache / "manifest.json").read_text())
        check("cache_manifest_identity", manifest["complete"] and
              manifest["schema"] == "h3-native-dlo-runtime-cache/v1")
        check("cache_files_intact", all(Path(n).name == n and v["bytes"] < 16 * 1024**2 and
              sha(a.cache / n) == v["sha256"] for n, v in manifest["files"].items()))
        fixture = json.loads((a.fixture / "fixture-manifest.json").read_text())
        check("fixture_intact", all(sha(a.fixture / n) == v["sha256"] for n, v in fixture.items()))
        setup_dist(a.work)
        config = json.loads((a.fixture / "dit/config.json").read_text())
        if config["hidden_size"] != 128 or config["num_layers"] != 2:
            raise ValueError("tiny only")
        quant = FastH3NVFP4Config.from_dit(a.fixture / "dit/nvfp4_weights.safetensors", rank=0)

        def build(cls):
            model = make_model(config, quant, device="meta").eval()
            state = _state(model)
            if set(state) != set(manifest["tensors"]):
                raise ValueError("target set")
            bindings = {}
            for name, target in state.items():
                info = manifest["tensors"][name]
                if list(target.shape) != info["shape"] or str(target.dtype) != info["dtype"]:
                    raise ValueError("shape/dtype")
                if name in manifest["nonpersistent"]:
                    with safe_open(a.cache / info["file"], framework="pt", device="cpu") as f:
                        value = f.get_tensor(name).clone()
                    parent, _, leaf = name.rpartition(".")
                    model.get_submodule(parent)._buffers[leaf] = value
                else:
                    bindings["transformer." + name] = TensorBinding(name, str(a.cache / info["file"]))
            pipeline = torch.nn.Module()
            pipeline.transformer = model
            pipeline._dit_modules = ["transformer"]
            pipeline._encoder_modules = []
            pipeline._vae_modules = []
            pipeline._resident_modules = []
            plan = HostWeightPlan("checkpoint_mmap", bindings, frozenset({"transformer."}))
            config_offload = OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                                          pin_cpu_memory=True, dp_size=1, dlo_use_allgather=False,
                                          components=frozenset({"dit"}))
            return model, pipeline, cls(config_offload, torch.device("cuda"), host_weight_plan=plan)

        model, pipeline, backend = build(native.DistributedLayerwiseOffloadBackend)
        check("two_distinct_blocks", model.blocks[0] is not model.blocks[1])
        check("all_targets_mapped", len(_state(model)) == 111 and len(backend.host_weight_plan.bindings) == 109)
        check("no_aliases", all(k == "transformer." + v.checkpoint_key
                               for k, v in backend.host_weight_plan.bindings.items()))
        from types import SimpleNamespace
        backend._using_rank_local_mmap = True
        backend._load_weights_via_mmap(pipeline, (SimpleNamespace(path="transformer", module=model),),
                                      backend.host_weight_plan)
        model.cuda()
        g = geometry("tiny")
        hidden = torch.full((g["text_tokens"], 128), 0.01, dtype=torch.bfloat16)
        baseline = sample(model, config, hidden, g)
        check("reference_finite_eight_steps", len(baseline["steps"]) == 8 and
              all(torch.isfinite(s[k]).all() for s in baseline["steps"] for k in ("video_rows", "audio_rows")))
        backend.shutdown()
        refs = [weakref.ref(v) for v in _state(model).values()]
        del model, pipeline, backend
        backend = None
        gc.collect()
        torch.cuda.empty_cache()
        check("reference_released", all(r() is None for r in refs) and
              str(a.cache.resolve()) not in Path("/proc/self/maps").read_text())
        torch.cuda.reset_peak_memory_stats()
        cls = StageDLOBackend if a.mode == "discard" else native.DistributedLayerwiseOffloadBackend
        model, pipeline, backend = build(cls)
        backend.enable(pipeline)
        hooks = [h for group in backend._all_hook_groups for h in group]
        check("native_dlo_two_hooks", backend.enabled and len(hooks) == 2)
        check("rank_local_mmap", backend._using_rank_local_mmap)
        gpu = sum(t.numel() * t.element_size() for slot in hooks[0].gpu_buffers for t in slot.values())
        cpu = sum(t.numel() * t.element_size() for slot in hooks[0].cpu_staging_buffers for t in slot.values())
        check("two_slot_sizes", gpu == cpu and gpu > 0 and
              hooks[0].gpu_buffers is hooks[1].gpu_buffers and
              hooks[0].cpu_staging_buffers is hooks[1].cpu_staging_buffers)
        slots = [weakref.ref(t) for collection in (hooks[0].gpu_buffers, hooks[0].cpu_staging_buffers)
                 for slot in collection for t in slot.values()]
        expected_restore_bytes = sum(m["numel"] * torch.empty((), dtype=d).element_size()
                                     for h in hooks for d, metas in h.metadata.items() for m in metas)
        actual = sample(model, config, hidden, g)
        check("candidate_finite_eight_steps", len(actual["steps"]) == 8 and
              all(torch.isfinite(s[k]).all() for s in actual["steps"] for k in ("video_rows", "audio_rows")))
        check("every_step_and_final_exact",
              all(torch.equal(x[k], y[k]) for x, y in zip(actual["steps"], baseline["steps"], strict=True)
                  for k in ("video_rows", "audio_rows")) and
              torch.equal(actual["video_latent"], baseline["video_latent"]) and
              torch.equal(actual["audio_rows"], baseline["audio_rows"]))
        restore_calls = []
        original_restore = native.restore_tensor_storage
        original_sync = native.current_omni_platform.synchronize

        def restore(target, value, *, device):
            size = value.numel() * value.element_size()
            event("restore_enter", logical_bytes=size, shape=list(value.shape), dtype=str(value.dtype))
            original_restore(target, value, device=device)
            restore_calls.append(size)
            event("restore_exit", logical_bytes=size)

        def synchronize():
            event("synchronize_enter")
            original_sync()
            event("synchronize_exit")

        native.restore_tensor_storage = restore
        native.current_omni_platform.synchronize = synchronize
        try:
            event("shutdown_enter")
            if a.mode == "discard":
                backend.shutdown(discard_stage=True)
            else:
                backend.shutdown()
            event("shutdown_exit")
        finally:
            native.restore_tensor_storage = original_restore
            native.current_omni_platform.synchronize = original_sync
        check("shutdown", not backend.enabled and not backend._all_hook_groups)
        check("restore_policy", (not restore_calls if a.mode == "discard" else
                                bool(restore_calls) and sum(restore_calls) == expected_restore_bytes))
        result["restore_calls"] = len(restore_calls)
        result["restored_logical_bytes"] = sum(restore_calls)
        result["expected_default_restore_bytes"] = expected_restore_bytes
        backend.shutdown()
        check("repeated_shutdown", not backend.enabled)
        if a.mode == "discard":
            try:
                backend.enable(pipeline)
            except RuntimeError:
                rejected = True
            else:
                rejected = False
            check("reuse_policy", rejected)
        else:
            check("reuse_policy", backend._poisoned_reason is None)
        refs = [weakref.ref(v) for v in _state(model).values()]
        del model, pipeline, backend, hooks
        backend = None
        gc.collect()
        torch.cuda.empty_cache()
        check("model_objects_released", all(r() is None for r in refs))
        check("slot_objects_released", all(r() is None for r in slots))
        check("maps_released", str(a.cache.resolve()) not in Path("/proc/self/maps").read_text() and
              str(a.fixture.resolve()) not in Path("/proc/self/maps").read_text())
        check("cache_unchanged", sha(a.cache / "manifest.json") == manifest_sha and
              all(sha(a.cache / n) == v["sha256"] for n, v in manifest["files"].items()))
        check("no_new_cache", not list(a.work.rglob("*.safetensors")))
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        check("cuda_budget", result["cuda_peak_allocated_bytes"] <= 4 * GIB)
        code = int({c["name"] for c in result["checks"]} != set(REQUIRED))
    except BaseException as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            if backend is not None:
                if a.mode == "discard" and getattr(backend, "_stage_discard_started", False):
                    backend.shutdown(discard_stage=True)
                else:
                    backend.shutdown()
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
    result["exit_code"] = code
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["checks_ok"] = sum(x["ok"] for x in result["checks"])
    result["checks_total"] = len(result["checks"])
    text = json.dumps(result, indent=2)
    for source, label in ((str(a.fixture), "<TINY_FIXTURE>"), (str(a.cache), "<TINY_CACHE>"),
                          (str(a.work), "<WORK_ROOT>"), (str(Path.home()), "<LINUX_HOME>"),
                          (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>")):
        text = text.replace(source, label)
    (a.work / "worker-result.json").write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
