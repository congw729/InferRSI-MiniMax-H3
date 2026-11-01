"""Frozen real-width one-block budget and bounded native DLO sampling experiment."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import time
import traceback
import weakref

GIB=1024**3
BUDGET_CHECKS=["fixture_bounded","fixture_hashes","real_width_one_block","all_meta",
 "complete_plan","known_groups","conversion_live_budget","cache_disk_budget",
 "native_resident_plus_slots_budget","initial_disk_headroom","budget_has_no_cache"]
EXEC_CHECKS=["baseline_eight_steps_finite","baseline_model_released","export_complete",
 "export_meta_and_source_unmapped","native_plan_complete","native_backend_one_hook",
 "native_slot_sizes_match","native_rank_local_mmap","candidate_eight_steps_finite",
 "every_step_and_latent_exact","backend_shutdown","model_objects_released",
 "all_mmaps_released","fixture_unchanged","cuda_peak_budget"]


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--mode",choices=("budget","execute"),required=True)
    p.add_argument("--work",type=Path,required=True)
    a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    result=dict(campaign_verdict="INCONCLUSIVE",scope="one real-width DiT and refiner; synthetic weights",
        mode=a.mode,real_weights=False,full_model_fit_proven=False,generic_cli_integrated=False,
        checks=[],phases=[])
    required=BUDGET_CHECKS+(EXEC_CHECKS if a.mode=="execute" else [])
    model=loader=backend=None
    code=1
    started=time.monotonic()
    def check(name,ok):
        if name not in required or any(c["name"]==name for c in result["checks"]):
            raise AssertionError("unknown or duplicate check")
        result["checks"].append(dict(name=name,ok=bool(ok)))
        if not ok:
            raise AssertionError(name)
    try:
        import torch
        from h3_omni_adapter.mmap_loader import H3MmapLoader,H3MmapPlan,_state
        from h3_omni_adapter.dlo_bridge import export_runtime_cache
        from validate_meta_coverage_v2 import make_model,setup_dist
        from dlo_temporal_common_v1 import geometry,sample
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        def mark(name):
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            row=dict(phase=name,elapsed=time.monotonic()-started,
                cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved(),
                cuda_peak=torch.cuda.max_memory_allocated(),
                worker_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            result["phases"].append(row)
            with (a.work/"phases.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            if row["cuda_peak"]>4*GIB:
                raise RuntimeError("CUDA peak allocation guard")
        manifest=json.loads((a.fixture/"fixture-manifest.json").read_text())
        check("fixture_bounded",sum(v["bytes"] for v in manifest)<3*GIB)
        check("fixture_hashes",all(sha(a.fixture/v["file"])==v["sha256"] for v in manifest))
        config=json.loads((a.fixture/"config.json").read_text())
        check("real_width_one_block",config["hidden_size"]==5376 and config["num_layers"]==1
            and config["num_refiner_layers"]==1 and config["text_dim"]==5120)
        setup_dist(a.work)
        quant=FastH3NVFP4Config.from_dit(a.fixture/"nvfp4_weights.safetensors",rank=0)
        model=make_model(config,quant,device="meta").eval()
        check("all_meta",all(t.is_meta for t in _state(model).values()))
        plan=H3MmapPlan.build(model,a.fixture,sidecars=["nvfp4_weights.safetensors"],max_total_file_bytes=3*GIB)
        check("complete_plan",set(plan.bindings)==set(_state(model)))
        groups={name:sum(math.prod(plan.bindings[n].shape)*torch.empty((),
                    dtype=plan.bindings[n].dtype,device="meta").element_size() for n in names)
                for name,names in plan.groups.items()}
        expected={"resident":267139392,"token_refiner.blocks.0":770725376,"blocks.0":758876756}
        check("known_groups",groups==expected)
        live=groups["resident"]+max(v for k,v in groups.items() if k!="resident")
        upper=4*1024**2+sum(groups.values())+sum(1024+sum(256+len(n.encode()) for n in names)
                                                        for names in plan.groups.values())
        resident=groups["resident"]+groups["token_refiner.blocks.0"]
        slots=2*groups["blocks.0"]
        result["budget"]={"groups":groups,"target_count":len(plan.bindings),
            "conversion_logical_live_bytes":live,"cache_payload_bytes":sum(groups.values()),
            "cache_conservative_upper_bytes":upper,"native_nonblock_resident_bytes":resident,
            "native_gpu_two_slots_bytes":slots,"native_cpu_two_staging_slots_bytes":slots,
            "native_gpu_weight_floor_bytes":resident+slots,
            "scope":"logical tensors only; excludes CUDA activations/workspace, allocator overhead and RSS/file pages"}
        check("conversion_live_budget",max(groups.values())<=GIB and live<=2*GIB)
        check("cache_disk_budget",upper<=2*GIB)
        check("native_resident_plus_slots_budget",resident+slots<3*GIB)
        check("initial_disk_headroom",shutil.disk_usage(a.work).free>=8*GIB+upper)
        check("budget_has_no_cache",not (a.work/"runtime-cache").exists())
        mark("meta_budget_complete")
        if a.mode=="execute":
            if torch.cuda.mem_get_info()[0]<5*GIB:
                raise RuntimeError("initial CUDA headroom")
            torch.cuda.set_per_process_memory_fraction(0.40)
            g=geometry("legal4s256")
            hidden=torch.full((g["text_tokens"],config["text_dim"]),0.01,dtype=torch.bfloat16)
            result["geometry"]=g
            loader=H3MmapLoader(model,plan,device="cuda",max_group_bytes=GIB)
            loader.load_group("resident")
            loader.install_block_hooks()
            mark("baseline_resident_loaded")
            baseline=sample(model,config,hidden,g)
            check("baseline_eight_steps_finite",len(baseline["steps"])==8 and all(
                torch.isfinite(row[key]).all() for row in baseline["steps"] for key in ("video_rows","audio_rows")))
            loader.close()
            refs=[weakref.ref(v) for v in _state(model).values()]
            del loader,model,plan
            loader=model=None
            gc.collect()
            torch.cuda.empty_cache()
            check("baseline_model_released",all(ref() is None for ref in refs)
                and str(a.fixture.resolve()) not in Path("/proc/self/maps").read_text())
            mark("baseline_released")
            model=make_model(config,quant,device="meta").eval()
            plan=H3MmapPlan.build(model,a.fixture,sidecars=["nvfp4_weights.safetensors"],max_total_file_bytes=3*GIB)
            cache=export_runtime_cache(model,plan,a.work/"runtime-cache",max_group_bytes=GIB,
                max_live_bytes=2*GIB,max_output_bytes=2*GIB,min_free_bytes=8*GIB)
            result["export_stats"]=cache.stats
            check("export_complete",cache.stats["groups_loaded"]==3 and cache.stats["max_active_streamed_groups"]==1
                and cache.stats["complete_cache_bytes"]<=2*GIB)
            check("export_meta_and_source_unmapped",all(t.is_meta for t in _state(model).values())
                and str(a.fixture.resolve()) not in Path("/proc/self/maps").read_text())
            mark("cpu_cache_export_complete")
            host_plan=cache.bind(model,component_path="transformer")
            check("native_plan_complete",len(host_plan.bindings)==len(plan.bindings)-2)
            from vllm_omni.diffusion.offloader.base import OffloadConfig
            from vllm_omni.diffusion.offloader.config import OffloadStrategy
            from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
            pipeline=torch.nn.Module()
            pipeline.transformer=model
            pipeline._dit_modules=["transformer"]
            pipeline._encoder_modules=[]
            pipeline._vae_modules=[]
            pipeline._resident_modules=[]
            settings=OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,pin_cpu_memory=True,
                dp_size=1,dlo_use_allgather=False,components=frozenset({"dit"}))
            backend=DistributedLayerwiseOffloadBackend(settings,torch.device("cuda"),host_weight_plan=host_plan)
            backend.enable(pipeline)
            hooks=[h for group in backend._all_hook_groups for h in group]
            check("native_backend_one_hook",backend.enabled and len(hooks)==1)
            actual_gpu=sum(t.numel()*t.element_size() for slot in hooks[0].gpu_buffers for t in slot.values())
            actual_cpu=sum(t.numel()*t.element_size() for slot in hooks[0].cpu_staging_buffers for t in slot.values())
            result["actual_slots"]={"gpu_bytes":actual_gpu,"cpu_bytes":actual_cpu}
            check("native_slot_sizes_match",actual_gpu==slots and actual_cpu==slots)
            check("native_rank_local_mmap",backend._using_mmap and backend._using_rank_local_mmap)
            mark("native_backend_enabled")
            actual=sample(model,config,hidden,g)
            check("candidate_eight_steps_finite",len(actual["steps"])==8 and all(
                torch.isfinite(row[key]).all() for row in actual["steps"] for key in ("video_rows","audio_rows")))
            check("every_step_and_latent_exact",all(torch.equal(x[key],y[key]) for x,y in
                zip(actual["steps"],baseline["steps"],strict=True) for key in ("video_rows","audio_rows"))
                and torch.equal(actual["video_latent"],baseline["video_latent"])
                and torch.equal(actual["audio_rows"],baseline["audio_rows"]))
            result["attention_backend"]=model.blocks[0].attn.attention.attn_backend.get_name()
            result["output_shapes"]={"video_latent":list(actual["video_latent"].shape),"audio_rows":list(actual["audio_rows"].shape)}
            mark("native_eight_steps_complete")
            backend.shutdown()
            check("backend_shutdown",not backend.enabled and not backend._all_hook_groups)
            refs=[weakref.ref(v) for v in _state(model).values()]
            del model,pipeline,cache,plan,host_plan,backend,hooks
            model=backend=None
            gc.collect()
            torch.cuda.empty_cache()
            check("model_objects_released",all(ref() is None for ref in refs))
            check("all_mmaps_released",str(a.fixture.resolve()) not in Path("/proc/self/maps").read_text()
                and str((a.work/"runtime-cache").resolve()) not in Path("/proc/self/maps").read_text())
            check("fixture_unchanged",all(sha(a.fixture/v["file"])==v["sha256"] for v in manifest))
            check("cuda_peak_budget",torch.cuda.max_memory_allocated()<=4*GIB)
            result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
            mark("complete")
        code=int(set(c["name"] for c in result["checks"])!=set(required))
    except BaseException as exc:
        result["execution_error"]={"type":type(exc).__name__,"message":str(exc)}
        result["traceback"]=traceback.format_exc()
        code=1
    finally:
        try:
            if loader is not None:
                loader.close()
            if backend is not None:
                backend.shutdown()
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"]=str(exc)
            code=1
    result["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    result["checks_total"]=len(result["checks"])
    result["exit_code"]=code
    text=json.dumps(result,indent=2)
    for source,label in ((str(a.fixture),"<REUSED_FIXTURE>"),(str(a.work),"<WORK_ROOT>"),
        (str(Path.home()),"<LINUX_HOME>"),(str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
        text=text.replace(source,label)
    (a.work/"worker-result.json").write_text(text+"\n")
    return code


if __name__=="__main__":
    raise SystemExit(main())
