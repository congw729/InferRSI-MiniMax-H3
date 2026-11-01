"""Frozen v1: bounded grouped H3 conversion into native DLO checkpoint mmap."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import traceback
from unittest.mock import patch

import torch
from safetensors import safe_open

from dlo_transport_worker_v1 import exact_steps
from native_chain_common_v1 import denoise
from validate_meta_coverage_v2 import make_model, setup_dist
from validate_mmap_v2 import native_load, state

REQUIRED = [
 "fixture_intact", "reference_finite", "reference_released", "output_budget_rejected_before_write",
 "group_budget_rejected_before_write", "live_budget_rejected_before_write",
 "write_failure_preserves_partial_evidence", "write_failure_releases_model",
 "export_all_model_state_meta", "all_four_groups_converted", "one_streamed_group_only",
 "canonical_values_exact", "all_111_targets_cached", "no_source_maps_retained",
 "cache_total_budget", "wrong_model_rejected", "tampered_cache_rejected",
 "tamper_rejection_keeps_model_meta", "native_plan_109_persistent_bindings",
 "pdd_nonpersistent_exact", "native_backend_enabled_two_blocks", "native_mmap_used",
 "two_native_sampling_loops_exact", "shutdown_removes_hooks", "all_mmaps_released",
 "source_fixture_unchanged", "cuda_budget"]
GIB=1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tiny-fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root=a.output.parent
    root.mkdir(parents=True,exist_ok=True)
    result={"campaign_verdict":"INCONCLUSIVE","scope":"grouped normalization bridge to real DLO",
            "generic_loader_integrated":False,"full_pipeline_constructed":False,"real_weights":False,"checks":[]}
    backend=None
    code=1

    def check(name,ok,**details):
        if name not in REQUIRED or any(c["name"]==name for c in result["checks"]):
            raise AssertionError("unknown/duplicate check: "+name)
        result["checks"].append({"name":name,"ok":bool(ok),**details})
        if not ok:
            raise AssertionError(name)

    def rejects(call):
        try:
            call()
        except (ValueError,RuntimeError,FileExistsError):
            return True
        return False

    try:
        torch.cuda.set_per_process_memory_fraction(0.40)
        if shutil.disk_usage(root).free < 8*GIB:
            raise RuntimeError("disk headroom")
        expected=json.loads((a.tiny_fixture/"fixture-manifest.json").read_text())
        check("fixture_intact",all(sha(a.tiny_fixture/n)==v for n,v in expected.items())
              and sum((a.tiny_fixture/n).stat().st_size for n in expected) < 16*1024**2)
        setup_dist(root)
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.offloader.base import OffloadConfig
        from vllm_omni.diffusion.offloader.config import OffloadStrategy
        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
        from h3_omni_adapter.mmap_loader import H3MmapPlan
        from h3_omni_adapter.dlo_bridge import export_runtime_cache

        config=json.loads((a.tiny_fixture/"config.json").read_text())
        if config["hidden_size"] != 128 or config["num_layers"] != 2:
            raise ValueError("tiny only")
        quant=FastH3NVFP4Config.from_dit(a.tiny_fixture/"nvfp4_weights.safetensors",rank=0)
        reference=make_model(config,quant,device="cpu").eval()
        native_load(reference,a.tiny_fixture)
        expected_state={n:v.detach().clone() for n,v in state(reference).items()}
        hidden=torch.randn(2,128,generator=torch.Generator().manual_seed(241)).bfloat16()
        baseline=denoise(reference.cuda(),config,hidden)
        check("reference_finite",torch.isfinite(baseline["video_latent"]).all() and torch.isfinite(baseline["audio_rows"]).all())
        import weakref
        ref=weakref.ref(reference)
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        check("reference_released",ref() is None)

        model=make_model(config,quant,device="meta").eval()
        plan=H3MmapPlan.build(model,a.tiny_fixture,sidecars=["nvfp4_weights.safetensors"],max_total_file_bytes=16*1024**2)
        options=dict(max_group_bytes=8*1024**2,max_live_bytes=16*1024**2,max_output_bytes=16*1024**2,min_free_bytes=8*GIB)
        for name,key in (("output","max_output_bytes"),("group","max_group_bytes"),("live","max_live_bytes")):
            dest=root/("reject-"+name)
            opts={**options,key:1}
            check(name+"_budget_rejected_before_write",rejects(lambda opts=opts,dest=dest:
                  export_runtime_cache(model,plan,dest,**opts)) and not dest.exists())
        partial=root/"injected-write-failure"
        with patch("h3_omni_adapter.dlo_bridge.save_file",side_effect=OSError("injected owned-output write failure")):
            try:
                export_runtime_cache(model,plan,partial,**options)
            except OSError:
                failed=True
            else:
                failed=False
        check("write_failure_preserves_partial_evidence",failed and partial.exists()
              and (partial/"failure.json").exists() and not (partial/"manifest.json").exists())
        check("write_failure_releases_model",all(v.is_meta for v in state(model).values()))
        cache=export_runtime_cache(model,plan,root/"runtime-cache",**options)
        check("export_all_model_state_meta",all(v.is_meta for v in state(model).values()))
        check("all_four_groups_converted",cache.stats["groups_loaded"]==4 and len(cache.manifest["files"])==4)
        check("one_streamed_group_only",cache.stats["max_active_streamed_groups"]==1)
        loaded={}
        for filename in cache.manifest["files"]:
            with safe_open(root/"runtime-cache"/filename,framework="pt",device="cpu") as reader:
                for name in reader.keys():
                    if name in loaded:
                        raise AssertionError("duplicate canonical target")
                    loaded[name]=reader.get_tensor(name).clone()
        check("canonical_values_exact",set(loaded)==set(expected_state) and all(torch.equal(loaded[n],v) for n,v in expected_state.items()))
        check("all_111_targets_cached",len(loaded)==111)
        del loaded
        gc.collect()
        check("no_source_maps_retained",all(str((a.tiny_fixture/n).resolve()) not in Path("/proc/self/maps").read_text()
              for n in ("dense.safetensors","nvfp4_weights.safetensors")))
        check("cache_total_budget",sum((root/"runtime-cache"/n).stat().st_size for n in cache.manifest["files"])<16*1024**2)
        other=make_model(config,quant,device="meta")
        check("wrong_model_rejected",rejects(lambda:cache.bind(other,component_path="transformer")))
        del other
        file=root/"runtime-cache"/next(iter(cache.manifest["files"]))
        with file.open("r+b") as stream:
            stream.seek(-1,2)
            original=stream.read(1)
            stream.seek(-1,2)
            stream.write(bytes([original[0]^1]))
        check("tampered_cache_rejected",rejects(lambda:cache.bind(model,component_path="transformer")))
        check("tamper_rejection_keeps_model_meta",all(v.is_meta for v in state(model).values()))
        with file.open("r+b") as stream:
            stream.seek(-1,2)
            stream.write(original)
        host_plan=cache.bind(model,component_path="transformer")
        check("native_plan_109_persistent_bindings",len(host_plan.bindings)==109
              and host_plan.backing_kind=="checkpoint_mmap"
              and all(v.transform is None for v in host_plan.bindings.values()))
        check("pdd_nonpersistent_exact",all(torch.equal(state(model)[n],expected_state[n]) for n in
              ("final_layer.video_out.coefficients","final_layer.audio_out.coefficients")))
        pipeline=torch.nn.Module()
        pipeline.transformer=model
        pipeline._dit_modules=["transformer"]
        pipeline._encoder_modules=[]
        pipeline._vae_modules=[]
        pipeline._resident_modules=[]
        offload=OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,pin_cpu_memory=True,
                             dp_size=1,dlo_use_allgather=False,components=frozenset({"dit"}))
        backend=DistributedLayerwiseOffloadBackend(offload,torch.device("cuda"),host_weight_plan=host_plan)
        backend.enable(pipeline)
        check("native_backend_enabled_two_blocks",backend.enabled and sum(len(g) for g in backend._all_hook_groups)==2)
        check("native_mmap_used",backend._using_mmap and backend._using_rank_local_mmap)
        repeated=[]
        for _ in range(2):
            actual=denoise(model,config,hidden)
            repeated.append(exact_steps(actual,baseline) and torch.equal(actual["video_latent"],baseline["video_latent"])
                            and torch.equal(actual["audio_rows"],baseline["audio_rows"]))
        check("two_native_sampling_loops_exact",all(repeated))
        result["cache_manifest"]=cache.manifest
        result["export_stats"]=cache.stats
        backend.shutdown()
        check("shutdown_removes_hooks",not backend.enabled and not backend._all_hook_groups
              and not hasattr(backend,"_mmap_file_cache"))
        del actual,backend,pipeline,model,host_plan,cache
        backend=None
        gc.collect()
        torch.cuda.empty_cache()
        check("all_mmaps_released",str((root/"runtime-cache").resolve()) not in Path("/proc/self/maps").read_text())
        check("source_fixture_unchanged",all(sha(a.tiny_fixture/n)==v for n,v in expected.items()))
        check("cuda_budget",torch.cuda.max_memory_allocated()<=4*GIB)
        result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        code=int(set(c["name"] for c in result["checks"])!=set(REQUIRED))
    except Exception as exc:
        result["execution_error"]={"type":type(exc).__name__,"message":str(exc)}
        result["traceback"]=traceback.format_exc()
    finally:
        try:
            if backend is not None:
                backend.shutdown()
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"]=str(exc)
            code=1
        result["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result["checks_total"]=len(result["checks"])
    result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    text=json.dumps(result,indent=2)
    for source,label in ((str(a.tiny_fixture),"<TINY_FIXTURE>"),(str(root),"<RUN_ROOT>"),
        (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"),(str(Path.home()),"<LINUX_HOME>")):
        text=text.replace(source,label)
    a.output.write_text(text+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","traceback","cache_manifest"}}))
    return code


if __name__=="__main__":
    raise SystemExit(main())
