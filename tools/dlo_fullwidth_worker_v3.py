"""Frozen v3 (same gates, durable phase receipts): two distinct real-width blocks reuse one verified canonical cache."""
import argparse
import gc
import json
from pathlib import Path
import resource
import traceback
import weakref
from types import SimpleNamespace

from dlo_fullwidth_worker_v1 import sha
GIB=1024**3
REQUIRED=["cache_manifest_identity","cache_files_intact","fixture_intact","two_distinct_blocks",
 "all_targets_mapped","alias_only_second_block","reference_finite_eight_steps","reference_released",
 "native_dlo_two_hooks","rank_local_mmap","two_slot_sizes","candidate_finite_eight_steps",
 "every_step_and_final_exact","shutdown","model_objects_released","maps_released",
 "cache_unchanged","no_new_cache","cuda_budget"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--cache-manifest-sha",required=True)
    p.add_argument("--work",type=Path,required=True)
    a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    result=dict(campaign_verdict="INCONCLUSIVE",real_weights=False,full_model_fit_proven=False,
        generic_cli_integrated=False,scope="two distinct real-width blocks with identical synthetic canonical weights",
        general_cache_reopen_implemented=False,checks=[])
    backend=None
    code=1
    def checkpoint(phase):
        import torch
        data={"phase":phase,"checks":result["checks"],
              "cuda_allocated":torch.cuda.memory_allocated(),
              "cuda_peak":torch.cuda.max_memory_allocated(),
              "ru_maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        with (a.work/"live-phases.jsonl").open("a") as stream:
            stream.write(json.dumps(data)+"\n")
            stream.flush()
    def check(name,ok):
        if name not in REQUIRED or any(c["name"]==name for c in result["checks"]):
            raise AssertionError("unknown or duplicate check")
        result["checks"].append(dict(name=name,ok=bool(ok)))
        checkpoint(name)
        if not ok:
            raise AssertionError(name)
    try:
        import torch
        from safetensors import safe_open
        from h3_omni_adapter.mmap_loader import _state
        from validate_meta_coverage_v2 import make_model,setup_dist
        from dlo_temporal_common_v1 import geometry,sample
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan,TensorBinding
        from vllm_omni.diffusion.offloader.base import OffloadConfig
        from vllm_omni.diffusion.offloader.config import OffloadStrategy
        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
        torch.cuda.set_per_process_memory_fraction(0.40)
        if torch.cuda.mem_get_info()[0]<5*GIB:
            raise RuntimeError("initial CUDA headroom")
        manifest=json.loads((a.cache/"manifest.json").read_text())
        check("cache_manifest_identity",sha(a.cache/"manifest.json")==a.cache_manifest_sha
            and manifest["schema"]=="h3-native-dlo-runtime-cache/v1" and manifest["complete"])
        check("cache_files_intact",all(Path(n).name==n and (a.cache/n).stat().st_size==v["bytes"]
            and sha(a.cache/n)==v["sha256"] for n,v in manifest["files"].items()))
        fixture=json.loads((a.fixture/"fixture-manifest.json").read_text())
        check("fixture_intact",all(sha(a.fixture/v["file"])==v["sha256"] for v in fixture))
        setup_dist(a.work)
        config=json.loads((a.fixture/"config.json").read_text())
        if config["hidden_size"]!=5376 or config["num_layers"]!=1 or config["num_refiner_layers"]!=1:
            raise ValueError("unexpected source shape")
        config["num_layers"]=2
        quant=FastH3NVFP4Config.from_dit(a.fixture/"nvfp4_weights.safetensors",rank=0)
        # The original packed sidecar contains only block zero metadata.
        # Duplicate that reviewed configuration for the distinct second block.
        import copy
        quant=copy.deepcopy(quant)
        result["quant_attributes"]=sorted(vars(quant))
        # Filled by the frozen native quant-config expansion below.
        expand_quant(quant)
        def build():
            model=make_model(config,quant,device="meta").eval()
            tensors=_state(model)
            bindings={}
            aliases={}
            for name,target in tensors.items():
                source="blocks.0."+name[len("blocks.1."):] if name.startswith("blocks.1.") else name
                info=manifest["tensors"][source]
                if list(target.shape)!=info["shape"] or str(target.dtype)!=info["dtype"]:
                    raise ValueError("canonical tensor metadata mismatch")
                aliases[name]=source
                if name in manifest["nonpersistent"]:
                    with safe_open(a.cache/info["file"],framework="pt",device="cpu") as reader:
                        value=reader.get_tensor(source).clone()
                    parent,_,leaf=name.rpartition(".")
                    model.get_submodule(parent)._buffers[leaf]=value
                else:
                    bindings["transformer."+name]=TensorBinding(source,str(a.cache/info["file"]))
            pipeline=torch.nn.Module()
            pipeline.transformer=model
            pipeline._dit_modules=["transformer"]
            pipeline._encoder_modules=[]
            pipeline._vae_modules=[]
            pipeline._resident_modules=[]
            host_plan=HostWeightPlan("checkpoint_mmap",bindings,frozenset({"transformer."}))
            settings=OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,pin_cpu_memory=True,
                dp_size=1,dlo_use_allgather=False,components=frozenset({"dit"}))
            transport=DistributedLayerwiseOffloadBackend(settings,torch.device("cuda"),host_weight_plan=host_plan)
            return model,pipeline,host_plan,transport,aliases
        model,pipeline,host_plan,backend,aliases=build()
        check("two_distinct_blocks",len(model.blocks)==2 and model.blocks[0] is not model.blocks[1]
            and all(a is not b for a,b in zip(model.blocks[0].parameters(),model.blocks[1].parameters(),strict=True)))
        check("all_targets_mapped",len(aliases)==111 and len(host_plan.bindings)==109)
        check("alias_only_second_block",all(name==source or
            (name.startswith("blocks.1.") and source=="blocks.0."+name[len("blocks.1."):])
            for name,source in aliases.items()) and sum(n!=s for n,s in aliases.items())==41)
        # Native mmap realization without offload hooks is the fully GPU-resident
        # reference. The candidate below enables the real two-slot transport.
        backend._using_rank_local_mmap=True
        backend._load_weights_via_mmap(pipeline,(SimpleNamespace(path="transformer",module=model),),host_plan)
        checkpoint("reference_before_cuda")
        model.cuda()
        checkpoint("reference_after_cuda")
        g=geometry("legal4s256")
        hidden=torch.full((g["text_tokens"],5120),0.01,dtype=torch.bfloat16)
        result["geometry"]=g
        checkpoint("reference_before_sampling")
        baseline=sample(model,config,hidden,g)
        check("reference_finite_eight_steps",len(baseline["steps"])==8 and all(torch.isfinite(row[k]).all()
            for row in baseline["steps"] for k in ("video_rows","audio_rows")))
        backend.shutdown()
        refs=[weakref.ref(v) for v in _state(model).values()]
        del model,pipeline,host_plan,backend,aliases
        backend=None
        gc.collect()
        torch.cuda.empty_cache()
        check("reference_released",all(r() is None for r in refs)
            and str(a.cache.resolve()) not in Path("/proc/self/maps").read_text())
        result["reference_cuda_peak"]=torch.cuda.max_memory_allocated()
        if result["reference_cuda_peak"]>4*GIB:
            raise RuntimeError("reference CUDA peak")
        torch.cuda.reset_peak_memory_stats()
        model,pipeline,host_plan,backend,aliases=build()
        checkpoint("candidate_before_enable")
        backend.enable(pipeline)
        checkpoint("candidate_after_enable")
        hooks=[h for group in backend._all_hook_groups for h in group]
        check("native_dlo_two_hooks",backend.enabled and len(hooks)==2)
        check("rank_local_mmap",backend._using_mmap and backend._using_rank_local_mmap)
        gpu=sum(t.numel()*t.element_size() for slot in hooks[0].gpu_buffers for t in slot.values())
        cpu=sum(t.numel()*t.element_size() for slot in hooks[0].cpu_staging_buffers for t in slot.values())
        result["actual_slots"]={"gpu_bytes":gpu,"cpu_bytes":cpu}
        check("two_slot_sizes",gpu==1517753512 and cpu==1517753512
            and hooks[0].gpu_buffers is hooks[1].gpu_buffers
            and hooks[0].cpu_staging_buffers is hooks[1].cpu_staging_buffers)
        checkpoint("candidate_before_sampling")
        actual=sample(model,config,hidden,g)
        check("candidate_finite_eight_steps",len(actual["steps"])==8 and all(torch.isfinite(row[k]).all()
            for row in actual["steps"] for k in ("video_rows","audio_rows")))
        check("every_step_and_final_exact",all(torch.equal(x[k],y[k]) for x,y in
            zip(actual["steps"],baseline["steps"],strict=True) for k in ("video_rows","audio_rows"))
            and torch.equal(actual["video_latent"],baseline["video_latent"])
            and torch.equal(actual["audio_rows"],baseline["audio_rows"]))
        result["attention_backend"]=model.blocks[0].attn.attention.attn_backend.get_name()
        backend.shutdown()
        check("shutdown",not backend.enabled and not backend._all_hook_groups)
        refs=[weakref.ref(v) for v in _state(model).values()]
        del model,pipeline,host_plan,backend,hooks,aliases
        backend=None
        gc.collect()
        torch.cuda.empty_cache()
        check("model_objects_released",all(r() is None for r in refs))
        check("maps_released",str(a.cache.resolve()) not in Path("/proc/self/maps").read_text()
            and str(a.fixture.resolve()) not in Path("/proc/self/maps").read_text())
        check("cache_unchanged",sha(a.cache/"manifest.json")==a.cache_manifest_sha
            and all(sha(a.cache/n)==v["sha256"] for n,v in manifest["files"].items()))
        check("no_new_cache",not list(a.work.rglob("*.safetensors")))
        result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        check("cuda_budget",result["cuda_peak_allocated_bytes"]<=4*GIB)
        code=int(set(c["name"] for c in result["checks"])!=set(REQUIRED))
    except BaseException as exc:
        result["execution_error"]={"type":type(exc).__name__,"message":str(exc)}
        result["traceback"]=traceback.format_exc()
    finally:
        try:
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
    for source,label in ((str(a.fixture),"<REUSED_FIXTURE>"),(str(a.cache),"<REUSED_CANONICAL_CACHE>"),
        (str(a.work),"<WORK_ROOT>"),(str(Path.home()),"<LINUX_HOME>"),
        (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
        text=text.replace(source,label)
    (a.work/"worker-result.json").write_text(text+"\n")
    return code


def expand_quant(quant):
    import copy
    additions={"blocks.1."+name[len("blocks.0."):]:copy.deepcopy(specs)
        for name,specs in quant.projections.items() if name.startswith("blocks.0.")}
    if not additions or any(name in quant.projections for name in additions):
        raise ValueError("unexpected packed projection expansion")
    # These copies describe distinct native modules. Final-layout HostWeightPlan
    # loads their tensors; source weight_map is deliberately not used here.
    quant.projections.update(additions)


if __name__=="__main__":
    raise SystemExit(main())
