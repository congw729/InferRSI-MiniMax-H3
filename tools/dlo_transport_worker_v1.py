"""Frozen v1: true native DLO transports against the complete tiny native PDD loop."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import traceback
import weakref

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from native_chain_common_v1 import denoise
from validate_meta_coverage_v2 import make_model, setup_dist
from validate_mmap_v2 import native_load, state

GIB = 1024**3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_steps(left, right):
    return len(left["steps"]) == len(right["steps"]) == 8 and all(
        x["step"] == y["step"] and torch.equal(x["video_rows"],y["video_rows"])
        and torch.equal(x["audio_rows"],y["audio_rows"]) for x,y in zip(left["steps"],right["steps"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiny-fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.parent
    root.mkdir(parents=True,exist_ok=True)
    result = {"campaign_verdict":"INCONCLUSIVE","scope":"actual native DLO transport on tiny H3",
              "full_pipeline_constructed":False,"generic_loader_integrated":False,
              "real_weights":False,"checks":[],"cases":{}}
    code = 1
    backend = None

    def check(name, ok, **details):
        result["checks"].append({"name":name,"ok":bool(ok),**details})
        if not ok:
            raise AssertionError(name)

    try:
        torch.cuda.set_per_process_memory_fraction(0.40)
        if shutil.disk_usage(root).free < 8*GIB:
            raise RuntimeError("disk headroom")
        expected=json.loads((a.tiny_fixture/"fixture-manifest.json").read_text())
        check("fixture_intact",all(sha(a.tiny_fixture/n)==v for n,v in expected.items())
              and sum((a.tiny_fixture/n).stat().st_size for n in expected) < 16*1024**2)
        config=json.loads((a.tiny_fixture/"config.json").read_text())
        if config["hidden_size"] != 128 or config["num_layers"] != 2:
            raise ValueError("tiny shape required")
        setup_dist(root)
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.offloader.base import OffloadConfig
        from vllm_omni.diffusion.offloader.config import OffloadStrategy
        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
        from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan, TensorBinding

        quant=FastH3NVFP4Config.from_dit(a.tiny_fixture/"nvfp4_weights.safetensors",rank=0)
        reference=make_model(config,quant,device="cpu").eval()
        native_load(reference,a.tiny_fixture)
        canonical=root/"runtime-layout.safetensors"
        canonical_values={name:value.detach().clone().contiguous() for name,value in state(reference).items()}
        save_file(canonical_values,canonical)
        check("runtime_cache_budget",canonical.stat().st_size < 16*1024**2)
        with safe_open(canonical,framework="pt",device="cpu") as reader:
            cache_exact=set(reader.keys())==set(canonical_values) and all(
                torch.equal(reader.get_tensor(n),v) for n,v in canonical_values.items())
        check("canonical_cache_exact",cache_exact)
        nonpersistent={}
        for name,value in reference.named_buffers():
            owner_name,_,leaf=name.rpartition(".")
            if leaf in reference.get_submodule(owner_name)._non_persistent_buffers_set:
                nonpersistent[name]=value.clone()
        result["nonpersistent_buffers"]=sorted(nonpersistent)
        result["canonical_file"]={"bytes":canonical.stat().st_size,"sha256":sha(canonical),
                                  "tensors":len(canonical_values),"origin":"tiny native_load final runtime layout"}
        del canonical_values
        hidden=torch.randn(2,128,generator=torch.Generator().manual_seed(241),dtype=torch.float32).bfloat16()
        reference.to("cuda")
        baseline=denoise(reference,config,hidden)
        check("reference_finite",torch.isfinite(baseline["video_latent"]).all()
              and torch.isfinite(baseline["audio_rows"]).all())
        ref=weakref.ref(reference)
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        check("reference_released",ref() is None)
        for mode in ("ordinary","canonical"):
            candidate=make_model(config,quant,device="cpu" if mode=="ordinary" else "meta").eval()
            pipeline=torch.nn.Module()
            pipeline.transformer=candidate
            pipeline._dit_modules=["transformer"]
            pipeline._encoder_modules=[]
            pipeline._vae_modules=[]
            pipeline._resident_modules=[]
            plan=None
            if mode=="ordinary":
                native_load(candidate,a.tiny_fixture)
            else:
                for name,value in nonpersistent.items():
                    owner_name,_,leaf=name.rpartition(".")
                    candidate.get_submodule(owner_name)._buffers[leaf]=value.clone()
                check("canonical_nonpersistent_buffers_restored",len(nonpersistent)==2
                      and all(torch.equal(state(candidate)[n],v) for n,v in nonpersistent.items()))
                bindings={}
                with safe_open(canonical,framework="pt",device="cpu") as reader:
                    for name,value in state(candidate).items():
                        if name in nonpersistent:
                            continue
                        source=reader.get_tensor(name)
                        if source.shape != value.shape or source.dtype != value.dtype:
                            raise ValueError("canonical metadata differs: "+name)
                        bindings["transformer."+name]=TensorBinding(name,str(canonical),None)
                        del source
                plan=HostWeightPlan("checkpoint_mmap",bindings,frozenset({"transformer."}))
            offload_config=OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                pin_cpu_memory=True,dp_size=1,dlo_use_allgather=False,components=frozenset({"dit"}))
            backend=DistributedLayerwiseOffloadBackend(offload_config,torch.device("cuda"),host_weight_plan=plan)
            backend.enable(pipeline)
            check(mode+"_enabled",backend.enabled)
            hooks=[hook for group in backend._all_hook_groups for hook in group]
            check(mode+"_two_blocks",len(hooks)==2)
            actual=denoise(candidate,config,hidden)
            check(mode+"_8_steps_exact",exact_steps(actual,baseline))
            check(mode+"_final_exact",torch.equal(actual["video_latent"],baseline["video_latent"])
                  and torch.equal(actual["audio_rows"],baseline["audio_rows"]))
            result["cases"][mode]={"native_hooks":len(hooks),"using_mmap":backend._using_mmap,
                "rank_local_mmap":backend._using_rank_local_mmap,
                "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "video_shape":list(actual["video_latent"].shape),"steps":len(actual["steps"]),
                "max_abs":float((actual["video_latent"]-baseline["video_latent"]).abs().max())}
            backend.shutdown()
            check(mode+"_shutdown",not backend.enabled and not hasattr(backend,"_mmap_file_cache")
                  and not backend._all_hook_groups)
            del hooks,actual,candidate,pipeline,plan
            backend=None
            gc.collect()
            torch.cuda.empty_cache()
        maps=Path("/proc/self/maps").read_text()
        check("mappings_released",str(canonical.resolve()) not in maps)
        check("fixture_unchanged",all(sha(a.tiny_fixture/n)==v for n,v in expected.items()))
        check("cuda_budget",torch.cuda.max_memory_allocated() <= 4*GIB)
        result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        code=0
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
        gc.collect()
        result["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result["checks_total"]=len(result["checks"])
    result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    text=json.dumps(result,indent=2)
    for source,label in ((str(a.tiny_fixture),"<TINY_FIXTURE>"),(str(root),"<RUN_ROOT>"),
        (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"),(str(Path.home()),"<LINUX_HOME>")):
        text=text.replace(source,label)
    a.output.write_text(text+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","traceback"}}))
    return code


if __name__=="__main__":
    raise SystemExit(main())
