"""Frozen worker: native reference or one owned streamed stage, no real weights."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import traceback
import weakref


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("reference","staged"),required=True)
    p.add_argument("--stage",choices=("encode","denoise","decode"),required=True)
    p.add_argument("--scenario",choices=("tiny","legal4s256"),required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--native-source",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--input",type=Path)
    a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    report=dict(campaign_verdict="INCONCLUSIVE",mode=a.mode,stage=a.stage,scenario=a.scenario,
                real_video_generated=False,generic_cli_integrated=False)
    code=1
    loader=backend=decoder=None
    try:
        import torch
        import native_chain_common_v1 as c
        import dlo_temporal_common_v1 as chain
        from validate_meta_coverage_v2 import setup_dist,make_model
        from validate_mmap_v2 import native_load,state
        from validate_text_streaming_v3 import reference_forward
        from validate_temporal_decoder_v2 import make_reference
        from h3_omni_adapter.stage_handoff import read_payload,write_payload
        from h3_omni_adapter.temporal_decoder import H3TemporalVideoDecoder
        from h3_omni_adapter.dlo_bridge import export_runtime_cache
        from h3_omni_adapter.mmap_loader import H3MmapPlan
        from vllm_omni.diffusion.offloader.base import OffloadConfig
        from vllm_omni.diffusion.offloader.config import OffloadStrategy
        from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        report["initial_cuda_allocated"]=torch.cuda.memory_allocated()
        torch.cuda.set_per_process_memory_fraction(0.40)
        sys.path.insert(0,str(a.native_source/"Ref2VA"))
        from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
        setup_dist(a.work)
        _ensure_vae_parallel_state()
        g=chain.geometry(a.scenario)
        report["geometry"]=g
        ids=(torch.arange(g["text_tokens"])*7+1)%250
        manifest=json.loads((a.checkpoint/"fixture-manifest.json").read_text())
        if not all(hashlib.sha256((a.checkpoint/n).read_bytes()).hexdigest()==v["sha256"] for n,v in manifest.items()):
            raise ValueError("fixture manifest mismatch")
        config=json.loads((a.checkpoint/"dit/config.json").read_text())
        if config["hidden_size"]!=128 or config["num_layers"]!=2:
            raise ValueError("tiny only")
        vae_config=json.loads((a.checkpoint/"decoder-config.json").read_text())
        outer=json.loads((a.checkpoint/"vae-normalization.json").read_text())
        if a.mode=="reference":
            text,_=c.text_model(a.checkpoint/"text",False)
            hidden=reference_forward(text,ids)
            dit,_,_=c.dit_model(a.checkpoint/"dit",False)
            sampled=chain.sample(dit,config,hidden,g)
            decoder=make_reference(vae_config,outer,a.checkpoint/"decoder.safetensors")
            decoded=chain.decode(decoder,a.checkpoint,sampled,streamed=False)
            output=dict(hidden=hidden,sampled=sampled,decoded=decoded)
            report["attention_backend"]=dit.blocks[0].attn.attention.attn_backend.get_name()
            refs=[weakref.ref(v) for model in (text,dit,decoder) for v in state(model).values()]
            del text,dit,decoder
            decoder=None
        elif a.stage=="encode":
            text,loader=c.text_model(a.checkpoint/"text",True)
            hidden=text.encode_ids(ids)
            report["loader_stats"]=c.close_loader(text,loader)
            report["all_meta_after_close"]=all(v.is_meta for v in state(text).values())
            refs=[weakref.ref(v) for v in state(text).values()]
            output=dict(kind="native_text_hidden",hidden=hidden,geometry=g)
            del text,loader
            loader=None
        elif a.stage=="denoise":
            incoming=read_payload(a.input)
            if incoming.get("kind")!="native_text_hidden" or incoming["geometry"]!=g or tuple(incoming["hidden"].shape)!=(g["text_tokens"],128):
                raise ValueError("invalid text handoff")
            quant=FastH3NVFP4Config.from_dit(a.checkpoint/"dit/nvfp4_weights.safetensors",rank=0)
            dit=make_model(config,quant,device="meta").eval()
            plan=H3MmapPlan.build(dit,a.checkpoint/"dit",sidecars=["nvfp4_weights.safetensors"],max_total_file_bytes=16*1024**2)
            cache=export_runtime_cache(dit,plan,a.work/"runtime-cache",max_group_bytes=8*1024**2,
                max_live_bytes=16*1024**2,max_output_bytes=16*1024**2,min_free_bytes=8*1024**3)
            host_plan=cache.bind(dit,component_path="transformer")
            pipeline=torch.nn.Module()
            pipeline.transformer=dit
            pipeline._dit_modules=["transformer"]
            pipeline._encoder_modules=[]
            pipeline._vae_modules=[]
            pipeline._resident_modules=[]
            settings=OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,pin_cpu_memory=True,
                dp_size=1,dlo_use_allgather=False,components=frozenset({"dit"}))
            backend=DistributedLayerwiseOffloadBackend(settings,torch.device("cuda"),host_weight_plan=host_plan)
            backend.enable(pipeline)
            report["native_dlo_hooks"]=sum(len(group) for group in backend._all_hook_groups)
            report["native_mmap"]=backend._using_mmap and backend._using_rank_local_mmap
            report["export_stats"]=cache.stats
            output=chain.sample(dit,config,incoming["hidden"],g)
            report["attention_backend"]=dit.blocks[0].attn.attention.attn_backend.get_name()
            backend.shutdown()
            report["dlo_shutdown"]=not backend.enabled and not backend._all_hook_groups
            refs=[weakref.ref(v) for v in state(dit).values()]
            del dit,pipeline,plan,cache,host_plan,backend
            backend=None
        else:
            incoming=read_payload(a.input)
            if incoming["geometry"]!=g:
                raise ValueError("decoder geometry mismatch")
            decoder=H3TemporalVideoDecoder(vae_config,outer,a.checkpoint/"decoder.safetensors",device="cuda",
                max_latent_tokens=g["latent_t"],max_height=g["height"],max_width=g["width"],
                max_output_frames=g["frames"],output_mode="uint8")
            refs=[weakref.ref(v) for v in state(decoder.native).values()]
            output=chain.decode(decoder,a.checkpoint,incoming,streamed=True)
            report["temporal_plan"]=decoder.last_plan
            decoder.close()
            del decoder
            decoder=None
        gc.collect()
        torch.cuda.synchronize()
        report["model_tensor_objects_released"]=all(ref() is None for ref in refs)
        maps=Path("/proc/self/maps").read_text()
        report["no_checkpoint_mmaps"]=str(a.checkpoint) not in maps and str(a.work/"runtime-cache") not in maps
        write_payload(a.work/"handoff",output)
        report["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        if report["cuda_peak_allocated_bytes"]>4*1024**3:
            raise RuntimeError("CUDA phase allocation guard")
        report["completed"]=True
        code=0
    except Exception as exc:
        report["execution_error"]={"type":type(exc).__name__,"message":str(exc)}
        report["traceback"]=traceback.format_exc()
    finally:
        try:
            if loader is not None:
                loader.close()
            if backend is not None:
                backend.shutdown()
            if decoder is not None and hasattr(decoder,"close"):
                decoder.close()
            import torch
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            report["cleanup_error"]=str(exc)
            code=1
    report["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    text=json.dumps(report,indent=2)
    for source,label in ((str(a.checkpoint),"<SYNTHETIC_CHECKPOINT>"),(str(a.work),"<STAGE_WORK>"),
        (str(a.native_source),"<NATIVE_SOURCE>"),(str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"),
        (str(Path.home()),"<LINUX_HOME>")):
        text=text.replace(source,label)
    (a.work/"worker-result.json").write_text(text+"\n")
    return code


if __name__=="__main__":
    raise SystemExit(main())
