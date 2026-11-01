"""Frozen v2: native VAE persistent mask/RoPE, full depth and lifecycle protocol."""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import traceback
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

GIB=1024**3
MIB=1024**2

def state(model):
    return dict(list(model.named_parameters())+list(model.named_buffers()))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def native_kwargs(outer):
    return {key:int(outer["vae_"+key]) for key in ("clip_length","token_drop",
        "encoder_tiling","decoder_tiling","parallel_tiling","tile_size","tile_overlap_min",
        "encoder_parallel","decoder_parallel","chunk_dim")}

@contextmanager
def temporal_mode(stream):
    wanted={"MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT":str(int(stream)),
        "MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE":"keep",
        "VLLM_OMNI_VAE_LEGACY_TEMPORAL":"0"}
    previous={k:os.environ.get(k) for k in wanted}
    os.environ.update(wanted)
    try:yield
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

def make_reference(config,outer):
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.base_module import RotaryEmbeddingND
    with torch.device("meta"):
        model=AutoencoderKLLegacy.from_config(copy.deepcopy(config),**native_kwargs(outer))
    if any(not value.is_meta for value in state(model).values()):
        raise AssertionError("full native reference constructor allocated storage")
    del model.encoder,model.quant_conv
    model.to_empty(device="cpu")
    options=config["vit_decoder_kwargs"]
    model.decoder.pos_embed=RotaryEmbeddingND(int(options["dim_head"]*options["rope_dim_ratio"]),
        options["rope_theta"],n_dim=3,use_angle=True)
    values={}
    for name,tensor in model.state_dict().items():
        seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%251
        data=((torch.arange(tensor.numel(),dtype=torch.float32)+seed).remainder(251)-125)/4096
        if "norm" in name and name.endswith("weight"):data+=1
        values[name]=data.reshape(tensor.shape)
    model.load_state_dict(values,strict=True)
    return model.eval(),values

def main():
    parser=argparse.ArgumentParser()
    for name in ("research","native-source","source-record","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    a=parser.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    out={"campaign_verdict":"INCONCLUSIVE","device":a.device,"real_weights":False,
         "checks":[],"cases":[],"scope":"native 36-layer meta; synthetic F32 narrow 3/36-layer decoder; native FP16 decode semantics"}
    cleanup=[]
    def ck(name,ok,**detail):
        out["checks"].append(dict(name=name,ok=bool(ok),**detail))
        if not ok:raise AssertionError(name)
    def reject(name,call):
        try:call()
        except (ValueError,TypeError,RuntimeError):ck(name,True)
        else:ck(name,False)
    try:
        torch.cuda.set_per_process_memory_fraction(.40)
        if a.device=="cuda" and torch.cuda.mem_get_info()[0]<2*GIB:
            raise RuntimeError("GPU free memory below resource boundary")
        ck("fixture_disk_headroom",shutil.disk_usage(a.output.parent).free>=8*GIB)
        provenance=json.loads((a.source_record/"plan/native-source-provenance.json").read_text())
        ck("native_source_fetch_complete",provenance.get("complete") is True)
        files=provenance["files"]
        ck("native_sources_present",bool(files))
        for item in files:
            relative=Path(item["path"]).relative_to("FL2VA")
            if relative.suffix==".py":
                ck("native_source_hash_"+relative.as_posix(),digest(a.native_source/relative)==item["sha256"])
        expected=json.loads((a.source_record/"plan/input-provenance.json").read_text())
        for name,sha in expected["metadata_files"].items():
            ck("metadata_hash_"+name,digest(a.research/"metadata"/name)==sha)
        sys.path.insert(0,str(a.native_source))
        from video_vae.minimax_h3_video_vae import _ensure_vae_parallel_state
        from video_vae.klvae import AutoencoderKLLegacy
        from vllm_omni.diffusion.models.minimax_h3 import vae as omni_vae
        from vllm_omni.diffusion.models.minimax_h3 import vae_temporal
        from h3_omni_adapter.native_vae_mmap import (
            build_native_decoder_meta,H3NativeVAEMmapPlan,H3NativeVAEMmapLoader)
        _ensure_vae_parallel_state()
        upstream=Path(inspect.getfile(omni_vae)).resolve()
        while upstream.name!="vllm_omni":upstream=upstream.parent
        for relative,sha in expected["upstream_sources"].items():
            ck("upstream_hash_"+relative,digest(upstream.parent/relative)==sha)
        cfg=json.loads((a.research/"metadata/FL2VA__video_vae__source__config.json").read_text())
        outer=json.loads((a.research/"metadata/FL2VA__video_vae__config.json").read_text())
        header=json.loads((a.research/"metadata/FL2VA__video_vae__source__model.safetensors.header.raw").read_bytes())
        header.pop("__metadata__",None)
        for info in header.values():
            info["shape"]=tuple(info["shape"])
            info["bytes"]=info["data_offsets"][1]-info["data_offsets"][0]
        full=build_native_decoder_meta(cfg)
        bindings,excluded=H3NativeVAEMmapPlan.audit_headers(full,header)
        with torch.device("meta"):
            native_full=AutoencoderKLLegacy.from_config(copy.deepcopy(cfg),**native_kwargs(outer))
        full_native_state=state(native_full)
        selected={k:v for k,v in full_native_state.items() if k.startswith(("decoder.","post_quant_conv."))}
        ck("native_full_constructor_all_meta",all(v.is_meta for v in full_native_state.values()))
        ck("full_exact_native_selected_topology",set(selected)==set(state(full)) and all(
            selected[k].shape==v.shape and selected[k].dtype==v.dtype for k,v in state(full).items()))
        ck("full_36_layers",len(full.decoder.transformer_blocks)==36)
        ck("full_all_parameters_buffers_covered",set(bindings)==set(selected))
        ck("full_source_partition",set(header)==set(excluded)|{s for b in bindings.values() for s in b.sources}
            and not set(excluded)&{s for b in bindings.values() for s in b.sources})
        ck("full_only_encoder_quant_excluded",set(excluded)=={k for k in header if k.startswith(("encoder.","quant_conv."))})
        ck("full_fp32_all_meta",all(v.dtype==torch.float32 and v.is_meta for v in state(full).values()))
        ck("full_native_buffers_exact",set(dict(full.named_buffers()))==
            {k for k in dict(native_full.named_buffers()) if k.startswith(("decoder.","post_quant_conv."))})
        ck("full_persistent_mask_and_nonpersistent_rope",
            "decoder.mask_token" in dict(full.named_buffers()) and "decoder.mask_token" in full.state_dict()
            and "decoder.pos_embed.inv_freq" not in full.state_dict()
            and "inv_freq" in full.decoder.pos_embed._non_persistent_buffers_set
            and set(full.state_dict())==set(bindings)-{"decoder.pos_embed.inv_freq"})
        ck("full_native_persistent_state_exact",set(full.state_dict())==
            {k for k in native_full.state_dict() if k.startswith(("decoder.","post_quant_conv."))})
        out["full_meta"]={"layers":36,"parameters":len(list(full.named_parameters())),
            "buffers":len(list(full.named_buffers())),"targets":len(bindings),
            "selected_sources":len(header)-len(excluded),"excluded_sources":len(excluded),
            "runtime_bytes":sum(v.numel()*v.element_size() for v in state(full).values()),
            "single_block_bytes":sum(v.numel()*v.element_size() for v in full.decoder.transformer_blocks[0].parameters())}
        inventory={k:{"shape":list(v.shape),"dtype":str(v.dtype),"sources":list(bindings[k].sources),
                       "default":bindings[k].default} for k,v in state(full).items()}
        (a.output.parent/"full-meta-inventory.json").write_text(json.dumps({"inventory":inventory,**out["full_meta"]},indent=2)+"\n")
        del selected,full_native_state,native_full,full,bindings,excluded
        fixture_hashes={}
        for depth in (3,36):
            config=copy.deepcopy(cfg);config["vit_decoder_kwargs"].update(heads=2,num_layers=depth)
            reference,values=make_reference(config,outer)
            path=a.output.parent/f"native-decoder-depth{depth}.safetensors"
            save_file(values,path);fixture_hashes[path.name]=digest(path)
            ck(f"d{depth}_fixture_budget",path.stat().st_size<64*MIB)
            candidate=build_native_decoder_meta(config)
            plan=H3NativeVAEMmapPlan.build(candidate,path,max_total_file_bytes=64*MIB)
            ck(f"d{depth}_groups",len(plan.groups)==depth+1 and "resident" in plan.groups)
            ck(f"d{depth}_coverage",set(plan.bindings)==set(state(reference)))
            ck(f"d{depth}_no_encoder_or_quant",not hasattr(candidate,"encoder") and not hasattr(candidate,"quant_conv"))
            expected_state={k:v.detach().clone() for k,v in state(reference).items()}
            loader=H3NativeVAEMmapLoader(candidate,plan,device=a.device,max_group_bytes=16*MIB);cleanup.append(loader)
            first="decoder.transformer_blocks.0"
            reject(f"d{depth}_resident_required",lambda:loader.load_group(first))
            loader.load_group("resident")
            for repeat in range(2):
                for group,names in plan.groups.items():
                    if group!="resident":loader.load_group(group)
                    ck(f"d{depth}_exact_{repeat}_{group}",all(
                        torch.equal(state(candidate)[k].cpu(),expected_state[k]) for k in names))
                    ck(f"d{depth}_device_{repeat}_{group}",all(state(candidate)[k].device.type==a.device for k in names))
                    if group!="resident":
                        other=first if group!=first else "decoder.transformer_blocks.1"
                        reject(f"d{depth}_one_block_{repeat}_{group}",lambda:loader.load_group(other))
                        tensors=[weakref.ref(state(candidate)[k]) for k in names]
                        with torch.autocast("cuda",dtype=torch.float16,enabled=a.device=="cuda"):
                            loader.release_group(group);gc.collect()
                            ck(f"d{depth}_meta_and_objects_released_{repeat}_{group}",
                                all(state(candidate)[k].is_meta for k in names) and all(t() is None for t in tensors))
            ck(f"d{depth}_mmap_observed",loader.stats["mmap_observed"])
            ck(f"d{depth}_single_active_block",loader.stats["max_active_streamed_groups"]==1)
            out.setdefault("loader_stats",{})[str(depth)]=dict(loader.stats)
            loader.close()
            ck(f"d{depth}_close_all_meta",all(v.is_meta for v in state(candidate).values()))
            ck(f"d{depth}_mmap_closed",str(path) not in Path("/proc/self/maps").read_text())
            if depth==3:
                reject("file_budget",lambda:H3NativeVAEMmapPlan.build(candidate,path,max_total_file_bytes=1))
                headers={k:{"dtype":"F32","shape":tuple(v.shape),"bytes":v.numel()*4} for k,v in values.items()}
                key=next(iter(values))
                for name,transform in (
                    ("missing",lambda h:h.pop(key)),
                    ("dtype",lambda h:h[key].update(dtype="BF16")),
                    ("shape",lambda h:h[key].update(shape=(1,))),
                    ("unknown_decoder",lambda h:h.update({"decoder.unexpected.weight":{"dtype":"F32","shape":(1,),"bytes":4}})),
                    ("unknown_excluded",lambda h:h.update({"encoder.unexpected.weight":{"dtype":"F32","shape":(1,),"bytes":4}}))):
                    bad=copy.deepcopy(headers);transform(bad)
                    reject("audit_"+name,lambda bad=bad:H3NativeVAEMmapPlan.audit_headers(candidate,bad))
                changed=H3NativeVAEMmapPlan.build(candidate,path,max_total_file_bytes=64*MIB)
                size=path.stat().st_size
                with path.open("ab") as f:f.write(b"\0")
                try:reject("changed_file",changed.check_files_unchanged)
                finally:
                    with path.open("r+b") as f:f.truncate(size)
                fresh=H3NativeVAEMmapPlan.build(candidate,path,max_total_file_bytes=64*MIB)
                fault=H3NativeVAEMmapLoader(candidate,fresh,device=a.device,max_group_bytes=16*MIB);cleanup.append(fault)
                fault.load_group("resident");original=fault._load_native_weights
                omitted=next(s for n in fresh.groups[first] for s in fresh.bindings[n].sources)
                fault._load_native_weights=lambda weights:original((k,v) for k,v in weights if k!=omitted)
                reject("partial_load_failure",lambda:fault.load_group(first))
                ck("partial_load_rollback",all(state(candidate)[k].is_meta for k in fresh.groups[first]))
                fault._load_native_weights=original;fault.load_group(first)
                ck("retry_exact",all(torch.equal(state(candidate)[k].cpu(),expected_state[k]) for k in fresh.groups[first]))
                fault.close()
                budget=H3NativeVAEMmapLoader(candidate,fresh,device=a.device,max_group_bytes=1);cleanup.append(budget)
                reject("group_budget",lambda:budget.load_group("resident"));budget.close()
                del fault,budget,original,fresh,changed
            if a.device=="cuda":
                from h3_omni_adapter.native_temporal_decoder import H3NativeTemporalVideoDecoder
                from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _prepare_minimax_h3_video_output
                reference=reference.to("cuda")
                arguments=dict(device="cuda",max_latent_tokens=32,max_height=272,max_width=64,
                    max_output_frames=107,max_file_bytes=64*MIB,max_group_bytes=16*MIB)
                raw=H3NativeTemporalVideoDecoder(config,outer,path,output_mode="raw",**arguments);cleanup.append(raw)
                pixels=H3NativeTemporalVideoDecoder(config,outer,path,output_mode="uint8",**arguments);cleanup.append(pixels)
                ck(f"d{depth}_facade_fp32_resident",all(v.dtype==torch.float32 for v in state(raw.native).values()))
                facade_refs=[weakref.ref(obj) for obj in (raw,pixels,raw.native,pixels.native,reference)]
                host_fields=("use_3d_conv","use_vit_decoder","causal_encoder","causal_decoder","slidedec",
                    "vae_ratio","vae_ratio_t","clip_length","token_drop","frame_drop","frame_pre_padding",
                    "tokens_chunk_size","token_overlap","frame_overlap","isolated_first_frame","isolated_last_frame",
                    "encoder_tiling","decoder_tiling","parallel_tiling","tile_size","tile_overlap_min",
                    "decoder_tile_size","decoder_tile_overlap_min","encoder_parallel","decoder_parallel","chunk_dim")
                ck(f"d{depth}_native_host_attributes_exact",all(
                    getattr(raw.native,k)==getattr(reference,k)==getattr(pixels.native,k) for k in host_fields))
                ck(f"d{depth}_native_processor_normalization_exact",all(
                    list(getattr(raw.native.processor.transform_rev,k))==
                    list(getattr(reference.processor.transform_rev,k))==
                    list(getattr(pixels.native.processor.transform_rev,k)) for k in ("mean","std")))
                counts={"reference":0,"raw":0,"uint8":0}
                handles=[]
                def count(name):
                    def pre(module,inputs):counts[name]+=1
                    return pre
                for name,model in (("reference",reference),("raw",raw.native),("uint8",pixels.native)):
                    handles.append(model.decoder.register_forward_pre_hook(count(name)))
                del model
                transient=[]
                def capture(module,inputs):
                    transient.extend(weakref.ref(v) for v in module.parameters())
                for block in raw.native.decoder.transformer_blocks:
                    handles.append(block.register_forward_pre_hook(capture))
                del block
                scenarios=([(t,2,2,torch.float32,s) for t in (7,13,22,32) for s in (False,True)]
                    +[(7,2,2,torch.float16,True),(7,16,2,torch.float32,True),(7,17,2,torch.float32,True)]
                    if depth==3 else [(7,2,2,torch.float32,True)])
                for ti,(length,height,width,dtype,stream) in enumerate(scenarios):
                    torch.manual_seed(3100+ti)
                    latent=torch.randn(1,24,length,height,width,device="cuda",dtype=dtype)
                    fake=SimpleNamespace(config_dict=outer,model=reference,_pixel_transform=None)
                    before=dict(counts)
                    with temporal_mode(stream),torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
                        denorm=omni_vae.MiniMaxH3VideoVAE._denormalize_latent(fake,latent)
                        expected_raw=reference.decode_base(denorm)
                        reverted=omni_vae.MiniMaxH3VideoVAE._revert_decoded_inplace(fake,expected_raw.clone())
                        normalized=omni_vae.MiniMaxH3VideoVAE._normalize_decoded_frames(reverted)
                        expected_pixels=_prepare_minimax_h3_video_output(normalized).permute(0,4,1,2,3)
                        actual_raw=raw.decode_latent(latent)
                        actual_pixels=pixels.decode_latent(latent)
                        gc.collect()
                        name=f"d{depth}_case{ti}_t{length}_h{height}_s{int(stream)}_{str(dtype)}"
                        ck(name+"_raw_exact",torch.equal(actual_raw,expected_raw))
                        ck(name+"_pixels_exact_native_omni",torch.equal(actual_pixels,expected_pixels))
                        ck(name+"_dtype",actual_raw.dtype==expected_raw.dtype and actual_pixels.dtype==torch.uint8)
                        expected_frames={7:22,13:43,22:73,32:107}[length]
                        ck(name+"_geometry",tuple(actual_raw.shape)==(1,3,expected_frames,height*16,width*16))
                        ck(name+"_finite_nonuniform",bool(torch.isfinite(actual_raw).all())
                            and actual_raw.min().item()!=actual_raw.max().item()
                            and actual_pixels.min().item()!=actual_pixels.max().item())
                        ck(name+"_native_clip_tile_count",len({counts[k]-before[k] for k in counts})==1
                            and counts["raw"]-before["raw"]>0)
                        ck(name+"_blocks_meta_inside_autocast",all(
                            v.is_meta for k,v in state(raw.native).items() if ".transformer_blocks." in k))
                        ck(name+"_old_parameters_dead_inside_autocast",bool(transient) and all(ref() is None for ref in transient))
                        ck(name+"_only_resident_active",raw.loader.active=={"resident"} and pixels.loader.active=={"resident"})
                        out["cases"].append({"depth":depth,"latent_t":length,"height":height,"width":width,
                            "stream":stream,"latent_dtype":str(dtype),"raw_dtype":str(actual_raw.dtype),
                            "frames":expected_frames,"native_decode_calls":counts["reference"]-before["reference"],
                            "raw_max_abs":(actual_raw.float()-expected_raw.float()).abs().max().item(),
                            "pixel_mismatches":torch.count_nonzero(actual_pixels!=expected_pixels).item()})
                        transient.clear()
                    del latent,denorm,expected_raw,reverted,normalized,expected_pixels,actual_raw,actual_pixels,fake
                    if torch.cuda.max_memory_allocated()>4*GIB:raise RuntimeError("CUDA allocation boundary")
                for handle in handles:handle.remove()
                del handles,handle
                valid=torch.zeros(1,24,7,2,2,device="cuda")
                for name,bad in (("channels",valid[:,:23]),("batch",valid.expand(2,-1,-1,-1,-1)),
                    ("integer",valid.to(torch.int32)),("cpu",valid.cpu()),("short",valid[:,:,:2]),
                    ("meta",torch.empty(1,24,7,2,2,device="meta")),
                    ("time_budget",torch.zeros(1,24,33,2,2,device="cuda")),
                    ("height_budget",torch.zeros(1,24,7,18,2,device="cuda")),
                    ("width_budget",torch.zeros(1,24,7,2,5,device="cuda")),
                    ("nan",torch.full_like(valid,float("nan")))):
                    reject(f"d{depth}_invalid_{name}",lambda bad=bad:raw.decode_latent(bad))
                reject(f"d{depth}_block_forward_failure",lambda:raw.native.decoder.transformer_blocks[0]())
                ck(f"d{depth}_forward_failure_releases",raw.loader.active=={"resident"}
                    and all(v.is_meta for k,v in state(raw.native).items() if ".transformer_blocks." in k))
                raw.close();pixels.close();raw.close()
                reject(f"d{depth}_closed",lambda:raw.decode_latent(valid))
                cleanup.remove(raw);cleanup.remove(pixels)
                del raw,pixels,valid,bad,reference
                gc.collect();torch.cuda.synchronize()
                ck(f"d{depth}_facades_and_native_models_released",all(ref() is None for ref in facade_refs))
            else:del reference
            ck(f"d{depth}_fixture_hash_unchanged",digest(path)==fixture_hashes[path.name])
            weak=weakref.ref(candidate)
            del candidate,loader,plan,expected_state,values
            cleanup.clear();gc.collect()
            ck(f"d{depth}_candidate_object_released",weak() is None)
            ck(f"d{depth}_all_mmaps_closed",str(path) not in Path("/proc/self/maps").read_text())
        (a.output.parent/"fixture-manifest.json").write_text(json.dumps(fixture_hashes,indent=2)+"\n")
        out["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        out["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        ck("cuda_resource_cap",out["cuda_peak_allocated_bytes"]<4*GIB)
    except BaseException as exc:
        out["error"]={"type":type(exc).__name__,"traceback":traceback.format_exc()}
    finally:
        for obj in cleanup:
            try:obj.close()
            except Exception as exc:out["cleanup_error"]=type(exc).__name__
        try:
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        except Exception as exc:out["cleanup_error"]=type(exc).__name__
    out["checks_total"]=len(out["checks"]);out["checks_ok"]=sum(c["ok"] for c in out["checks"])
    out["exit_code"]=int("error" in out or "cleanup_error" in out or out["checks_ok"]!=out["checks_total"])
    text=json.dumps(out,indent=2)
    for source,label in ((str(a.output.parent),"<RUN_ROOT>"),(str(a.source_record),"<ITERATION_ROOT>"),
        (str(a.research),"<RESEARCH_ROOT>"),(str(a.native_source),"<NATIVE_SOURCE>"),
        (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"),(str(Path.home()),"<LINUX_HOME>")):
        text=text.replace(source,label)
    a.output.write_text(text+"\n")
    print(json.dumps({k:v for k,v in out.items() if k not in {"checks","cases","error"}}))
    return out["exit_code"]

if __name__=="__main__":raise SystemExit(main())
