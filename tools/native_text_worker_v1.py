"""Frozen original H3 dense text coverage, native loading and streaming protocol."""
import argparse,gc,hashlib,inspect,json,math,traceback,weakref
from pathlib import Path
import torch
from safetensors.torch import save_file
from validate_meta_coverage_v2 import setup_dist

def state(m):
    return dict(list(m.named_parameters())+list(m.named_buffers()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--research",type=Path,required=True);p.add_argument("--source-record",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    out={"campaign_verdict":"INCONCLUSIVE","device":a.device,"checks":[],"real_weights":False,
         "scope":"full layer-50 meta plus reduced dense text weights; CUDA-only forward; no tokenizer/pipeline/video"}
    loaders=[]
    def ck(name,ok,**detail):
        out["checks"].append(dict(name=name,ok=bool(ok),**detail))
        if not ok:raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError,RuntimeError):ck(name,True)
        else:ck(name,False)
    try:
        torch.cuda.set_per_process_memory_fraction(.40)
        if a.device=="cuda" and torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError("GPU free-memory guard")
        setup_dist(a.output.parent)
        from transformers import Qwen3VLConfig
        from vllm.distributed import get_tp_group
        from vllm_omni.diffusion.models.minimax_h3.encoder import (
            MiniMaxH3Qwen3VLEncoder,_get_rope_index,_fused_source_shards)
        from h3_omni_adapter.native_text_mmap import H3NativeTextOnly,H3NativeTextMmapPlan,H3NativeTextMmapLoader
        source=json.loads((a.source_record/"plan/input-provenance.json").read_text())
        ck("pinned_encoder_source",hashlib.sha256(Path(inspect.getfile(MiniMaxH3Qwen3VLEncoder)).read_bytes()).hexdigest()==source["encoder_source"]["sha256"])
        cfg=json.loads((a.research/"metadata/FL2VA__text_encoder__config.json").read_text())
        headers={}
        locations={}
        for file in source["files"]:
            if file["kind"]!="header":continue
            path=a.research/"metadata"/file["saved"]
            ck("header_hash_"+Path(file["path"]).name,hashlib.sha256(path.read_bytes()).hexdigest()==file["sha256"])
            h=json.loads(path.read_bytes());h.pop("__metadata__",None)
            ck("header_unique_"+Path(file["path"]).name,not headers.keys()&h.keys())
            for k,v in h.items():v["shape"]=tuple(v["shape"]);v["bytes"]=v["data_offsets"][1]-v["data_offsets"][0]
            headers.update(h);locations.update({k:Path(file["path"]).name for k in h})
        index=json.loads((a.research/"metadata/FL2VA__text_encoder__model.safetensors.index.json").read_text())["weight_map"]
        ck("full_index_matches_headers",locations==index)
        full=H3NativeTextOnly(Qwen3VLConfig.from_dict(cfg),get_tp_group(),max_tokens=64)
        bindings,routes,excluded=H3NativeTextMmapPlan.audit_headers(full,headers)
        ck("full_50_retained_layers",len(full.text_model.layers)==50)
        ck("full_meta_no_vision_no_final_norm",all(t.is_meta for t in state(full).values()) and not hasattr(full,"vision") and not hasattr(full.text_model,"norm"))
        ck("full_coverage_every_parameter_buffer",set(bindings)==set(state(full)))
        ck("full_sources_partition",set(routes).isdisjoint(excluded) and set(routes)|set(excluded)==set(headers))
        ck("native_excluded_last14_layers",all(k in excluded for k in headers if k.startswith("model.language_model.layers.") and int(k.split(".")[3])>=50))
        out["full_meta"]={"targets":len(bindings),"selected_sources":len(routes),"excluded_sources":len(excluded),
           "buffers":len(dict(full.named_buffers())),"layers":50,"runtime_bytes":sum(t.numel()*t.element_size() for t in state(full).values()),
           "embedding_bytes":full.text_model.embed_tokens.weight.numel()*2,
           "single_layer_bytes":sum(t.numel()*t.element_size() for t in full.text_model.layers[0].parameters())}
        inventory={"bindings":{k:{"sources":list(v.sources),"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in bindings.items()},
                   "routes":routes,"excluded":excluded,**out["full_meta"]}
        (a.output.parent/"full-meta-inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
        del full,bindings,routes,excluded
        cfg["text_config"].update(hidden_size=256,intermediate_size=512,num_hidden_layers=3,
          num_attention_heads=4,num_key_value_heads=2,head_dim=128,vocab_size=257)
        cfg["vision_config"].update(depth=1,hidden_size=128,intermediate_size=128,num_heads=1,
          out_hidden_size=256,deepstack_visual_indexes=[0],num_position_embeddings=16)
        cfg.update(image_token_id=255,video_token_id=256)
        root=a.output.parent/"fixture";root.mkdir()
        (root/"config.json").write_text(json.dumps(cfg))
        ordinary=MiniMaxH3Qwen3VLEncoder(str(root),device=torch.device(a.device),load_model=True,encoder_group=get_tp_group(),quant_config=None)
        values={}
        def value(key,shape,dtype):
            seed=int(hashlib.sha256(key.encode()).hexdigest()[:6],16)%251
            data=((torch.arange(math.prod(shape),dtype=torch.float32)+seed).remainder(251)-125)/2048
            if "norm" in key:data+=1
            return data.reshape(shape).to(dtype)
        for name,t in ordinary.named_parameters():
            parent,_,field=name.rpartition(".");owner=ordinary.get_submodule(parent)
            fused=_fused_source_shards(owner) if field=="weight" else None
            if fused:
                for label,shard in fused:
                    rows=(owner.local_num_heads if label=="q" else owner.local_num_kv_heads)*owner.head_dim if isinstance(shard,str) else owner.intermediate_size
                    source_name=name.replace("text_model.","model.language_model.",1)
                    source_name=source_name.replace("qkv_proj",label+"_proj").replace("gate_up_proj",label+"_proj")
                    values[source_name]=value(source_name,(rows,t.shape[1]),t.dtype)
            else:
                source_name=name.replace("vision.","model.visual.",1) if name.startswith("vision.") else name.replace("text_model.","model.language_model.",1)
                values[source_name]=value(source_name,tuple(t.shape),t.dtype)
        values["model.language_model.norm.weight"]=torch.ones(256,dtype=torch.bfloat16)*2
        values["lm_head.weight"]=torch.zeros(257,256,dtype=torch.bfloat16)
        keys=sorted(values);files=("a.safetensors","b.safetensors","c.safetensors")
        for i,name in enumerate(files):save_file({k:values[k] for j,k in enumerate(keys) if j%3==i},root/name)
        index={"weight_map":{k:files[i%3] for i,k in enumerate(keys)}}
        ix=root/"model.safetensors.index.json";ix.write_text(json.dumps(index))
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir()}
        out["fixture_bytes"]=sum(f.stat().st_size for f in root.iterdir())
        ck("fixture_budget",out["fixture_bytes"]<64*1024**2)
        ck("ordinary_strict_loader_complete",ordinary.load_weights(iter(values.items()))==set(dict(ordinary.named_parameters())))
        ref={k:t.detach().clone() for k,t in state(ordinary).items() if k.startswith("text_model.")}
        stem="model.language_model.layers.0."
        qkv=torch.cat([values[stem+"self_attn."+k+"_proj.weight"] for k in ("q","k","v")])
        gateup=torch.cat([values[stem+"mlp."+k+"_proj.weight"] for k in ("gate","up")])
        ck("independent_gqa_qkv_oracle",torch.equal(ref["text_model.layers.0.self_attn.qkv_proj.weight"],qkv))
        ck("independent_gate_up_oracle",torch.equal(ref["text_model.layers.0.mlp.gate_up_proj.weight"],gateup))
        reject("ordinary_strict_missing_k",lambda:ordinary.load_weights((k,v) for k,v in values.items() if k!=stem+"self_attn.k_proj.weight"))
        ordinary.load_weights(iter(values.items()))
        candidate=H3NativeTextOnly(Qwen3VLConfig.from_dict(cfg),get_tp_group(),max_tokens=64)
        plan=H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2)
        ck("tiny_full_text_coverage",set(plan.bindings)==set(ref))
        ck("tiny_no_vision",not hasattr(candidate,"vision"))
        ck("explicit_vision_norm_head_exclusions",set(plan.excluded)=={k for k in values if k.startswith("model.visual.") or k in {"lm_head.weight","model.language_model.norm.weight"}})
        loader=H3NativeTextMmapLoader(candidate,plan,device=a.device,max_group_bytes=16*1024**2);loaders.append(loader)
        reject("resident_required",lambda:loader.load_group("text_model.layers.0"))
        loader.load_group("resident")
        for repeat in range(2):
            for group,names in plan.groups.items():
                if group!="resident":loader.load_group(group)
                ck(f"exact_{repeat}_{group}",all(torch.equal(state(candidate)[k].cpu(),ref[k]) for k in names))
                ck(f"device_{repeat}_{group}",all(state(candidate)[k].device.type==a.device for k in names))
                if group!="resident":
                    other="text_model.layers.0" if group!="text_model.layers.0" else "text_model.layers.1"
                    reject(f"one_layer_{repeat}_{group}",lambda:loader.load_group(other))
                    loader.release_group(group)
                    ck(f"release_{repeat}_{group}",all(state(candidate)[k].is_meta for k in names))
        ids=torch.tensor([1,8,3,9,8,4,6,2],dtype=torch.long)
        if a.device=="cuda":
            ordinary.text_model.to("cuda");loader.install_block_hooks()
            for length in (1,8,17,64,8):
                tok=ids.repeat(8)[:length];batch=tok.cuda()[None]
                pos,_=_get_rope_index(batch,mm_token_type_ids=torch.zeros_like(batch,dtype=torch.int32),image_grid_thw=None,video_grid_thw=None,attention_mask=torch.ones_like(batch),spatial_merge_size=2)
                previous=torch.backends.cuda.cudnn_sdp_enabled()
                torch.backends.cuda.enable_cudnn_sdp(True)
                try:
                    with torch.inference_mode():expected=ordinary.text_model(ordinary.text_model.embed_tokens(batch),pos)[0].cpu()
                finally:torch.backends.cuda.enable_cudnn_sdp(previous)
                actual=candidate.encode_ids(tok)
                ck("hidden_exact_"+str(length),torch.equal(expected,actual) and bool(torch.isfinite(actual).all()),shape=list(actual.shape))
                ck("sdpa_flag_restored_"+str(length),torch.backends.cuda.cudnn_sdp_enabled()==previous)
                ck("layers_released_after_forward_"+str(length),loader.active=={"resident"} and all(t.is_meta for k,t in state(candidate).items() if ".layers." in k))
            reject("forward_exception",lambda:candidate.text_model.layers[0]())
            ck("forward_exception_releases",all(state(candidate)[k].is_meta for k in plan.groups["text_model.layers.0"]))
        invalid=[("image",torch.tensor([255])),("video",torch.tensor([256])),("negative",torch.tensor([-1])),
          ("outside",torch.tensor([257])),("empty",ids[:0]),("batched",ids[None]),("float",ids.float()),
          ("long",torch.zeros(65,dtype=torch.long)),("meta",torch.empty(1,dtype=torch.long,device="meta"))]
        for name,bad in invalid:reject("invalid_input_"+name,lambda bad=bad:candidate.encode_ids(bad))
        reject("pixel_input",lambda:candidate.encode_ids(ids,pixel_values=torch.zeros(1)))
        ck("mmap_observed",loader.stats["mmap_observed"])
        ck("single_active_layer",loader.stats["max_active_streamed_groups"]==1)
        out["loader_stats"]=dict(loader.stats);loader.close()
        ck("close_all_meta",all(t.is_meta for t in state(candidate).values()))
        reject("closed_encode",lambda:candidate.encode_ids(ids))
        ck("mmap_closed",str(root) not in Path("/proc/self/maps").read_text())
        reject("tp2",lambda:H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2,tensor_parallel_size=2))
        reject("file_budget",lambda:H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=1))
        tinyheaders={k:{"dtype":"BF16","shape":tuple(v.shape),"bytes":v.numel()*v.element_size()} for k,v in values.items()}
        for name,transform in (
          ("missing_k",lambda h:h.pop(stem+"self_attn.k_proj.weight")),
          ("missing_up",lambda h:h.pop(stem+"mlp.up_proj.weight")),
          ("unknown",lambda h:h.update({"model.language_model.unexpected.weight":{"dtype":"BF16","shape":(1,),"bytes":2}})),
          ("wrong_q_shape",lambda h:h[stem+"self_attn.q_proj.weight"].update(shape=(256,256))),
          ("wrong_dtype",lambda h:h[stem+"mlp.down_proj.weight"].update(dtype="F32")),
          ("unknown_layer",lambda h:h.update({"model.language_model.layers.64.mlp.down_proj.weight":{"dtype":"BF16","shape":(256,512),"bytes":262144}}))):
            h={k:dict(v) for k,v in tinyheaders.items()};transform(h)
            reject("audit_"+name,lambda h=h:H3NativeTextMmapPlan.audit_headers(candidate,h))
        saved=ix.read_bytes()
        try:
            bad=json.loads(saved);bad["weight_map"].pop(keys[0]);ix.write_text(json.dumps(bad))
            reject("index_incomplete",lambda:H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2))
            bad=json.loads(saved);bad["weight_map"][keys[0]]="../escape";ix.write_text(json.dumps(bad))
            reject("index_path_escape",lambda:H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2))
        finally:ix.write_bytes(saved)
        fresh=H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2)
        file=root/"a.safetensors";size=file.stat().st_size
        with file.open("ab") as f:f.write(b"\0")
        try:reject("file_changed",fresh.check_files_unchanged)
        finally:
            with file.open("r+b") as f:f.truncate(size)
        fresh=H3NativeTextMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2)
        fault=H3NativeTextMmapLoader(candidate,fresh,device=a.device,max_group_bytes=16*1024**2);loaders.append(fault)
        fault.load_group("resident");original=fault._load_native_weights
        fault._load_native_weights=lambda weights:original((k,v) for k,v in weights if k!=stem+"self_attn.k_proj.weight")
        reject("runtime_missing_fused_shard",lambda:fault.load_group("text_model.layers.0"))
        ck("partial_fused_load_rollback",all(state(candidate)[k].is_meta for k in fresh.groups["text_model.layers.0"]))
        fault._load_native_weights=original;fault.load_group("text_model.layers.0")
        ck("retry_after_failure_exact",all(torch.equal(state(candidate)[k].cpu(),ref[k]) for k in fresh.groups["text_model.layers.0"]))
        fault.close()
        budget=H3NativeTextMmapLoader(candidate,fresh,device=a.device,max_group_bytes=1);loaders.append(budget)
        reject("group_budget",lambda:budget.load_group("resident"));budget.close()
        ck("all_closed_meta",all(t.is_meta for t in state(candidate).values()))
        ck("all_mmaps_closed",str(root) not in Path("/proc/self/maps").read_text())
        ck("fixture_hashes_unchanged",hashes=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir()})
        (a.output.parent/"fixture-manifest.json").write_text(json.dumps(hashes,indent=2)+"\n")
        refweak=weakref.ref(candidate)
        del candidate,loader,fault,budget,original,plan,fresh
        loaders.clear();gc.collect()
        ck("model_object_released",refweak() is None)
        out["cuda_context_initialized"]=torch.cuda.is_initialized()
        out["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated();out["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        ck("cuda_resource_cap",out["cuda_peak_allocated_bytes"]<4*1024**3)
    except BaseException as exc:
        trace=traceback.format_exc()
        for source,label in ((str(a.source_record),"<ITERATION_ROOT>"),(str(a.research),"<RESEARCH_ROOT>"),(str(a.output.parent),"<RUN_ROOT>"),(str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"),(str(Path.home()),"<LINUX_HOME>")):trace=trace.replace(source,label)
        out["error"]={"type":type(exc).__name__,"traceback":trace}
    finally:
        for loader in loaders:
            try:loader.close()
            except Exception as exc:out["cleanup_error"]=type(exc).__name__
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        except Exception as exc:out["cleanup_error"]=type(exc).__name__
    out["checks_total"]=len(out["checks"]);out["checks_ok"]=sum(c["ok"] for c in out["checks"])
    out["exit_code"]=int("error" in out or "cleanup_error" in out or out["checks_ok"]!=out["checks_total"])
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k not in {"checks","error"}}))
    return out["exit_code"]
if __name__=="__main__":raise SystemExit(main())
