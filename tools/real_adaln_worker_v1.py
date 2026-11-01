"""Real ADALN sample native-loading protocol draft; never execute before separate freeze.
Uses independent predeclared real-data oracle and exact fixed downloader identity.
Resident tensors are zero-touched placeholders; no forward or complete block loading.
"""
import argparse, gc, hashlib, inspect, json, math, os, shutil, struct, time, traceback, types, weakref
from pathlib import Path
import torch
from safetensors import safe_open
from native_dense_worker_v5 import make, state, setup_dist
from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields
MiB=1024**2
SCENARIOS={"adaln":{"key":"blocks.0.adaln_proj.linear.weight","shape":(96768,2688),"group":64}}
WORKSPACE=512*1024
WRITE_CHUNK=4*MiB

def sha(path):
    d=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(WRITE_CHUNK): d.update(chunk)
    return d.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--research",type=Path,required=True)
    p.add_argument("--static-budget",type=Path,required=True)
    
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sample",type=Path,required=True)
    p.add_argument("--download-receipt",type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.scenario="adaln"
    case=SCENARIOS[a.scenario];rows,cols=case["shape"];key=case["key"];g=case["group"]
    result={"campaign_verdict":"INCONCLUSIVE","scenario":a.scenario,"device":a.device,"checks":[],
      "scope":"real three-key ADALN sample plus zero-touched synthetic resident; no forward, no full block",
      "oracle_scope":"predeclared F32-product/FP64-fsum bound at 2112 values; full target transfer equality",
      "not_proven":["full block","full model","quality","speedup","original BF16 parity","entire real model file integrity"],
      "phases":[]}
    model=templates=target=source=scales=output=reader=None
    shim=actual_loader=observed_loader=target_rows=hm=expect=scale_bits=None
    resident_names=[];source_refs=[];output_refs=[];target_refs=[];copy_refs=[]
    original_qkv=None;native_module=None
    def ck(name,ok,**more):
        result["checks"].append({"name":name,"ok":bool(ok),**more})
        a.output.with_name("live-checks.json").write_text(json.dumps(result,indent=2)+"\n")
        if not ok:raise AssertionError(name)
    def phase(name,hold=False,**more):
        item={"phase":name,"elapsed":time.monotonic()-started,
          "self_memory":fields(Path("/proc/self/status"),{"VmRSS","RssAnon","RssFile","RssShmem"}),
          "cuda_allocated":torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,**more}
        result["phases"].append(item)
        a.output.with_name("live-phases.json").write_text(json.dumps(result["phases"],indent=2)+"\n")
        if hold:time.sleep(.3)
    started=time.monotonic()
    try:
      with _termination_guard():
        torch.set_num_threads(2)
        ck("initial_system_available_5GiB",fields(Path("/proc/meminfo"),{"MemAvailable"})["MemAvailable"]>=5*1024**3)
        ck("disk_available_8GiB",shutil.disk_usage(a.output.parent).free>=8*1024**3)
        torch.cuda.set_per_process_memory_fraction(.40)
        ck("initial_cuda_free_2GiB",torch.cuda.mem_get_info()[0]>=2*1024**3)
        setup_dist(a.output.parent)
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        from h3_omni_adapter.mmap_loader import H3MmapLoader
        from h3_omni_adapter.convrot import parse_comfy_quant,recover_bf16_projection
        source_record=json.loads((a.research/"source-record.json").read_text())
        ck("pinned_native_source",sha(Path(inspect.getfile(MiniMaxH3DiTModel)))==source_record["files"]["vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"]["sha256"])
        budget=json.loads(a.static_budget.read_text())
        b=next(x for x in budget["projections"] if x["name"]==key)
        ck("audited_real_shape",b["shape"]==list(case["shape"]) and b["rotation_group"]==g)
        ck("audited_real_bytes",b["source_INT8_bytes"]==rows*cols and b["recovery_BF16_output_bytes"]==rows*cols*2)
        result["logical_tensor_bytes"]={"INT8_source":rows*cols,"F32_scale":rows*4,
          "BF16_output":rows*cols*2,"native_QKV_copy":0,
          "single_target":rows*cols*2,"resident":budget["resident_runtime_bytes"],
          "explicit_workspace":WORKSPACE,
          "exclusions":["other block targets","allocator/BLAS","library/context","OS page cache"]}
        result["static_budget_sha256"]=sha(a.static_budget)
        from real_adaln_oracle_v3 import verify_sample_identity,validate_sampled_real_projection,_decode_binary_bits
        result["sample_identity"]=verify_sample_identity(a.sample,a.download_receipt)
        ck("sample_identity_from_frozen_receipt",True)
        config=json.loads((a.research/"metadata/FL2VA__transformer__config.json").read_text())
        model=make(config);templates=state(model)
        ck("original_full_model_535_meta",len(templates)==535 and all(t.is_meta for t in templates.values()))
        ck("single_target_true_shape",tuple(templates[key].shape)==case["shape"] and templates[key].dtype==torch.bfloat16)
        resident_names=[n for n in templates if not n.startswith(("blocks.","token_refiner.blocks."))]
        resident_bytes=sum(templates[n].numel()*templates[n].element_size() for n in resident_names)
        ck("resident_exact_static_bytes",resident_bytes==budget["resident_runtime_bytes"])
        shim=types.SimpleNamespace(model=model,templates=templates,parameter_names=set(dict(model.named_parameters())))
        # Reuse the unchanged adapter replacement primitive to preserve native parameter subclass/loaders.
        for name in resident_names+[key]:
            template=templates[name]
            H3MmapLoader._replace(shim,name,torch.zeros(tuple(template.shape),dtype=template.dtype,device=a.device))
        ck("only_selected_resident_and_target_materialized",
          {n for n,t in state(model).items() if not t.is_meta}==set(resident_names+[key]))
        phase("resident_and_target_touched",hold=True)
        target=state(model)[key];target_refs.append(weakref.ref(target))
        fixture=a.sample
        result["fixture"]={"bytes":fixture.stat().st_size,"sha256":result["sample_identity"]["sample_sha256"],"source_kind":"real_three_key_repacked_sample"}
        phase("real_sample_identity_ready")
        with safe_open(fixture,framework="pt",device="cpu") as reader:
            prefix=key[:-7]
            ck("fixture_exact_keys",set(reader.keys())=={key,prefix+".weight_scale",prefix+".comfy_quant"})
            source=reader.get_tensor(key);scales=reader.get_tensor(prefix+".weight_scale")
            source_refs.extend((weakref.ref(source),weakref.ref(scales)))
            meta=reader.get_tensor(prefix+".comfy_quant")
            spec=parse_comfy_quant(bytes(meta.tolist()),max_metadata_bytes=4096);meta=None
            ck("fixture_true_source_and_scale_shape_dtype",source.dtype==torch.int8 and tuple(source.shape)==case["shape"] and scales.dtype==torch.float32 and tuple(scales.shape)==(rows,1) and spec.convrot_groupsize==g)
            ck("fixture_mapping_observed",str(fixture) in Path("/proc/self/maps").read_text())
            # Entry-only defense: decode every scale from integer bits so DAZ cannot
            # turn a negative subnormal into zero before the nonnegative check.
            # The frozen runtime itself is unchanged; -0 remains accepted.
            scale_cap=float(torch.finfo(torch.bfloat16).max)/(2.0*128.0*math.sqrt(g))
            scale_checked=scale_subnormal=scale_negative_zero=0
            for start in range(0,rows,4096):
                scale_bits=scales.view(torch.int32)[start:min(rows,start+4096),0].tolist()
                for signed_bits in scale_bits:
                    bits=signed_bits&0xffffffff
                    scale_value=_decode_binary_bits(bits,23)
                    if not math.isfinite(scale_value) or not 0<=scale_value<=scale_cap:
                        raise ValueError("raw F32 scale outside unchanged finite/nonnegative/cap domain")
                    scale_checked+=1
                    scale_subnormal+=int((bits&0x7f800000)==0 and (bits&0x007fffff)!=0)
                    scale_negative_zero+=int(bits==0x80000000)
                scale_bits=None
            result["scale_bit_domain"]={"checked":scale_checked,"subnormal_count":scale_subnormal,
              "negative_zero_count":scale_negative_zero,"max_safe_scale":scale_cap,
              "scope":"real experiment entry defense; frozen runtime unchanged"}
            ck("full_scale_bit_domain_validated",scale_checked==rows)
            t0=time.monotonic()
            output,stats=recover_bf16_projection(source,scales,spec,max_output_bytes=rows*cols*2,
              max_workspace_bytes=WORKSPACE,row_layout="plain")
            result["recovery_seconds"]=time.monotonic()-t0;result["recovery_stats"]=stats
            output_refs.append(weakref.ref(output))
            ck("bounded_output_and_workspace",stats["output_bytes"]==rows*cols*2 and stats["workspace_peak_bytes"]<=WORKSPACE)
            phase("source_output_target_resident_coexist",hold=True)
            result["numerical_oracle"]=validate_sampled_real_projection(source,scales,output)
            ck("independent_sampled_inverse_rotation_within_frozen_bound",
              result["numerical_oracle"]["values_checked"]==2112 and result["numerical_oracle"]["frozen_underflow_extended_bound_violations"]==0)
            from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import default_weight_loader
            actual_loader=getattr(target,"weight_loader",default_weight_loader)
            def observed_loader(parameter,loaded,*args,**kwargs):
                ck("native_parameter_subclass_and_identity",parameter is target)
                ck("target_has_distinct_storage",loaded.data_ptr()!=parameter.data_ptr())
                ck("native_plain_source_is_recovery",loaded.data_ptr()==output.data_ptr())
                phase("native_weight_loader_entry_joint_peak",hold=True,
                  input_bytes=loaded.numel()*loaded.element_size(),input_device=str(loaded.device))
                return actual_loader(parameter,loaded,*args,**kwargs)
            target.weight_loader=observed_loader
            t0=time.monotonic()
            ck("native_exact_single_source_consumed",model.load_weights(iter(((key,output),)))=={key})
            result["native_load_seconds_including_0p3s_probe_hold"]=time.monotonic()-t0
            target.weight_loader=actual_loader;observed_loader=None;actual_loader=None
            transfer_exact=True;finite=True
            # ADALN has plain row order; compare every target value in bounded stripes.
            for row in range(0,rows,128):
                target_rows=target[row:row+128].detach().cpu()
                outrow=row
                transfer_exact &= torch.equal(target_rows,output[outrow:outrow+target_rows.shape[0]])
                finite &= bool(torch.isfinite(target_rows).all())
            target_rows=None
            ck("complete_native_target_transfer_exact",transfer_exact,compared_values=rows*cols)
            ck("complete_target_finite",finite)
            output=source=scales=None
        reader=None;gc.collect()
        ck("source_and_output_weakrefs_released",all(ref() is None for ref in source_refs+output_refs))
        ck("source_mapping_released",str(fixture) not in Path("/proc/self/maps").read_text())
        phase("only_target_and_resident_remain",hold=True)
        if a.device=="cuda":torch.cuda.synchronize()
        for name in resident_names+[key]:
            H3MmapLoader._replace(shim,name,torch.empty(tuple(templates[name].shape),dtype=templates[name].dtype,device="meta"))
        target=None;gc.collect()
        ck("target_released_and_model_all_meta",all(ref() is None for ref in target_refs) and all(t.is_meta for t in state(model).values()))
        ck("fixture_unchanged",sha(fixture)==result["fixture"]["sha256"])
        phase("all_model_tensors_meta")
        result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        ck("cuda_peak_below_4GiB",result["cuda_peak_allocated_bytes"]<4*1024**3)
        result["cuda_context_initialized"]=torch.cuda.is_initialized()
        result["fixture_name"]=fixture.name
    except BaseException as e:
        result["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
    finally:
        try:
            if target is not None and actual_loader is not None:target.weight_loader=actual_loader
        except BaseException as cleanup_error:
            result["instrumentation_cleanup_error"]=type(cleanup_error).__name__
        shim=actual_loader=observed_loader=target_rows=hm=expect=scale_bits=None
        source=scales=output=reader=target=model=templates=None
        if torch.distributed.is_initialized():
            try:torch.distributed.destroy_process_group()
            except BaseException as e:result["distributed_cleanup_error"]=type(e).__name__
        gc.collect()
    result["checks_total"]=len(result["checks"]);result["checks_ok"]=sum(x["ok"] for x in result["checks"])
    result["elapsed_seconds"]=time.monotonic()-started
    result["exit_code"]=int("error" in result or "distributed_cleanup_error" in result or "instrumentation_cleanup_error" in result or result["checks_ok"]!=result["checks_total"])
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("checks","error","phases")}))
    return result["exit_code"]
if __name__=="__main__":raise SystemExit(main())
