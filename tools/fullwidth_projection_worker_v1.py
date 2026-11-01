"""Frozen full-width single-projection ConvRot capacity protocol (draft until manifest).
Native model remains meta except one target and touched synthetic resident tensors.
No forward; numerical inverse-rotation oracle is sampled, transfer compare is full.
"""
import argparse, gc, hashlib, inspect, json, math, os, shutil, struct, time, traceback, types, weakref
from pathlib import Path
import torch
from safetensors import safe_open
from native_dense_worker_v5 import make, state, setup_dist
from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields
MiB=1024**2
SCENARIOS={
 "adaln":{"key":"blocks.0.adaln_proj.linear.weight","shape":(96768,2688),"group":64},
 "qkv":{"key":"blocks.0.attn.qkv_proj.weight","shape":(21504,5376),"group":256}}
WORKSPACE=512*1024
WRITE_CHUNK=4*MiB

def sha(path):
    d=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(WRITE_CHUNK): d.update(chunk)
    return d.hexdigest()

def write_fixture(path,case):
    rows,cols=case["shape"];key=case["key"];prefix=key[:-7]
    raw=json.dumps({"format":"int8_tensorwise","convrot":True,"convrot_groupsize":case["group"]},separators=(",",":")).encode()
    wbytes=rows*cols;sbytes=rows*4
    head={key:{"dtype":"I8","shape":[rows,cols],"data_offsets":[0,wbytes]},
      prefix+".weight_scale":{"dtype":"F32","shape":[rows,1],"data_offsets":[wbytes,wbytes+sbytes]},
      prefix+".comfy_quant":{"dtype":"U8","shape":[len(raw)],"data_offsets":[wbytes+sbytes,wbytes+sbytes+len(raw)]}}
    encoded=json.dumps(head,separators=(",",":")).encode();encoded+=b" "*((-len(encoded))%8)
    expected=8+len(encoded)+wbytes+sbytes+len(raw)
    if expected>300*MiB or shutil.disk_usage(path.parent).free<8*1024**3+expected:
        raise RuntimeError("fixture disk/file bound")
    started=time.monotonic()
    with path.open("xb") as f:
        f.write(struct.pack("<Q",len(encoded)));f.write(encoded)
        c=torch.arange(cols,dtype=torch.int32).view(1,-1).mul_(7)
        rowchunk=max(1,WRITE_CHUNK//cols)
        for start in range(0,rows,rowchunk):
            rr=torch.arange(start,min(rows,start+rowchunk),dtype=torch.int32).view(-1,1).mul_(13)
            q=(rr+c).remainder_(255).sub_(127).to(torch.int8)
            f.write(q.numpy().tobytes())
            del q,rr
        del c
        # Exactly dyadic, nonzero and row varying; no large F32 allocation.
        for start in range(0,rows,4096):
            s=torch.arange(start,min(rows,start+4096),dtype=torch.float32).remainder_(7).add_(1).div_(512)
            f.write(s.numpy().tobytes());del s
        f.write(raw);f.flush();os.fsync(f.fileno())
    if path.stat().st_size!=expected:raise RuntimeError("fixture short write")
    return {"bytes":expected,"sha256":sha(path),"write_seconds":time.monotonic()-started,
      "max_write_payload_bytes":WRITE_CHUNK,"pattern":"q=(13*row+7*column)%255-127; scale=(row%7+1)/512",
      "scope":"INT8+scale+format only, no BF16 cache"}

def hmatrix(group):
    base=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float64)
    m=torch.ones((1,1),dtype=torch.float64)
    while m.shape[0]<group:m=torch.kron(m,base)
    return m/math.sqrt(group)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--research",type=Path,required=True)
    p.add_argument("--static-budget",type=Path,required=True)
    p.add_argument("--scenario",choices=tuple(SCENARIOS),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--fixture-sha256")
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    case=SCENARIOS[a.scenario];rows,cols=case["shape"];key=case["key"];g=case["group"]
    result={"campaign_verdict":"INCONCLUSIVE","scenario":a.scenario,"device":a.device,"checks":[],
      "scope":"synthetic full-width single native projection plus zero-touched resident; no forward, no full block",
      "oracle_scope":"independent inverse rotation at selected rows and column groups; full transfer equality",
      "not_proven":["full block","full model","quality","speedup","real values","entire real model file integrity"],
      "phases":[]}
    model=templates=target=source=scales=output=reader=None
    shim=actual_loader=observed_loader=target_rows=hm=expect=None
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
          "BF16_output":rows*cols*2,"native_QKV_copy":rows*cols*2 if a.scenario=="qkv" else 0,
          "single_target":rows*cols*2,"resident":budget["resident_runtime_bytes"],
          "explicit_workspace":WORKSPACE,
          "exclusions":["other block targets","allocator/BLAS","library/context","OS page cache"]}
        result["static_budget_sha256"]=sha(a.static_budget)
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
        fixture=a.fixture or (a.output.parent/("synthetic-"+a.scenario+".safetensors"))
        if a.fixture:
            if not a.fixture_sha256:raise ValueError("explicit fixture needs hash")
            ck("reused_fixture_size",fixture.stat().st_size<=300*MiB)
            ck("reused_fixture_hash",sha(fixture)==a.fixture_sha256)
            result["fixture"]={"bytes":fixture.stat().st_size,"sha256":a.fixture_sha256,"reused":True}
        else:
            result["fixture"]=write_fixture(fixture,case)
        phase("fixture_ready")
        with safe_open(fixture,framework="pt",device="cpu") as reader:
            prefix=key[:-7]
            ck("fixture_exact_keys",set(reader.keys())=={key,prefix+".weight_scale",prefix+".comfy_quant"})
            source=reader.get_tensor(key);scales=reader.get_tensor(prefix+".weight_scale")
            source_refs.extend((weakref.ref(source),weakref.ref(scales)))
            meta=reader.get_tensor(prefix+".comfy_quant")
            spec=parse_comfy_quant(bytes(meta.tolist()),max_metadata_bytes=4096);meta=None
            ck("fixture_true_source_and_scale_shape_dtype",source.dtype==torch.int8 and tuple(source.shape)==case["shape"] and scales.dtype==torch.float32 and tuple(scales.shape)==(rows,1) and spec.convrot_groupsize==g)
            ck("fixture_mapping_observed",str(fixture) in Path("/proc/self/maps").read_text())
            t0=time.monotonic()
            output,stats=recover_bf16_projection(source,scales,spec,max_output_bytes=rows*cols*2,
              max_workspace_bytes=WORKSPACE,row_layout="grouped_qkv" if a.scenario=="qkv" else "plain",
              qkv_heads=56 if a.scenario=="qkv" else None,qkv_head_dim=128 if a.scenario=="qkv" else None)
            result["recovery_seconds"]=time.monotonic()-t0;result["recovery_stats"]=stats
            output_refs.append(weakref.ref(output))
            ck("bounded_output_and_workspace",stats["output_bytes"]==rows*cols*2 and stats["workspace_peak_bytes"]<=WORKSPACE)
            phase("source_output_target_resident_coexist",hold=True)
            # Small independent FP64 Kronecker oracle, fixed boundary/interior rows and all
            # first/middle/last column groups. It never constructs a full FP64 projection.
            hm=hmatrix(g)
            sampled_rows=sorted({0,1,31,32,127,128,rows//3-1,rows//3,rows//2,rows-2,rows-1})
            sampled_columns=sorted({0,(cols//g//2)*g,cols-g})
            exact=True;sample_values=0
            for sr in sampled_rows:
                dr=sr
                if a.scenario=="qkv":
                    projection,rem=divmod(sr,56*128);head,dim=divmod(rem,128);dr=(3*head+projection)*128+dim
                for col in sampled_columns:
                    expect=(source[sr:sr+1,col:col+g].to(torch.float64)*scales[sr,0].item())@hm.t()
                    exact &= torch.equal(output[dr,col:col+g],expect[0].to(torch.bfloat16));sample_values+=g
            del hm,expect
            ck("independent_sampled_inverse_rotation_exact",exact,sampled_rows=sampled_rows,column_group_starts=sampled_columns,values=sample_values)
            from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import default_weight_loader
            actual_loader=getattr(target,"weight_loader",default_weight_loader)
            def observed_loader(parameter,loaded,*args,**kwargs):
                ck("native_parameter_subclass_and_identity",parameter is target)
                ck("target_has_distinct_storage",loaded.data_ptr()!=parameter.data_ptr())
                if a.scenario=="qkv":
                    ck("native_qkv_copy_distinct",loaded.data_ptr()!=output.data_ptr())
                    copy_refs.append(weakref.ref(loaded))
                else:ck("native_plain_source_is_recovery",loaded.data_ptr()==output.data_ptr())
                phase("native_weight_loader_entry_joint_peak",hold=True,
                  input_bytes=loaded.numel()*loaded.element_size(),input_device=str(loaded.device))
                return actual_loader(parameter,loaded,*args,**kwargs)
            target.weight_loader=observed_loader
            t0=time.monotonic()
            ck("native_exact_single_source_consumed",model.load_weights(iter(((key,output),)))=={key})
            result["native_load_seconds_including_0p3s_probe_hold"]=time.monotonic()-t0
            target.weight_loader=actual_loader;observed_loader=None;actual_loader=None
            ck("native_qkv_temporary_released",all(ref() is None for ref in copy_refs))
            transfer_exact=True;finite=True
            # Native QKV target restores [Q/K/V,head,dim], output is [head,Q/K/V,dim].
            for row in range(0,rows,128):
                target_rows=target[row:row+128].detach().cpu()
                if a.scenario=="qkv":
                    projection,rem=divmod(row,56*128);head,dim=divmod(rem,128)
                    outrow=(3*head+projection)*128+dim
                else:outrow=row
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
        shim=actual_loader=observed_loader=target_rows=hm=expect=None
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
