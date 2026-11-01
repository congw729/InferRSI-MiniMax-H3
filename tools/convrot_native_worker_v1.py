"""Frozen synthetic ConvRot projection -> actual original H3 loader and tiny forward."""
import argparse,gc,hashlib,json,math,weakref,traceback
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from native_dense_worker_v5 import make,state,setup_dist

def hmatrix(g):
    # Independent dense Kronecker oracle; never call the recovery implementation.
    b=torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]],dtype=torch.float64)
    h=b
    while h.shape[0]<g:h=torch.kron(h,b)
    return h/math.sqrt(g)

def native_order(t):
    return t.reshape(3,2,128,*t.shape[1:]).permute(1,0,2,*range(3,t.ndim+2)).reshape(t.shape).contiguous()

def source_order(t):
    return t.reshape(2,3,128,*t.shape[1:]).permute(1,0,2,*range(3,t.ndim+2)).reshape(t.shape).contiguous()

def oracle(q,s,g):
    rows,cols=q.shape
    # scales are dyadic, so this oracle is unambiguous before final BF16 rounding.
    return ((q.double()*s.double()).reshape(rows,cols//g,g)@hmatrix(g).T).reshape(rows,cols).float().bfloat16()

def forward_compare(reference,candidate,config,check):
    from vllm_omni.diffusion.data import OmniDiffusionConfig,TransformerConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    od=OmniDiffusionConfig(model=None,model_class_name="MiniMaxH3Pipeline",tf_model_config=TransformerConfig.from_dict(config),
        cache_config={"minimax_h3_adaln_cache":False,"minimax_h3_adaln_offload":False})
    torch.manual_seed(19)
    def ids(v):return {"position_ids":torch.tensor(v,device="cuda")}
    kwargs={"x":torch.randn(1,8,reference.video_patch_proj.input_size,device="cuda"),
      "audio_x":torch.randn(1,8,reference.audio_patch_proj.input_size,device="cuda"),
      "img_position_ids":torch.zeros(1,8,3,device="cuda"),"unique_timesteps":torch.tensor([0.5],device="cuda"),
      "inverse_indices":torch.zeros(8,dtype=torch.long,device="cuda"),"update_mask":torch.tensor([1,1,1,0],device="cuda"),
      "token_tags":torch.tensor([0,0,1,1,1,1,2,2],device="cuda"),
      "prompt_embeds":torch.randn(2,128,dtype=torch.bfloat16,device="cuda"),
      "img_pos_info":ids([2,3,4,5]),"audio_pos_info":ids([6,7]),"text_pos_info":ids([0,1]),
      "img_pos_for_infer_output_info":ids([2,3,4,5]),
      "packed_seq_params":{"cu_seqlens_q":torch.tensor([0,8],dtype=torch.int32,device="cuda"),"max_seqlen_q":8,"num_requests":1},
      "refiner_packed_seq_params":{"cu_seqlens_q":torch.tensor([0,2],dtype=torch.int32,device="cuda"),"max_seqlen_q":2,"num_requests":1}}
    backend=reference.blocks[0].attn.attention.attn_backend.get_name()
    with torch.inference_mode(),set_forward_context(omni_diffusion_config=od):
        for step in (0.1,0.5,0.9):
            kwargs["unique_timesteps"].fill_(step)
            expected=reference(**kwargs);actual=candidate(**kwargs);torch.cuda.synchronize()
            check("forward_exact_"+str(step),all(torch.equal(x,y) for x,y in zip(expected,actual)))
            check("forward_finite_"+str(step),all(torch.isfinite(x).all().item() for x in actual))
    return {"backend":backend,"shapes":[list(x.shape) for x in actual]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--research",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",choices=("cpu","cuda"),required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    r={"campaign_verdict":"INCONCLUSIVE","scope":"synthetic original H3 2 DiT blocks + 2 refiners; no sampling or real model","checks":[],"device":a.device}
    models=[];generator=None
    def ck(name,ok,**extras):
        r["checks"].append(dict(name=name,ok=bool(ok),**extras))
        if not ok:raise AssertionError(name)
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(0.40)
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError("insufficient initial free CUDA")
        setup_dist(a.output.parent)
        import inspect
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        record=json.loads((a.research/"source-record.json").read_text())
        ck("pinned_original_loader_source",hashlib.sha256(Path(inspect.getfile(MiniMaxH3DiTModel)).read_bytes()).hexdigest()==record["files"]["vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"]["sha256"])
        config=json.loads((a.research/"metadata/FL2VA__transformer__config.json").read_text())
        config.update(num_layers=2,token_refiner_num_layers=2,hidden_size=256,num_attention_heads=2,attention_head_dim=128,
            ffn_hidden_size=512,text_dim=128,timestep_input_dim=64,time_embed_hidden_size=256,time_embed_dim=128,
            adaln_out_features=18*256,final_adaln_out_features=2*256)
        template=make(config);source={};expected={};quant={}
        quant_suffixes=("adaln_proj.linear.weight","attn.out_proj.weight","attn.qkv_proj.weight","mlp.fc1.weight","mlp.fc2.weight")
        for i,(k,t) in enumerate(state(template).items()):
            vals=((torch.arange(t.numel(),dtype=torch.float32).remainder(251)+i*7).remainder(257)-128)/4096
            if "norm" in k:vals=vals+1
            dense=vals.reshape(t.shape).to(t.dtype)
            if k.startswith("blocks.") and k.endswith(quant_suffixes):
                rows,cols=t.shape;g=64 if "adaln_proj" in k else 256
                q=((torch.arange(rows*cols,dtype=torch.int32).reshape(rows,cols)*7+i*11)%255-127).to(torch.int8)
                scale=torch.pow(2.,-(torch.arange(rows)%3+14).float()).reshape(rows,1)
                base=k[:-7];meta={"format":"int8_tensorwise","convrot":True,"convrot_groupsize":g}
                source[k]=q;source[base+".weight_scale"]=scale
                source[base+".comfy_quant"]=torch.tensor(list(json.dumps(meta,sort_keys=True).encode()),dtype=torch.uint8)
                recovered=oracle(q,scale,g)
                expected[k]=native_order(recovered) if "qkv_proj" in k else recovered
                quant[k]=meta
            else:
                expected[k]=dense
                source[k]=source_order(dense) if "qkv_proj" in k else dense
        del vals,dense,t,template,q,scale,recovered
        ck("ten_quantized_projections",len(quant)==10)
        root=a.output.parent/"fixture";root.mkdir();file=root/"tiny-convrot.safetensors"
        if __import__("shutil").disk_usage(root).free<8*1024**3:raise RuntimeError("insufficient fixture disk")
        save_file(source,file)
        r["fixture_bytes"]=file.stat().st_size
        ck("fixture_below_64MiB",r["fixture_bytes"]<64*1024**2)
        r["fixture_sha256"]=hashlib.sha256(file.read_bytes()).hexdigest()
        del source
        reference=make(config,"cpu");models.append(reference)
        ck("reference_native_loader_coverage",reference.load_weights(iter(expected.items()))==set(state(reference)))
        reference.post_load_weights()
        from h3_omni_adapter.convrot import parse_comfy_quant,recover_bf16_projection
        stats=[];recover_refs=[]
        def weights(cancel=False):
            output=None;qw=None;sc=None;tensor=None
            try:
                with safe_open(file,framework="pt",device="cpu") as f:
                    for k in expected:
                        if k in quant:
                            base=k[:-7];qw=f.get_tensor(k);sc=f.get_tensor(base+".weight_scale")
                            spec=parse_comfy_quant(bytes(f.get_tensor(base+".comfy_quant").tolist()),max_metadata_bytes=256)
                            output,st=recover_bf16_projection(qw,sc,spec,max_output_bytes=4*1024**2,max_workspace_bytes=512*1024,
                                row_layout="grouped_qkv" if "qkv_proj" in k else "plain",
                                qkv_heads=2 if "qkv_proj" in k else None,qkv_head_dim=128 if "qkv_proj" in k else None,
                                cancelled=(lambda:True) if cancel else None)
                            ck("restored_"+k,torch.equal(output,expected[k]))
                            stats.append(st);recover_refs.append(weakref.ref(output))
                            yield k,output
                            output=qw=sc=None
                        else:
                            tensor=f.get_tensor(k)
                            yield k,native_order(tensor) if "qkv_proj" in k else tensor
                            tensor=None
            finally:output=qw=sc=tensor=None
        candidate=make(config,"cpu");models.append(candidate)
        for repeat in range(2):
            generator=weights()
            try:loaded=candidate.load_weights(generator)
            finally:generator.close();generator=None
            ck("candidate_native_loader_coverage_"+str(repeat),loaded==set(state(candidate)))
            candidate.post_load_weights()
            ck("all_native_parameters_exact_"+str(repeat),all(torch.equal(v,state(reference)[k]) for k,v in state(candidate).items()),targets=len(state(candidate)))
            gc.collect();ck("projection_temporaries_released_"+str(repeat),all(ref() is None for ref in recover_refs))
            ck("fixture_mapping_released_"+str(repeat),str(file) not in Path("/proc/self/maps").read_text())
        # Recovery failure before native loader submission: explicitly close suspended generator.
        generator=weights(cancel=True)
        try:
            for item in generator:del item
        except RuntimeError:ck("cancelled_generator_raises",True)
        else:ck("cancelled_generator_raises",False)
        finally:generator.close();generator=None
        gc.collect()
        ck("cancelled_mapping_released",str(file) not in Path("/proc/self/maps").read_text())
        ck("cancelled_recovery_did_not_mutate_model",all(torch.equal(v,state(reference)[k]) for k,v in state(candidate).items()))
        ck("quant_workspace_accounting_bounded",all(st["workspace_peak_bytes"]<=512*1024 for st in stats))
        r["recovery_stats"]=stats;r["max_workspace_bytes"]=max(st["workspace_peak_bytes"] for st in stats)
        if a.device=="cuda":
            reference.eval().to("cuda");candidate.eval().to("cuda")
            r["forward"]=forward_compare(reference,candidate,config,ck)
        parameter_refs=[weakref.ref(v) for m in models for v in state(m).values()]
        del reference,candidate,expected,models,weights
        models=[];gc.collect()
        ck("model_tensors_released",all(ref() is None for ref in parameter_refs))
        r["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        ck("cuda_peak_under_4GiB",r["cuda_peak_allocated_bytes"]<4*1024**3)
    except BaseException as e:
        r["error"]={"type":type(e).__name__,"traceback":traceback.format_exc()}
    finally:
        if generator is not None:generator.close()
        models.clear()
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        except Exception as e:r["cleanup_error"]=type(e).__name__
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();torch.cuda.empty_cache()
            r["cuda_allocated_after_cleanup"]=torch.cuda.memory_allocated()
    r["checks_total"]=len(r["checks"]);r["checks_ok"]=sum(x["ok"] for x in r["checks"]);r["exit_code"]=int("error" in r or "cleanup_error" in r or r["checks_total"]!=r["checks_ok"])
    text=json.dumps(r,indent=2)
    for val,label in ((str(a.output.parent),"<TRIAL_ROOT>"),(str(a.research),"<RESEARCH_ROOT>"),(str(Path.home()),"<LINUX_HOME>"),(str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):text=text.replace(val,label)
    a.output.write_text(text+"\n");print(json.dumps({k:v for k,v in r.items() if k not in ("checks","recovery_stats","error")}));return r["exit_code"]
if __name__=="__main__":raise SystemExit(main())
