"""Frozen original FL2VA full-meta inventory and tiny dense mmap CPU protocol."""
import argparse,gc,hashlib,json,os,traceback
from pathlib import Path
import torch
from safetensors.torch import save_file
from validate_meta_coverage_v2 import setup_dist

def state(m):
    return dict(list(m.named_parameters())+list(m.named_buffers()))

def make(config,device="meta"):
    from vllm_omni.diffusion.data import OmniDiffusionConfig,TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    od=OmniDiffusionConfig(model=None,model_class_name="MiniMaxH3Pipeline",
       tf_model_config=TransformerConfig.from_dict(config),
       cache_config={"minimax_h3_adaln_cache":False,"minimax_h3_adaln_offload":False})
    with set_current_diffusion_config(od),torch.device(device):
        return MiniMaxH3DiTModel(od,quant_config=None,diffusers_weights=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--research",type=Path,required=True)
    p.add_argument("--source-record",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result={"campaign_verdict":"INCONCLUSIVE","scope":"CPU full native DiT meta coverage and tiny dense mmap loading; no forward or real weights","checks":[]}
    def ck(name,ok,**kw):
        result["checks"].append(dict(name=name,ok=bool(ok),**kw))
        if not ok:raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError):ck(name,True)
        else:ck(name,False)
    try:
        import inspect
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        path=Path(inspect.getfile(MiniMaxH3DiTModel))
        source=json.loads((a.source_record/"source-record.json").read_text())
        ck("pinned_upstream_source",hashlib.sha256(path.read_bytes()).hexdigest()==source["files"]["vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"]["sha256"])
        setup_dist(a.output.parent)
        config=json.loads((a.research/"metadata/FL2VA__transformer__config.json").read_text())
        headers=json.loads((a.research/"results/transformer-tensors.json").read_text())
        full=make(config);s=state(full)
        ck("full_model_stays_meta",all(t.is_meta for t in s.values()))
        ck("full_exact_key_coverage",set(s)==set(headers),targets=len(s),sources=len(headers))
        ck("full_exact_shape_coverage",all(list(t.shape)==headers[k]["shape"] for k,t in s.items()))
        dtypes={"BF16":torch.bfloat16,"F32":torch.float32}
        ck("full_exact_dtype_coverage",all(t.dtype==dtypes[headers[k]["dtype"]] for k,t in s.items()))
        ck("original_not_pdd_quantized_vsa",full.arch.pdd_steps is None and not full._serialized_weight_map and not full.vsa_gates_enabled and not full._diffusers_weights)
        inventory={k:{"shape":list(t.shape),"dtype":str(t.dtype),"bytes":t.numel()*t.element_size(),"meta":t.is_meta} for k,t in s.items()}
        (a.output.parent/"full-meta-inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
        del s,full
        tiny=dict(config,num_layers=2,token_refiner_num_layers=2,hidden_size=256,
          num_attention_heads=2,attention_head_dim=128,ffn_hidden_size=512,text_dim=128,
          timestep_input_dim=64,time_embed_hidden_size=256,time_embed_dim=128,
          adaln_out_features=18*256,final_adaln_out_features=2*256)
        template=make(tiny);raw={}
        for i,(k,t) in enumerate(state(template).items()):
            # Variation across rows and heads is required to detect grouped QKV order.
            values=((torch.arange(t.numel(),dtype=torch.float32).remainder(1009)+i*7).remainder(1013)-506)/2048
            if "norm" in k:values=values+1
            raw[k]=values.reshape(t.shape).to(t.dtype)
        root=a.output.parent/"fixture";root.mkdir()
        keys=sorted(raw);half=len(keys)//2
        for name,part in (("a.safetensors",keys[:half]),("b.safetensors",keys[half:])):
            save_file({k:raw[k] for k in part},root/name)
        index={"weight_map":{k:"a.safetensors" if i<half else "b.safetensors" for i,k in enumerate(keys)}}
        ix=root/"model.safetensors.index.json";ix.write_text(json.dumps(index))
        fixture_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir()}
        result["fixture_bytes"]=sum(f.stat().st_size for f in root.iterdir())
        ck("fixture_under_64mib",result["fixture_bytes"]<64*1024**2)
        ordinary=make(tiny,"cpu")
        ck("ordinary_native_loader_complete",ordinary.load_weights(iter(raw.items()))==set(state(ordinary)))
        ordinary.post_load_weights()
        reference={k:t.detach().clone() for k,t in state(ordinary).items()}
        qkey="blocks.0.attn.qkv_proj.weight"
        sourceq=raw[qkey].reshape(2,3,128,256)
        independent=torch.cat([sourceq[:,i].reshape(256,256) for i in range(3)],dim=0)
        ck("nontrivial_grouped_qkv_oracle",not torch.equal(raw[qkey],independent) and torch.equal(reference[qkey],independent))
        ck("native_gate_up_order",torch.equal(reference["blocks.0.mlp.fc1.weight"],raw["blocks.0.mlp.fc1.weight"]))
        from h3_omni_adapter.native_mmap import H3NativeMmapPlan,H3NativeMmapLoader
        def plan_for(m=template,**kw):return H3NativeMmapPlan.build(m,root,max_total_file_bytes=64*1024**2,**kw)
        plan=plan_for();ck("tiny_all_tensors_planned",set(plan.bindings)==set(reference))
        loader=H3NativeMmapLoader(template,plan,device="cpu",max_group_bytes=32*1024**2)
        try:
            reject("resident_required",lambda:loader.load_group("blocks.0"))
            loader.load_group("resident")
            for roundno in range(2):
                for group,names in plan.groups.items():
                    if group!="resident":loader.load_group(group)
                    ck("exact_"+str(roundno)+"_"+group,all(torch.equal(state(template)[k],reference[k]) for k in names))
                    if group!="resident":
                        other="blocks.1" if group!="blocks.1" else "blocks.0"
                        reject("single_group_"+str(roundno)+"_"+group,lambda:loader.load_group(other))
                        loader.release_group(group)
                        ck("released_"+str(roundno)+"_"+group,all(state(template)[k].is_meta for k in names))
            ck("native_rope_loaded_from_checkpoint",torch.equal(template.rope.inv_freq,raw["rope.inv_freq"]))
            ck("mmap_observed",loader.stats["mmap_observed"])
            ck("one_streamed_group",loader.stats["max_active_streamed_groups"]==1)
            result["loader_stats"]=dict(loader.stats)
        finally:loader.close()
        ck("close_all_meta",all(t.is_meta for t in state(template).values()) and not loader.active)
        ck("mmap_released",not any(str(root) in line for line in Path("/proc/self/maps").read_text().splitlines()))
        loader.close();reject("closed_loader_rejects_reuse",lambda:loader.load_group("resident"))
        small=H3NativeMmapLoader(template,plan,device="cpu",max_group_bytes=1)
        try:reject("allocation_budget_before_load",lambda:small.load_group("resident"))
        finally:small.close()
        ck("budget_failure_all_meta",all(t.is_meta for t in state(template).values()))
        reject("file_budget",lambda:H3NativeMmapPlan.build(template,root,max_total_file_bytes=1))
        reject("tp2_rejected",lambda:plan_for(tensor_parallel_size=2))
        changed=make(dict(tiny,ffn_hidden_size=256))
        reject("shape_mismatch",lambda:plan_for(changed))
        saved=ix.read_bytes()
        try:
            bad=json.loads(saved);bad["weight_map"].pop(keys[0]);ix.write_text(json.dumps(bad))
            reject("missing_index_key",plan_for)
            bad=json.loads(saved);bad["weight_map"]["unknown.weight"]="a.safetensors";ix.write_text(json.dumps(bad))
            reject("unknown_index_key",plan_for)
            bad=json.loads(saved);bad["weight_map"][keys[0]]="../escape.safetensors";ix.write_text(json.dumps(bad))
            reject("path_traversal",plan_for)
        finally:ix.write_bytes(saved)
        fresh=plan_for();target=root/"a.safetensors";st=target.stat()
        with target.open("ab") as changed_file:changed_file.write(b"\0")
        try:reject("changed_file_after_plan",fresh.check_files_unchanged)
        finally:
            with target.open("r+b") as changed_file:changed_file.truncate(st.st_size)
        fresh=plan_for();fault=H3NativeMmapLoader(template,fresh,device="cpu",max_group_bytes=32*1024**2)
        fault.load_group("resident")
        native=template.load_weights
        def failing(weights):
            next(weights);raise RuntimeError("injected loader failure")
        template.load_weights=failing
        try:
            reject("native_exception",lambda:fault.load_group("blocks.0"))
            ck("native_exception_releases_block",all(state(template)[k].is_meta for k in fresh.groups["blocks.0"]))
        finally:template.load_weights=native;fault.close()
        ck("native_exception_releases_mmaps",not any(str(root) in line for line in Path("/proc/self/maps").read_text().splitlines()))
        ck("fixture_unchanged",fixture_hashes=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir()})
        (a.output.parent/"fixture-manifest.json").write_text(json.dumps(fixture_hashes,indent=2)+"\n")
        ck("no_cuda_context",not torch.cuda.is_initialized())
        gc.collect()
    except BaseException as exc:
        trace=traceback.format_exc()
        for value,label in ((str(a.source_record),"<ITERATION_ROOT>"),(str(a.research),"<RESEARCH_ROOT>"),(str(Path.home()),"<LINUX_HOME>"),(str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
            trace=trace.replace(value,label)
        result["error"]={"type":type(exc).__name__,"traceback":trace}
    result["checks_total"]=len(result["checks"]);result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    result["exit_code"]=int("error" in result or result["checks_total"]!=result["checks_ok"])
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","error"}}))
    return result["exit_code"]
if __name__=="__main__":raise SystemExit(main())
