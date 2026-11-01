"""Static checkpoint header audit. No tensor payload, imports or model execution."""
import argparse,collections,hashlib,json,math,re,struct
from pathlib import Path
SIZES={"BOOL":1,"U8":1,"I8":1,"F8_E4M3":1,"F8_E5M2":1,"BF16":2,"F16":2,"I16":2,"U16":2,"F32":4,"I32":4,"U32":4,"F64":8,"I64":8,"U64":8}
EXTRA=(".comfy_quant",".weight_scale",".weight_scale_2",".input_scale",".pre_quant_scale")
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def nodup(items):
    d={}
    for k,v in items:
        if k in d: raise ValueError("duplicate JSON key")
        d[k]=v
    return d
def strict_header(raw,file_size,prefix):
    if prefix:
        n=struct.unpack("<Q",raw[:8])[0]
        if n!=len(raw)-8: raise ValueError("header length")
        data=raw[8:]
    else: data=raw
    d=json.loads(data,object_pairs_hook=nodup)
    spans=[]
    for k,v in d.items():
        if k=="__metadata__":
            if not isinstance(v,dict) or not all(isinstance(a,str) and isinstance(b,str) for a,b in v.items()):
                raise ValueError("metadata must be string mapping")
            continue
        if set(v)!={"dtype","shape","data_offsets"} or v["dtype"] not in SIZES: raise ValueError("header fields/dtype")
        if not isinstance(v["shape"],list) or not all(type(i) is int and i>=0 for i in v["shape"]): raise ValueError("shape")
        offsets=v["data_offsets"]
        if not isinstance(offsets,list) or len(offsets)!=2 or not all(type(i) is int for i in offsets): raise ValueError("offset types")
        lo,hi=offsets
        if lo<0 or hi<lo or hi-lo!=math.prod(v["shape"])*SIZES[v["dtype"]]: raise ValueError("dtype shape size mismatch")
        spans.append((lo,hi,k))
    cursor=0
    for lo,hi,k in sorted(spans):
        if lo!=cursor: raise ValueError("overlap or gap")
        cursor=hi
    if cursor+len(data)+8!=file_size: raise ValueError("total serialized length")
    return d
def tensors(h): return {k:v for k,v in h.items() if k!="__metadata__"}
def nbytes(v): return v["data_offsets"][1]-v["data_offsets"][0]
def group(k):
    m=re.match(r"(blocks\.\d+|token_refiner\.blocks\.\d+|model\.layers\.\d+)\.",k)
    if m: return m[1]
    if k.startswith("visual."): return "vision"
    if k.startswith("model.embed_tokens."): return "embedding"
    return "resident"
def overview(h):
    ts=tensors(h); gs=collections.defaultdict(int); ds=collections.defaultdict(int); cats=collections.defaultdict(int)
    for k,v in ts.items():
        b=nbytes(v); gs[group(k)]+=b; ds[v["dtype"]]+=b
        cat="quant_json" if k.endswith(".comfy_quant") else "pre_quant_scale" if k.endswith(".pre_quant_scale") else "scales" if k.endswith((".weight_scale",".weight_scale_2",".input_scale")) else "weights_other"
        cats[cat]+=b
    return {"tensor_count":len(ts),"tensor_bytes":sum(nbytes(v) for v in ts.values()),"dtype_bytes":dict(ds),"category_bytes":dict(cats),"group_bytes":dict(gs),
            "largest_tensor":{"key":max(ts,key=lambda k:nbytes(ts[k])),"bytes":max(nbytes(v) for v in ts.values())},
            "metadata":h.get("__metadata__",{}),"quant_json_contents_read":False}
def compare(base,candidate,packed4=False):
    b=tensors(base); c=tensors(candidate)
    extras=[k for k in c if k.endswith(EXTRA)]
    ck={k for k in c if k not in extras}
    missing=sorted(set(b)-ck); unexpected=sorted(ck-set(b)); mismatch=[]
    for k in sorted(set(b)&ck):
        sh=c[k]["shape"].copy()
        if packed4 and k.endswith(".weight") and c[k]["dtype"]=="U8":
            sh[-1]*=2
        if sh!=b[k]["shape"]: mismatch.append({"key":k,"baseline":b[k]["shape"],"candidate_logical":sh})
    return {"base_count":len(b),"candidate_base_count":len(ck),"extra_count":len(extras),"missing":missing,"unexpected":unexpected,"shape_mismatch":mismatch,
            "shape_coverage_complete":not (missing or unexpected or mismatch),"payload_equivalence_verified":False,
            "packed4_shape_interpretation":"U8 weight last dimension x2; format-content/kernel validity not proven" if packed4 else None}
def main():
    p=argparse.ArgumentParser();p.add_argument("--iteration",type=Path,required=True);p.add_argument("--original",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    q=a.iteration; original=a.original;files={}; headers={}; provenance=[]
    for sub in ("comfy-headers-v2-retry","comfy-nvfp4-v2"):
        rp=q/"metadata"/sub/"receipt.json"; receipt=read(rp)
        provenance.append({"path":rp.relative_to(q).as_posix(),"sha256":digest(rp),"execution_complete":receipt["execution_complete"]})
        for f in receipt["files"]:
            name=Path(f["path"]).name
            if name in files: raise ValueError("duplicate source file")
            rawp=rp.parent/(name+".header.raw");raw=rawp.read_bytes()
            if digest(rawp)!=f["header_sha256"]: raise ValueError("raw header hash mismatch")
            h=strict_header(raw,f["file_bytes"],True)
            if h!=read(rp.parent/(name+".header.json")): raise ValueError("parsed header mismatch")
            files[name]={"file_bytes":f["file_bytes"],"header_bytes":f["header_bytes"],"header_sha256":f["header_sha256"],"source_url":f["source_url"],"lfs_identity":f["lfs_source_identity"],**overview(h)}
            headers[name]=h
    rec=read(original/"metadata/fetch-record.json"); official={"transformer":{},"text_encoder":{},"video_vae":{},"audio_vae":{}}; original_sizes=collections.defaultdict(int)
    for f in rec["files"]:
        if f["kind"]!="header": continue
        component=f["path"].split("/")[1];rawp=original/"metadata"/f["saved"];raw=rawp.read_bytes()
        if digest(rawp)!=f["sha256"]: raise ValueError("historical header hash mismatch")
        h=strict_header(raw,f["file_bytes"],False)
        for k,v in tensors(h).items():
            if k in official[component]: raise ValueError("duplicate official key across shards")
            official[component][k]=v
        original_sizes[component]+=f["file_bytes"]
    bf=headers["minimax_h3_fl2va_bf16.safetensors"];i8=headers["minimax_h3_fl2va_int8_convrot.safetensors"]
    te=headers["qwen3vl_32b_minimax_h3_bf16.safetensors"];tei=headers["qwen3vl_32b_minimax_h3_int8_convrot.safetensors"];ten=headers["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]
    ot={}
    omitted=[]
    for k,v in official["text_encoder"].items():
        nk=k.replace("model.language_model.","model.").replace("model.visual.","visual.")
        m=re.match(r"model.layers.(\d+)\.",nk)
        if nk.startswith("visual.") or nk=="model.embed_tokens.weight" or (m and int(m[1])<50): ot[nk]=v
        else: omitted.append(k)
    coverage={"official_to_comfy_bf16_dit":compare(official["transformer"],bf),"bf16_to_int8_dit":compare(bf,i8),
              "official_native_50_and_vision_to_comfy_bf16_te":compare(ot,te),"bf16_to_int8_te":compare(te,tei),"bf16_to_nvfp4_te":compare(te,ten,True),
              "pruned_fp8_to_full_dit":compare(bf,headers["minimax_h3_fl2va_pruned_fp8_scaled.safetensors"])}
    # The first five must cover all original native target logical shapes.
    for name,item in list(coverage.items())[:5]:
        if not item["shape_coverage_complete"]: raise ValueError("coverage mismatch: "+name)
    if coverage["pruned_fp8_to_full_dit"]["shape_coverage_complete"]: raise ValueError("pruned exclusion reference unexpectedly complete")
    budget={}
    for label,h in (("dit_bf16",bf),("dit_int8_convrot",i8),("te_bf16",te),("te_int8_convrot",tei),("te_nvfp4_awq",ten)):
        gs=overview(h)["group_bytes"]
        stage="dit" if label.startswith("dit") else "text"
        prefix="blocks." if stage=="dit" else "model.layers."
        groups={k:v for k,v in gs.items() if k.startswith(prefix)}
        refs={k:v for k,v in gs.items() if k.startswith("token_refiner.blocks.")}
        resident=gs.get("resident",0)+gs.get("embedding",0)
        maxgroup=max(list(groups.values())+list(refs.values()))
        budget[label]={"resident_bytes":resident,"main_layer_count":len(groups),"main_layer_min_bytes":min(groups.values()),"main_layer_max_bytes":max(groups.values()),
          "refiner_count":len(refs),"refiner_max_bytes":max(refs.values(),default=0),"max_group_bytes":maxgroup,
          "resident_plus_max_group_serialized_bytes":resident+maxgroup,"vision_excluded_bytes":gs.get("vision",0),
          "retained_tensor_bytes":sum(gs.values())-gs.get("vision",0),"entire_stage_resident_feasibility":"INCONCLUSIVE","entire_stage_fits_existing_6GiB_worker_RSS_guard_by_tensor_bytes":sum(gs.values())-gs.get("vision",0)<=6*1024**3}
    originalbudget=read(original/"results/native-budget-detail.json")
    # Arithmetic examples, not peak estimates: destination CPU + packed source; staged GPU is separate.
    db=budget["dit_bf16"]; di=budget["dit_int8_convrot"]; tb=budget["te_bf16"];ti=budget["te_int8_convrot"];tn=budget["te_nvfp4_awq"]
    overlap={}
    for label,b,base in (("dit_int8_to_bf16",di,db),("te_int8_to_bf16",ti,tb),("te_nvfp4_to_bf16",tn,tb)):
        overlap[label]={"packed_main_group_bytes":b["main_layer_max_bytes"],"dense_main_group_bytes":base["main_layer_max_bytes"],
          "cpu_dense_resident_plus_one_dense_and_one_packed_main_group_bytes":base["resident_bytes"]+base["main_layer_max_bytes"]+b["main_layer_max_bytes"],
          "additional_same_size_pinned_copy_bytes":base["main_layer_max_bytes"],
          "gpu_dense_resident_plus_two_dense_main_slots_bytes":base["resident_bytes"]+2*base["main_layer_max_bytes"],
          "excluded":["dequant intermediate FP32","rotation/QKV copies","activations","kernels/workspaces","allocator","imports/page-cache","runtime remapping","embedding dequant temporary"]}
    fallback={"CPU two full FP32 linear intermediates for largest projection":2*max(math.prod(v["shape"])*4 for k,v in tensors(bf).items() if k.endswith(".weight")),
              "historical_DiT_QKV_copy_bytes":originalbudget["dit"]["qkv_reorder_cpu_temporary_bytes"],
              "NVFP4_embedding_packed_plus_scale_bytes":sum(nbytes(v) for k,v in tensors(ten).items() if k.startswith("model.embed_tokens.") and not k.endswith(".comfy_quant")),
              "NVFP4_embedding_BF16_destination_bytes":tb["resident_bytes"]}
    common=original_sizes["video_vae"]+original_sizes["audio_vae"]
    dit_size=files["minimax_h3_fl2va_int8_convrot.safetensors"]["file_bytes"]
    bundles={"official_all":sum(original_sizes.values()),
      "int8_dit_plus_official_full_text_original_vaes":dit_size+original_sizes["text_encoder"]+common,
      "int8_dit_plus_native_50_bf16_community_text_original_vaes":dit_size+files["qwen3vl_32b_minimax_h3_bf16.safetensors"]["file_bytes"]+common,
      "int8_dit_int8_text_original_vaes":dit_size+files["qwen3vl_32b_minimax_h3_int8_convrot.safetensors"]["file_bytes"]+common,
      "int8_dit_nvfp4_text_original_vaes":dit_size+files["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]["file_bytes"]+common}
    device=read(q/"metadata/resource-snapshot.json");reserve=8*1024**3
    storage={k:{"files_bytes":v,"GiB":v/1024**3,"remaining_guest_after_single_copy_bytes":device["guest"]["ext4_bytes"]["available"]-v,
                "fits_single_copy_plus_8GiB_operational_margin":v+reserve<=device["guest"]["ext4_bytes"]["available"],
                "two_identical_copies_plus_margin_fit":2*v+reserve<=device["guest"]["ext4_bytes"]["available"]} for k,v in bundles.items()}
    dio=overview(i8)["group_bytes"]
    logical50=50*sum(v for k,v in dio.items() if k.startswith(("blocks.","token_refiner.blocks.")))
    result={"campaign_verdict":"INCONCLUSIVE","scope":"static headers and hypothetical liveness arithmetic; no tensor values, RSS/VRAM or speed measured",
      "source_receipts":provenance,"files":files,"coverage":coverage,"official_omitted_text_keys":omitted,
      "original_component_file_bytes":dict(original_sizes),"group_budgets":budget,"overlap_arithmetic":overlap,"temporary_examples":fallback,
      "disk_bundle_budgets":storage,"disk_scope":"complete named component files only; config/tokenizer/cache/env/download temporary extra. Operational margin is capacity planning, not an acceptance gate.",
      "fifty_steps_logical_int8_dit_groups_bytes":logical50,"fifty_steps_logical_original_dit_groups_bytes":originalbudget["dit"]["fifty_step_streamed_source_bytes"],
      "logical_stream_warning":"Assumes every block/refiner read every step; no persistent GPU weights. Not actual SSD I/O or inference latency.",
      "metadata_payload_not_read":True,"quality_equivalence":"INCONCLUSIVE","peak_memory":"INCONCLUSIVE"}
    (a.output/"header-budget-audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"files":len(files),"header_tensors":sum(f["tensor_count"] for f in files.values()),"coverage":{k:v["shape_coverage_complete"] for k,v in coverage.items()},"bundles":storage,"campaign_verdict":"INCONCLUSIVE"}))
if __name__=="__main__": main()
