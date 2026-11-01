"""Read only selected small .comfy_quant U8 JSON records, never weight/scale tensors."""
import argparse,hashlib,json,ssl,time
from pathlib import Path
from urllib.request import Request,urlopen
def context():
    c=ssl.create_default_context()
    if hasattr(ssl,"enum_certificates"):
        for store in ("ROOT","CA"):
            for cert,encoding,trust in ssl.enum_certificates(store):
                if encoding=="x509_asn":
                    try:c.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert))
                    except ssl.SSLError:pass
    return c
def main():
    p=argparse.ArgumentParser();p.add_argument("--iteration",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    selected=[
      ("comfy-headers-v2-retry","minimax_h3_fl2va_int8_convrot.safetensors","blocks.0.adaln_proj.linear.comfy_quant"),
      ("comfy-headers-v2-retry","minimax_h3_fl2va_int8_convrot.safetensors","blocks.0.attn.out_proj.comfy_quant"),
      ("comfy-headers-v2-retry","qwen3vl_32b_minimax_h3_int8_convrot.safetensors","model.layers.0.mlp.down_proj.comfy_quant"),
      ("comfy-nvfp4-v2","qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","model.embed_tokens.comfy_quant"),
      ("comfy-nvfp4-v2","qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","model.layers.0.mlp.down_proj.comfy_quant"),
      ("comfy-nvfp4-v2","qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","model.layers.0.self_attn.q_proj.comfy_quant")]
    result={"campaign_verdict":"INCONCLUSIVE","purpose":"selected non-weight JSON metadata only","records":[],"body_bytes_read":0,
            "weight_and_numeric_scale_payload_bytes_requested":0,"not_all_quant_metadata_audited":True,"errors":[]}
    ctx=context()
    try:
        for sub,name,key in selected:
            rp=a.iteration/"metadata"/sub/"receipt.json";receipt=json.loads(rp.read_text(encoding="utf-8"));f=next(x for x in receipt["files"] if Path(x["path"]).name==name)
            raw=(rp.parent/(name+".header.raw")).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=f["header_sha256"]:raise ValueError("header identity")
            d=json.loads(raw[8:]);v=d[key];lo,hi=v["data_offsets"];n=hi-lo
            if not key.endswith(".comfy_quant") or v["dtype"]!="U8" or v["shape"]!=[n] or not 1<=n<=256:raise ValueError("non-JSON tensor/bounds rejected")
            if result["body_bytes_read"]+n>4096:raise ValueError("body budget")
            remaining=120-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError("overall request budget")
            offset=f["header_bytes"]+lo;end=offset+n-1
            req=Request(f["source_url"]+f"?h3_quant_json={offset}-{end}",headers={"Range":f"bytes={offset}-{end}","Accept-Encoding":"identity","User-Agent":"H3-small-JSON-metadata-audit"})
            with urlopen(req,context=ctx,timeout=min(30,remaining)) as res:
                if res.status!=206 or res.headers.get("Content-Range")!=f'bytes {offset}-{end}/{f["file_bytes"]}' or res.headers.get("Content-Length")!=str(n) or res.headers.get("Content-Encoding") not in (None,"identity"):raise ValueError("range not exact; body not read")
                blob=res.read(n);result["body_bytes_read"]+=len(blob)
                if len(blob)!=n:raise ValueError("short body")
            metadata=json.loads(blob)
            if not isinstance(metadata,dict):raise ValueError("not object metadata")
            result["records"].append({"source_url":f["source_url"],"key":key,"absolute_byte_range":[offset,end],"bytes":n,"sha256":hashlib.sha256(blob).hexdigest(),"metadata":metadata})
        result["execution_complete"]=True
    except Exception as exc:
        reason=getattr(exc,"reason",None)
        result["errors"].append({"type":type(exc).__name__,"reason_type":type(reason).__name__ if reason else None,"message":str(exc) if isinstance(exc,(ValueError,TimeoutError)) else "network failure"})
        result["execution_complete"]=False
    result["elapsed_seconds"]=time.monotonic()-start
    (a.output/"receipt.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))
    return int(not result["execution_complete"])
if __name__=="__main__":raise SystemExit(main())
