"""Bounded HF metadata / safetensors header reads; never request tensor payload."""
import argparse,hashlib,json,re,ssl,struct,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import quote
MAX_METADATA=2*1024**2
MAX_HEADER=4*1024**2
TOTAL_BUDGET=24*1024**2
def context():
    ctx=ssl.create_default_context()
    # Honor the existing Windows trust store; never disable TLS verification.
    if hasattr(ssl,"enum_certificates"):
        for store in ("ROOT","CA"):
            for cert,encoding,trust in ssl.enum_certificates(store):
                if encoding=="x509_asn":
                    try: ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert))
                    except ssl.SSLError: pass
    return ctx
def nodup(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError("duplicate JSON key")
        result[k]=v
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("inventory","headers"),required=True)
    p.add_argument("--repository",required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--selection",type=Path)
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",a.repository): raise ValueError("repository")
    if not re.fullmatch(r"[a-f0-9]{40}",a.revision): raise ValueError("pinned revision required")
    a.output.mkdir(parents=True,exist_ok=False)
    result={"repository":a.repository,"revision":a.revision,"mode":a.mode,"campaign_verdict":"INCONCLUSIVE",
            "files":[],"http_body_bytes":0,"tensor_payload_requested_bytes":0,"errors":[]}
    ctx=context()
    start=time.monotonic()
    def get(url,offset=None,count=None,total=None):
        if time.monotonic()-start>240: raise TimeoutError("whole request budget")
        if count is not None and result["http_body_bytes"]+count>TOTAL_BUDGET:
            raise ValueError("remaining body budget insufficient")
        headers={"Accept-Encoding":"identity","User-Agent":"H3-metadata-audit"}
        if offset is not None: headers["Range"]=f"bytes={offset}-{offset+count-1}"
        with urlopen(Request(url,headers=headers),context=ctx,timeout=30) as response:
            status=response.status
            encoding=response.headers.get("Content-Encoding")
            if encoding not in (None,"identity"): raise ValueError("unexpected encoding")
            if offset is not None:
                wanted=f"bytes {offset}-{offset+count-1}/{total}"
                if status!=206 or response.headers.get("Content-Range")!=wanted:
                    raise ValueError("range not honored exactly; response body not read")
                if int(response.headers.get("Content-Length","-1"))!=count:
                    raise ValueError("range length not exact; body not read")
                data=response.read(count)
                if len(data)!=count: raise ValueError("short range body")
            else:
                if status!=200: raise ValueError("metadata status")
                declared=response.headers.get("Content-Length")
                if declared is not None and int(declared)>MAX_METADATA: raise ValueError("metadata too large")
                data=response.read(MAX_METADATA+1)
                if len(data)>MAX_METADATA: raise ValueError("metadata body limit")
            result["http_body_bytes"]+=len(data)
            if result["http_body_bytes"]>TOTAL_BUDGET: raise ValueError("total body budget")
            return data
    try:
        if a.mode=="inventory":
            url=f"https://huggingface.co/api/models/{a.repository}/revision/{a.revision}?blobs=true"
            raw=get(url)
            doc=json.loads(raw,object_pairs_hook=nodup)
            if doc["sha"]!=a.revision: raise ValueError("revision mismatch")
            result["source_url"]=url
            result["files"]=[{"path":x["rfilename"],"bytes":x.get("size"),"lfs":x.get("lfs"),"blob_id":x.get("blobId")}
                             for x in doc["siblings"]]
            (a.output/"inventory-response.raw.json").write_bytes(raw)
        else:
            selected=json.loads(a.selection.read_text(encoding="utf-8"))
            if selected["repository"]!=a.repository or selected["revision"]!=a.revision: raise ValueError("selection mismatch")
            if len(selected["files"])>12: raise ValueError("file-count budget")
            for item in selected["files"]:
                path=item["path"]
                if path.startswith("/") or ".." in path.split("/") or not path.endswith(".safetensors"): raise ValueError("source path")
                size=item["bytes"]
                url=f"https://huggingface.co/{a.repository}/resolve/{a.revision}/{quote(path,safe='/')}"
                prefix=get(url+"?h3_metadata_range=0-7",0,8,size)
                length=struct.unpack("<Q",prefix)[0]
                if length<2 or length>MAX_HEADER or length+8>size: raise ValueError("header bounds")
                raw=get(url+f"?h3_metadata_range=8-{length+7}",8,length,size)
                header=json.loads(raw,object_pairs_hook=nodup)
                spans=[]
                for key,value in header.items():
                    if key=="__metadata__": continue
                    if set(value)!={"dtype","shape","data_offsets"}: raise ValueError("unexpected tensor header fields")
                    lo,hi=value["data_offsets"]
                    if not 0<=lo<=hi<=size-length-8: raise ValueError("tensor offsets out of file")
                    spans.append((lo,hi,key))
                spans.sort()
                cursor=0
                for lo,hi,key in spans:
                    if lo!=cursor: raise ValueError("gap or overlapping tensor spans")
                    cursor=hi
                if cursor!=size-length-8: raise ValueError("unaccounted file bytes")
                name=Path(path).name
                (a.output/(name+".header.raw")).write_bytes(prefix+raw)
                (a.output/(name+".header.json")).write_text(json.dumps(header,indent=2)+"\n",encoding="utf-8")
                result["files"].append({"path":path,"file_bytes":size,"header_bytes":length+8,"tensor_count":len(spans),
                    "header_sha256":hashlib.sha256(prefix+raw).hexdigest(),"source_url":url,
                    "payload_sha256_verified":False,"lfs_source_identity":item.get("lfs")})
                (a.output/"receipt.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        result["execution_complete"]=True
    except BaseException as exc:
        # Do not persist signed redirect URLs or full network exception strings.
        result["errors"].append({"type":type(exc).__name__,"message":str(exc) if isinstance(exc,(ValueError,TimeoutError)) else "network or filesystem operation failed"})
        result["execution_complete"]=False
    result["elapsed_seconds"]=time.monotonic()-start
    (a.output/"receipt.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="files"}))
    return int(not result["execution_complete"])
if __name__=="__main__": raise SystemExit(main())
