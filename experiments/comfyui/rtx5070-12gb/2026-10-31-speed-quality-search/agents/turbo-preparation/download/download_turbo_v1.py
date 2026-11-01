"""Q33 fixed Turbo single-file downloader. Stdlib only, no automatic retries.
Application-direct mirror HTTPS does not prove bypass of a system VPN/TUN.
Old ledger remains immutable; all legacy downloaders must account for this
new subledger before further network work. No model is imported or executed.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import ssl
import sys
import time
import urllib.parse
import uuid

FILE_BYTES=779849816
FILE_SHA="5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3"
REVISION="7b7ac96b0616100db75ea285090210c3ddf37c04"
NAME="minimax_h3_turbo_v4_step600_ema.safetensors"
CANONICAL="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/"+REVISION+"/"+NAME
MIRROR=CANONICAL.replace("https://huggingface.co/","https://hf-mirror.com/",1)
RELATIVE="comfy-original/models/loras/"+NAME
CHUNK=4*1024**2
SUBCAP=1024**3
GLOBALCAP=128*1024**3
BASE_RESERVED=81401170863
BASE_BYTES=4568504
BASE_SHA="02de3d2c8c4d4f988655f00e21b335cebd47a1351bd4f0e0aeb73f0ed5a90da2"
CDNS=frozenset({"us.aws.cdn.hf.co","cas-bridge.xethub.hf.co","cdn-lfs.huggingface.co","cdn-lfs-us-1.huggingface.co","cdn-lfs-eu-1.huggingface.co"})
OLD_REL="iterations/2026-10-27-original-sequential-entry/agents/download-preparation/state/download-ledger-v1.jsonl"
STOP=False

class Gate(ValueError):pass

def require(ok,code):
    if not ok:raise Gate(code)

def unique(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,"duplicate_json_key");out[k]=v
    return out

def linkfree(path):
    return all(not p.is_symlink() and not (hasattr(p,"is_junction") and p.is_junction()) for p in (Path(path),*Path(path).parents))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while b:=f.read(512*1024):h.update(b)
    return h.hexdigest()

def json_read(path,cap=128*1024):
    p=Path(path);require(linkfree(p) and p.stat().st_size<=cap,"json_size_or_link")
    return json.loads(p.read_bytes(),object_pairs_hook=unique)

def emit_new(path,value):
    b=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    require(len(b)<=256*1024 and linkfree(path),"output_bound")
    with Path(path).open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())

def fixed_profile():
    return {"schema":"q33-turbo-download-profile/v1","id":"turbo-v4-step600-ema","revision":REVISION,
      "asset_url":CANONICAL,"asset_bytes":FILE_BYTES,"asset_sha256":FILE_SHA,"relative_path":RELATIVE,
      "body_cap_bytes":SUBCAP,"global_cap_bytes":GLOBALCAP,"old_ledger_bytes":BASE_BYTES,
      "old_ledger_sha256":BASE_SHA,"old_reserved_body_bytes":BASE_RESERVED,
      "route":"fixed_mirror_application_direct_TLS_TUN_unproven","chunk_bytes":CHUNK,"attempts_per_range":3,
      "authorization_source":"parent Q33 delegation: user speed/quality scope; parent selected 1GiB subcap within previously authorized total128GiB",
      "reserved_base_source":"parent completed Q32 accounting handoff; old ledger hash and size are mandatory immutable binding",
      "legacy_transport_notice":"Do not run legacy downloaders unless they also account for this subledger; shared old lock alone does not update their budget totals."}

def validate_profile(p):
    require(p==fixed_profile(),"fixed_profile_identity");return True

def range_at(index):
    require(type(index)is int and 0<=index<((FILE_BYTES+CHUNK-1)//CHUNK),"chunk_index")
    start=index*CHUNK;return start,min(CHUNK,FILE_BYTES-start)

def check_budget(reserved,n,base):
    require(type(reserved)is int and type(n)is int and type(base)is int and 0<=reserved<=SUBCAP and 0<n<=CHUNK and 0<=base<=GLOBALCAP,"budget_types_or_bounds")
    require(reserved+n<=SUBCAP and base+reserved+n<=GLOBALCAP,"body_budget_exhausted");return True

def check_base(path):
    p=Path(path);require(linkfree(p) and p.stat().st_size==BASE_BYTES,"old_ledger_size_or_link")
    s=p.stat();require(sha(p)==BASE_SHA,"old_ledger_sha")
    t=p.stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns),"old_ledger_changed")

class WorkerLock:
    # Same lock byte/protocol as frozen Q28 WorkerLock. No old ledger mutation.
    def __init__(self,path):self.path=Path(path);self.f=None
    def __enter__(self):
        require(linkfree(self.path),"lock_link")
        self.f=self.path.open("a+b");self.f.seek(0,2)
        if self.f.tell()==0:self.f.write(b"0");self.f.flush()
        self.f.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.f.close();self.f=None;raise Gate("another_download_worker_owns_lock")
        return self
    def __exit__(self,*args):
        if self.f is not None:self.f.close();self.f=None

class Clock:
    def __init__(self,seconds,stop):self.deadline=time.monotonic()+seconds;self.stop=stop
    def check(self,new=False):
        require(not STOP and not (self.stop is not None and self.stop.exists()),"operator_stop")
        left=self.deadline-time.monotonic();require(left>(15 if new else 0),"deadline");return left

class Ledger:
    def __init__(self,path,profile_sha):
        self.path=Path(path);self.reserved=0;self.counts={};self.attempts=[];self.done={}
        self.init={"event":"init","schema":"q33-turbo-body-ledger/v1","profile_sha256":profile_sha,"base_sha256":BASE_SHA,"base_reserved":BASE_RESERVED,"subcap":SUBCAP,"globalcap":GLOBALCAP}
        require(linkfree(self.path),"ledger_link")
        if not self.path.exists():
            with self.path.open("xb") as f:f.write((json.dumps(self.init,separators=(",",":"))+"\n").encode());f.flush();os.fsync(f.fileno())
        require(self.path.stat().st_size<=2*1024**2,"ledger_size")
        b=self.path.read_bytes();require(b.endswith(b"\n"),"ledger_partial_event")
        rows=[json.loads(x,object_pairs_hook=unique)for x in b.splitlines()]
        require(rows and rows[0]==self.init,"ledger_identity")
        for row in rows[1:]:self.consume(row)
    def allowed(self,index):
        _,n=range_at(index);check_budget(self.reserved,n,BASE_RESERVED)
        require(self.counts.get(index,0)<3 and index not in self.done,"attempt_limit_or_complete")
    def consume(self,e):
        if e.get("event")=="reserve":
            require(set(e)=={"event","id","index"} and e["id"]==len(self.attempts),"reserve_shape")
            index=e["index"];self.allowed(index);self.reserved+=range_at(index)[1]
            self.counts[index]=self.counts.get(index,0)+1;self.attempts.append(dict(e))
        elif e.get("event")=="finish":
            require(set(e)=={"event","id","ok","sha256","error","body_read"} and type(e["id"])is int and 0<=e["id"]<len(self.attempts),"finish_shape")
            a=self.attempts[e["id"]];n=range_at(a["index"])[1]
            require("ok"not in a and type(e["ok"])is bool and type(e["body_read"])is int and 0<=e["body_read"]<=n,"finish_values")
            require(not e["ok"] or (e["body_read"]==n and isinstance(e["sha256"],str) and re.fullmatch("[0-9a-f]{64}",e["sha256"]) and e["error"] is None),"success_values")
            a.update(e)
            if e["ok"]:self.done[a["index"]]=a
        else:raise Gate("ledger_event")
    def append(self,e):
        b=(json.dumps(e,separators=(",",":"))+"\n").encode()
        require(self.path.stat().st_size+len(b)<=2*1024**2,"ledger_capacity")
        self.consume(e)
        with self.path.open("ab") as f:f.write(b);f.flush();os.fsync(f.fileno())
    def reserve(self,index):
        self.allowed(index);i=len(self.attempts);self.append({"event":"reserve","id":i,"index":index});return i
    def finish(self,i,ok,h,error,n):
        self.append({"event":"finish","id":i,"ok":ok,"sha256":h,"error":error,"body_read":n})
    def summary(self):
        return {"reserved_body_bytes":self.reserved,"recorded_body_bytes":sum(a.get("body_read",0)for a in self.attempts),
          "unfinished_reservations":sum("ok"not in a for a in self.attempts),"complete_chunks":len(self.done),
          "base_reserved_body_bytes":BASE_RESERVED,"combined_reserved_body_bytes":BASE_RESERVED+self.reserved,
          "subcap_bytes":SUBCAP,"globalcap_bytes":GLOBALCAP,
          "accounting":"Failed/unfinished reservations persist. Unknown unfinished read <= reservation. Application body only, not TCP/TLS/VPN overhead."}

def route(url,hop):
    u=urllib.parse.urlsplit(url)
    require(u.scheme=="https" and not u.username and not u.password and u.port in (None,443) and not u.fragment,"https_authority")
    require(url==MIRROR if hop==0 else u.hostname in CDNS,"fixed_mirror_or_CDN")
    return u

def tls_context():
    c=ssl.create_default_context()
    require(c.check_hostname and c.verify_mode==ssl.CERT_REQUIRED,"TLS_verification")
    return c

def check_response(r,start,n):
    require(r.status!=429,"HTTP429_stop")
    require(r.status==206,"exact206_required")
    require(r.getheader("Content-Range")==f"bytes {start}-{start+n-1}/{FILE_BYTES}","content_range")
    require(r.getheader("Content-Length")==str(n),"content_length")
    require(r.getheader("Content-Encoding","identity")in ("identity","") and r.getheader("Transfer-Encoding")is None,"encoding")
    return True

def open_response(start,n,clock):
    url=MIRROR;ctx=tls_context()
    for hop in range(6):
        clock.check();u=route(url,hop)
        conn=http.client.HTTPSConnection(u.hostname,443,timeout=max(.1,min(30,clock.check())),context=ctx);response=None
        try:
            conn.request("GET",urllib.parse.urlunsplit(("","",u.path,u.query,"")),headers={"Range":f"bytes={start}-{start+n-1}","Accept-Encoding":"identity","Connection":"close","User-Agent":"H3-Q33-fixed-turbo/1"})
            response=conn.getresponse()
            if response.status in (301,302,303,307,308):
                require(hop<5,"redirect_limit");loc=response.getheader("Location")
                require(bool(loc),"redirect_location")
                target=urllib.parse.urljoin(url,loc);route(target,hop+1)
                response.close();conn.close();url=target;continue
            check_response(response,start,n)
            return conn,response
        except BaseException:
            if response is not None:response.close()
            conn.close();raise
    raise Gate("redirect_limit")

def copy_body(response,stream,start,n,check,progress=None):
    stream.seek(start);left=n;read=0;h=hashlib.sha256()
    while left:
        check();b=response.read(min(65536,left))
        require(bool(b) and len(b)<=left,"body_short_or_oversize")
        read+=len(b)
        if progress is not None:progress[0]=read
        require(stream.write(b)==len(b),"short_write")
        h.update(b);left-=len(b)
    return read,h.hexdigest()

def hash_region(f,start,n,clock):
    f.seek(start);h=hashlib.sha256()
    while n:
        clock.check();b=f.read(min(512*1024,n));require(bool(b),"file_short");h.update(b);n-=len(b)
    return h.hexdigest()

def verify_freeze(path,want,profile,profile_sha):
    require(sha(path)==want,"freeze_sha");d=json_read(path)
    require(d.get("schema")=="q33-turbo-deployment-freeze/v1","freeze_schema")
    require(d.get("candidate_sha256")==sha(__file__) and d.get("profile_sha256")==profile_sha,"frozen_source_or_profile")
    require(sha(profile)==profile_sha,"profile_sha")

def run(args):
    global STOP
    require(os.name=="nt","Windows_execution_required")
    root=Path(args.model_root);state=Path(args.state_dir);notes=Path(args.notes_root)
    require(root.is_absolute() and root.drive.upper()=="E:" and linkfree(root),"E_model_root_only")
    require(state.is_absolute() and linkfree(state) and notes.is_absolute() and linkfree(notes),"state_or_notes_path")
    require(1<=args.max_ranges<=187 and 30<=args.max_seconds<=3600,"run_limits")
    profile_path=Path(args.profile);profile=json_read(profile_path);validate_profile(profile)
    require(sha(profile_path)==args.profile_sha256,"profile_sha")
    verify_freeze(Path(args.freeze),args.freeze_sha256,profile_path,args.profile_sha256)
    state.mkdir(parents=True,exist_ok=True)
    clock=Clock(args.max_seconds,Path(args.stop_file)if args.stop_file else None)
    base=notes/OLD_REL;lock=base.parent/"single-worker.lock"
    require(base.exists() and lock.parent.exists(),"old_accounting_missing")
    target=root/RELATIVE;partial=Path(str(target)+".partial")
    require(linkfree(target) and linkfree(partial),"target_link")
    target.parent.mkdir(parents=True,exist_ok=True)
    session=uuid.uuid4().hex
    result={"schema":"q33-turbo-session/v1","session":session,"profile_sha256":args.profile_sha256,
      "route":"application_direct_https_TUN_unproven","complete":False,"real_network_attempted":False,
      "model_root":"<MODEL_ROOT>","partial_relative_path":RELATIVE+".partial"}
    ledger=None;start_time=time.monotonic()
    old_handler=signal.signal(signal.SIGTERM,lambda *x:globals().__setitem__("STOP",True))
    try:
        with WorkerLock(lock):
            check_base(base);clock.check(True)
            owner=state/"ownership.json"
            identity={"schema":"q33-turbo-owned-partial/v1","profile_sha256":args.profile_sha256,"partial_relative_path":RELATIVE+".partial","asset_bytes":FILE_BYTES,"asset_sha256":FILE_SHA}
            if not owner.exists():
                require(not partial.exists() and not (state/"ledger.jsonl").exists(),"unowned_partial_or_ledger")
                emit_new(owner,identity)
            require(json_read(owner)==identity,"ownership_identity")
            ledger=Ledger(state/"ledger.jsonl",args.profile_sha256)
            if target.exists():
                require(target.stat().st_size==FILE_BYTES,"existing_final_size")
                with target.open("rb")as f:require(hash_region(f,0,FILE_BYTES,clock)==FILE_SHA,"existing_final_sha")
                result["complete"]=True;return result
            require(shutil.disk_usage(root).free>=2*1024**3,"disk_free_below2GiB")
            if not partial.exists():
                require(ledger.reserved==0,"reserved_partial_missing")
                with partial.open("xb")as f:f.truncate(FILE_BYTES);f.flush();os.fsync(f.fileno())
            require(partial.stat().st_size==FILE_BYTES,"partial_size")
            with partial.open("r+b")as f:
                for index,row in sorted(ledger.done.items()):
                    a,n=range_at(index);require(hash_region(f,a,n,clock)==row["sha256"],"resume_chunk_sha")
                completed_now=0
                for index in range((FILE_BYTES+CHUNK-1)//CHUNK):
                    if index in ledger.done:continue
                    if completed_now>=args.max_ranges:result["stop_reason"]="max_ranges";break
                    clock.check(True);require(shutil.disk_usage(root).free>=1024**3,"disk_running_floor")
                    a,n=range_at(index);attempt=ledger.reserve(index);progress=[0];conn=response=None
                    result["real_network_attempted"]=True
                    try:
                        conn,response=open_response(a,n,clock)
                        count,h=copy_body(response,f,a,n,clock.check,progress)
                        f.flush();os.fsync(f.fileno())
                        require(hash_region(f,a,n,clock)==h,"chunk_readback_sha")
                        ledger.finish(attempt,True,h,None,count)
                        completed_now+=1
                        print(json.dumps({"event":"chunk_complete","index":index,"bytes":n,"complete_chunks":len(ledger.done),"total_chunks":(FILE_BYTES+CHUNK-1)//CHUNK}),flush=True)
                    except BaseException as e:
                        ledger.finish(attempt,False,None,type(e).__name__,progress[0]);raise
                    finally:
                        if response is not None:response.close()
                        if conn is not None:conn.close()
                if len(ledger.done)==(FILE_BYTES+CHUNK-1)//CHUNK:
                    require(hash_region(f,0,FILE_BYTES,clock)==FILE_SHA,"whole_file_sha")
                    check_base(base)
                    require(not target.exists(),"final_already_exists")
                    os.link(partial,target)
                    receipt={"schema":"q33-asset-receipt/v1","complete":True,"id":"turbo-v4-step600-ema","bytes":FILE_BYTES,"sha256":FILE_SHA,"relative_path":RELATIVE,"plan_sha256":args.profile_sha256,"profile_sha256":args.profile_sha256,"ledger_sha256":sha(state/"ledger.jsonl"),"base_ledger_sha256":BASE_SHA,"route":"application_direct_https_TUN_unproven"}
                    emit_new(state/("asset-receipt-"+session+".json"),receipt)
                    result["complete"]=True;result["receipt_relative_path"]="asset-receipt-"+session+".json"
    except BaseException as e:
        result["error_type"]=type(e).__name__
        result["reason"]=str(e)if isinstance(e,Gate)else "external_operation_failed_details_omitted"
    finally:
        signal.signal(signal.SIGTERM,old_handler)
        result["elapsed_seconds"]=time.monotonic()-start_time
        if ledger is not None:result["accounting"]=ledger.summary()
        emit_new(state/("session-"+session+".json"),result)
    return result

def main(argv=None):
    ap=argparse.ArgumentParser()
    for name in ("model-root","state-dir","notes-root","profile","profile-sha256","freeze","freeze-sha256"):ap.add_argument("--"+name,required=True)
    ap.add_argument("--stop-file");ap.add_argument("--max-ranges",type=int,default=187);ap.add_argument("--max-seconds",type=int,default=900)
    args=ap.parse_args(argv)
    try:r=run(args)
    except BaseException as e:
        print(json.dumps({"complete":False,"preflight_error":type(e).__name__,"reason":str(e)if isinstance(e,Gate)else "details_omitted"}),flush=True);return 2
    print(json.dumps(r),flush=True);return 0 if r.get("complete")or r.get("stop_reason")=="max_ranges"else 2
if __name__=="__main__":sys.exit(main())
