"""Frozen Q33 offline boundary checks; no model data and no network."""
import argparse, hashlib, importlib.util, io, json, os, ssl, sys, time, uuid
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location("q33candidate",a.candidate);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    calls=[]
    def no_network(*args,**kwargs):
        calls.append(True);raise AssertionError("network_forbidden")
    m.http.client.HTTPSConnection=no_network
    checks=[]
    def check(name,fn):
        try:
            value=fn();assert value is not False
            checks.append({"id":name,"ok":True})
        except Exception as e:checks.append({"id":name,"ok":False,"error_type":type(e).__name__})
    def rejected(fn):
        try:fn()
        except (m.Gate,ValueError,FileExistsError):return True
        return False
    p=m.fixed_profile()
    check("profile_fixed",lambda:m.validate_profile(p))
    for key,val in [("asset_bytes",1),("asset_sha256","0"*64),("body_cap_bytes",2**31),("global_cap_bytes",129*2**30),("relative_path","../x"),("revision","main")]:
        check("profile_reject_"+key,lambda key=key,val=val:rejected(lambda:m.validate_profile(dict(p,**{key:val}))))
    check("full_chunk",lambda:m.range_at(0)==(0,4194304))
    last=(m.FILE_BYTES-1)//m.CHUNK
    check("last_chunk",lambda:m.range_at(last)==(last*m.CHUNK,m.FILE_BYTES-last*m.CHUNK))
    check("past_end",lambda:rejected(lambda:m.range_at(last+1)))
    check("negative_index",lambda:rejected(lambda:m.range_at(-1)))
    check("budget_exact",lambda:m.check_budget(m.SUBCAP-m.CHUNK,m.CHUNK,m.BASE_RESERVED))
    check("budget_over",lambda:rejected(lambda:m.check_budget(m.SUBCAP-m.CHUNK+1,m.CHUNK,m.BASE_RESERVED)))
    check("global_over",lambda:rejected(lambda:m.check_budget(0,m.CHUNK,m.GLOBALCAP-m.CHUNK+1)))
    check("mirror_first",lambda:bool(m.route(m.MIRROR,0)))
    for i,url in enumerate(["http://hf-mirror.com/x","https://hf-mirror.com:444/x","https://user@hf-mirror.com/x","https://example.invalid/x",m.CANONICAL]):
        check("bad_first_"+str(i),lambda url=url:rejected(lambda:m.route(url,0)))
    check("cdn_allowed",lambda:bool(m.route("https://us.aws.cdn.hf.co/fake?signature=synthetic",1)))
    check("cdn_not_wildcard",lambda:rejected(lambda:m.route("https://sub.us.aws.cdn.hf.co/fake",1)))
    check("tls_checked",lambda:m.tls_context().check_hostname and m.tls_context().verify_mode==ssl.CERT_REQUIRED)
    class R:
        def __init__(self,status=206,headers=None,body=b"12345678"):
            self.status=status;self.headers=headers or {"Content-Length":"8","Content-Range":f"bytes 0-7/{m.FILE_BYTES}"};self.body=io.BytesIO(body);self.reads=0
        def getheader(self,k,default=None):return self.headers.get(k,default)
        def read(self,n):self.reads+=1;return self.body.read(n)
    check("response_exact",lambda:m.check_response(R(),0,8))
    for name,status,headers in [
      ("status200",200,None),("429",429,None),
      ("wrongrange",206,{"Content-Length":"8","Content-Range":f"bytes 1-8/{m.FILE_BYTES}"}),
      ("wronglength",206,{"Content-Length":"9","Content-Range":f"bytes 0-7/{m.FILE_BYTES}"}),
      ("encoded",206,{"Content-Length":"8","Content-Range":f"bytes 0-7/{m.FILE_BYTES}","Content-Encoding":"gzip"}),
      ("chunked",206,{"Content-Length":"8","Content-Range":f"bytes 0-7/{m.FILE_BYTES}","Transfer-Encoding":"chunked"})]:
        r=R(status,headers)
        check("header_reject_"+name,lambda r=r:rejected(lambda:m.check_response(r,0,8)) and r.reads==0)
    def copy_ok():
        f=io.BytesIO();n,h=m.copy_body(R(),f,0,8,lambda:None)
        return n==8 and h==hashlib.sha256(b"12345678").hexdigest() and f.getvalue()==b"12345678"
    check("copy_exact_hash",copy_ok)
    check("short_body",lambda:rejected(lambda:m.copy_body(R(body=b"123"),io.BytesIO(),0,8,lambda:None)))
    def stop():raise m.Gate("stop")
    check("cancel_mid_copy",lambda:rejected(lambda:m.copy_body(R(),io.BytesIO(),0,8,stop)))
    lp=root/"ledger.jsonl";l=m.Ledger(lp,"testprofile")
    for i in range(3):l.reserve(0)
    check("three_attempts_persist",lambda:m.Ledger(lp,"testprofile").counts[0]==3)
    check("fourth_attempt_rejected",lambda:rejected(lambda:m.Ledger(lp,"testprofile").reserve(0)))
    check("unfinished_not_refunded",lambda:m.Ledger(lp,"testprofile").reserved==3*m.CHUNK)
    lp2=root/"ledger2.jsonl";l2=m.Ledger(lp2,"testprofile");idx=l2.reserve(1);l2.finish(idx,True,"a"*64,None,m.CHUNK)
    check("complete_duplicate_refused",lambda:rejected(lambda:m.Ledger(lp2,"testprofile").reserve(1)))
    check("profile_drift_refused",lambda:rejected(lambda:m.Ledger(lp2,"otherprofile")))
    with lp2.open("ab") as f:f.write(b"{")
    check("partial_event_refused",lambda:rejected(lambda:m.Ledger(lp2,"testprofile")))
    lock=root/"lock"
    def locking():
        with m.WorkerLock(lock):
            if not rejected(lambda:m.WorkerLock(lock).__enter__()):return False
        try:
            with m.WorkerLock(lock):raise RuntimeError("synthetic")
        except RuntimeError:pass
        with m.WorkerLock(lock):pass
        return True
    check("lock_exclusion_and_exception_release",locking)
    sf=root/"stop";sf.write_text("stop",encoding="ascii")
    check("stop_before_reserve",lambda:rejected(lambda:m.Clock(30,sf).check(True)))
    check("deadline_before_reserve",lambda:rejected(lambda:m.Clock(0,None).check(True)))
    check("network_calls_zero",lambda:len(calls)==0)
    result={"schema":"q33-turbo-download-offline/v1","candidate_sha256":hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest(),"validator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"checks":checks,"passed":sum(x["ok"]for x in checks),"total":len(checks),"network_calls":len(calls),"limits":"Offline synthetic function/ledger/Windows lock gates only; no real HTTP/TLS/redirect lifecycle, GPU, weights, full-file download or throughput test."}
    (root/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"passed":result["passed"],"total":result["total"],"network_calls":len(calls)}))
    return 0 if all(x["ok"]for x in checks)else 1
if __name__=="__main__":sys.exit(main())
