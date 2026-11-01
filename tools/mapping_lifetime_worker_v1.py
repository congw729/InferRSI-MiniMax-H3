"""CPU-only follow-up: retain versus reopen the same synthetic file mapping."""
import argparse,hashlib,json,mmap,os,resource,time
from pathlib import Path
from h3_omni_adapter.stage_process import _termination_guard
TOTAL=512*1024**2
BLOCK=8*1024**2
GIB=1024**3
def counters():
    r=resource.getrusage(resource.RUSAGE_SELF)
    d={"minor_faults":r.ru_minflt,"major_faults":r.ru_majflt}
    for line in Path("/proc/self/io").read_text().splitlines():
        k,_,v=line.partition(":")
        if k in ("read_bytes","rchar","syscr"): d[k]=int(v)
    return d
def digest(path):
    h=hashlib.sha256()
    with path.open("rb",buffering=0) as f:
        for _ in range(TOTAL//BLOCK):
            data=f.read(BLOCK)
            if len(data)!=BLOCK: raise RuntimeError("short source")
            h.update(data)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result={"campaign_verdict":"INCONCLUSIVE","bytes_per_pass":TOTAL,"block_bytes":BLOCK,
            "cases":[],"note":"hash preparation prewarms; first pass is not cold; process I/O counters are not host NVMe counters"}
    def save():
        pending=a.output.with_suffix(".pending")
        pending.write_text(json.dumps(result,indent=2)+"\n")
        os.replace(pending,a.output)
    try:
        with _termination_guard():
            expected=digest(a.source)
            result["prefix_sha256_before"]=expected
            buffer=bytearray(BLOCK)
            for mode,retained in (("readinto",False),("mmap",False),("readinto",True),("mmap",True)):
                f=None; mm=None
                samples=[]
                try:
                    if retained:
                        f=a.source.open("rb",buffering=0)
                        if mode=="mmap": mm=mmap.mmap(f.fileno(),TOTAL,access=mmap.ACCESS_READ)
                    for repeat in range(4):
                        if not retained:
                            f=a.source.open("rb",buffering=0)
                            if mode=="mmap": mm=mmap.mmap(f.fileno(),TOTAL,access=mmap.ACCESS_READ)
                        f.seek(0)
                        before=counters()
                        start=time.perf_counter()
                        for offset in range(0,TOTAL,BLOCK):
                            if mode=="readinto":
                                if f.readinto(buffer)!=BLOCK: raise RuntimeError("short read")
                            else:
                                view=memoryview(mm)[offset:offset+BLOCK]
                                memoryview(buffer)[:]=view
                                view.release()
                        elapsed=time.perf_counter()-start
                        after=counters()
                        samples.append({"repeat":repeat,"seconds":elapsed,"GiB_s":0.5/elapsed,
                                        "counter_delta":{k:after[k]-before[k] for k in before}})
                        if not retained:
                            if mm is not None: mm.close(); mm=None
                            f.close(); f=None
                finally:
                    if mm is not None: mm.close()
                    if f is not None: f.close()
                result["cases"].append({"mode":mode,"retained":retained,"samples":samples})
                save()
            result["prefix_sha256_after"]=digest(a.source)
            result["source_unchanged"]=result["prefix_sha256_after"]==expected
            result["execution_complete"]=result["source_unchanged"]
    except BaseException as exc:
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
    finally: save()
    return int(not result.get("execution_complete",False))
if __name__=="__main__": raise SystemExit(main())
