"""v2 mapping lifetime follow-up; unavailable process I/O telemetry remains null."""
import argparse,hashlib,json,mmap,os,resource,time
from pathlib import Path
from h3_omni_adapter.stage_process import _termination_guard
TOTAL=512*1024**2
BLOCK=8*1024**2
def counters():
    r=resource.getrusage(resource.RUSAGE_SELF)
    d={"minor_faults":r.ru_minflt,"major_faults":r.ru_majflt,"read_bytes":None,"rchar":None,"syscr":None}
    try:
        lines=Path("/proc/self/io").read_text().splitlines()
    except OSError:
        return d
    for line in lines:
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
    result={"campaign_verdict":"INCONCLUSIVE","bytes_per_pass":TOTAL,"block_bytes":BLOCK,"cases":[],
       "note":"hash prewarms; first pass is not cold; unavailable /proc/self/io fields are null, not zero"}
    def save():
        pending=a.output.with_suffix(".pending")
        pending.write_text(json.dumps(result,indent=2)+"\n")
        os.replace(pending,a.output)
    try:
        with _termination_guard():
            expected=digest(a.source)
            result["prefix_sha256_before"]=expected
            result["proc_io_available"]=counters()["read_bytes"] is not None
            buffer=bytearray(BLOCK)
            cases=(("readinto_reopen",False,False,False),("mmap_reopen",True,False,False),
                   ("readinto_retained",False,True,False),("mmap_fd_retained",True,True,False),
                   ("mmap_retained",True,True,True))
            for label,use_map,keep_fd,keep_map in cases:
                f=None; mm=None; samples=[]
                initial=time.perf_counter()
                try:
                    if keep_fd:
                        f=a.source.open("rb",buffering=0)
                        if keep_map: mm=mmap.mmap(f.fileno(),TOTAL,access=mmap.ACCESS_READ)
                    initial_setup=time.perf_counter()-initial
                    for repeat in range(4):
                        start_setup=time.perf_counter()
                        if not keep_fd: f=a.source.open("rb",buffering=0)
                        if use_map and not keep_map: mm=mmap.mmap(f.fileno(),TOTAL,access=mmap.ACCESS_READ)
                        f.seek(0)
                        setup_seconds=time.perf_counter()-start_setup
                        before=counters()
                        start=time.perf_counter()
                        for offset in range(0,TOTAL,BLOCK):
                            if not use_map:
                                if f.readinto(buffer)!=BLOCK: raise RuntimeError("short read")
                            else:
                                view=memoryview(mm)[offset:offset+BLOCK]
                                memoryview(buffer)[:]=view
                                view.release()
                        elapsed=time.perf_counter()-start
                        after=counters()
                        close_start=time.perf_counter()
                        if use_map and not keep_map: mm.close(); mm=None
                        if not keep_fd: f.close(); f=None
                        teardown_seconds=time.perf_counter()-close_start
                        samples.append({"repeat":repeat,"seconds":elapsed,"GiB_s":0.5/elapsed,
                            "setup_seconds":setup_seconds,"teardown_seconds":teardown_seconds,
                            "counter_delta":{k:after[k]-before[k] if before[k] is not None and after[k] is not None else None for k in before}})
                finally:
                    close_start=time.perf_counter()
                    if mm is not None: mm.close()
                    if f is not None: f.close()
                    final_teardown=time.perf_counter()-close_start
                result["cases"].append({"mode":label,"initial_setup_seconds":initial_setup,
                    "final_teardown_seconds":final_teardown,"samples":samples})
                save()
            result["prefix_sha256_after"]=digest(a.source)
            result["source_unchanged"]=result["prefix_sha256_after"]==expected
            result["execution_complete"]=result["source_unchanged"]
    except BaseException as exc:
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
    finally: save()
    return int(not result.get("execution_complete",False))
if __name__=="__main__": raise SystemExit(main())
