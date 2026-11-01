"""Bounded owned-process monitor for the frozen native ConvRot mmap protocol."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields,group_sample,terminate_owned

def main():
    p=argparse.ArgumentParser()
    for k in ("research","fixture","coverage-spec","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--worker",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    a=p.parse_args()
    root=a.output.parent;root.mkdir(parents=True,exist_ok=True)
    worker=Path(__file__).with_name(a.worker)
    if worker.parent!=Path(__file__).parent or not worker.is_file():
        raise ValueError("worker must be an existing local tools file")
    if a.fixture.stat().st_size>64*1024**2 or a.coverage_spec.stat().st_size>4*1024**2:
        raise ValueError('fixture or coverage metadata exceeds bounded protocol')
    r={"campaign_verdict":"INCONCLUSIVE",
       "worker_sha256":hashlib.sha256(worker.read_bytes()).hexdigest(),
       "fixture_sha256":hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
       "coverage_spec_sha256":hashlib.sha256(a.coverage_spec.read_bytes()).hexdigest(),
       "resources":{"RSS":6*1024**3,"start_available":5*1024**3,
                    "runtime_available":2*1024**3,"worker_seconds":180,
                    "disk_available":8*1024**3}}
    child=None
    try:
        with _termination_guard():
            before=fields(Path("/proc/meminfo"),{"MemAvailable","MemTotal"})
            r["memory_before"]=before
            if before["MemAvailable"]<5*1024**3 or shutil.disk_usage(root).free<8*1024**3:
                raise RuntimeError("resource preflight")
            if a.fixture.stat().st_size>64*1024**2 or a.coverage_spec.stat().st_size>4*1024**2:
                raise RuntimeError("fixture or metadata exceeds small protocol budget")
            cmd=[sys.executable,"-B",str(worker),"--research",str(a.research),
                 "--fixture",str(a.fixture),"--coverage-spec",str(a.coverage_spec),
                 "--device",a.device,"--output",str(root/"worker-result.json")]
            peaks={"VmRSS":0,"RssAnon":0,"RssFile":0,"RssShmem":0,"members":0}
            start=time.monotonic()
            with (root/"worker.stdout.txt").open("wb") as out,(root/"worker.stderr.txt").open("wb") as err,(root/"samples.jsonl").open("w") as sample:
                child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,
                    HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"),start_new_session=True)
                while child.poll() is None:
                    now=group_sample(child.pid)
                    available=fields(Path("/proc/meminfo"),{"MemAvailable"})["MemAvailable"]
                    elapsed=time.monotonic()-start
                    sample.write(json.dumps({"elapsed":elapsed,"group":now,"available":available})+"\n");sample.flush()
                    for k,v in now.items():peaks[k]=max(peaks[k],v)
                    reason="rss" if now["VmRSS"]>6*1024**3 else "available" if available<2*1024**3 else "timeout" if elapsed>180 else None
                    if reason:
                        r["guard_stop_reason"]=reason;terminate_owned(child);break
                    time.sleep(.1)
                child.wait(timeout=5)
            r.update(worker_exit_code=child.returncode,sampled_group_peaks=peaks,
                     elapsed_seconds=time.monotonic()-start)
            if (root/"worker-result.json").exists():
                d=json.loads((root/"worker-result.json").read_text())
                r["worker"]={k:v for k,v in d.items() if k not in
                    ("checks","cases","error","recovery_stats","inventory","headers","bindings")}
    except BaseException as e:
        r["guard_error"]={"type":type(e).__name__,"message":str(e)}
    finally:
        if child is not None:terminate_owned(child)
        r["owned_process_reaped"]=child is None or child.poll() is not None
    r["exit_code"]=int(r.get("worker_exit_code")!=0 or "guard_error" in r or "guard_stop_reason" in r)
    a.output.write_text(json.dumps(r,indent=2)+"\n")
    return r["exit_code"]
if __name__=="__main__":raise SystemExit(main())
