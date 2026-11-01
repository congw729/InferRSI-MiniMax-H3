"""Frozen monitor: absolute deadline, bounded resources, owned worker process group."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields,group_sample,terminate_owned

def main():
    p=argparse.ArgumentParser()
    for key in ("research","static-budget","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--scenario",choices=("adaln","qkv"),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),required=True)
    p.add_argument("--deadline-utc",required=True)
    p.add_argument("--fixture",type=Path);p.add_argument("--fixture-sha256")
    a=p.parse_args();root=a.output.parent;root.mkdir(parents=True,exist_ok=True)
    deadline=datetime.datetime.fromisoformat(a.deadline_utc.replace("Z","+00:00")).timestamp()
    if not datetime.datetime.fromisoformat(a.deadline_utc.replace("Z","+00:00")).tzinfo:
        raise ValueError("timezone required")
    worker=Path(__file__).with_name("fullwidth_projection_worker_v1.py")
    r={"campaign_verdict":"INCONCLUSIVE","deadline_utc":a.deadline_utc,
      "worker_sha256":hashlib.sha256(worker.read_bytes()).hexdigest(),
      "resources":{"group_RSS":6*1024**3,"start_available":5*1024**3,"runtime_available":2*1024**3,
        "worker_seconds":180,"disk_available":8*1024**3,"cuda_fraction":.40,"cuda_peak":4*1024**3,
        "no_new_worker_with_less_than_seconds":190}}
    child=None
    try:
      with _termination_guard():
        if deadline-time.time()<190:raise RuntimeError("not enough time before absolute deadline")
        before=fields(Path("/proc/meminfo"),{"MemAvailable","MemTotal"});r["memory_before"]=before
        if before["MemAvailable"]<5*1024**3 or shutil.disk_usage(root).free<8*1024**3:
            raise RuntimeError("resource preflight")
        cmd=[sys.executable,"-B",str(worker),"--research",str(a.research),"--static-budget",str(a.static_budget),
          "--scenario",a.scenario,"--device",a.device,"--output",str(root/"worker-result.json")]
        if a.fixture:
            if not a.fixture_sha256:raise ValueError("fixture SHA required")
            if a.fixture.stat().st_size>300*1024**2:raise ValueError("fixture file bound")
            cmd.extend(("--fixture",str(a.fixture),"--fixture-sha256",a.fixture_sha256))
        peaks={"VmRSS":0,"RssAnon":0,"RssFile":0,"RssShmem":0,"members":0};start=time.monotonic()
        with (root/"worker.stdout.txt").open("wb") as out,(root/"worker.stderr.txt").open("wb") as err,(root/"samples.jsonl").open("w") as sample:
            child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,
                HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"),start_new_session=True)
            while child.poll() is None:
                now=group_sample(child.pid);available=fields(Path("/proc/meminfo"),{"MemAvailable"})["MemAvailable"]
                elapsed=time.monotonic()-start
                sample.write(json.dumps({"elapsed":elapsed,"group":now,"available":available})+"\n");sample.flush()
                for k,v in now.items():peaks[k]=max(peaks[k],v)
                reason=("rss" if now["VmRSS"]>6*1024**3 else "available" if available<2*1024**3 else
                    "timeout" if elapsed>180 else "absolute_deadline" if time.time()>=deadline-5 else None)
                if reason:r["guard_stop_reason"]=reason;terminate_owned(child);break
                time.sleep(.1)
            child.wait(timeout=5)
        r.update(worker_exit_code=child.returncode,sampled_group_peaks=peaks,elapsed_seconds=time.monotonic()-start)
        if (root/"worker-result.json").exists():
            d=json.loads((root/"worker-result.json").read_text())
            r["worker"]={k:v for k,v in d.items() if k not in ("checks","error","phases")}
    except BaseException as e:r["guard_error"]={"type":type(e).__name__,"message":str(e)}
    finally:
        if child is not None:terminate_owned(child)
        r["owned_process_reaped"]=child is None or child.poll() is not None
    r["exit_code"]=int(r.get("worker_exit_code")!=0 or "guard_error" in r or "guard_stop_reason" in r)
    a.output.write_text(json.dumps(r,indent=2)+"\n")
    return r["exit_code"]
if __name__=="__main__":raise SystemExit(main())
