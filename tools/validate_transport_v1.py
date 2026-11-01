"""Transport monitor: owned group cleanup on SIGTERM, resource stops and partial evidence."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields, group_sample, terminate_owned

GIB=1024**3
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--staging",required=True)
    p.add_argument("--scratch-root",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root=a.output.parent
    root.mkdir(parents=True,exist_ok=True)
    worker=Path(__file__).with_name("transport_worker_v1.py")
    result={"campaign_verdict":"INCONCLUSIVE","worker_sha256":hashlib.sha256(worker.read_bytes()).hexdigest(),
            "resource_limits_not_acceptance_gates":{"rss_bytes":6*GIB,"initial_available_bytes":5*GIB,
            "minimum_available_bytes":2*GIB,"worker_seconds":180,"sample_seconds":0.1}}
    child=None
    try:
        with _termination_guard():
            before=fields(Path("/proc/meminfo"),{"MemTotal","MemAvailable","SwapFree"})
            result["memory_before"]=before
            if before["MemAvailable"]<5*GIB: raise RuntimeError("insufficient memory")
            peaks={"VmRSS":0,"RssAnon":0,"RssFile":0,"RssShmem":0,"members":0}
            started=time.monotonic()
            with (root/"worker.stdout.txt").open("wb") as out,(root/"worker.stderr.txt").open("wb") as err,(root/"samples.jsonl").open("w") as samples:
                child=subprocess.Popen([sys.executable,"-B",str(worker),"--source",a.source,
                    "--staging",a.staging,"--scratch-root",a.scratch_root,"--output",str(root/"worker-result.json")],
                    stdout=out,stderr=err,start_new_session=True,
                    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"))
                while child.poll() is None:
                    current=group_sample(child.pid)
                    available=fields(Path("/proc/meminfo"),{"MemAvailable"})["MemAvailable"]
                    elapsed=time.monotonic()-started
                    samples.write(json.dumps({"elapsed":elapsed,"group":current,"available":available})+"\n")
                    samples.flush()
                    for k,v in current.items(): peaks[k]=max(peaks[k],v)
                    reason="rss" if current["VmRSS"]>6*GIB else "available" if available<2*GIB else "timeout" if elapsed>180 else None
                    if reason:
                        result["guard_stop_reason"]=reason
                        terminate_owned(child)
                        break
                    time.sleep(0.1)
                child.wait(timeout=5)
            result.update(worker_exit_code=child.returncode,sampled_group_peaks=peaks,elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        result["guard_error"]={"type":type(exc).__name__,"message":str(exc)}
    finally:
        if child is not None: terminate_owned(child)
        result["owned_process_reaped"]=child is None or child.poll() is not None
        # Only exact owned temporary directory and one known filename, no recursive deletion.
        try:
            record=root/"worker-result.json"
            if record.exists():
                wr=json.loads(record.read_text())
                name=wr.get("temporary_directory_name","")
                if name:
                    if Path(name).name!=name or not name.startswith("h3-io-"):
                        raise RuntimeError("invalid owned temporary name")
                    scratch=Path(a.scratch_root).resolve()
                    owned=(scratch/name).resolve()
                    if owned.parent!=scratch: raise RuntimeError("temporary path escape")
                    if owned.exists():
                        (owned/"prefix.bin").unlink(missing_ok=True)
                        owned.rmdir()
                    result["ext4_temporary_removed_by_monitor"]=not owned.exists()
        except BaseException as exc:
            result["guard_error"]={"type":type(exc).__name__,"message":str(exc)}
    result["exit_code"]=int(result.get("worker_exit_code")!=0 or "guard_error" in result or "guard_stop_reason" in result)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    return result["exit_code"]
if __name__=="__main__":
    raise SystemExit(main())
