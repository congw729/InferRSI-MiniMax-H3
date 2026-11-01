"""Frozen original dense CPU-only monitor; reuses owned-group monitoring, not capacity claims."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields, group_sample, terminate_owned

GIB = 1024**3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--source-record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.parent
    root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("native_dense_worker_v1.py")
    result = {"campaign_verdict":"INCONCLUSIVE", "worker_sha256":hashlib.sha256(worker.read_bytes()).hexdigest(),
              "resource_limits_not_acceptance_gates": {"rss_bytes":6*GIB,
              "initial_available_bytes":5*GIB, "minimum_available_bytes":2*GIB, "worker_seconds":150},
              "cpu_only":True}
    child = None
    try:
        with _termination_guard():
            before = fields(Path("/proc/meminfo"), {"MemAvailable","MemTotal","SwapFree"})
            result["memory_before"] = before
            if before["MemAvailable"] < 5*GIB:
                raise RuntimeError("insufficient initial memory")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            peaks = {"VmRSS":0,"RssAnon":0,"RssFile":0,"RssShmem":0,"members":0}
            started = time.monotonic()
            with (root/"worker.stdout.txt").open("wb") as stdout, (root/"worker.stderr.txt").open("wb") as stderr, (root/"samples.jsonl").open("w") as samples:
                child = subprocess.Popen([sys.executable,"-B",str(worker),"--research",str(a.research),
                    "--source-record",str(a.source_record),"--output",str(root/"worker-result.json")],
                    env=env, stdout=stdout,stderr=stderr,start_new_session=True)
                while child.poll() is None:
                    current = group_sample(child.pid)
                    available = fields(Path("/proc/meminfo"),{"MemAvailable"})["MemAvailable"]
                    elapsed = time.monotonic()-started
                    samples.write(json.dumps({"elapsed":elapsed,"group":current,"available":available})+"\n")
                    samples.flush()
                    for k,v in current.items():
                        peaks[k] = max(peaks[k],v)
                    reason = ("rss" if current["VmRSS"]>6*GIB else "available" if available<2*GIB else "timeout" if elapsed>150 else None)
                    if reason:
                        result["guard_stop_reason"] = reason
                        terminate_owned(child)
                        break
                    time.sleep(0.1)
                child.wait(timeout=5)
            result.update(worker_exit_code=child.returncode, sampled_group_peaks=peaks,
                          elapsed_seconds=time.monotonic()-started)
            if (root/"worker-result.json").exists():
                data=json.loads((root/"worker-result.json").read_text())
                result["worker"]={k:v for k,v in data.items() if k not in {"checks","error","cases"}}
    except BaseException as exc:
        result["guard_error"] = {"type":type(exc).__name__,"message":str(exc)}
    finally:
        if child is not None:
            terminate_owned(child)
            result["owned_process_reaped"] = child.poll() is not None
        else:
            result["owned_process_reaped"] = True
    result["exit_code"] = int(result.get("worker_exit_code") != 0 or "guard_error" in result or "guard_stop_reason" in result)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    return result["exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
