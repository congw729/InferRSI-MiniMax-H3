"""Frozen v1: 100ms owned-group monitor for one complete 8-layer compact decoder capacity case."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields, group_sample, terminate_owned

GIB = 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.parent
    root.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "guarded complete 107-frame temporal decoder capacity",
        "sample_interval_seconds": 0.1, "resource_limits_not_acceptance_gates": {
            "group_rss_bytes": 6*GIB, "minimum_available_bytes": 2*GIB, "initial_available_bytes": 5*GIB,
            "worker_timeout_seconds": 180, "torch_allocator_fraction": 0.40, "phase_peak_allocation_bytes": 4*GIB}}
    worker = Path(__file__).with_name("temporal_capacity_worker_v1.py")
    result["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
    before = fields(Path("/proc/meminfo"), {"MemAvailable","Cached","SwapFree","MemTotal"})
    result["memory_before"] = before
    child = None
    monitor_code = None
    stdout_path, stderr_path = root/"worker.stdout.txt", root/"worker.stderr.txt"
    try:
        with _termination_guard():
            if before["MemAvailable"] < 5*GIB:
                raise RuntimeError("insufficient initial host memory headroom")
            started = time.monotonic()
            peaks = {"VmRSS":0,"RssAnon":0,"RssFile":0,"RssShmem":0,"members":0}
            minimum_available, minimum_swap = before["MemAvailable"], before["SwapFree"]
            count = 0
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr, (root/"samples.jsonl").open("w") as samples:
                command = [sys.executable,"-B",str(worker),"--research",str(a.research),
                           "--native-source",str(a.native_source),
                           "--output",str(root/"worker-result.json")]
                if a.fixture is not None:
                    command.extend(["--fixture",str(a.fixture)])
                child = subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
                while child.poll() is None:
                    elapsed = time.monotonic()-started
                    current = group_sample(child.pid)
                    memory = fields(Path("/proc/meminfo"), {"MemAvailable","Cached","SwapFree"})
                    samples.write(json.dumps({"elapsed_seconds":elapsed,"owned_process_group":current,
                                               "system_memory":memory})+"\n")
                    samples.flush()
                    count += 1
                    for key,value in current.items():
                        peaks[key] = max(peaks[key],value)
                    minimum_available = min(minimum_available,memory.get("MemAvailable",minimum_available))
                    minimum_swap = min(minimum_swap,memory.get("SwapFree",minimum_swap))
                    reason = ("group RSS limit" if current["VmRSS"] > 6*GIB else
                              "system available-memory limit" if memory.get("MemAvailable",0) < 2*GIB else
                              "worker timeout" if elapsed > 180 else None)
                    if reason:
                        result["guard_stop_reason"] = reason
                        terminate_owned(child)
                        break
                    time.sleep(0.1)
                child.wait(timeout=5)
            result.update(worker_exit_code=child.returncode,owned_process_reaped=child.poll() is not None,
                sampled_group_peaks=peaks,minimum_system_available_bytes=minimum_available,
                minimum_swap_free_bytes=minimum_swap,sample_count=count,elapsed_seconds=time.monotonic()-started)
            if (root/"worker-result.json").exists():
                data = json.loads((root/"worker-result.json").read_text())
                result["worker"] = {k:v for k,v in data.items() if k not in {"traceback","checks"}}
    except BaseException as exc:
        result["guard_error"] = {"type":type(exc).__name__,"message":str(exc)}
        monitor_code = int(exc.code) if isinstance(exc,SystemExit) and isinstance(exc.code,int) else 1
    finally:
        if child is not None:
            terminate_owned(child)
            result["owned_process_reaped"] = child.poll() is not None
        result["memory_after"] = fields(Path("/proc/meminfo"), {"MemAvailable","Cached","SwapFree","MemTotal"})
        for path in (stdout_path,stderr_path):
            if path.exists():
                text = path.read_text(errors="replace")
                for source,label in ((str(root),"<RUN_ROOT>"),(str(a.research),"<RESEARCH_ROOT>"),
                    (str(a.native_source),"<NATIVE_SOURCE>"),(str(Path.home()),"<LINUX_HOME>"),
                    (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
                    text = text.replace(source,label)
                if a.fixture is not None:
                    text = text.replace(str(a.fixture),"<REUSED_FIXTURE>")
                path.write_text(text)
    code = monitor_code or int(result.get("worker_exit_code") != 0 or "guard_stop_reason" in result or "guard_error" in result)
    result["exit_code"] = code
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
