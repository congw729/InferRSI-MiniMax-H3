"""Frozen v1: external memory guard and sampler for an owned capacity worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GIB = 1024**3


def fields(path, wanted):
    result = {}
    try:
        for line in path.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key in wanted:
                result[key] = int(value.strip().split()[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return result


def group_sample(pid):
    totals = {"VmRSS": 0, "RssAnon": 0, "RssFile": 0, "RssShmem": 0}
    members = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tail = (entry / "stat").read_text().rpartition(")")[2].split()
            if int(tail[2]) != pid:
                continue
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
        memory = fields(entry / "status", totals)
        for key, value in memory.items():
            totals[key] += value
        members += 1
    return {**totals, "members": members}


def terminate_owned(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.parent
    root.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "resource-guarded full-width synthetic single-block capacity",
              "sample_interval_seconds": 0.1, "resource_limits_not_acceptance_gates": {
                  "group_rss_bytes": 6 * GIB, "minimum_available_bytes": 2 * GIB,
                  "initial_available_bytes": 5 * GIB, "worker_timeout_seconds": 300}}
    worker = Path(__file__).with_name("capacity_worker_v1.py")
    result["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
    before = fields(Path("/proc/meminfo"), {"MemAvailable", "Cached", "SwapFree", "MemTotal"})
    result["memory_before"] = before
    child = None
    stdout_path, stderr_path = root / "worker.stdout.txt", root / "worker.stderr.txt"
    try:
        if before["MemAvailable"] < 5 * GIB:
            raise RuntimeError("insufficient initial host memory headroom")
        started = time.monotonic()
        peaks = {"VmRSS": 0, "RssAnon": 0, "RssFile": 0, "RssShmem": 0, "members": 0}
        minimum_available, maximum_cached = before["MemAvailable"], before["Cached"]
        count = 0
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr, (root / "samples.jsonl").open("w") as samples:
            child = subprocess.Popen([sys.executable, "-B", str(worker), "--research", str(a.research),
                "--output", str(root / "worker-result.json")], stdout=stdout, stderr=stderr, start_new_session=True)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                current = group_sample(child.pid)
                memory = fields(Path("/proc/meminfo"), {"MemAvailable", "Cached", "SwapFree"})
                row = {"elapsed_seconds": elapsed, "owned_process_group": current, "system_memory": memory}
                samples.write(json.dumps(row) + "\n")
                samples.flush()
                count += 1
                for key, value in current.items():
                    peaks[key] = max(peaks[key], value)
                minimum_available = min(minimum_available, memory.get("MemAvailable", minimum_available))
                maximum_cached = max(maximum_cached, memory.get("Cached", maximum_cached))
                reason = ("group RSS limit" if current["VmRSS"] > 6 * GIB else
                          "system available-memory limit" if memory.get("MemAvailable", 0) < 2 * GIB else
                          "worker timeout" if elapsed > 300 else None)
                if reason:
                    result["guard_stop_reason"] = reason
                    terminate_owned(child)
                    break
                time.sleep(0.1)
            child.wait(timeout=5)
        result.update(worker_exit_code=child.returncode, owned_process_reaped=child.poll() is not None,
                      sampled_group_peaks=peaks, minimum_system_available_bytes=minimum_available,
                      maximum_system_cached_bytes=maximum_cached, sample_count=count,
                      elapsed_seconds=time.monotonic() - started)
        if (root / "worker-result.json").exists():
            worker_result = json.loads((root / "worker-result.json").read_text())
            result["worker"] = {k: v for k, v in worker_result.items()
                                if k not in {"architecture", "source_header_provenance", "traceback", "checks"}}
    except Exception as exc:
        result["guard_error"] = str(exc)
    finally:
        if child is not None:
            terminate_owned(child)
        result["memory_after"] = fields(Path("/proc/meminfo"), {"MemAvailable", "Cached", "SwapFree", "MemTotal"})
        for path in (stdout_path, stderr_path):
            if path.exists():
                text = path.read_text(errors="replace")
                for source, label in ((str(a.research), "<RESEARCH_ROOT>"), (str(root), "<RUN_ROOT>"),
                                      (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"),
                                      (str(Path.home()), "<LINUX_HOME>")):
                    text = text.replace(source, label)
                path.write_text(text)
    code = int(result.get("worker_exit_code") != 0 or "guard_stop_reason" in result or "guard_error" in result)
    result["exit_code"] = code
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
