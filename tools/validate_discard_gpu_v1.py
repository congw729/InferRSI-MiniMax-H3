"""Frozen owner: separate native and terminal-discard tiny workers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from validate_dlo_temporal_chain_v1 import run_owned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output.parent
    if shutil.disk_usage(root).free < 8 * 1024**3:
        raise RuntimeError("disk headroom")
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "tiny only", "runs": []}
    for mode in ("native", "discard"):
        work = root / mode
        command = [sys.executable, "-B", str(Path(__file__).with_name("discard_gpu_worker_v1.py")),
                   "--fixture", str(a.fixture), "--cache", str(a.cache), "--mode", mode]
        monitor = run_owned(command, work)
        record = {"mode": mode, "monitor": monitor}
        if (work / "worker-result.json").is_file():
            record["worker"] = json.loads((work / "worker-result.json").read_text())
        result["runs"].append(record)
        if monitor["exit_code"] or monitor.get("guard_stop_reason"):
            break
    code = int(len(result["runs"]) != 2 or any(r["monitor"]["exit_code"] or
               r["monitor"].get("guard_stop_reason") or not r["monitor"]["process_reaped"]
               for r in result["runs"]))
    result["exit_code"] = code
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exit_code": code, "runs": [
        {"mode": r["mode"], "checks": r.get("worker", {}).get("checks_ok"),
         "error": r.get("worker", {}).get("execution_error"), "monitor": r["monitor"]}
        for r in result["runs"]]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
