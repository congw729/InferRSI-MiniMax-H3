"""Use InferRSI for full-module WSL validation; no model downloads or benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from inferrsi.infra.executors.local import execute_command

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--iteration", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--setup-timeout", type=int, default=180)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    freeze_path = args.iteration / "plan/runtime-validator-freeze-v1.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    if sha(project / "tools/validate_runtime.py") != freeze["validator_sha256"]:
        raise ValueError("runtime validator changed after freeze")
    historical = json.loads((args.iteration / "plan/tensor-validator-freeze-v1.json").read_text(encoding="utf-8-sig"))
    if sha(project / "tools/validate_tensors.py") != historical["validator_sha256"]:
        raise ValueError("base validator changed after freeze")
    run = args.iteration / "results" / ("linux-" + args.device + "-" +
                                      datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir()
    sources = {path.relative_to(project).as_posix(): sha(path)
               for folder in ("src", "tools") for path in (project / folder).rglob("*.py")}
    (run / "runspec.json").write_text(json.dumps({
        "schema": "local-validation-runspec/v1", "plane": "infra", "repo": "<H3_PROJECT_ROOT>",
        "revision": {"files_sha256": sources}, "backend": args.device, "device": "RTX 5070 / WSL2",
        "model": "synthetic H3 one-block packed weights", "scenario": "full module import and loader parity",
        "metric": None, "campaign_verdict": "INCONCLUSIVE", "acceptance_issue": None,
        "setup_timeout_seconds": args.setup_timeout, "runtime_validator_freeze_sha256": sha(freeze_path)}, indent=2) + "\n", encoding="utf-8")
    trial = run / "trial"
    command = (sys.executable, "-B", "-X", "utf8", str(project / "tools/wsl_validation_entry.py"),
               "--distro", "H3Lab", "--linux-python", "<LINUX_HOME>/venvs/omni/bin/python",
               "--omni-root", "<LINUX_HOME>/work/vllm-omni", "--prototype", str(args.prototype),
               "--device", args.device, "--output", "{result}", "--timeout-seconds", str(args.setup_timeout))
    execution = execute_command(command, project, trial, time.monotonic() + args.setup_timeout + 30)
    def sanitize(value):
        if isinstance(value, str):
            for source, label in ((str(args.prototype), "<PROTOTYPE_ROOT>"),
                                  (str(args.iteration), "<ITERATION_ROOT>"),
                                  (str(project), "<H3_PROJECT_ROOT>"), (str(project.parent), "<WORKSPACE_ROOT>"),
                                  ("<LINUX_HOME>", "<LINUX_HOME>"), (str(Path.home()), "<USER_HOME>")):
                value = value.replace(source, label)
            return value
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        return value
    (trial / "execution.json").write_text(json.dumps(sanitize(execution), indent=2) + "\n", encoding="utf-8")
    result = json.loads((trial / "result.json").read_text(encoding="utf-8")) if (trial / "result.json").exists() else {}
    print(json.dumps({"run": run.name, "returncode": execution["returncode"],
                      "checks_ok": result.get("checks_ok"), "checks_total": result.get("checks_total"),
                      "execution_error": str(result.get("execution_error"))[-1600:], "campaign_verdict": "INCONCLUSIVE"}))
    return 0 if execution["returncode"] == 0 and result.get("full_runtime_imported") else 1
if __name__ == "__main__":
    raise SystemExit(main())
