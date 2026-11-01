"""Run correctness-only child jobs with the external InferRSI executor."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from inferrsi.infra.executors.local import execute_command
import inferrsi.infra.executors.local as executor

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--iteration", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    freeze_path = args.iteration / "plan/tensor-validator-freeze-v1.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    if digest(project / "tools/validate_tensors.py") != freeze["validator_sha256"]:
        raise ValueError("validator no longer matches freeze")
    run = args.iteration / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    mappings = [(str(args.prototype), "<PROTOTYPE_ROOT>"), (str(args.iteration), "<ITERATION_ROOT>"),
                (str(project), "<H3_PROJECT_ROOT>"), (str(project.parent), "<WORKSPACE_ROOT>"),
                (str(args.runtime_python), "<RUNTIME_PYTHON>"), (str(Path.home()), "<USER_HOME>")]
    def sanitize(value):
        if isinstance(value, str):
            for source, label in mappings:
                value = value.replace(source, label).replace(source.replace("\\", "/"), label)
            return value
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        return value
    def save(path, value):
        path.write_text(json.dumps(sanitize(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sources = {path.relative_to(project).as_posix(): digest(path)
               for directory in ("src", "tools") for path in (project / directory).rglob("*.py")}
    save(run / "runspec.json", {
        "schema": "local-validation-runspec/v1", "plane": "infra",
        "repo": "<H3_PROJECT_ROOT>", "revision": {"files_sha256": sources},
        "backend": ["cpu", "cuda"], "device": "one RTX 5070; observed details in child result",
        "model": "synthetic single block, seven H3 NVFP4 projections",
        "scenario": "real tensor staging and isolated upstream loader parity",
        "validator_freeze_sha256": digest(freeze_path), "acceptance_issue": None,
        "inferrsi_version": importlib.metadata.version("inferrsi"),
        "executor_source_sha256": digest(Path(executor.__file__)),
        "performance_metric": None, "campaign_verdict": "INCONCLUSIVE"})
    trials = []
    for label, device, fault in (("cpu", "cpu", None), ("cuda", "cuda", None),
                                 ("fault-swap-qk", "cpu", "swap-qk"),
                                 ("fault-omit-gate-swap", "cpu", "omit-gate-swap")):
        trial = run / label
        command = (str(args.runtime_python), "-B", "-X", "utf8", str(project / "tools/validation_entry.py"),
                   "--prototype", str(args.prototype), "--device", device, "--output", "{result}")
        if fault:
            command += ("--fault", fault)
        execution = execute_command(command, project, trial, time.monotonic() + 120)
        save(trial / "execution.json", execution)
        result = json.loads((trial / "result.json").read_text(encoding="utf-8")) if (trial / "result.json").exists() else {}
        expected_exit = 1 if fault else 0
        observed_rejections = [c["name"] for c in result.get("checks", []) if not c["ok"]]
        matched = execution["returncode"] == expected_exit and "execution_error" not in result
        matched &= (bool(observed_rejections) if fault else result.get("checks_total", 0) > 0)
        trials.append({"label": label, "expected_exit": expected_exit, "observed_exit": execution["returncode"],
                       "expectation_met": matched, "checks_ok": result.get("checks_ok"),
                       "checks_total": result.get("checks_total"), "rejected_checks": observed_rejections,
                       "execution_error": result.get("execution_error")})
    unchanged = all(digest(project / path) == value for path, value in sources.items())
    summary = {"trials": trials, "all_expectations_met": all(t["expectation_met"] for t in trials),
               "source_unchanged": unchanged, "campaign_verdict": "INCONCLUSIVE",
               "full_vllm_import_verified": False, "model_inference_executed": False,
               "independent_review": "not performed; author self-review"}
    save(run / "summary.json", summary)
    save(args.iteration / "results/latest.json", {"run": run.relative_to(args.iteration).as_posix()})
    print(json.dumps(summary))
    return 0 if summary["all_expectations_met"] and unchanged else 1

if __name__ == "__main__":
    raise SystemExit(main())
