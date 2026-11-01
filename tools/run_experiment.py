"""Bounded isolated validation using the external InferRSI executor."""
import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from inferrsi.infra.executors.local import execute_command


def lp(path):
    value = str(path.resolve()).replace("\\", "/")
    return "/mnt/" + value[0].lower() + value[2:]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record-root", type=Path, required=True)
    p.add_argument("--iteration", type=Path, required=True)
    p.add_argument("--linux-home", required=True)
    p.add_argument("--lab-root", type=Path, required=True)
    p.add_argument("--wsl-distro", default="H3Lab")
    p.add_argument("--script", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    script = project / "tools" / a.script
    if script.parent != project / "tools" or not script.is_file():
        raise ValueError("script must be a local tools filename")
    run = a.iteration / "results" / (a.name + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir(parents=True)
    trial = run / "trial"
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {path.relative_to(project).as_posix(): sha(path)
               for folder in ("src", "tools") for path in (project / folder).rglob("*.py")}
    (run / "runspec.json").write_text(json.dumps({"schema": "local-validation-runspec/v1",
        "plane": "infra", "backend": "WSL2 / torch / upstream vLLM-Omni", "device": "RTX 5070",
        "model": "saved header metadata or bounded synthetic weights", "scenario": a.name,
        "acceptance_issue": None, "campaign_verdict": "INCONCLUSIVE",
        "source_hashes": sources, "timeout_seconds": a.timeout,
        "validator_sha256": sha(script)}, indent=2) + "\n")
    extra = a.args[1:] if a.args[:1] == ["--"] else a.args
    args = [lp(Path(x)) if len(x) >= 3 and x[1:3] in (":\\", ":/") else x for x in extra]
    linux_home = a.linux_home.rstrip("/")
    if not linux_home.startswith("/") or len(linux_home) < 2:
        raise ValueError("absolute isolated Linux home required")
    cuda = linux_home + "/toolchains/cu130/lib/python3.12/site-packages/nvidia/cu13"
    child = ["wsl.exe", "-d", a.wsl_distro, "-u", Path(linux_home).name, "--exec", "env",
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
        "VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1", "MAX_JOBS=2",
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + lp(project / "src"),
        "CUDA_HOME=" + cuda, "LIBRARY_PATH=" + cuda + "/lib:/usr/lib/wsl/lib",
        "PATH=" + linux_home + "/venvs/omni/bin:" + cuda + "/bin:/usr/local/bin:/usr/bin:/bin",
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", str(a.timeout) + "s",
        linux_home + "/venvs/omni/bin/python", "-B", lp(script), *args, "--output", lp(trial / "result.json")]
    command = [sys.executable, "-B", "-X", "utf8", str(project / "tools/capture_command.py"),
        "--output", str(trial / "capture.json"), "--record-root", str(a.record_root),
        "--lab-root", str(a.lab_root), "--linux-home", linux_home, "--", *child]
    execution = execute_command(tuple(command), project, trial, time.monotonic() + a.timeout + 20)
    text = json.dumps(execution, indent=2)
    for source, label in ((str(a.record_root), "<RECORD_ROOT>"), (str(project), "<H3_PROJECT_ROOT>"),
                          (str(project.parent), "<WORKSPACE_ROOT>"), (str(Path.home()), "<USER_HOME>")):
        text = text.replace(source.replace("\\", "\\\\"), label).replace(source, label)
    (trial / "execution.json").write_text(text + "\n")
    result = json.loads((trial / "result.json").read_text()) if (trial / "result.json").exists() else {}
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items()
                    if k not in {"inventory", "headers", "traceback", "bindings", "checks", "defaults"}}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    print(json.dumps({"run": run.name, "exit_code": execution["returncode"], "summary": compact(result)}))
    return int(execution["returncode"] != 0)
if __name__ == "__main__":
    raise SystemExit(main())


