"""Bounded WSL child entry; outputs stay in the canonical Windows evidence tree."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from capture_command import decode

def linux_path(path):
    value = str(path.resolve()).replace("\\", "/")
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("expected a Windows drive path")
    return "/mnt/" + value[0].lower() + value[2:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distro", required=True)
    parser.add_argument("--linux-python", required=True)
    parser.add_argument("--omni-root", required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    command = ["wsl.exe", "-d", args.distro, "-u", "<LINUX_USER>", "--exec", "env",
               "PYTHONDONTWRITEBYTECODE=1", "PYTHONUTF8=1",
               "PYTHONPATH=" + linux_path(project / "src"),
               "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "MAX_JOBS=2",
               "CUDA_HOME=<LINUX_HOME>/toolchains/cu130/lib/python3.12/site-packages/nvidia/cu13",
               "LIBRARY_PATH=<LINUX_HOME>/toolchains/cu130/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib",
               "PATH=<LINUX_HOME>/venvs/omni/bin:<LINUX_HOME>/toolchains/cu130/lib/python3.12/site-packages/nvidia/cu13/bin:/usr/local/bin:/usr/bin:/bin",
               "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", str(args.timeout_seconds) + "s",
               args.linux_python, "-B", linux_path(project / "tools/validate_runtime.py"),
               "--omni-root", args.omni_root, "--prototype", linux_path(args.prototype),
               "--output", linux_path(args.output), "--device", args.device]
    result = subprocess.run(command, capture_output=True)
    def redact(text):
        for source, label in ((linux_path(args.prototype), "<PROTOTYPE_ROOT>"),
                              (linux_path(project), "<H3_PROJECT_ROOT>"),
                              (args.omni_root, "<OMNI_ROOT>"),
                              (str(Path(args.linux_python).parent.parent), "<LINUX_ENV>"),
                              ("<LINUX_HOME>", "<LINUX_HOME>"), (str(Path.home()), "<USER_HOME>")):
            text = text.replace(source, label)
        return text
    print(redact(decode(result.stdout)), end="")
    print(redact(decode(result.stderr)), end="", file=sys.stderr)
    if not args.output.exists():
        args.output.write_text(json.dumps({"campaign_verdict": "INCONCLUSIVE",
            "execution_error": {"type": "ChildDidNotComplete", "exit_code": result.returncode}}, indent=2) + "\n",
            encoding="utf-8")
    return result.returncode
if __name__ == "__main__":
    raise SystemExit(main())
