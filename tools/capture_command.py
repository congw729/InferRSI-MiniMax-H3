"""Capture one authorized setup command, redacting local paths before persistence."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

def decode(data):
    # WSL may prefix UTF-16 Windows warnings to UTF-8 Linux stderr.
    parts = []
    while data:
        sample = data[:80]
        if data.startswith(b"\xff\xfe") or sample.count(b"\0") > len(sample) // 8:
            end = data.find(b"\n\0")
            count = end + 2 if end >= 0 else len(data)
            parts.append(data[:count].decode("utf-16-le", errors="replace").lstrip("\ufeff"))
            data = data[count:]
        else:
            parts.append(data.decode("utf-8", errors="replace"))
            break
    return "".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--linux-home", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    project = Path(__file__).resolve().parents[1]
    mappings = [(str(args.record_root.resolve()), "<RECORD_ROOT>"),
                (str(project), "<H3_PROJECT_ROOT>"),
                (str(project.parent), "<WORKSPACE_ROOT>"),
                (str(args.lab_root.resolve()), "<LAB_ROOT>"),
                (str(Path.home()), "<USER_HOME>")]
    if args.linux_home:
        mappings.append((args.linux_home, "<LINUX_HOME>"))
    for source, target in list(mappings):
        if len(source) > 2 and source[1] == ":":
            mappings.append(("/mnt/" + source[0].lower() + source[2:].replace("\\", "/"), target))
    mappings.sort(key=lambda item: len(item[0]), reverse=True)
    def redact(text):
        for source, target in mappings:
            text = re.sub(re.escape(source), lambda match: target, text, flags=re.I)
            text = re.sub(re.escape(source.replace("\\", "/")), lambda match: target, text, flags=re.I)
        return text
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(args.lab_root / "cache" / "uv"), UV_NO_CONFIG="1",
               PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE="1",
               PYTHONUTF8="1", TMP=str(args.lab_root / "tmp"), TEMP=str(args.lab_root / "tmp"))
    started = time.time()
    result = subprocess.run(command, capture_output=True, env=env)
    record = {"argv": [redact(x) for x in command], "exit_code": result.returncode,
              "elapsed_seconds": time.time() - started, "stdout": redact(decode(result.stdout)),
              "stderr": redact(decode(result.stderr))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("capture output already exists; choose a new attempt")
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"exit_code": result.returncode, "stdout_tail": record["stdout"][-2500:],
                      "stderr_tail": record["stderr"][-3500:]}, ensure_ascii=False))
    return result.returncode
if __name__ == "__main__":
    raise SystemExit(main())
