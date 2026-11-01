"""Linux serial stage ownership: reap each child before launching the next."""
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _termination_guard():
    # Used on the main thread so a controller SIGTERM takes the same owned-child
    # cleanup path as an interrupted communicate(). Restore the caller's handler.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_serial_stages(stages, root, *, timeout_seconds):
    root = Path(root)
    if root.exists():
        raise FileExistsError("use a fresh stage run directory")
    root.mkdir(parents=True)
    records = []
    previous = None
    for stage, command in stages:
        if stage not in {"encode", "denoise", "decode"} or any(s["stage"] == stage for s in records):
            raise ValueError("invalid or duplicate stage")
        work = root / stage
        work.mkdir()
        argv = [*command, "--work", str(work)]
        if previous is not None:
            argv += ["--input", str(previous / "handoff")]
        started = time.time()
        with _termination_guard():
            child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            timed_out = False
            try:
                stdout, stderr = child.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    stdout, stderr = child.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    stdout, stderr = child.communicate(timeout=5)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.communicate(timeout=5)
                raise
        reaped = child.poll() is not None
        ended = time.time()
        def redact(data, command=command):
            value = data.decode("utf-8", errors="replace")
            mappings = [(str(root), "<STAGE_RUN>"), (str(Path.home()), "<LINUX_HOME>"),
                        (str(Path(__file__).resolve().parents[2]), "<H3_PROJECT_ROOT>")]
            if "--checkpoint" in command:
                mappings.insert(0, (str(command[command.index("--checkpoint") + 1]), "<SYNTHETIC_CHECKPOINT>"))
            for source, label in mappings:
                value = value.replace(source, label)
            return value
        record = {"stage": stage, "started_at": started, "reaped_at": ended,
                  "exit_code": child.returncode, "timed_out": timed_out,
                  "process_reaped": reaped, "elapsed_seconds": ended - started}
        (work / "process-result.json").write_text(json.dumps(record, indent=2) + "\n")
        (work / "stdout.txt").write_text(redact(stdout))
        (work / "stderr.txt").write_text(redact(stderr))
        records.append(record)
        if child.returncode or timed_out:
            break
        if not (work / "handoff/manifest.json").is_file():
            record["handoff_missing"] = True
            break
        previous = work
    return records
