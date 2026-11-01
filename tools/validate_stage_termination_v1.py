"""Frozen v1: external SIGTERM cleanup of only the stage controller's owned group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def live(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("test", "controller", "worker", "sentinel"), default="test")
    p.add_argument("--output", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--work", type=Path)
    p.add_argument("--input", type=Path)
    a = p.parse_args()
    if a.role == "worker":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
        (a.root / "owned.json").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(), "descendant": child.pid}))
        child.wait(timeout=20)
        return 0
    if a.role == "sentinel":
        (a.root / "unexpected-next-stage").write_text("started")
        return 0
    if a.role == "controller":
        from h3_omni_adapter.stage_process import run_serial_stages
        commands = [(stage, [sys.executable, "-B", __file__, "--role", role, "--root", str(a.root)])
                    for stage, role in (("encode", "worker"), ("decode", "sentinel"))]
        run_serial_stages(commands, a.root / "stages", timeout_seconds=30)
        return 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    root = a.output.parent / "termination"
    root.mkdir()
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "owned-stage outer termination only",
              "checks": []}
    owned = None
    controller = subprocess.Popen([sys.executable, "-B", __file__, "--role", "controller",
                                   "--root", str(root)], start_new_session=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    code = 1
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
    try:
        deadline = time.monotonic() + 10
        while not (root / "owned.json").exists() and time.monotonic() < deadline:
            if controller.poll() is not None:
                raise RuntimeError("controller exited before ready")
            time.sleep(0.05)
        owned = json.loads((root / "owned.json").read_text())
        if owned["pid"] != owned["pgid"] or os.getpgid(owned["descendant"]) != owned["pgid"]:
            raise RuntimeError("ownership handshake mismatch")
        controller.send_signal(signal.SIGTERM)
        out, err = controller.communicate(timeout=7)
        deadline = time.monotonic() + 2
        while any(live(owned[n]) for n in ("pid", "descendant")) and time.monotonic() < deadline:
            time.sleep(0.05)
        check("controller_reaped", controller.poll() is not None)
        check("owned_worker_not_running", not live(owned["pid"]))
        check("owned_descendant_not_running", not live(owned["descendant"]))
        check("downstream_not_launched", not (root / "unexpected-next-stage").exists())
        check("termination_status_preserved", controller.returncode in (143, -15))
        result["controller_exit_code"] = controller.returncode
        result["checks_total"] = len(result["checks"])
        result["checks_ok"] = sum(c["ok"] for c in result["checks"])
        result["scope_limit"] = "not-running includes zombies; unrelated processes never signaled"
        result["stderr"] = err.decode(errors="replace").replace(str(Path.home()), "<LINUX_HOME>").replace(
            str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>").replace(str(root), "<RUN_ROOT>")
        code = int(result["checks_ok"] != result["checks_total"])
    finally:
        if owned is not None and any(live(owned[n]) for n in ("pid", "descendant")):
            try:
                os.killpg(owned["pgid"], signal.SIGKILL)
                result["supervisor_emergency_cleanup_used"] = True
            except ProcessLookupError:
                pass
        if controller.poll() is None:
            os.killpg(controller.pid, signal.SIGKILL)
            controller.communicate(timeout=5)
        a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
