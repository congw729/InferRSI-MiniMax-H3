"""Frozen CPU check: SIGTERM of the new direct stage owner reaps its worker."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root=a.output.parent
    root.mkdir(parents=True,exist_ok=True)
    result={"campaign_verdict":"INCONCLUSIVE","checks":[]}
    controller=None
    worker_pid=None
    def check(name,ok):
        result["checks"].append({"name":name,"ok":bool(ok)})
        if not ok:raise AssertionError(name)
    try:
        code=("import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
              "from validate_dlo_temporal_chain_v1 import run_owned; "
              "run_owned([sys.executable,'-c','import time; time.sleep(120)','H3_CHAIN_GUARD_SENTINEL'],Path(sys.argv[2]))")
        with (root/"controller.stdout.txt").open("wb") as stdout,(root/"controller.stderr.txt").open("wb") as stderr:
            controller=subprocess.Popen([sys.executable,"-B","-c",code,str(Path(__file__).parent),str(root/"owned-stage")],
                                        start_new_session=True,stdout=stdout,stderr=stderr)
            until=time.monotonic()+10
            while time.monotonic()<until:
                path=Path(f"/proc/{controller.pid}/task/{controller.pid}/children")
                if path.exists():
                    children=path.read_text().split()
                    if children:
                        worker_pid=int(children[0])
                        cmd=Path(f"/proc/{worker_pid}/cmdline").read_bytes()
                        if b"H3_CHAIN_GUARD_SENTINEL" in cmd:
                            break
                time.sleep(0.05)
            check("owned_worker_observed",worker_pid is not None and b"H3_CHAIN_GUARD_SENTINEL" in Path(f"/proc/{worker_pid}/cmdline").read_bytes())
            os.killpg(controller.pid,signal.SIGTERM)
            controller.wait(timeout=10)
            check("controller_sigterm_exit",controller.returncode==143)
            check("owned_worker_reaped",not Path(f"/proc/{worker_pid}").exists())
            saved=json.loads((root/"owned-stage/process-result.json").read_text())
            check("cleanup_receipt_saved",saved["process_reaped"] and saved["exit_code"]!=0)
        result["exit_code"]=0
    except Exception as exc:
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
        result["exit_code"]=1
    finally:
        if controller is not None and controller.poll() is None:
            os.killpg(controller.pid,signal.SIGTERM)
            controller.wait(timeout=10)
        # Only the exact observed sentinel belongs to this test.
        if worker_pid is not None and Path(f"/proc/{worker_pid}/cmdline").exists():
            if b"H3_CHAIN_GUARD_SENTINEL" in Path(f"/proc/{worker_pid}/cmdline").read_bytes():
                os.killpg(worker_pid,signal.SIGTERM)
    result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    result["checks_total"]=len(result["checks"])
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    return result["exit_code"]


if __name__=="__main__":
    raise SystemExit(main())
