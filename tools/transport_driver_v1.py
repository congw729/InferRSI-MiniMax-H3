"""InferRSI-controlled Windows + isolated Linux diagnostics; exact owned staging cleanup."""
import argparse, hashlib, json, subprocess, sys, time, uuid
from pathlib import Path
from inferrsi.infra.executors.local import execute_command

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--record-root",type=Path,required=True)
    p.add_argument("--iteration",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--lab-root",type=Path,required=True)
    p.add_argument("--linux-home",required=True)
    a=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    scratch=(a.lab_root/"tmp").resolve()
    if not scratch.is_dir(): raise RuntimeError("existing lab tmp required")
    staging=scratch/("h3-transport-"+uuid.uuid4().hex+".bin")
    if staging.exists(): raise FileExistsError("staging collision")
    trial=a.iteration/"results"/("windows-"+uuid.uuid4().hex[:10])
    receipt={"campaign_verdict":"INCONCLUSIVE","staging_file_name":staging.name,
             "windows_timeout_seconds":240,"linux_outer_seconds":240}
    def redact(value):
        text=json.dumps(value,indent=2)
        for old,new in ((str(a.record_root),"<RECORD_ROOT>"),(str(project),"<H3_PROJECT_ROOT>"),
                        (str(project.parent),"<WORKSPACE_ROOT>"),(str(a.lab_root),"<LAB_ROOT>"),
                        (a.linux_home,"<LINUX_HOME>"),(str(Path.home()),"<USER_HOME>")):
            text=text.replace(json.dumps(old)[1:-1],new).replace(old,new)
        return text
    try:
        command=(sys.executable,"-B",str(project/"tools/windows_transport_v1.py"),
                 "--source",str(a.source),"--staging",str(staging),"--output",str(trial/"result.json"))
        execution=execute_command(command,project,trial,time.monotonic()+240)
        (trial/"execution.json").write_text(redact(execution)+"\n")
        receipt["windows_exit_code"]=execution["returncode"]
        if execution["returncode"]!=0: raise RuntimeError("Windows probe incomplete")
        cmd=[sys.executable,"-B",str(project/"tools/run_experiment.py"),
             "--record-root",str(a.record_root),"--iteration",str(a.iteration),"--linux-home",a.linux_home,
             "--lab-root",str(a.lab_root),"--script","validate_transport_v1.py","--name","transport-linux-v1",
             "--timeout","240","--","--source",str(a.source),"--staging",str(staging),
             "--scratch-root",a.linux_home.rstrip("/")+"/io-staging"]
        child=subprocess.run(cmd,capture_output=True,timeout=290)
        receipt["linux_launcher_exit_code"]=child.returncode
        (a.iteration/"results/launcher-output.json").write_text(redact({
            "stdout":child.stdout.decode("utf-8",errors="replace"),
            "stderr":child.stderr.decode("utf-8",errors="replace")})+"\n")
        receipt["execution_complete"]=child.returncode==0
    except BaseException as exc:
        receipt["error"]={"type":type(exc).__name__,"message":str(exc)}
    finally:
        try:
            if staging.resolve().parent!=scratch or not staging.name.startswith("h3-transport-"):
                raise RuntimeError("staging ownership mismatch")
            staging.unlink(missing_ok=True)
            receipt["E_temporary_removed"]=not staging.exists()
        except BaseException as exc:
            receipt["cleanup_error"]={"type":type(exc).__name__,"message":str(exc)}
            receipt["execution_complete"]=False
        (a.iteration/"results/driver-receipt.json").write_text(redact(receipt)+"\n")
    print(json.dumps(receipt))
    return int(not receipt.get("execution_complete",False))
if __name__=="__main__": raise SystemExit(main())
