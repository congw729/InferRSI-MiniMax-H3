"""Frozen v3 direct owner for two real-width synthetic DiT blocks."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from validate_dlo_temporal_chain_v1 import run_owned


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--cache-manifest-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root=a.output.parent
    root.mkdir(parents=True,exist_ok=True)
    result=dict(campaign_verdict="INCONCLUSIVE",mode="two-blocks-instrumented")
    code=1
    try:
        record=run_owned([sys.executable,"-B",str(Path(__file__).with_name("dlo_fullwidth_worker_v3.py")),
            "--fixture",str(a.fixture),"--cache",str(a.cache),
            "--cache-manifest-sha",a.cache_manifest_sha],root/"worker")
        result["monitor"]=record
        path=root/"worker/worker-result.json"
        if path.exists():
            result["worker"]=json.loads(path.read_text())
        code=int(record.get("exit_code")!=0 or not record.get("process_reaped")
            or bool(record.get("guard_stop_reason")))
    except BaseException as exc:
        result["execution_error"]={"type":type(exc).__name__,"message":str(exc)}
        result["traceback"]=traceback.format_exc()
        code=int(exc.code) if isinstance(exc,SystemExit) and isinstance(exc.code,int) else 1
    result["exit_code"]=code
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json",".jsonl",".txt"}:
            text=path.read_text()
            for source,label in ((str(a.fixture),"<REUSED_FIXTURE>"),(str(a.cache),"<REUSED_CANONICAL_CACHE>"),
                (str(root),"<RUN_ROOT>"),(str(Path.home()),"<LINUX_HOME>"),
                (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
                text=text.replace(source,label)
            path.write_text(text)
    print(json.dumps({"mode":"two-blocks-instrumented","exit_code":code,"checks":result.get("worker",{}).get("checks_ok"),
                      "guard":result.get("monitor",{}).get("guard_stop_reason")}))
    return code


if __name__=="__main__":
    raise SystemExit(main())
