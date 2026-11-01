"""Checkpoint local DLO evidence without changing frozen validation sources."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--iteration",type=Path,required=True)
    p.add_argument("--record-root",type=Path,required=True)
    p.add_argument("--linux-home",required=True)
    p.add_argument("--lab-root",type=Path,required=True)
    a=p.parse_args()
    root,iteration=a.record_root.resolve(),a.iteration.resolve()
    project=Path(__file__).resolve().parents[1]
    if not iteration.is_relative_to(root):
        raise ValueError("iteration outside record root")
    replacements=[]
    for path,label in ((project,"<H3_PROJECT_ROOT>"),(root,"<RECORD_ROOT>"),
                       (project.parent,"<WORKSPACE_ROOT>"),(a.lab_root,"<LAB_ROOT>"),
                       (Path.home(),"<USER_HOME>")):
        native=str(path)
        variants=[native,native.replace("\\","/")]
        if len(native)>2 and native[1]==":":
            variants.append("/mnt/"+native[0].lower()+native[2:].replace("\\","/"))
        for variant in variants:
            replacements.extend(((variant,label),(json.dumps(variant)[1:-1],label)))
    replacements.append((a.linux_home,"<LINUX_HOME>"))
    replacements.sort(key=lambda row:len(row[0]),reverse=True)
    redacted=[]
    for area in ("results","logs"):
        for path in sorted((iteration/area).rglob("*")):
            if not path.is_file() or path.suffix not in {".json",".jsonl",".txt",".log"}:
                continue
            before=path.read_text(encoding="utf-8")
            after=before
            for source,label in replacements:
                after=after.replace(source,label)
            if before!=after:
                path.write_text(after,encoding="utf-8")
                redacted.append(path.relative_to(iteration).as_posix())
    intact={}
    for path in (iteration/"plan").glob("*freeze*.json"):
        frozen=json.loads(path.read_text())
        for name,digest in frozen["validator_sha256"].items():
            intact[name]=sha(project/"tools"/name)==digest==sha(iteration/"validator"/name)
        gate=("BRIDGE-GATES.zh-CN.txt" if path.name.startswith("bridge") else
              "TRANSPORT-GATES.zh-CN.txt" if path.name.startswith("transport") else "FROZEN-GATES.zh-CN.txt")
        intact[path.name+":gates"]=sha(iteration/"plan"/gate)==frozen["gates_sha256"]
    old=root/"prototypes/2026-10-22-synthetic-loader"
    original=json.loads((old/"plan/freeze.json").read_text())["files"]
    old_intact=all(sha(old/name)==digest for name,digest in original.items())
    if not all(intact.values()) or not old_intact:
        raise RuntimeError("frozen validator mismatch")
    for version,path in ((1,iteration/"patch/candidates/dlo_bridge-v1.py"),
                         (2,project/"src/h3_omni_adapter/dlo_bridge.py")):
        frozen=json.loads((iteration/f"plan/bridge-candidate-v{version}.json").read_text())
        if sha(path)!=frozen["sha256"]:
            raise RuntimeError("candidate snapshot mismatch")
    runs=[]
    for wp in sorted((iteration/"results").glob("*/trial/worker-result.json")):
        worker=json.loads(wp.read_text())
        monitor=json.loads((wp.parent/"result.json").read_text())
        name=wp.parent.parent.name
        if "preflight-v1-" in name:
            freeze="preflight-freeze-v1.json"
        elif "preflight-v2-" in name:
            freeze="preflight-freeze-v2.json"
        elif "transport-v1-" in name:
            freeze="transport-freeze-v1.json"
        else:
            freeze="bridge-freeze-v2.json"
        frozen=json.loads((iteration/"plan"/freeze).read_text())
        checks=worker["checks"]
        complete=len(checks)==frozen["expected_worker_checks"] and all(c["ok"] for c in checks)
        if "required_worker_checks" in frozen:
            complete=complete and set(c["name"] for c in checks)==set(frozen["required_worker_checks"])
        probes={key:{"plan_available":v["plan_available"],"fallback_reason":v["fallback_reason"],
                       "required":len(v.get("required_names",[])),"sources":len(v.get("source_bindings",[])),
                       "missing":len(v.get("missing_names",[]))} for key,v in worker.get("probes",{}).items()}
        runs.append({"run":name,"exit_code":monitor["exit_code"],
            "checks_ok":worker["checks_ok"],"checks_total":worker["checks_total"],
            "frozen_expected_checks":frozen["expected_worker_checks"],"frozen_checks_complete":complete,
            "probes":probes,"export_stats":worker.get("export_stats"),"cases":worker.get("cases"),
            "cuda_peak_allocated_bytes":worker.get("cuda_peak_allocated_bytes"),
            "cuda_peak_reserved_bytes":worker.get("cuda_peak_reserved_bytes"),
            "sampled_group_peak_rss_bytes":monitor["sampled_group_peaks"]["VmRSS"],
            "worker_ru_maxrss_bytes":worker["peak_rss_bytes"],
            "minimum_available_bytes":monitor["minimum_system_available_bytes"],
            "minimum_swap_free_bytes":monitor["minimum_swap_free_bytes"],
            "owned_process_reaped":monitor["owned_process_reaped"],"guard_error":monitor.get("guard_error"),
            "guard_stop_reason":monitor.get("guard_stop_reason")})
    summary={"updated_utc":datetime.now(UTC).isoformat(),"campaign_verdict":"INCONCLUSIVE",
        "scope":"actual generic DLO preflight plus native backend runtime-cache bridge",
        "real_model_downloaded":False,"real_video_generated":False,"full_model_fit_proven":False,
        "generic_dlo_planner_integrated":False,"native_dlo_transport_executed":True,
        "generic_cli_integrated":False,"upstream_clean":True,
        "upstream_revision":"325d5713503dc6a64a773d8afdc372838568a94d",
        "runs":runs,"validator_freeze_intact":intact,"original_freeze_intact":old_intact,
        "candidate_snapshots_intact":True,"independent_review":False,"path_redacted_files":redacted,
        "static_failure_logs":["logs/03-preflight-static.json","logs/11-bridge-fatal-static.json",
                               "logs/13-bridge-candidate-static.json"],
        "known_failed_gate":"preflight v1 count 28 vs frozen29; preserved, v2 completes29",
        "next":"plan/NEXT-STEPS.zh-CN.txt"}
    dump(iteration/"results/iteration-summary.json",summary)
    snapshot=iteration/"patch/project-snapshot"
    sources={}
    snapshot_redactions={}
    for path in sorted(project.rglob("*")):
        rel=path.relative_to(project)
        if not path.is_file() or any(part in {".git",".venv",".ruff_cache","__pycache__"}
                or part.endswith(".egg-info") for part in rel.parts):
            continue
        target=snapshot/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        sources[rel.as_posix()]=sha(path)
        if path.suffix in {".py",".toml",".md",".txt",".json",".yaml",".yml"}:
            before=path.read_text(encoding="utf-8")
            after=before
            for original,label in replacements:
                after=after.replace(original,label)
            if before!=after:
                target.write_text(after,encoding="utf-8")
                snapshot_redactions[rel.as_posix()]={"original_sha256":sha(path),"archived_sha256":sha(target)}

    dump(iteration/"project-source-manifest.json",sources)
    dump(iteration/"snapshot-redactions.json",snapshot_redactions)
    summary["sanitized_source_snapshot_files"]=snapshot_redactions
    dump(iteration/"results/iteration-summary.json",summary)
    for name in ("STATUS.zh-CN.txt","README.zh-CN.txt"):
        backup=iteration/("before-"+name)
        if not backup.exists():
            shutil.copyfile(root/name,backup)
    shutil.copyfile(iteration/"plan/ROOT-STATUS.zh-CN.txt",root/"STATUS.zh-CN.txt")
    (root/"README.zh-CN.txt").write_text(
        "H3 / FastH3 on RTX 5070 — local InferRSI records\n"
        "Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/\n"
        "Current status: STATUS.zh-CN.txt\n"
        f"Latest: iterations/{iteration.name}/REPORT.zh-CN.txt\n"
        "Historical failures and frozen evidence retained.\n"
        "No real model weights or real video. Full deployment, quality and performance INCONCLUSIVE.\n",encoding="utf-8")
    files={path.relative_to(iteration).as_posix():{"sha256":sha(path),"bytes":path.stat().st_size}
           for path in sorted(iteration.rglob("*")) if path.is_file() and path.name!="artifact-manifest.json"}
    dump(iteration/"artifact-manifest.json",{"campaign_verdict":"INCONCLUSIVE","files":files})
    print(json.dumps({"runs":[{"run":r["run"],"checks_ok":r["checks_ok"],
          "frozen_checks_complete":r["frozen_checks_complete"]} for r in runs],
          "frozen_intact":all(intact.values()),"original_intact":old_intact,
          "source_files":len(sources),"artifact_files":len(files),"redacted_files":len(redacted)}))


if __name__=="__main__":
    main()

