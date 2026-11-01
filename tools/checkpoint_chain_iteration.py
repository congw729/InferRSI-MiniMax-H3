"""Archive the frozen integrated chain and preserve historical source evidence."""
import argparse
from datetime import UTC,datetime
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
    p.add_argument("--checkpoint",type=Path,required=True)
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
    def redact(text):
        for source,label in replacements:
            text=text.replace(source,label)
        return text
    redacted=[]
    for area in ("results","logs"):
        for path in sorted((iteration/area).rglob("*")):
            if path.is_file() and path.suffix in {".json",".jsonl",".txt",".log"}:
                before=path.read_text(encoding="utf-8")
                after=redact(before)
                if before!=after:
                    path.write_text(after,encoding="utf-8")
                    redacted.append(path.relative_to(iteration).as_posix())
    frozen=json.loads((iteration/"plan/freeze-v1.json").read_text())
    guard=json.loads((iteration/"plan/guard-freeze-v1.json").read_text())
    intact={}
    for name,digest in {**frozen["files"],**guard["files"]}.items():
        intact[name]=sha(project/"tools"/name)==digest==sha(iteration/"validator"/name)
    for name,digest in frozen["runtime"].items():
        intact["runtime:"+name]=sha(project/"src/h3_omni_adapter"/name)==digest
    intact["gates"]=sha(iteration/"plan/FROZEN-GATES.zh-CN.txt")==frozen["gates_sha256"]
    intact["fixture_manifest"]=sha(a.checkpoint/"fixture-manifest.json")==frozen["fixture_manifest_sha256"]
    fixture=json.loads((a.checkpoint/"fixture-manifest.json").read_text())
    intact["fixture_files"]=all(sha(a.checkpoint/n)==v["sha256"] for n,v in fixture.items())
    old=root/"prototypes/2026-10-22-synthetic-loader"
    original=json.loads((old/"plan/freeze.json").read_text())["files"]
    old_intact=all(sha(old/name)==digest for name,digest in original.items())
    if not all(intact.values()) or not old_intact:
        raise RuntimeError("frozen source or fixture mismatch")
    runs=[]
    for path in sorted((iteration/"results").glob("*/trial/result.json")):
        value=json.loads(path.read_text())
        checks=value["checks"]
        expected=guard["expected_checks"] if "chain-owner-" in path.parent.parent.name else frozen["expected_checks"]
        complete=len(checks)==expected and all(c["ok"] for c in checks)
        if expected==frozen["expected_checks"]:
            complete=complete and set(c["name"] for c in checks)==set(frozen["required_checks"])
        runs.append({"run":path.parent.parent.name,"exit_code":value["exit_code"],
                     "checks_ok":sum(c["ok"] for c in checks),"checks_total":len(checks),
                     "frozen_expected":expected,"frozen_checks_complete":complete,
                     "scenario":value.get("scenario"),"geometry":value.get("geometry"),
                     "output_summary":value.get("output_summary"),"process_runs":value.get("process_runs"),
                     "worker_reports":value.get("worker_reports")})
    summary={"updated_utc":datetime.now(UTC).isoformat(),"campaign_verdict":"INCONCLUSIVE",
             "scope":"tiny native text / native DLO eight-step sampling / native temporal decoder in three processes",
             "real_model_downloaded":False,"real_video_generated":False,"audio_decoded":False,
             "video_only_72_frame_crop_checked":True,"full_model_fit_proven":False,
             "generic_cli_integrated":False,"runtime_changed_this_iteration":False,
             "independent_review":False,"runs":runs,"frozen_intact":intact,
             "original_freeze_intact":old_intact,"path_redacted_files":redacted,
             "next":"plan/NEXT-STEPS.zh-CN.txt"}
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
            after=redact(before)
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
    print(json.dumps({"runs":[{"run":r["run"],"checks":r["checks_ok"],
         "complete":r["frozen_checks_complete"]} for r in runs],
         "frozen_intact":all(intact.values()),"original_intact":old_intact,
         "sources":len(sources),"artifacts":len(files),"redacted_files":len(redacted)}))


if __name__=="__main__":
    main()
