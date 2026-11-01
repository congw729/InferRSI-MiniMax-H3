"""Frozen controller: directly own and guard reference plus three serial stages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from h3_omni_adapter.stage_process import _termination_guard
from validate_capacity_v1 import fields,group_sample,terminate_owned

GIB=1024**3
REQUIRED=["reference_completed","three_stages_completed","serial_reap_before_next",
 "all_workers_reaped","candidate_model_objects_released","all_mmaps_closed","text_restored_meta",
 "fresh_cuda_start","geometry_matches","text_hidden_exact","eight_pdd_steps",
 "step_0_exact","step_1_exact","step_2_exact","step_3_exact","step_4_exact","step_5_exact","step_6_exact","step_7_exact",
 "final_latent_exact","final_audio_rows_exact","full_pixels_exact","crop_pixels_exact",
 "full_pixel_geometry","crop_geometry","crop_duration","finite_nonuniform_output",
 "native_dlo_mmap_hooks","dlo_shutdown","temporal_clip_count","handoff_budgets","fixture_unchanged"]


def run_owned(command,work):
    work.mkdir(parents=True)
    before=fields(Path("/proc/meminfo"),{"MemAvailable","SwapFree"})
    if before["MemAvailable"]<5*GIB:
        raise RuntimeError("initial host headroom")
    started=time.time()
    tick=time.monotonic()
    child=None
    record=dict(started_at=started,samples=0,peaks={"VmRSS":0,"RssAnon":0,"RssFile":0,"RssShmem":0,"members":0},
        minimum_available_bytes=before["MemAvailable"],minimum_swap_free_bytes=before["SwapFree"])
    try:
        with _termination_guard():
            with (work/"stdout.txt").open("wb") as stdout,(work/"stderr.txt").open("wb") as stderr,(work/"samples.jsonl").open("w") as samples:
                child=subprocess.Popen([*command,"--work",str(work)],stdout=stdout,stderr=stderr,start_new_session=True)
                while child.poll() is None:
                    now=time.monotonic()-tick
                    memory=fields(Path("/proc/meminfo"),{"MemAvailable","SwapFree"})
                    current=group_sample(child.pid)
                    record["samples"]+=1
                    for key,value in current.items():
                        record["peaks"][key]=max(record["peaks"][key],value)
                    record["minimum_available_bytes"]=min(record["minimum_available_bytes"],memory.get("MemAvailable",0))
                    record["minimum_swap_free_bytes"]=min(record["minimum_swap_free_bytes"],memory.get("SwapFree",0))
                    samples.write(json.dumps(dict(elapsed=now,memory=memory,owned_group=current))+"\n")
                    samples.flush()
                    reason=("owned RSS limit" if current["VmRSS"]>6*GIB else
                        "system headroom" if memory.get("MemAvailable",0)<2*GIB else
                        "stage timeout" if now>180 else None)
                    if reason:
                        record["guard_stop_reason"]=reason
                        terminate_owned(child)
                        break
                    time.sleep(0.1)
                child.wait(timeout=5)
    finally:
        if child is not None:
            terminate_owned(child)
            record["exit_code"]=child.returncode
            record["process_reaped"]=child.poll() is not None
        record["reaped_at"]=time.time()
        record["elapsed_seconds"]=time.monotonic()-tick
        (work/"process-result.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--native-source",type=Path,required=True)
    p.add_argument("--scenario",choices=("tiny","legal4s256"),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root=a.output.parent
    root.mkdir(parents=True,exist_ok=True)
    result=dict(campaign_verdict="INCONCLUSIVE",real_video_generated=False,full_model_fit_proven=False,
                generic_cli_integrated=False,scenario=a.scenario,checks=[],process_runs={})
    code=1
    def check(name,ok):
        if name not in REQUIRED or any(c["name"]==name for c in result["checks"]):
            raise AssertionError("unknown or repeated check")
        result["checks"].append(dict(name=name,ok=bool(ok)))
        if not ok:
            raise AssertionError(name)
    try:
        if shutil.disk_usage(root).free<8*GIB:
            raise RuntimeError("disk headroom")
        worker=Path(__file__).with_name("dlo_temporal_chain_worker_v1.py")
        base=[sys.executable,"-B",str(worker),"--checkpoint",str(a.checkpoint),
              "--native-source",str(a.native_source),"--scenario",a.scenario]
        reference=run_owned([*base,"--mode","reference","--stage","encode"],root/"reference")
        result["process_runs"]["reference"]=reference
        check("reference_completed",reference["exit_code"]==0 and not reference.get("guard_stop_reason"))
        stages=[]
        previous=None
        for stage in ("encode","denoise","decode"):
            command=[*base,"--mode","staged","--stage",stage]
            if previous is not None:
                command+=["--input",str(previous/"handoff")]
            current=root/"candidate"/stage
            record=run_owned(command,current)
            record["stage"]=stage
            stages.append(record)
            result["process_runs"]["candidate"]=stages
            if record["exit_code"]!=0 or record.get("guard_stop_reason"):
                break
            previous=current
        check("three_stages_completed",len(stages)==3 and all(r["exit_code"]==0 for r in stages))
        check("serial_reap_before_next",reference["reaped_at"]<=stages[0]["started_at"] and
              all(stages[i]["reaped_at"]<=stages[i+1]["started_at"] for i in range(2)))
        check("all_workers_reaped",all(r["process_reaped"] for r in [reference,*stages]))
        reports=[json.loads((root/"candidate"/s/"worker-result.json").read_text()) for s in ("encode","denoise","decode")]
        result["worker_reports"]=reports
        check("candidate_model_objects_released",all(r["model_tensor_objects_released"] for r in reports))
        check("all_mmaps_closed",all(r["no_checkpoint_mmaps"] for r in reports))
        check("text_restored_meta",reports[0]["all_meta_after_close"])
        check("fresh_cuda_start",all(r["initial_cuda_allocated"]==0 for r in reports))
        import torch
        from h3_omni_adapter.stage_handoff import read_payload
        expected=read_payload(root/"reference/handoff")
        text=read_payload(root/"candidate/encode/handoff")
        sampled=read_payload(root/"candidate/denoise/handoff")
        decoded=read_payload(root/"candidate/decode/handoff")
        g=decoded["geometry"]
        result["geometry"]=g
        check("geometry_matches",all(r["geometry"]==g for r in reports) and expected["decoded"]["geometry"]==g)
        check("text_hidden_exact",torch.equal(text["hidden"],expected["hidden"]))
        check("eight_pdd_steps",[s["step"] for s in sampled["steps"]]==list(range(8)))
        for i,(got,want) in enumerate(zip(sampled["steps"],expected["sampled"]["steps"],strict=True)):
            check(f"step_{i}_exact",torch.equal(got["video_rows"],want["video_rows"]) and torch.equal(got["audio_rows"],want["audio_rows"]))
        check("final_latent_exact",torch.equal(sampled["video_latent"],expected["sampled"]["video_latent"]))
        check("final_audio_rows_exact",torch.equal(sampled["audio_rows"],expected["sampled"]["audio_rows"]))
        check("full_pixels_exact",torch.equal(decoded["full_pixels"],expected["decoded"]["full_pixels"]))
        check("crop_pixels_exact",torch.equal(decoded["crop_pixels"],expected["decoded"]["crop_pixels"])
              and torch.equal(decoded["crop_pixels"],decoded["full_pixels"][:,:,:g["crop_frames"]]))
        check("full_pixel_geometry",tuple(decoded["full_pixels"].shape)==(1,3,g["frames"],g["height"],g["width"])
              and decoded["full_pixels"].dtype==torch.uint8)
        check("crop_geometry",tuple(decoded["crop_pixels"].shape)==(1,3,g["crop_frames"],g["height"],g["width"]))
        check("crop_duration",decoded["crop_seconds"]==g["crop_frames"]/24 and decoded["video_only"] and not decoded["audio_decoded"])
        check("finite_nonuniform_output",torch.isfinite(sampled["video_latent"]).all() and
              torch.isfinite(sampled["audio_rows"]).all() and decoded["full_pixels"].min()!=decoded["full_pixels"].max())
        check("native_dlo_mmap_hooks",reports[1]["native_dlo_hooks"]==2 and reports[1]["native_mmap"])
        check("dlo_shutdown",reports[1]["dlo_shutdown"])
        check("temporal_clip_count",reports[2]["temporal_plan"]["chunks"]==(1 if a.scenario=="tiny" else 6))
        paths=[root/"reference",*(root/"candidate"/s for s in ("encode","denoise","decode"))]
        check("handoff_budgets",all(json.loads((path/"handoff/manifest.json").read_text())["logical_tensor_bytes"]<=64*1024**2 for path in paths))
        manifest=json.loads((a.checkpoint/"fixture-manifest.json").read_text())
        check("fixture_unchanged",all(hashlib.sha256((a.checkpoint/n).read_bytes()).hexdigest()==v["sha256"] for n,v in manifest.items()))
        result["output_summary"]={"full_shape":list(decoded["full_pixels"].shape),"crop_shape":list(decoded["crop_pixels"].shape),
            "crop_seconds":decoded["crop_seconds"],"full_sha256":hashlib.sha256(decoded["full_pixels"].numpy().tobytes()).hexdigest(),
            "crop_sha256":hashlib.sha256(decoded["crop_pixels"].numpy().tobytes()).hexdigest(),"pixels_mismatch":0,
            "max_sampled_latent_abs_difference":0.0,"real_video":False,"audio_decoded":False}
        code=int(set(c["name"] for c in result["checks"])!=set(REQUIRED))
    except BaseException as exc:
        result["execution_error"]={"type":type(exc).__name__,"message":str(exc)}
        result["traceback"]=traceback.format_exc()
        code=int(exc.code) if isinstance(exc,SystemExit) and isinstance(exc.code,int) else 1
    result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    result["checks_total"]=len(result["checks"])
    result["exit_code"]=code
    replacements=((str(a.checkpoint),"<SYNTHETIC_CHECKPOINT>"),(str(root),"<RUN_ROOT>"),
        (str(a.native_source),"<NATIVE_SOURCE>"),(str(Path.home()),"<LINUX_HOME>"),
        (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>"))
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json",".jsonl",".txt"}:
            text=path.read_text()
            for source,label in replacements:
                text=text.replace(source,label)
            path.write_text(text)
    text=json.dumps(result,indent=2)
    for source,label in replacements:
        text=text.replace(source,label)
    a.output.write_text(text+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","traceback","worker_reports"}}))
    return code


if __name__=="__main__":
    raise SystemExit(main())
