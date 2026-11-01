"""Frozen v1: sequential process release, real handoff validation, no video claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "model_video_generated": False,
              "scope": "synthetic three-stage protocol with real H3 conditioning / small NVFP4 projection",
              "checks": []}
    def check(name, ok, **details):
        result["checks"].append({"name": name, "ok": bool(ok), **details})
    code = 1
    try:
        from h3_omni_adapter.stage_process import run_serial_stages
        worker = Path(__file__).with_name("stage_probe_worker_v1.py")
        common = [sys.executable, "-B", str(worker), "--checkpoint", str(a.checkpoint), "--device", a.device]
        commands = [(stage, [*common, "--stage", stage]) for stage in ("encode", "denoise", "decode")]
        good = run_serial_stages(commands, a.output.parent / "success", timeout_seconds=100)
        result["successful_stages"] = good
        check("three_stages_completed", len(good) == 3 and all(s["exit_code"] == 0 for s in good))
        check("previous_process_reaped_before_next_start",
              len(good) == 3 and all(good[i]["reaped_at"] <= good[i + 1]["started_at"] for i in range(2)))
        check("all_processes_reaped", all(s["process_reaped"] for s in good))
        reports = [json.loads((a.output.parent / "success" / s["stage"] / "worker-result.json").read_text()) for s in good]
        check("real_conditioning_roundtrip", reports[0].get("conditioning_roundtrip"))
        check("denoise_loader_closed", reports[1].get("loader_all_meta_after_close"))
        check("text_audio_video_metadata_preserved", reports[2].get("text_audio_video_metadata_preserved"))
        if a.device == "cuda":
            check("each_fresh_process_starts_without_live_torch_cuda_tensors", all(r.get("initial_cuda_allocated") == 0 for r in reports))
            check("actual_nvfp4_projection_in_middle_stage",
                  reports[1].get("real_nvfp4_projection_executed") and reports[1].get("projection_finite"))
            check("each_stage_allocates_cuda_probe", all(r.get("cuda_peak_allocated", 0) >= 16 * 1024**2 for r in reports))
        failed = run_serial_stages([("encode", [*common, "--stage", "encode", "--fail"]),
                                    ("decode", [*common, "--stage", "decode"])],
                                   a.output.parent / "failure", timeout_seconds=30)
        result["failed_stages"] = failed
        check("failure_stops_next_stage", len(failed) == 1 and failed[0]["exit_code"] != 0 and
              not (a.output.parent / "failure/decode").exists())
        check("failed_process_reaped", failed[0]["process_reaped"])
        timed = run_serial_stages([("encode", [*common, "--stage", "encode", "--delay", "15"]),
                                   ("decode", [*common, "--stage", "decode"])],
                                  a.output.parent / "timeout", timeout_seconds=2)
        result["timed_out_stages"] = timed
        check("timeout_stops_next_stage", len(timed) == 1 and timed[0]["timed_out"])
        check("timed_out_process_reaped", timed[0]["process_reaped"])
        from h3_omni_adapter.stage_handoff import read_payload
        artifact = a.output.parent / "success/encode/handoff"
        manifest_path = artifact / "manifest.json"
        original = manifest_path.read_text()
        manifest = json.loads(original)
        manifest["tensor_sha256"] = "0" * 64
        corrupt = a.output.parent / "corrupt"
        corrupt.mkdir()
        (corrupt / "manifest.json").write_text(json.dumps(manifest))
        (corrupt / "tensors.safetensors").write_bytes((artifact / "tensors.safetensors").read_bytes())
        try:
            read_payload(corrupt)
        except ValueError:
            check("handoff_digest_mismatch_rejected", True)
        else:
            check("handoff_digest_mismatch_rejected", False)
        result["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
        result["checks_total"] = len(result["checks"])
        result["checks_ok"] = sum(c["ok"] for c in result["checks"])
        code = int(result["checks_ok"] != result["checks_total"])
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    text = json.dumps(result, indent=2)
    for source, label in ((str(a.output.parent), "<RUN_ROOT>"), (str(a.checkpoint), "<SYNTHETIC_CHECKPOINT>"),
                          (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    a.output.write_text(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"checks", "traceback", "successful_stages", "failed_stages", "timed_out_stages"}}))
    return code
if __name__ == "__main__":
    raise SystemExit(main())
