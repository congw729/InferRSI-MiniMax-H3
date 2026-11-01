"""Frozen v1: serial native tiny model chain vs single-process native load reference."""
import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--text-fixture", type=Path, required=True)
    p.add_argument("--decoder-fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "model_video_generated": False,
              "scope": "real tiny native text / 8-step DiT / compact video decoder; synthetic weights",
              "checks": [], "process_runs": {}}
    def check(name, ok, **details):
        result["checks"].append({"name": name, "ok": bool(ok), **details})
        if not ok:
            raise AssertionError(name)
    code = 1
    try:
        import torch

        from h3_omni_adapter.stage_handoff import read_payload
        from h3_omni_adapter.stage_process import run_serial_stages
        worker = Path(__file__).with_name("native_chain_worker_v1.py")
        checkpoint = a.output.parent / "checkpoint"
        base = [sys.executable, "-B", str(worker), "--checkpoint", str(checkpoint),
                "--native-source", str(a.native_source)]
        def run(name, commands, timeout=120):
            records = run_serial_stages(commands, a.output.parent / name, timeout_seconds=timeout)
            result["process_runs"][name] = records
            return records
        prep = run("prepare", [("encode", [*base, "--mode", "prepare", "--stage", "encode",
            "--research", str(a.research), "--text-fixture", str(a.text_fixture),
            "--decoder-fixture", str(a.decoder_fixture)])])
        check("bounded_synthetic_fixtures_prepared", len(prep) == 1 and prep[0]["exit_code"] == 0)
        baseline = run("reference", [("encode", [*base, "--mode", "reference", "--stage", "encode"])])
        check("native_single_process_reference_completed", len(baseline) == 1 and baseline[0]["exit_code"] == 0)
        commands = [(stage, [*base, "--mode", "staged", "--stage", stage]) for stage in ("encode", "denoise", "decode")]
        stages = run("candidate", commands)
        check("all_three_native_stages_completed", len(stages) == 3 and all(r["exit_code"] == 0 for r in stages))
        check("serial_reap_before_next_start", all(stages[i]["reaped_at"] <= stages[i+1]["started_at"] for i in range(2)))
        check("all_stage_processes_reaped", all(r["process_reaped"] and not r["timed_out"] for r in stages))
        reports = [json.loads((a.output.parent / "candidate" / s["stage"] / "worker-result.json").read_text()) for s in stages]
        result["worker_reports"] = reports
        check("fresh_stage_cuda_allocations_start_at_zero", all(r["initial_cuda_allocated"] == 0 for r in reports))
        check("text_and_dit_restored_to_meta", all(r["all_meta_after_close"] for r in reports[:2]))
        check("decoder_tensor_objects_released", reports[2]["all_decoder_tensor_objects_released"])
        check("all_checkpoint_mmaps_closed", all(r["no_checkpoint_mmaps"] for r in reports))
        expected = read_payload(a.output.parent / "reference/encode/handoff")
        text = read_payload(a.output.parent / "candidate/encode/handoff")
        sampled = read_payload(a.output.parent / "candidate/denoise/handoff")
        decoded = read_payload(a.output.parent / "candidate/decode/handoff")
        check("native_text_hidden_exact", torch.equal(text["hidden"], expected["hidden"]))
        check("eight_native_pdd_steps", [s["step"] for s in sampled["steps"]] == list(range(8)))
        for step, (got, want) in enumerate(zip(sampled["steps"], expected["sampled"]["steps"], strict=True)):
            check("sampled_video_audio_exact_step_" + str(step),
                  torch.equal(got["video_rows"], want["video_rows"]) and torch.equal(got["audio_rows"], want["audio_rows"]))
        check("native_patch_roundtrip", sampled["patch_roundtrip"])
        check("native_geometry_6_used_64_padded", sampled["used_tokens"] == 6 and sampled["padded_tokens"] == 64)
        check("sampled_latent_payload_semantics", sampled["kind"] == "native_sampled_latents" and sampled["not_raw_velocity"])
        check("final_video_latent_exact", torch.equal(sampled["video_latent"], expected["sampled"]["video_latent"]))
        check("final_audio_rows_exact", torch.equal(sampled["audio_rows"], expected["sampled"]["audio_rows"]))
        check("native_latent_channels_and_geometry", tuple(sampled["video_latent"].shape) == (1, 24, 2, 2, 2))
        check("raw_decoder_tensor_exact", torch.equal(decoded["raw_frames"], expected["decoded"]["raw_frames"]))
        check("decoder_geometry_8_frames_32x32", tuple(decoded["raw_frames"].shape) == (1, 3, 8, 32, 32))
        check("all_chain_outputs_finite", all(torch.isfinite(x).all() for x in
              (text["hidden"], sampled["video_latent"], sampled["audio_rows"], decoded["raw_frames"])))
        check("all_expected_dit_groups_each_step", reports[1]["loader_stats"]["groups_loaded"] == 25)
        failed = run("intentional_failure", [("encode", [*base, "--mode", "staged", "--stage", "encode", "--fail"]),
                                               commands[1]])
        check("failure_blocks_next_stage", len(failed) == 1 and failed[0]["exit_code"] != 0 and
              not (a.output.parent / "intentional_failure/denoise").exists())
        timed = run("intentional_timeout", [("encode", [*base, "--mode", "staged", "--stage", "encode", "--delay", "15"]),
                                              commands[1]], timeout=2)
        check("timeout_reaped_and_blocks_next_stage", len(timed) == 1 and timed[0]["timed_out"]
              and timed[0]["process_reaped"] and not (a.output.parent / "intentional_timeout/denoise").exists())
        corrupt = a.output.parent / "corrupt-handoff"
        corrupt.mkdir()
        source = a.output.parent / "candidate/encode/handoff"
        shutil.copyfile(source / "tensors.safetensors", corrupt / "tensors.safetensors")
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["tensor_sha256"] = "0"*64
        (corrupt / "manifest.json").write_text(json.dumps(manifest))
        bad = run("corrupt_input", [("denoise", [*commands[1][1], "--input", str(corrupt)]), commands[2]])
        check("invalid_digest_blocks_decoder", len(bad) == 1 and bad[0]["exit_code"] != 0
              and not (a.output.parent / "corrupt_input/decode").exists())
        check("negative_case_children_reaped", all(r["process_reaped"] for rows in (failed, timed, bad) for r in rows))
        files = json.loads((checkpoint / "fixture-manifest.json").read_text())
        check("fixture_hashes_unchanged", all(hashlib.sha256((checkpoint/k).read_bytes()).hexdigest() == v["sha256"]
                                             for k,v in files.items()))
        result["fixture_bytes"] = sum(v["bytes"] for v in files.values())
        result["checks_total"] = len(result["checks"])
        result["checks_ok"] = sum(c["ok"] for c in result["checks"])
        result["final_tensor"] = {"shape": list(decoded["raw_frames"].shape),
                                 "dtype": str(decoded["raw_frames"].dtype),
                                 "max_abs_vs_reference": float((decoded["raw_frames"].float()
                                     - expected["decoded"]["raw_frames"].float()).abs().max())}
        code = 0
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["traceback"] = traceback.format_exc()
    text = json.dumps(result, indent=2)
    for source, label in ((str(a.output.parent), "<RUN_ROOT>"), (str(a.native_source), "<NATIVE_SOURCE>"),
            (str(a.research), "<RESEARCH_ROOT>"), (str(a.text_fixture), "<TEXT_FIXTURE>"),
            (str(a.decoder_fixture), "<DECODER_FIXTURE>"),
            (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(source, label)
    a.output.write_text(text + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","traceback","process_runs","worker_reports"}}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
