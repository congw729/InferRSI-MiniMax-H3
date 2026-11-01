"""Archive a bounded capacity iteration, including failures and frozen evidence."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--linux-home", required=True)
    parser.add_argument("--lab-root", type=Path, required=True)
    args = parser.parse_args()
    root, iteration = args.record_root.resolve(), args.iteration.resolve()
    project = Path(__file__).resolve().parents[1]
    if not iteration.is_relative_to(root):
        raise ValueError("iteration outside record root")

    # Redact this iteration's command evidence before producing its final manifest.
    # Preserve all numbers, errors, verdicts, source hashes and frozen validators.
    replacements = []
    for path, label in [(project, "<H3_PROJECT_ROOT>"), (root, "<RECORD_ROOT>"),
                        (project.parent, "<WORKSPACE_ROOT>"), (args.lab_root, "<LAB_ROOT>")]:
        native = str(path)
        variants = [native, native.replace("\\", "/")]
        if len(native) > 2 and native[1] == ":":
            variants.append("/mnt/" + native[0].lower() + native[2:].replace("\\", "/"))
        for variant in variants:
            replacements.append((variant, label))
            replacements.append((json.dumps(variant)[1:-1], label))
    replacements.append((args.linux_home, "<LINUX_HOME>"))
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)
    redacted = []
    for area in ("results", "logs"):
        for path in sorted((iteration / area).rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".log", ".txt"}:
                continue
            before = path.read_text(encoding="utf-8")
            after = before
            for original, replacement in replacements:
                after = after.replace(original, replacement)
            if after != before:
                path.write_text(after, encoding="utf-8")
                redacted.append(path.relative_to(iteration).as_posix())

    intact = {}
    for path in (iteration / "plan").glob("*freeze*.json"):
        frozen = json.loads(path.read_text(encoding="utf-8"))
        for name, digest in frozen["files"].items():
            intact[name] = sha(project / "tools" / name) == digest == sha(iteration / "validator" / name)
    old = root / "prototypes/2026-10-22-synthetic-loader"
    old_freeze = json.loads((old / "plan/freeze.json").read_text(encoding="utf-8"))["files"]
    old_intact = all(sha(old / name) == digest for name, digest in old_freeze.items())
    if not all(intact.values()) or not old_intact:
        raise RuntimeError("frozen artifact mismatch")

    runs = []
    for execution_path in sorted((iteration / "results").glob("*/trial/execution.json")):
        trial = execution_path.parent
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        result_path = trial / "result.json"
        item = {"run": trial.parent.name, "exit_code": execution["returncode"],
                "result_available": result_path.exists(),
                "capture": (trial / "capture.json").relative_to(iteration).as_posix()}
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            worker = data.get("worker", data)
            checks = worker.get("checks", [])
            item.update(checks_ok=worker.get("checks_ok", sum(c["ok"] for c in checks)),
                        checks_total=worker.get("checks_total", len(checks)))
            if "worker" in data:
                phases = worker.get("phases", [])
                item.update(
                    geometry=worker.get("geometry"),
                    fixture=worker.get("reused_fixture", worker.get("fixture")),
                    output_shape=worker.get("output_shape"),
                    output_sha256=worker.get("output_sha256"),
                    cases=worker.get("cases"),
                    group_logical_bytes=worker.get("group_logical_bytes"),
                    logical_model_bytes=worker.get("logical_model_bytes"),
                    cuda_peak_allocated_bytes=max((p["cuda_max_allocated"] for p in phases), default=worker.get("cuda_peak_allocated_bytes", 0)),
                    cuda_peak_reserved_bytes=max((p["cuda_max_reserved"] for p in phases), default=worker.get("cuda_peak_reserved_bytes", 0)),
                    sampled_group_peak_rss_bytes=data["sampled_group_peaks"]["VmRSS"],
                    worker_ru_maxrss_bytes=worker["peak_rss_bytes"],
                    minimum_available_bytes=data["minimum_system_available_bytes"],
                    minimum_swap_free_bytes=data["minimum_swap_free_bytes"],
                    guard_error=data.get("guard_error"),
                    owned_process_reaped=data["owned_process_reaped"],
                    attention_backend=worker.get("attention_backend"))
        else:
            item["failure_note"] = "Geometry v1 omitted required explicit aspect_ratio; capture retained."
        runs.append(item)
    capacity_freeze = json.loads((iteration / "plan/temporal-capacity-freeze-v1.json").read_text(encoding="utf-8"))
    required = set(capacity_freeze["required_worker_checks"])
    tiny_freeze = json.loads((iteration / "plan/temporal-freeze-v2.json").read_text(encoding="utf-8"))
    check_evidence = {}
    for path in (iteration / "results").glob("*/trial/worker-result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        checks = data.get("checks", [])
        if path.parent.parent.name.startswith("tiny-temporal"):
            complete = len(checks) == tiny_freeze["expected_worker_checks"] and len(data.get("cases", [])) == 8
            complete = complete and data.get("fixture_sha256") == tiny_freeze["fixture_sha256"]
        else:
            complete = set(c["name"] for c in checks) == required
        check_evidence[path.parent.parent.name] = complete and all(c["ok"] for c in checks)
    candidate_intact = sha(project / "src/h3_omni_adapter/temporal_decoder.py") == capacity_freeze["candidate_sha256"]
    if not candidate_intact:
        raise RuntimeError("candidate changed since capacity freeze")
    summary = {
        "updated_utc": datetime.now(UTC).isoformat(), "campaign_verdict": "INCONCLUSIVE",
        "scope": "native temporal decoder equality and complete 107-frame capacity",
        "real_model_downloaded": False, "real_video_generated": False, "full_model_fit_proven": False,
        "request_constraint": "Native t2va minimum is 4 seconds; 96 frames align to 107. Internal 3-second geometry is not a valid native request.",
        "runs": runs, "candidate_freeze_intact": candidate_intact,
        "static_failure_logs": ["logs/01-frozen-validator-static.json"], "required_worker_checks_complete": check_evidence, "validator_freeze_intact": intact,
        "original_freeze_intact": old_intact, "independent_review": False,
        "generic_dlo_or_cli_integrated": False, "path_redaction_files": redacted,
        "next": "plan/NEXT-STEPS.zh-CN.txt"}
    dump(iteration / "results/iteration-summary.json", summary)
    snapshot = iteration / "patch/project-snapshot"
    sources = {}
    for path in sorted(project.rglob("*")):
        rel = path.relative_to(project)
        if not path.is_file() or any(part in {".git", ".venv", ".ruff_cache", "__pycache__"}
                or part.endswith(".egg-info") for part in rel.parts):
            continue
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[rel.as_posix()] = sha(path)
    dump(iteration / "project-source-manifest.json", sources)
    for name in ("STATUS.zh-CN.txt", "README.zh-CN.txt"):
        backup = iteration / ("before-" + name)
        if not backup.exists():
            shutil.copyfile(root / name, backup)
    shutil.copyfile(iteration / "plan/ROOT-STATUS.zh-CN.txt", root / "STATUS.zh-CN.txt")
    (root / "README.zh-CN.txt").write_text(
        "H3 / FastH3 on RTX 5070 — local InferRSI records\n"
        "Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/\n"
        "Current status: STATUS.zh-CN.txt\n"
        f"Latest: iterations/{iteration.name}/REPORT.zh-CN.txt\n"
        "Earlier iterations and all frozen failures remain in iterations/.\n"
        "No real model weights or real video. Deployment, quality and performance INCONCLUSIVE.\n",
        encoding="utf-8")
    files = {path.relative_to(iteration).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(iteration.rglob("*"))
             if path.is_file() and path.name != "artifact-manifest.json"}
    dump(iteration / "artifact-manifest.json", {"campaign_verdict": "INCONCLUSIVE", "files": files})
    print(json.dumps({"runs": runs, "frozen_intact": all(intact.values()),
                      "original_intact": old_intact,
                      "source_files": len(sources), "artifact_files": len(files),
                      "redacted_files": len(redacted)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
