"""Archive the bounded long-sequence capacity iteration, including failed runs."""
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
    long_freeze = json.loads((iteration / "plan/long-capacity-freeze-v1.json").read_text(encoding="utf-8"))
    geometry_intact = sha(iteration / "plan/native-request-geometry.json") == long_freeze["geometry_sha256"]
    old = root / "prototypes/2026-10-22-synthetic-loader"
    old_freeze = json.loads((old / "plan/freeze.json").read_text(encoding="utf-8"))["files"]
    old_intact = all(sha(old / name) == digest for name, digest in old_freeze.items())
    if not all(intact.values()) or not geometry_intact or not old_intact:
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
                phases = worker["phases"]
                item.update(
                    geometry=worker["geometry"],
                    cuda_peak_allocated_bytes=max(p["cuda_max_allocated"] for p in phases),
                    cuda_peak_reserved_bytes=max(p["cuda_max_reserved"] for p in phases),
                    sampled_group_peak_rss_bytes=data["sampled_group_peaks"]["VmRSS"],
                    worker_ru_maxrss_bytes=worker["peak_rss_bytes"],
                    minimum_available_bytes=data["minimum_system_available_bytes"],
                    minimum_swap_free_bytes=data["minimum_swap_free_bytes"],
                    guard_error=data.get("guard_error"),
                    owned_process_reaped=data["owned_process_reaped"],
                    attention_backend=worker["attention_backend"])
        else:
            item["failure_note"] = "Geometry v1 omitted required explicit aspect_ratio; capture retained."
        runs.append(item)
    summary = {
        "updated_utc": datetime.now(UTC).isoformat(), "campaign_verdict": "INCONCLUSIVE",
        "scope": "native request geometry and guarded real-width single-block long-sequence capacity",
        "real_model_downloaded": False, "real_video_generated": False, "full_model_fit_proven": False,
        "request_constraint": "Native t2va minimum is 4 seconds; 96 frames align to 107. Internal 3-second geometry is not a valid native request.",
        "runs": runs, "validator_freeze_intact": intact, "geometry_freeze_intact": geometry_intact,
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
    (root / "STATUS.zh-CN.txt").write_text("""H3 / RTX 5070 当前状态（2026-10-22 夜间）
最新迭代：2026-10-22-three-second-capacity，报告/summary/plan与清单已保存。
原生请求/几何14/14：当前t2va最少4秒且必须显式aspect_ratio；3秒原生请求被拒绝。
3秒内部几何为73帧/latent_t22/audio_t122；合法4秒对齐107帧/latent_t32/audio_t178。
复用1.60GiB真实宽度单DiT+refiner合成fixture，3秒128²、3秒256²、合法4秒256²各14/14。
最大场景2496 tokens：torch allocated峰值1452322304 B（1.353GiB），
reserved1673527296 B，采样所属进程组RSS3026546688 B（2.819GiB）。
这只是单块/PDD step-0/CUDNN_ATTN容量，不是全50层、完整8步、生产VSA或真实视频证明。
6GiB RSS/2GiB系统剩余/CUDA fraction0.40/180秒worker等保护未触发；worker均回收。
首版几何漏aspect_ratio的失败及冻结验证器保留，v2仅补参数。没有新大文件或真实权重下载。
此前小型三阶段34/34、SIGTERM清理5/5、CPU协议11/11、text44/44、DiT CUDA35/35有效。
下一优先真实encoder embedding+单text层容量，约1.70GiB合成fixture；注意encoder点号量化key和scale语义。
具体见最新迭代plan/NEXT-STEPS.zh-CN.txt。随后原生compact decoder真实clip容量。
约3秒输出目标保留，合法长输出裁剪/音频同步尚未实现。
上游未修改，generic DLO/CLI未接线；全模型/质量/性能INCONCLUSIVE。
实际Linux home与记录路径从参数传入。保持隔离、冻结、失败归档；北京时间07:30截止。
""", encoding="utf-8")
    (root / "README.zh-CN.txt").write_text("""H3 / FastH3 on RTX 5070 — local InferRSI records
Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/
Current status: STATUS.zh-CN.txt
Latest: iterations/2026-10-22-three-second-capacity/REPORT.zh-CN.txt
Previous chain: iterations/2026-10-22-native-three-stage/
Previous text: iterations/2026-10-22-encoder-streaming/
Reusable real-width fixture: iterations/2026-10-22-single-block-capacity/
Previous coverage: iterations/2026-10-22-full-coverage-mmap/
Original frozen prototype: prototypes/2026-10-22-synthetic-loader/
No real model weights or real model video. Deployment, quality and performance INCONCLUSIVE.
""", encoding="utf-8")
    files = {path.relative_to(iteration).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(iteration.rglob("*"))
             if path.is_file() and path.name != "artifact-manifest.json"}
    dump(iteration / "artifact-manifest.json", {"campaign_verdict": "INCONCLUSIVE", "files": files})
    print(json.dumps({"runs": runs, "frozen_intact": all(intact.values()),
                      "geometry_intact": geometry_intact, "original_intact": old_intact,
                      "source_files": len(sources), "artifact_files": len(files),
                      "redacted_files": len(redacted)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
