"""Checkpoint the current local coverage/mmap iteration and preserve evidence."""
import argparse
import ast
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record-root", type=Path, required=True)
    p.add_argument("--iteration", type=Path, required=True)
    p.add_argument("--linux-home", required=True)
    a = p.parse_args()
    root, iteration = a.record_root.resolve(), a.iteration.resolve()
    project = Path(__file__).resolve().parents[1]
    if not iteration.is_relative_to(root):
        raise ValueError("iteration outside canonical record root")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))
    def write(path, obj):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Redact newly created JSON text before final manifests. Binary fixtures and
    # frozen Python validators are never rewritten.
    mappings = [(str(root), "<RECORD_ROOT>"), (str(project), "<H3_PROJECT_ROOT>"),
                (str(project.parent), "<WORKSPACE_ROOT>"), (str(Path.home()), "<USER_HOME>"),
                (a.linux_home, "<LINUX_HOME>")]
    for path, label in list(mappings):
        if len(path) >= 3 and path[1:3] == ":\\":
            mappings.append(("/mnt/" + path[0].lower() + path[2:].replace("\\", "/"), label))
    old_summary = iteration / "results/iteration-summary.json"
    privacy_changes = read(old_summary).get("privacy_redactions", []) if old_summary.exists() else []
    for path in iteration.rglob("*.json"):
        if path.name in {"artifact-manifest.json", "project-source-manifest.json"}:
            continue
        old = path.read_text(encoding="utf-8-sig")
        new = old
        for source, label in sorted(mappings, key=lambda item: len(item[0]), reverse=True):
            new = new.replace(source.replace("\\", "\\\\"), label).replace(source, label)
        if new != old:
            path.write_text(new, encoding="utf-8")
            privacy_changes.append(path.relative_to(iteration).as_posix())
    trials = []
    for folder in sorted((iteration / "results").iterdir()):
        if not folder.is_dir() or not (folder / "trial/execution.json").is_file():
            continue
        execution = read(folder / "trial/execution.json")
        result = read(folder / "trial/result.json") if (folder / "trial/result.json").exists() else {}
        trials.append({"run": folder.name, "exit_code": execution["returncode"],
                       "checks_ok": result.get("checks_ok", result.get("checks_passed")),
                       "checks_total": result.get("checks_total"),
                       "error": result.get("error", result.get("execution_error")),
                       "scope": result.get("scope")})
    original = root / "prototypes/2026-10-22-synthetic-loader"
    old_freeze = read(original / "plan/freeze.json")["files"]
    frozen = [{"file": name, "intact": sha(original / name) == digest} for name, digest in old_freeze.items()]
    validators = []
    for path in sorted((iteration / "validator").glob("*.py")):
        live = project / "tools" / path.name
        validators.append({"file": path.name, "sha256": sha(path),
                           "live_matches": live.is_file() and sha(live) == sha(path)})
    protocol_checks = []
    for freeze_path in sorted((iteration / "plan").glob("*freeze*.json")):
        freeze = read(freeze_path)
        pairs = freeze.get("files", {})
        if "validator" in freeze and "sha256" in freeze:
            pairs = {freeze["validator"]: freeze["sha256"]}
        for name, digest in pairs.items():
            candidate = iteration / "validator" / name
            protocol_checks.append({"protocol": freeze_path.name, "file": name,
                                    "intact": candidate.is_file() and sha(candidate) == digest})
    native = read(iteration / "sources/native-vae/source-manifest.json")
    native_intact = all(sha(iteration / "sources/native-vae" / row["path"]) == row["sha256"]
                        for row in native["files"])
    syntax = []
    for path in project.rglob("*.py"):
        if not any(x in {".git", ".venv", ".ruff_cache", "__pycache__"} for x in path.relative_to(project).parts):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
            syntax.append(path.relative_to(project).as_posix())
    summary = {"updated_utc": datetime.now(UTC).isoformat(), "campaign_verdict": "INCONCLUSIVE",
        "real_model_payload_downloaded": False, "real_model_video_generated": False,
        "production_dlo_cli_integrated": False, "independent_review": False,
        "trials": trials, "original_freeze_file_count": len(frozen),
        "original_freeze_intact": all(x["intact"] for x in frozen),
        "validators": validators, "reviewed_native_source_count": len(native["files"]),
        "reviewed_native_sources_intact": native_intact, "syntax_file_count": len(syntax),
        "privacy_redactions": sorted(set(privacy_changes)),
        "freeze_protocols": protocol_checks,
        "next_step": "plan/NEXT-STEPS.zh-CN.txt",
        "stop_new_experiments_at": "2026-10-22T07:30:00+08:00"}
    write(iteration / "results/iteration-summary.json", summary)
    for name in ("STATUS.zh-CN.txt", "README.zh-CN.txt"):
        backup = iteration / ("before-" + name)
        if not backup.exists():
            shutil.copyfile(root / name, backup)
    status = """H3 / RTX 5070 当前状态（2026-10-22）
首个目标：约 3 秒、低分辨率文生视频，vLLM-Omni 优先。
本轮已完成真实子模块的 meta 参数/buffer 覆盖：
DiT 2087 项；text/vision encoder 1952 项；原生 video VAE 561 项；audio VAE 1087 项；compact decoder 378 项。
mmap CUDA 最终 35/35；完整小型 DiT 前向 18/18；compact decoder 17/17；三阶段协议 14/14。
已补齐隔离 WSL 的 Python 开发头文件，Triton JIT 小型 DiT 可执行。
适配代码位于独立 InferRSI-MiniMax-H3 项目；InferRSI 工作台和上游 runtime 源码未修改。
泛用 DLO/CLI、真实形状容量、完整真实三阶段和视频质量仍待验证；INCONCLUSIVE。
没有下载真实模型权重或生成真实模型视频。没有 issue/PR/push 或公开性能结论。
最新报告：iterations/2026-10-22-full-coverage-mmap/REPORT.zh-CN.txt。
机器摘要、所有失败和候选源码快照均在该迭代目录。
下一步先做有资源限制的真实单块形状容量试验，详见 plan/NEXT-STEPS.zh-CN.txt。
已安排本线程继续推进至北京时间 2026-10-22 07:30，并收尾；依赖本机和应用保持运行。
"""
    (root / "STATUS.zh-CN.txt").write_text(status, encoding="utf-8")
    (root / "README.zh-CN.txt").write_text("""H3 / FastH3 on RTX 5070 — local InferRSI records
Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/
Current status: STATUS.zh-CN.txt
Latest iteration: iterations/2026-10-22-full-coverage-mmap/REPORT.zh-CN.txt
Its plan/, validator/, results/, logs/, sources/ and patch/ preserve protocols, failures, provenance and candidates.
Previous environment iteration: iterations/2026-10-22-environment-bootstrap/
Earlier frozen byte-only prototype: prototypes/2026-10-22-synthetic-loader/
Original research, planning, preflight and sync records remain preserved.
Model quality, full deployment and service performance: INCONCLUSIVE.
No real model payload downloaded and no real-model video generated.
""", encoding="utf-8")
    snapshot = iteration / "patch/project-snapshot"
    source_manifest = {}
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if not path.is_file() or any(x in {".git", ".venv", ".ruff_cache", "__pycache__"}
                                    or x.endswith(".egg-info") for x in relative.parts):
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_manifest[relative.as_posix()] = sha(path)
    write(iteration / "project-source-manifest.json", source_manifest)
    manifest = {path.relative_to(iteration).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(iteration.rglob("*")) if path.is_file() and path.name != "artifact-manifest.json"}
    write(iteration / "artifact-manifest.json", {"campaign_verdict": "INCONCLUSIVE", "files": manifest})
    print(json.dumps({"trials": len(trials), "original_freeze_intact": summary["original_freeze_intact"],
        "validator_count": len(validators), "validators_intact": all(x["live_matches"] for x in validators),
        "native_sources_intact": native_intact,
        "freeze_protocols_intact": all(x["intact"] for x in protocol_checks), "artifact_files": len(manifest),
        "source_snapshot_files": len(source_manifest), "privacy_redacted_files": len(privacy_changes)}))


if __name__ == "__main__":
    main()

