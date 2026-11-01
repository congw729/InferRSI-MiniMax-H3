"""Checkpoint local text-streaming evidence, retaining all historical runs."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=Path, required=True)
    p.add_argument("--record-root", type=Path, required=True)
    a = p.parse_args()
    root, iteration = a.record_root.resolve(), a.iteration.resolve()
    if not iteration.is_relative_to(root):
        raise ValueError("iteration outside canonical evidence root")
    project = Path(__file__).resolve().parents[1]
    runs = []
    for path in sorted((iteration / "results").glob("*/trial/result.json")):
        data = json.loads(path.read_text())
        execution = json.loads((path.parent / "execution.json").read_text())
        runs.append({"run": path.parent.parent.name, "exit_code": execution["returncode"],
            "checks_ok": sum(c["ok"] for c in data.get("checks", [])),
            "checks_total": len(data.get("checks", [])),
            "execution_error": data.get("execution_error"),
            "cuda_peak_allocated": data.get("cuda_peak_allocated", data.get("cuda_peak_allocated_bytes")),
            "cuda_peak_reserved": data.get("cuda_peak_reserved"),
            "peak_rss_bytes": data.get("peak_rss_bytes")})
    intact = {}
    for version in (1, 2, 3):
        frozen = json.loads((iteration / f"plan/validator-freeze-v{version}.json").read_text())
        name = f"validate_text_streaming_v{version}.py"
        intact[name] = (sha(iteration / "validator" / name) == frozen["validator_sha256"]
                        == sha(project / "tools" / name))
    old = root / "prototypes/2026-10-22-synthetic-loader"
    old_freeze = json.loads((old / "plan/freeze.json").read_text())["files"]
    old_intact = all(sha(old / name) == digest for name, digest in old_freeze.items())
    prev = root / "iterations/2026-10-22-full-coverage-mmap/validator"
    regression_intact = {name: sha(project / "tools" / name) == sha(prev / name)
                        for name in ("validate_mmap_v3.py", "validate_dit_forward_v1.py")}
    summary = {"updated_utc": datetime.now(UTC).isoformat(), "campaign_verdict": "INCONCLUSIVE",
        "real_model_downloaded": False, "real_video_generated": False, "full_model_fit_proven": False,
        "scope": "native tiny two-layer text-only encoder mmap; all hidden values and release verified",
        "runs": runs, "validator_freeze_intact": intact, "regression_validators_intact": regression_intact,
        "original_freeze_intact": old_intact, "independent_review": False,
        "generic_dlo_or_cli_integrated": False, "next": "plan/NEXT-STEPS.zh-CN.txt"}
    (iteration / "results/iteration-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    snapshot = iteration / "patch/project-snapshot"
    sources = {}
    for path in sorted(project.rglob("*")):
        rel = path.relative_to(project)
        if not path.is_file() or any(x in {".git", ".venv", ".ruff_cache", "__pycache__"}
                or x.endswith(".egg-info") for x in rel.parts):
            continue
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[rel.as_posix()] = sha(path)
    (iteration / "project-source-manifest.json").write_text(json.dumps(sources, indent=2)+"\n")
    for name in ("STATUS.zh-CN.txt", "README.zh-CN.txt"):
        backup = iteration / ("before-" + name)
        if not backup.exists():
            shutil.copyfile(root / name, backup)
    (root / "STATUS.zh-CN.txt").write_text("""H3 / RTX 5070 当前状态（2026-10-22 夜间）
目标仍是约3秒低分辨率文生视频，vLLM-Omni优先。
最新：纯文本encoder逐层mmap已实现，默认meta构造；原生小模型数值/释放44项全部完成。
8、17、8 tokens前向与原生参照逐值一致；vision不构造，层用完回meta；关闭后无权重映射。
DiT共享加载器回归：CPU34/34、CUDA35/35、完整小型前向18/18。
先前真实宽度单DiT/refiner短序列12项完成，GPU allocated约1.032GiB、进程组RSS约2.819GiB；
这不是3秒视频长度。真实encoder大embedding尚未容量实测。
最新报告：iterations/2026-10-22-encoder-streaming/REPORT.zh-CN.txt。
下一步见同目录plan/NEXT-STEPS.zh-CN.txt：完整小型text→DiT→compact decoder三阶段；
先对齐24通道latent和原生patch/采样语义，再冻结验证器。之后真实长度激活测试可复用1.60GiB fixture。
上游未修改，generic DLO/CLI未接线，无真实权重或真实视频，质量/整模型/性能INCONCLUSIVE。
记录统一当前目录，运行参数传实际Linux home与目录；不下载、不发布、不改驱动或无关进程。
北京时间2026-10-22 07:30截止，届时只收尾。
""", encoding="utf-8")
    (root / "README.zh-CN.txt").write_text("""H3 / FastH3 on RTX 5070 — local InferRSI records
Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/
Current status: STATUS.zh-CN.txt
Latest: iterations/2026-10-22-encoder-streaming/REPORT.zh-CN.txt
Previous capacity: iterations/2026-10-22-single-block-capacity/
Previous coverage: iterations/2026-10-22-full-coverage-mmap/
Previous environment: iterations/2026-10-22-environment-bootstrap/
Original frozen prototype: prototypes/2026-10-22-synthetic-loader/
No real model weights or real model video. Deployment, quality and performance INCONCLUSIVE.
""", encoding="utf-8")
    manifest = {path.relative_to(iteration).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(iteration.rglob("*"))
                if path.is_file() and path.name != "artifact-manifest.json"}
    (iteration / "artifact-manifest.json").write_text(json.dumps(
        {"campaign_verdict": "INCONCLUSIVE", "files": manifest}, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
