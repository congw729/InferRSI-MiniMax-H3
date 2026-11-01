"""Checkpoint single-block capacity evidence without touching older iterations."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=Path, required=True)
    p.add_argument("--record-root", type=Path, required=True)
    a = p.parse_args()
    root, iteration = a.record_root.resolve(), a.iteration.resolve()
    if not iteration.is_relative_to(root):
        raise ValueError("iteration is outside canonical record root")
    project = Path(__file__).resolve().parents[1]
    run = next((iteration / "results").glob("full-width-capacity-v1-*/trial"))
    result = json.loads((run / "result.json").read_text())
    samples = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
    freeze = json.loads((iteration / "plan/capacity-freeze-v1.json").read_text(encoding="utf-8-sig"))
    intact = {name: sha(iteration / "validator" / name) == digest and sha(project / "tools" / name) == digest
              for name, digest in freeze["files"].items()}
    fixture = json.loads((run / "fixture/fixture-manifest.json").read_text())
    fixtures_intact = all(sha(run / "fixture" / row["file"]) == row["sha256"] for row in fixture)
    old = root / "prototypes/2026-10-22-synthetic-loader"
    old_freeze = json.loads((old / "plan/freeze.json").read_text())["files"]
    old_intact = all(sha(old / name) == digest for name, digest in old_freeze.items())
    summary = {"updated_utc": datetime.now(UTC).isoformat(), "campaign_verdict": "INCONCLUSIVE",
        "real_model_downloaded": False, "real_video_generated": False, "full_model_fit_proven": False,
        "run": run.parent.name, "exit_code": result["exit_code"],
        "worker_checks_passed": result["worker"]["checks_passed"],
        "worker_checks_total": result["worker"]["checks_total"],
        "sampled_process_group_peak_rss": result["sampled_group_peaks"]["VmRSS"],
        "worker_peak_rss": result["worker"]["peak_rss_bytes"],
        "cuda_peak_allocated": max(x["cuda_max_allocated"] for x in result["worker"]["phases"]),
        "cuda_peak_reserved": max(x["cuda_max_reserved"] for x in result["worker"]["phases"]),
        "minimum_system_available": min(x["system_memory"]["MemAvailable"] for x in samples),
        "minimum_swap_free": min(x["system_memory"]["SwapFree"] for x in samples),
        "initial_swap_free": result["memory_before"]["SwapFree"],
        "guard_stop_reason": result.get("guard_stop_reason"),
        "worker_reaped": result["owned_process_reaped"],
        "validator_freeze_intact": intact, "fixtures_intact": fixtures_intact,
        "original_freeze_intact": old_intact,
        "scope_limit": "one real-width DiT block, one refiner; only 8 tokens; warm synthetic files",
        "next": "plan/ENCODER-STREAMING-NEXT.zh-CN.txt"}
    (iteration / "results/iteration-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    snapshot = iteration / "patch/project-snapshot"
    source_manifest = {}
    for path in sorted(project.rglob("*")):
        rel = path.relative_to(project)
        if not path.is_file() or any(x in {".git", ".venv", ".ruff_cache", "__pycache__"}
                                    or x.endswith(".egg-info") for x in rel.parts):
            continue
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_manifest[rel.as_posix()] = sha(path)
    (iteration / "project-source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    for name in ("STATUS.zh-CN.txt", "README.zh-CN.txt"):
        backup = iteration / ("before-" + name)
        if not backup.exists():
            shutil.copyfile(root / name, backup)
    (root / "STATUS.zh-CN.txt").write_text("""H3 / RTX 5070 当前状态（2026-10-22 夜间）
目标仍是约 3 秒、低分辨率文生视频，vLLM-Omni 优先。
上一迭代完成完整子模块 meta 覆盖、mmap CUDA 35/35、小型完整 DiT 18/18、compact decoder17/17、阶段协议14/14。
最新单块容量试验：真实宽度的1个DiT块+1个refiner、8-token前向，12/12。
峰值 torch CUDA allocated约1.032GiB、reserved约1.477GiB；所属进程组采样RSS约2.819GiB。
这不是3秒视频激活长度，也未运行完整50层或真实模型。全模型可运行性与质量仍INCONCLUSIVE。
没有下载真实权重；本地合成文件约1.60GiB，保存在本轮记录目录。
最新报告：iterations/2026-10-22-single-block-capacity/REPORT.zh-CN.txt。
下一切片：纯文本encoder逐层mmap。已核对原生严格加载、设备入口与vision无关构造的问题；
详见该目录plan/ENCODER-STREAMING-NEXT.zh-CN.txt。
继续使用隔离环境，不下载真实模型，不发布远端；北京时间07:30收尾。
""", encoding="utf-8")
    (root / "README.zh-CN.txt").write_text("""H3 / FastH3 on RTX 5070 — local InferRSI records
Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/
Current status: STATUS.zh-CN.txt
Latest iteration: iterations/2026-10-22-single-block-capacity/REPORT.zh-CN.txt
Previous coverage/mmap/stage iteration: iterations/2026-10-22-full-coverage-mmap/
Previous environment iteration: iterations/2026-10-22-environment-bootstrap/
Earlier frozen byte-only prototype: prototypes/2026-10-22-synthetic-loader/
No real model download or model video. Full deployment, quality and performance remain INCONCLUSIVE.
""", encoding="utf-8")
    manifest = {path.relative_to(iteration).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(iteration.rglob("*")) if path.is_file() and path.name != "artifact-manifest.json"}
    (iteration / "artifact-manifest.json").write_text(json.dumps({
        "campaign_verdict": "INCONCLUSIVE", "files": manifest}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
