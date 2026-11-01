"""Archive this local iteration without rewriting frozen prototype evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    args = parser.parse_args()
    root, iteration = args.record_root.resolve(), args.iteration.resolve()
    if not iteration.is_relative_to(root):
        raise ValueError("iteration is outside record root")
    project = Path(__file__).resolve().parents[1]
    windows = read(iteration / "results/latest.json")
    windows_summary = read(iteration / windows["run"] / "summary.json")
    linux = []
    for folder in sorted((iteration / "results").glob("linux-*")):
        if not (folder / "trial/execution.json").exists():
            continue
        execution = read(folder / "trial/execution.json")
        result = read(folder / "trial/result.json") if (folder / "trial/result.json").exists() else {}
        error = result.get("execution_error")
        if execution["returncode"] == 124:
            reason = "first-use JIT setup timed out; intermediate tensor checks are not a completed kernel result"
        elif error:
            message = error.get("message", "")
            lines = [line for line in message.splitlines() if any(word in line for word in
                     ("fatal error:", "cannot find", "incompatible", "Could not find"))]
            reason = "\n".join(lines[-4:]) if lines else message[-600:]
        else:
            reason = None
        linux.append({"run": folder.name, "exit_code": execution["returncode"],
                      "checks_ok": result.get("checks_ok"), "checks_total": result.get("checks_total"),
                      "full_module_imported": result.get("full_runtime_imported", False),
                      "nvfp4_forward_completed": execution["returncode"] == 0 and "nvfp4_probe" in result,
                      "reason": reason})
    completed_cuda = [item for item in linux if item["nvfp4_forward_completed"]]
    cpu_ok = any("linux-cpu-" in item["run"] and item["exit_code"] == 0 for item in linux)
    old_freeze = read(args.prototype / "plan/freeze.json")
    frozen_ok = all(sha(args.prototype / name) == value for name, value in old_freeze["files"].items())
    validation_freezes = {
        "tensor": sha(project / "tools/validate_tensors.py") == read(iteration / "plan/tensor-validator-freeze-v1.json")["validator_sha256"],
        "runtime": sha(project / "tools/validate_runtime.py") == read(iteration / "plan/runtime-validator-freeze-v1.json")["validator_sha256"],
    }
    metadata = read(iteration / "results/full-header-name-audit.json")
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "goal": "approximately three-second low-resolution text-to-video, prioritized vLLM-Omni",
        "campaign_verdict": "INCONCLUSIVE", "model_downloaded": False, "video_generated": False,
        "windows_tensor_suite": windows_summary,
        "linux_trials": linux, "full_linux_cpu_loader_ok": cpu_ok,
        "nvfp4_forward_completed": bool(completed_cuda),
        "old_frozen_file_count": len(old_freeze["files"]), "old_frozen_files_intact": frozen_ok,
        "current_validators_intact": validation_freezes,
        "metadata_mapping": {k: metadata[k] for k in (
            "dense_source_keys", "dense_resolved_keys", "dense_unique_target_names",
            "packed_source_keys", "packed_runtime_names", "packed_default_buffers",
            "actual_model_parameter_coverage_verified")},
        "independent_review": "not performed; author self-review only",
    }
    write(iteration / "results/iteration-summary.json", summary)
    kernel = ("完成：完整模块、真实加载与 FlashInfer CUTLASS NVFP4 小算子共 20/20 项检查满足规则。"
              if completed_cuda else "未完成：完整模块/张量加载已验证，NVFP4 算子仍需解决最后一次运行记录中的阻塞。")
    report = f"""H3 / RTX 5070 环境与真实张量迭代报告
目标已确认：先跑通约 3 秒、低分辨率文生视频，优先 vLLM-Omni。
模型、质量和端到端性能 verdict：INCONCLUSIVE；尚未下载模型或生成视频。

执行结果
1. 已创建独立本地项目 InferRSI-MiniMax-H3，参照 InferRSI engine delivery layout。
   实际适配代码在 src/h3_omni_adapter/，测试/归档入口在 tools/。本地分支 codex/h3-bootstrap，无远程发布。
2. 已安装 WSL 3.0.3，Ubuntu 24.04 H3Lab 发行版放在 E 盘隔离目录；安装后可直接启动，无自动重启。
   WSL 的 Windows 系统组件仍由 Windows 管理；E 盘承载发行版、虚拟环境、源码 checkout 与大缓存。
   Windows 驱动 610.88 由 WSL 使用；未安装 Linux 显示驱动。
   Windows tensor venv、Linux venv、源码 checkout 与安装缓存均独立于原有项目环境。
3. Windows 官方 safetensors + PyTorch 检查：
   CPU 15/15，CUDA 16/16；Q/K 交换和 gate/up 缺失两种故障注入均被检出。
   实测 RTX 5070 capability 为 (12, 0)，Windows wheel 包含 sm_120。
4. Linux：完整原始 H3 模块可导入，CPU 参数加载 15/15；CUDA 加载检查在算子尝试中完成 16/16。
   NVFP4 计算状态：{kernel}
5. 完整头信息审计：338 个 dense 源名称全部解析为 334 个唯一目标；1,500 个 packed 源字段映射到
   1,750 个运行时名称（含 250 个默认 input_scale buffer）；两组无目标名冲突，另需恢复 rope.inv_freq。
   这是名称审计，尚未逐项与完整模型对象核对 shape/dtype/参数和 buffer 集合。

环境记录
Windows：Python 3.12.8 / PyTorch 2.14.1+cu130 / safetensors 0.8.0 / NumPy 2.5.2。
Linux：Python 3.12.3 / vLLM 0.32.0 / PyTorch 2.13.0+cu130 /
vLLM-Omni 0.32.0rc2.dev41+g325d57135；固定 checkout 分支 codex/h3-nvfp4-loader，未修改上游源码。
首次 JIT 使用独立 CUDA 13.0 工具链、g++ 和进程级 CUDA_HOME/PATH/LIBRARY_PATH；不依赖运行环境自动解析的 NVCC 13.4。
完整依赖清单在 logs/24、31、45 对应 JSON；依赖一致性检查在 logs/30、47。
WSL 当前可见约 15.3 GiB RAM、4 GiB swap；Windows 整机约 31.35 GiB RAM。
隔离环境并不会自动获得整个主机的内存。新模型预算需包含页缓存、pinned buffers、临时副本和运行库工作空间。

失败与修复
- 未提升权限的 WSL 初始安装未成功；正常 UAC 流程后成功。
- 初始非交互 Linux PATH 不含 sbin，useradd 调用失败，后续用户相关两次调用也失败；改用完整路径后恢复。
- WSL stderr 混合 UTF-16/UTF-8 导致少数早期日志尾部乱码；保留早期记录并修正后续捕获工具。
- Windows uv 曾遇到缓存重命名的临时访问错误，自动重试后安装成功。
- NVFP4 首次运行依次定位到：编译工具路径缺失、NVCC 13.4 与 13.0 头文件不匹配、
  独立工具链缺 cuBLAS/cuRAND 头文件、pip 包布局缺 lib64/libcudart 链接名及 WSL 驱动库查找路径。
- 首次 CUTLASS GEMM 编译超过原 180 秒预算。保留该次 exit 124，单独将首次编译预算调整为 600 秒，MAX_JOBS 保持 2。
  正确性 validator 字节未改变；安装和 JIT 时间不是模型推理性能。
每次尝试的结果、退出码、源文件哈希和错误保存在 results/linux-* 下，不能只看中间 16 项完成状态。

实现限制
- 原型仍限制单文件 8 MiB；只支持两类 dense 名称、七种 packed projection 和 TP1。
- 暂存统计为两份 byte slot 加两份 tensor slot，各组 32 KiB；包含在实验内，但不是 H3 RSS 峰值上限。
- CUDA pinned slot 的复用由事件等待保护，当前只验证单 stream。
- 单个 NVFP4 forward 是普通加载与暂存加载的相同性及零输入检查，不是独立量化精度、视频质量或全模型推理证明。
- 尚未完成生产 DLO 接线、完整 DiT 构造、分阶段 encoder/DiT/decoder 生命周期或真实视频基线。
- 旧原型冻结的 {len(old_freeze["files"])} 个文件保持不变；新验证器冻结校验结果见 iteration-summary.json。
- 仅作者自查，未进行独立 review。未建立正式 acceptance issue，未发布或推送。

下一顺序
按 plan/NEXT-ITERATION.zh-CN.txt 推进完整参数/buffer 覆盖、mmap 规划失败时禁止超内存回退、
text-to-video 的完整 handoff、阶段进程退出与实测容量，再形成可审查的模型文件清单和基线参数。
当前不进入大模型下载，也不将现有默认 CPU offload 配方视为可在本机运行。

目录
计划：plan/ROADMAP.zh-CN.txt、BASELINE-SCOPE.zh-CN.txt、NEXT-ITERATION.zh-CN.txt
机器可读汇总：results/iteration-summary.json
安装和修复：logs/
原始运行与失败：results/
冻结验证器：snapshots/ 与 plan/*freeze*.json
当前项目快照：patch/project-snapshot/
"""
    (iteration / "REPORT.zh-CN.txt").write_text(report, encoding="utf-8")
    status = f"""H3 / RTX 5070 当前状态（2026-10-22）
首个目标：约 3 秒、低分辨率文生视频；vLLM-Omni 优先。
当前已进入真实环境验证：独立 H3 项目、E 盘 Windows tensor venv、WSL2 Ubuntu 与 Linux vLLM/Omni 环境已建立。
Windows CPU 15/15、CUDA 16/16；两种故障注入检出。Linux 完整 H3 模块和 CPU loader 15/15；CUDA 加载16项完成。
NVFP4 小算子：{kernel}
完整头名称审计已完成，实际整模型参数覆盖、mmap 运行接线与分阶段生命周期仍待开发。
没有下载大模型、生成视频或执行优化 A/B。整体 verdict 仍 INCONCLUSIVE。
最新报告：iterations/2026-10-22-environment-bootstrap/REPORT.zh-CN.txt。
完整计划及下一开发切片见该目录的 plan/。
此前状态已备份为该目录 before-STATUS.zh-CN.txt；原始调查和冻结原型保持不变。
"""
    (root / "STATUS.zh-CN.txt").write_text(status, encoding="utf-8")
    (root / "README.zh-CN.txt").write_text("""H3 / FastH3 on RTX 5070 — local InferRSI records

Canonical root: <VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/
Current status: STATUS.zh-CN.txt
Latest executed iteration: iterations/2026-10-22-environment-bootstrap/REPORT.zh-CN.txt
Its plan/ contains the roadmap, confirmed text-to-video objective, freezes and next implementation slices.
Its logs/ preserves environment setup and fixes; results/ preserves every completed/failed/timed-out attempt.

Historical:
- prototypes/2026-10-22-synthetic-loader/: frozen byte-only fixture, validators and source provenance.
- planning/2026-10-22-environment-and-repository/: original environment/project proposal.
- planning/2026-10-22-goal-alignment/: original objective alignment.
- research/2026-10-22/loading-options/: mmap, offload, staging audit and external 5070 cases.
- research/2026-10-22/memory-budget/: saved headers, source audit and static capacity ledger.
- preflight/: original environment and migration records.

Model/quality/performance verdict remains INCONCLUSIVE. No model payload has been downloaded and no output video generated.
Runtime environments and caches live separately on E; experiment evidence remains here.
""", encoding="utf-8")
    files = [path for path in project.rglob("*") if path.is_file()
             and not any(part in {".git", "__pycache__", ".venv", ".ruff_cache"} or part.endswith(".egg-info")
                         for part in path.relative_to(project).parts)]
    snapshot = iteration / "patch/project-snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    source_manifest = {}
    for path in files:
        relative = path.relative_to(project)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_manifest[relative.as_posix()] = sha(path)
    write(iteration / "project-source-manifest.json", source_manifest)
    manifest = {path.relative_to(iteration).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(iteration.rglob("*")) if path.is_file() and path.name != "artifact-manifest.json"}
    write(iteration / "artifact-manifest.json", {"files": manifest,
        "campaign_verdict": "INCONCLUSIVE", "source_snapshot_files": len(source_manifest)})
    print(json.dumps({"linux_nvfp4_completed": bool(completed_cuda), "old_freeze_intact": frozen_ok,
                      "current_validators_intact": validation_freezes, "files": len(manifest),
                      "source_snapshot_files": len(source_manifest)}))

if __name__ == "__main__":
    main()
