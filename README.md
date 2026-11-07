# InferRSI-MiniMax-H3

**Run a short H3-family video on an RTX 5070, and keep the configuration and timings.**

Experimental preview for an **existing ComfyUI installation**. The default is **FastH3 Trim, 8 steps, 384 × 256, 73 frames at 24 fps (~3.04 s), with audio**. Trim is a pruned/distilled H3 variant; it is not the original full H3 model and quality can differ.

中文：这是面向 RTX 5070 的 H3 运行与实验工具。首版只提供一个配置：连接已有 ComfyUI，输入提示词，生成约 3 秒视频并保存耗时。请先看下面的准备与运行步骤。

## What you need

- A dedicated, idle **ComfyUI v0.43.0** instance with native H3 nodes; no custom nodes are needed for this preset.
- Target machine: **RTX 5070 12GB / 32GB system RAM**. Close memory-heavy applications before the first run; the launcher checks resource headroom.
- The four model files listed in [MODELS.md](quickstart/MODELS.md), about **28.87 GiB** in total. Model licenses are separate from this project's code license. Download them from their official sources and place them in the listed ComfyUI model folders.
- Python 3.10+ for the small client script (standard library only). ComfyUI's portable Python can also run it.

This repository does not download models or install/change drivers. The experimental vLLM-Omni adapter and historical experiments are documented separately below.

## Quick start — Windows portable ComfyUI

Get the client once:

```powershell
git clone https://github.com/congw729/InferRSI-MiniMax-H3.git
```

**1. Start a dedicated ComfyUI instance.** From the ComfyUI portable folder, run:

```powershell
.\python_embeded\python.exe -s ComfyUI\main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-all-custom-nodes --disable-partner-nodes --offline --disable-assets --disable-pinned-memory --disable-smart-memory --cache-none --preview-method none --vram-headroom 1.0 --reserve-vram 3.0
```

Keep this terminal open. Wait for the server to be ready. Use this instance only for this client while generating: it releases loaded models between text encoding and video generation to fit the memory budget. The recorded runtime and version details are in [MODELS.md](quickstart/MODELS.md).

**2. Check the environment.** In another terminal, from this repository:

```powershell
python quickstart/run_h3.py doctor --comfy-root "<path-to-ComfyUI>"
```

Replace `<path-to-ComfyUI>` with the folder containing `main.py`, not the outer portable folder. If `python` is not available, use the full path to ComfyUI's `python_embeded/python.exe` instead. The default server address is `http://127.0.0.1:8188`.

**3. Generate a video.**

```powershell
python quickstart/run_h3.py generate --comfy-root "<path-to-ComfyUI>" --prompt "A red toy car rolls slowly across a wooden table. Static camera, continuous shot. Audio: quiet room ambience." --output "./outputs/demo.mp4"
```

Use `--help` for the supported options. Choose a new MP4 filename for each run; existing files are preserved. ComfyUI stores its intermediate files under a unique run directory. The result includes an MP4 and a small record of the prompt, seed, configuration and observed elapsed times. Text encoding and video-stage wall times are reported separately from total client time; server startup and prior model downloads are outside that total.

The preview has a [recorded functional smoke](quickstart/SMOKE-TEST.json): one new prompt produced 73 decoded video frames and an audio stream, with model cleanup and owned-service teardown checked. This is not a quality benchmark.

The first request can be slower due to model loading and kernel initialization. This preview does not promise a fixed speed or original-model quality.

## Troubleshooting

- **Cannot connect:** start ComfyUI first, check its terminal, and match the client's `--url` to the server port.
- **Missing node or model:** use the documented ComfyUI version and [model filenames/folders](quickstart/MODELS.md); check `doctor` before generating.
- **Not enough memory:** close unnecessary GPU/RAM-heavy applications and retry. The client does not terminate other applications.
- **A failed or cancelled request:** inspect the run record and ComfyUI terminal. Keep the diagnostic output; do not submit multiple jobs to the same instance while a request is still stopping.
- **Custom model/output locations:** create a custom ComfyUI output directory before starting the server; see `--help` for the explicit path options. The client and ComfyUI must refer to the same local installation and output folders.

## Project scope

- `quickstart/`: the maintained minimal ComfyUI preview.
- `src/h3_omni_adapter/`: experimental vLLM-Omni loading/streaming adapters. The quick start does not use this backend; full vLLM-Omni serving integration is a separate research goal.
- `experiments/` and `archive/`: preserved development history, including failed attempts and frozen identities. These are not additional user-facing launchers and may reference external evidence that is not bundled.
- [Adapter research notes](docs/ADAPTER-RESEARCH.md) and [source attribution](docs/provenance.md).

InferRSI organizes the research comparisons and evidence. The quick-start client does not require installing InferRSI. Future benchmark results must distinguish model changes, cache conditions and complete output workloads; a working demo is not a formal quality or performance acceptance result.

## License

Original project code is licensed under [Apache-2.0](LICENSE). ComfyUI, other dependencies and model weights retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No model weights are distributed here.
