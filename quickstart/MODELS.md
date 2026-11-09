# Models and ComfyUI environment

This preview uses **FastH3 Trim NVFP4, 8 steps, 384 x 256, 73 frames at 24 fps** (about 3.04 seconds). Trim is a pruned/distilled H3 variant; it is not the original 50-block model or an assertion of unchanged quality.

中文：先准备已有 ComfyUI 环境，将下面四个模型放入对应目录，再按首页运行。脚本不会下载模型；已有文件校验一致即可复用。

## Four required files

Download the exact files below from their publishers. Keep the filenames unchanged. Folder paths are relative to `ComfyUI/models/`. For an external model root, configure its subfolders in ComfyUI itself first; the client's `--model-root` only selects files for hash verification and does not change the server's model search paths. The four files total **30,998,085,709 bytes = 31.00 GB / 28.87 GiB**; also leave space for ComfyUI, caches, conditioning and video output.

| Role | Download | Folder | Bytes |
| --- | --- | --- | ---: |
| trim-dit | [`fastvideo_fasth3_trim_8step_nvfp4.safetensors`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-Comfy/resolve/f66c13dc17391dd4da99242a407918b92bf7d5fc/diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors) | `diffusion_models/` | 11,894,623,166 |
| text-encoder | [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `text_encoders/` | 15,687,142,551 |
| video-vae | [`minimax_h3_video_vae_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_video_vae_int8_convrot.safetensors) | `vae/` | 2,811,065,184 |
| audio-vae | [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/vae/minimax_h3_audio_vae_fp32.safetensors) | `vae/` | 605,254,808 |

The fixed revisions, SHA-256 hashes and sizes are in [models.json](models.json). After downloading, compare SHA-256 as well as size. With the dedicated ComfyUI server running, verify all four files from this repository:

```powershell
python quickstart/run_h3.py doctor --comfy-root "<path-to-ComfyUI>" --verify-weights
```

For an external model root containing `diffusion_models/`, `text_encoders/` and `vae/`, add `--model-root "<path-to-model-root>"`. `doctor` checks the server and starting memory margins as well; it does not submit a generation job. To hash one file without starting ComfyUI, use PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 "PATH_TO_DOWNLOADED_FILE.safetensors"
```

These links pin `FastVideo/FastVideo-FastH3-Trim-Comfy` at `f66c13dc17391dd4da99242a407918b92bf7d5fc` and `Comfy-Org/MiniMax-H3` at `e5eb578a89295337b8ff433a035929ce0279e0b6`. Only the listed files are required; a full repository download is unnecessary.

## Existing ComfyUI installation

Use an isolated **Windows NVIDIA ComfyUI installation**. The measured reference is [ComfyUI v0.43.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0), source commit `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`. Its recorded portable environment used Python 3.13.14, PyTorch 2.14.1+cu130, comfy-kitchen 0.2.40 and comfy-aimdo 0.6.0. The current client accepts only ComfyUI 0.43.0 and rejects other server versions; supporting another version requires a client change and separate verification. Copying only the Python client does not install ComfyUI or its CUDA dependencies.

The workflow uses native ComfyUI nodes; no custom-node pack is required. Start a separate, idle ComfyUI instance for this preview because the client frees models between text encoding and video generation. Do not share that instance with other active jobs.

For the reference Windows portable package, run this from its top-level directory:

```powershell
.\python_embeded\python.exe -s -B .\ComfyUI\main.py `
  --listen 127.0.0.1 --port 8188 `
  --disable-auto-launch --disable-all-custom-nodes --disable-partner-nodes `
  --offline --disable-assets --disable-pinned-memory --disable-smart-memory `
  --cache-none --preview-method none --vram-headroom 1.0 --reserve-vram 3.0
```

This reproduces the recorded memory policy; individual flags have not each been proven necessary by ablation. `--offline` is a ComfyUI frontend setting, not an operating-system network firewall. In the reference runtime, DynamicVRAM is detected automatically and NVIDIA asynchronous offloading defaults to two streams; the startup log should show that mode. Do not add `--disable-async-offload` when reproducing this preset.

The host used an RTX 5070 12GB and 32GB RAM. Nominal capacity is not free memory: close other memory-heavy applications before running. The current client requires at startup at least **6 GiB free VRAM, 12 GiB available physical RAM and 16.5 GiB available Windows commit memory**. These are admission checks, not a guarantee that every prompt or runtime version fits. First startup can include kernel preparation and model loading.

## Model terms and attribution

This repository does not contain or redistribute model weights. Both fixed export model cards identify the **MiniMax H3 Community License Agreement**: [Trim publisher card](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-Comfy/blob/f66c13dc17391dd4da99242a407918b92bf7d5fc/README.md), [H3 export card](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/README.md), and [official agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/5d9b308a59ab12e67147f191e184baf704185bd1/LICENSE). It includes territory, use and redistribution conditions; check the actual terms before downloading or using the models. The code license does not replace those terms.

The H3 export card attributes the text-encoder conversion to [cybermotaz/Qwen3-VL-32B-Instruct-NVFP4](https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4/blob/dfcab1395094f56d3c78ba472158284def0fc278/README.md), based on [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct). Their license metadata differs from the H3 export card; this guide preserves the source chain without asserting that one label overrides the others.
