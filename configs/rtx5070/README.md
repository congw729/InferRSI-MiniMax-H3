# RTX 5070 configuration

The tested host has one RTX 5070 with 12GB VRAM and 32GB system RAM. Nominal capacity is not free memory; the current client checks starting headroom as described in [the model/environment guide](../../quickstart/MODELS.md).

The maintained ComfyUI configuration is [quickstart/trim8.json](../../quickstart/trim8.json): FastH3 Trim NVFP4, 8 steps, 384 × 256, 73 frames at 24 fps, with audio. Follow the [startup commands](../../README.md#quick-start--windows-portable-comfyui); no separate configuration in this directory is required.

Other original-H3/Turbo/Trim settings are documented with their [historical outputs and timings](../../docs/SHOWCASE.md). They are not extra quick-start presets. Full vLLM-Omni deployment and formal quality/service-performance acceptance remain **INCONCLUSIVE**.
