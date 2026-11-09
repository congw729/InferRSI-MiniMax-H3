# Architecture and current scope

This public experimental repository has two runtime paths. The maintained user entry is the **ComfyUI preview**; the **vLLM-Omni adapter** remains research code. InferRSI organizes the recorded experiments externally and is not required by the preview client.

## ComfyUI preview

`quickstart/run_h3.py` is a Python 3.10+ standard-library client for an existing, dedicated local ComfyUI v0.43.0 server. It validates the environment, submits text encoding, releases loaded models, then submits video generation and records the output and stage times. ComfyUI performs inference; the client does not start the server or download weights.

The supplied `quickstart/trim8.json` preset uses FastH3 Trim NVFP4, 8 steps, 384 × 256, 73 frames at 24 fps, with audio. Trim is pruned/distilled and differs from original H3. See the [quick start](../README.md#quick-start--windows-portable-comfyui), [model requirements](../quickstart/MODELS.md) and [functional smoke record](../quickstart/SMOKE-TEST.json).

Actual original-H3, Turbo and Trim comparison outputs are available in [the showcase](SHOWCASE.md). Those historical configurations are not additional maintained quick-start presets; the timing conditions and quality limitations are stated with the results.

## Experimental vLLM-Omni adapter

The code under `src/h3_omni_adapter/` uses separately installed upstream model classes and loaders. It is not the backend of the ComfyUI preview.

| Components | Research responsibility |
| --- | --- |
| `byte_adapter`, `tensor_loader` | Preserved bounded byte/layout prototype and tensor staging; not whole-model offload. |
| `mmap_loader`, `text_streaming` | Packed-weight loading plans and native text-layer streaming for the earlier FastH3 experiments. |
| `dlo_bridge`, `stage_dlo` | Grouped runtime-layout caches for native DLO, plus an explicit terminal-stage discard implementation. Discard forbids model reuse and is not a serving API. |
| `compact_decoder`, `temporal_decoder` | Released compact decoder construction and native temporal/spatial decode semantics. |
| `native_mmap`, `native_text_mmap`, `native_vae_mmap`, `native_temporal_decoder` | Separate original-H3 dense loading and decode paths. |
| `convrot`, `native_convrot_mmap` | Bounded INT8 ConvRot recovery and native loading experiments. |
| `stage_handoff`, `stage_process` | Bounded artifact handoff and owned-process lifetime control. |

Frozen studies cover parameter/buffer inventories, tiny numerical equivalence, component capacity and selected lifecycle behavior. These are distinct from complete model deployment. The generic loader, full pipeline construction and serving CLI are not integrated; full vLLM-Omni deployment, quality and service performance remain **INCONCLUSIVE**.

[Adapter overview](ADAPTER-RESEARCH.md), [historical tiny-chain reproduction](reproducibility.md) and [DLO lifecycle findings](dlo-lifecycle-gap.md) describe the evidence scope. Their dated experiments are not current deployment instructions or proof that every adapter path is validated.
