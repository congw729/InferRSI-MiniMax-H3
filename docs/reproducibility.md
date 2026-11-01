# Reproduction and boundaries

The canonical evidence root is `<RECORD_ROOT>`. Current integrated iteration:
`iterations/2026-10-22-dlo-temporal-chain/`.
Read its report, frozen plan, results and manifest before rerunning. Earlier environment setup, failed attempts and byte-only fixtures remain in their original directories.

## Environment

The runtime uses a dedicated Ubuntu 24.04 WSL2 distribution, a separate Python 3.12 environment and an independent vLLM-Omni checkout. Recorded versions are torch 2.13.0+cu130, vLLM 0.32.0 and FlashInfer 0.7.0.post1. NVFP4 JIT uses a separate CUDA 13.0 compiler environment. The evidence records exact package/source identities.

The VM exposes approximately 15.3 GiB RAM and 4 GiB swap, distinct from host RAM. Installed dependencies and reviewed source/config downloads are not real model weights. All synthetic runs use offline flags.

## Commands

Supply real paths explicitly. `<TINY_CHECKPOINT>` is the unchanged checkpoint from the native-three-stage iteration. `<REVIEWED_NATIVE_SOURCE_BUNDLE>` is the recorded MiniMax source bundle. These are local prerequisites; the runner does not download them.

```text
<INFERRSI_PYTHON> -B <H3_PROJECT>/tools/run_experiment.py --record-root <RECORD_ROOT> --iteration <NEW_ITERATION> --linux-home <LINUX_HOME> --lab-root <LAB_ROOT> --wsl-distro H3Lab --script validate_dlo_temporal_chain_v1.py --name dlo-temporal-tiny --timeout 240 -- --checkpoint <TINY_CHECKPOINT> --native-source <REVIEWED_NATIVE_SOURCE_BUNDLE> --scenario tiny
<INFERRSI_PYTHON> -B <H3_PROJECT>/tools/run_experiment.py --record-root <RECORD_ROOT> --iteration <NEW_ITERATION> --linux-home <LINUX_HOME> --lab-root <LAB_ROOT> --wsl-distro H3Lab --script validate_dlo_temporal_chain_v1.py --name dlo-temporal-legal4s256 --timeout 240 -- --checkpoint <TINY_CHECKPOINT> --native-source <REVIEWED_NATIVE_SOURCE_BUNDLE> --scenario legal4s256
<INFERRSI_PYTHON> -B <H3_PROJECT>/tools/run_experiment.py --record-root <RECORD_ROOT> --iteration <NEW_ITERATION> --linux-home <LINUX_HOME> --lab-root <LAB_ROOT> --wsl-distro H3Lab --script validate_chain_guard_v1.py --name chain-owner-sigterm --timeout 60 --
```

Each invocation creates a new run through InferRSI's local executor and saves source hashes. Copy and verify the frozen plan/validators before execution; do not overwrite historical experiments. This is a validation entry point, not a user-facing inference CLI. Recorded process elapsed times include startup/import/JIT and are not inference performance measurements.

The frozen integrated controller directly owns each reference or stage process and reaps it before starting the next. It limits each worker to 180 seconds and the complete run to 240 seconds. Local experiment resource gates live in the evidence plan; they are not deployment recommendations or acceptance thresholds.

## Implementation limits

`H3MmapPlan` requires TP1, supported Diffusers H3 keys, an explicit index and packed sidecars. It inventories all parameters and buffers, including defined defaults. File budgets are explicit, with separate larger limits only in frozen capacity protocols. Header/file identity checks do not establish complete payload integrity.

`H3MmapLoader` uses the original native loader and retains resident tensors plus one streamed group, with synchronous release. It preserves native parameter classes and loader metadata. Logical group limits are not process RSS guarantees. It does not provide prefetch or concurrent request support.

`text_streaming` keeps the actual native text model and weight loader, constructs on meta, and runs layer by layer. It excludes vision and consumes already prepared token IDs. The integrated fixture uses bounded synthetic IDs, without a tokenizer, special-token processing or meaningful prompts. The real-size embedding plus one encoder layer has separate capacity evidence; the full encoder is untested.

`dlo_bridge` converts one group at a time on CPU into final runtime-layout safetensors, then emits the actual HostWeightPlan and restores defined nonpersistent PDD buffers. The native DLO backend handles two GPU slots and rank-local mmap. The cache adds disk usage and conversion work. Its manifest includes file hashes, but its current in-memory object binds to the same model identity once; reopening a cache in a fresh process is not implemented. Native DLO replaces parameter classes, so the validated scope is forward execution, not hot reloading custom loader metadata. Tiny two-block eight-step equivalence does not establish full-width DLO memory or full-model fit.

`compact_decoder` constructs only the released compact ViT depth on meta. `temporal_decoder` hosts the original temporal/spatial overlap, blending, padding and trim methods and preserves the released FP32 pixel conversion. Tiny nonuniform fixtures compare raw tensors and pixels against the native reference. A real-shape eight-layer synthetic decoder has separate full temporal clip capacity evidence.

`stage_handoff` uses bounded JSON/safetensors with digests. The current integrated chain runs actual tiny text encoding, eight native PDD updates and native temporal decoding in three fresh processes. The legal four-second request becomes 107 aligned frames; the first 72 frames represent a three-second video-only crop at 24 fps. Audio rows participate in denoising but audio decoding and synchronization are absent. Output artifacts contain synthetic tensors and hashes, without MP4 or real video.

The integrated DiT uses CUDNN_ATTN. Production VSA, all released layers, real weights, meaningful prompt generation, end-to-end memory, quality and service performance are unverified. Individual component memory peaks cannot be added together as full-model evidence.

FlashInfer and cuBLAS retain process-owned CUDA workspaces after tensor deletion. Stage process exit supplies the final lifetime boundary. The controller's SIGTERM cleanup is separately exercised with a sentinel process.

Historical `byte_adapter.py` and frozen validators remain unchanged.

## Status

Campaign verdict: INCONCLUSIVE. No acceptance issue or independent review exists. The actual native DLO backend bridge runs; the generic planner, eager full pipeline construction and serving CLI remain unintegrated. No upstream runtime patch is applied.
