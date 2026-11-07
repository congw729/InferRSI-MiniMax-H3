> Historical adapter/backup overview, preserved from the pre-preview README. For the maintained ComfyUI user entry, see [the project README](../README.md).

# InferRSI-MiniMax-H3

Local H3 loading experiments organized by [InferRSI](https://github.com/InferMatrixLab/InferRSI), with vLLM-Omni as the preferred runtime. This follows InferRSI's engine delivery layout. It is an experimental integration project prepared for private development backup, not an independent inference engine release.

The external adapter provides strict TP1 mmap loading, native text layer streaming, a compact NVFP4 temporal video decoder, a runtime-layout cache bridge to the actual vLLM-Omni DLO backend, and serial process handoff. It uses upstream model classes and native weight loaders. InferRSI is the external experiment controller, not a runtime dependency.

- Implementation: `src/h3_omni_adapter/`
- [Reproduction and current boundaries](reproducibility.md)
- [Source attribution](provenance.md)
- [Local evidence ownership](../evidence/README.md)
- [Upstream integration status](../patches/vllm-omni/README.md)

Local frozen checks cover full meta parameter/buffer inventories, exact tiny native-loader equivalence, bounded real-shape component capacity, and an integrated tiny text → native DLO with eight PDD sampling steps → temporal decoder chain. The chain also exercises the native legal four-second request geometry and a video-only 72-frame crop at 24 fps. It produces synthetic tensors; it does not demonstrate meaningful video generation.

Full model deployment, real video quality and service performance remain **INCONCLUSIVE**. The generic loader does not automatically select this bridge, full pipeline construction and the serving CLI remain unintegrated, and the full-width DLO lifecycle remains blocked by a CPU memory peak during shutdown. A bounded two-block synthetic run reached exact eight-step outputs before its RSS guard stopped cleanup; this is not a completed lifecycle check. No real model payload is included.

Use a dedicated matched Linux environment; see the reproduction guide. Historical failures and frozen validators are preserved. Private source backup does not constitute an upstream contribution, a public release, or an acceptance verdict.

Original code is experimental local work. Upstream source snapshots retain their notices; this repository grants no model-weight license.

由 [InferRSI](https://github.com/InferMatrixLab/InferRSI) 生成。

## Development backup and experiment progress

Primary measured host: **RTX 5070 12GB, 32GB RAM**. Hardware is recorded per experiment rather than in the repository name.

- `src/h3_omni_adapter/`: experimental vLLM-Omni integration; the limitations above apply to this path.
- `experiments/comfyui/rtx5070-12gb/2026-10-31-speed-quality-search/`: archived actual-video ComfyUI experiments, including historical controller versions and frozen identities. These results do not establish vLLM-Omni service performance.
- [Progress, comparison results and reproduction boundaries](PROGRESS-AND-REPRODUCTION.zh-CN.txt): the canonical dashboard and notes remain in vllm-omni-draft; no dashboard migration.
- [Backup identities and exclusions](backup-manifest-2026-11-01.json): source/export hashes and privacy copies. Two historical local-path tools remain excluded; their labelled redacted copies are under `archive/legacy-tools-redacted/`.

Model weights, generated videos, environments and raw large logs are not stored in Git. Original local evidence is retained. This backup makes no new license grant for third-party code or model weights.
