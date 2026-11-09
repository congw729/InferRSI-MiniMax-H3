# vLLM-Omni adapter research

For video generation, use the maintained [ComfyUI quick start](../README.md). This page describes the separate experimental adapter and preserved research history; it is not another ready-to-run inference entry point.

The project follows [InferRSI](https://github.com/InferMatrixLab/InferRSI)'s engine delivery layout. InferRSI organized research execution and evidence; upstream model classes and native weight loaders perform the adapter's model operations. The repository is now a **public experimental preview**, with real ComfyUI output examples in [the showcase](SHOWCASE.md). Public availability does not imply upstream acceptance or complete vLLM-Omni serving support.

## Adapter capabilities and limits

The external adapter includes TP1 mmap loading, native text-layer streaming, compact temporal decoding, a runtime-layout cache bridge to the actual vLLM-Omni DLO backend, and serial process handoff. Later modules also cover original-H3 dense text/VAE loading and bounded INT8 ConvRot recovery; see [the module map](architecture.md#experimental-vllm-omni-adapter).

The earlier FastH3 studies cover full meta parameter/buffer inventories, exact tiny native-loader equivalence, bounded real-shape component capacity, and an integrated tiny text → native DLO with eight PDD sampling steps → temporal decoder chain. That chain exercises the native legal four-second geometry and a video-only 72-frame crop at 24 fps. Its outputs are synthetic tensors, not meaningful generated video. The separate ComfyUI experiments did generate real video.

The generic loader does not automatically select the DLO bridge; full pipeline construction and the serving CLI remain unintegrated. A historical full-width two-block run reached exact eight-step outputs before an RSS guard interrupted cleanup. An opt-in terminal-discard implementation was added later, but does not retroactively complete that failed lifecycle check. See [the dated DLO findings](dlo-lifecycle-gap.md). Full vLLM-Omni deployment, quality and service performance remain **INCONCLUSIVE**.

## Research and record locations

- [Historical tiny-chain reproduction](reproducibility.md): dedicated Linux environment and external prerequisites; not a ComfyUI setup guide.
- [Source attribution](provenance.md) and [evidence ownership](../evidence/README.md).
- [Upstream integration status](../patches/vllm-omni/README.md).
- [Archived ComfyUI experiments](../experiments/comfyui/rtx5070-12gb/2026-10-31-speed-quality-search/): actual-video controllers, measurements, failed attempts and frozen identities. These results do not establish vLLM-Omni performance.
- [Historical progress ownership](PROGRESS-AND-REPRODUCTION.zh-CN.txt): the canonical local dashboard and notes remain in vllm-omni-draft; the dashboard was not migrated.
- [2026-11-01 backup manifest](backup-manifest-2026-11-01.json): a historical export inventory, not a manifest of today's repository. Two excluded local-path tools have labelled copies under `archive/legacy-tools-redacted/`.

The measured host is an RTX 5070 12GB with 32GB RAM; hardware and software identities belong to each experiment. Selected generated comparison media are now included under `docs/assets/showcase/`. Weights, environments, original raw run collections and large logs remain external. Historical failures, frozen validators and source identities are preserved.

Original project code is experimental work under the repository's code license. External source/model terms remain separate; see [third-party notices](../THIRD_PARTY_NOTICES.md).
