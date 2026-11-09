# vLLM-Omni integration status

The adapter studies used a dedicated Linux checkout in isolated WSL2, on `codex/h3-nvfp4-loader`; the audited source identity is recorded in [provenance](../../docs/provenance.md). The behavior described here is specific to that research runtime, not a claim about every newer upstream release. No upstream framework patch is included in this repository.

The external H3 adapter constructs native models, uses native weight loaders and exports final runtime-layout grouped caches. These caches bind through the real HostWeightPlan to DistributedLayerwiseOffloadBackend. Frozen tiny tests exercise its actual mmap, block hooks and repeated PDD steps, including the integrated text/temporal chain.

The unmodified generic planner in the audited runtime was insufficient: its indexed source path omits packed sidecars, source names differ from runtime targets, and refiner QKV needs multiple sources combined into one binding. Name remapping alone is insufficient. The external bridge resolves these before native DLO loading, but is not automatically selected by the generic loader.

Full pipeline construction happens before generic planning, VAEs load eagerly, and text/vision construction needs explicit separation. Therefore `vllm serve` and the normal diffusion CLI have not been demonstrated. A future framework patch belongs in the isolated checkout after reading its repository instructions, with a reproducible diff saved in canonical evidence.

A local [StageDLOBackend](../../src/h3_omni_adapter/stage_dlo.py) adds an explicit terminal-discard path without changing upstream defaults; see [its scope and the historical lifecycle gap](../../docs/dlo-lifecycle-gap.md). The maintained [ComfyUI preview](../../README.md) uses a separate runtime and does not depend on these adapters.
