# vLLM-Omni integration status

A dedicated Linux checkout exists in isolated WSL2, on `codex/h3-nvfp4-loader`. Its identity is recorded as research provenance. No upstream framework patch is applied.

The external H3 adapter constructs native models, uses native weight loaders and exports final runtime-layout grouped caches. These caches bind through the real HostWeightPlan to DistributedLayerwiseOffloadBackend. Frozen tiny tests exercise its actual mmap, block hooks and repeated PDD steps, including the integrated text/temporal chain.

The unmodified generic planner remains insufficient: its indexed source path omits packed sidecars, source names differ from runtime targets, and refiner QKV needs multiple sources combined into one binding. Name remapping alone is insufficient. The external bridge resolves these before native DLO loading, but is not automatically selected by the generic loader.

Full pipeline construction happens before generic planning, VAEs load eagerly, and text/vision construction needs explicit separation. Therefore `vllm serve` and the normal diffusion CLI have not been demonstrated. A future framework patch belongs in the isolated checkout after reading its repository instructions, with a reproducible diff saved in canonical evidence.
