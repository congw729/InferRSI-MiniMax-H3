# Evidence ownership

The maintainer's canonical local experiment root is `<VLLM_OMNI_DRAFT_ROOT>/src/dev_notes/h3-5070-inferrsi/`, supplied to research commands as an explicit path. Public quick-start users do not need this directory or access to the separate notes repository.

This repository includes selected archived controllers/results under `experiments/`, the preview's [functional smoke record](../quickstart/SMOKE-TEST.json), and [showcase data/media](../docs/SHOWCASE.md). It is not a full copy of raw run collections, model weights, environments or earlier external source bundles. The [2026-11-01 backup manifest](../docs/backup-manifest-2026-11-01.json) describes that export date, not later additions.

New preview runs save an MP4 and a JSON record beside the requested output; the dedicated ComfyUI installation retains intermediate artifacts in its unique run directory. Research reproduction needs the original protocol's explicit source/fixture/output paths. Preserve failed attempts and frozen validators.

Formal campaign exports use `plan/`, `results/`, `patch/` and `validator/` after acceptance scope and gates are established. Recorded functional results and observed timings are distinct from a formal model-quality/service-performance verdict, which remains **INCONCLUSIVE**.
