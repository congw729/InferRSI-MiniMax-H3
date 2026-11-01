# Provenance

The layout is adapted from InferMatrixLab/InferRSI's engine delivery template. The project is locally created under the user's instruction, not an organization-owned remote repository.

The initial `byte_adapter.py` is copied without modification from:
`<RECORD_ROOT>/prototypes/2026-10-22-synthetic-loader/patch/adapter.py`.
Its original fixtures, validator freeze and source snapshots remain preserved there.

The serialized NVFP4 layout and native target mapping were audited from vLLM-Omni source revision `325d5713503dc6a64a773d8afdc372838568a94d`:
[checkpoint_quantization.py](https://github.com/vllm-project/vllm-omni/blob/325d5713503dc6a64a773d8afdc372838568a94d/vllm_omni/diffusion/models/minimax_h3/checkpoint_quantization.py).
The local source manifest records the authoritative upstream path and hash; source snapshots retain Apache-2.0 notices.

This revision is research provenance, not a frozen baseline/candidate performance campaign. Verify the recorded source URL against the source manifest before citing it.
