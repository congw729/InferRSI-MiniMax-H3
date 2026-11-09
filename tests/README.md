# Validation entry points

## ComfyUI preview client

Run the offline client checks from the repository root:

```text
python -B -m unittest discover -s quickstart -p test_run_h3.py
```

[quickstart/test_run_h3.py](../quickstart/test_run_h3.py) uses Python 3.10+ and the standard library. It exercises graph construction, local API boundaries, cancellation/cleanup ownership, artifact paths, memory gates and file verification without a server, models or GPU.

The separate [functional smoke record](../quickstart/SMOKE-TEST.json) documents an actual generated clip and cleanup. Neither offline checks nor one functional clip establishes model quality or service performance.

## Experimental adapter

[validate_tensors.py](../tools/validate_tensors.py) and [validate_runtime.py](../tools/validate_runtime.py) are early research validators, not the entire current test suite. Later protocols cover mmap, native text/VAE loading, ConvRot and DLO lifetime behavior. The `test_stage_dlo_discard_v1.py` and `test_stage_dlo_discard_v2.py` files are preserved versions tied to their frozen protocols, not two interchangeable current smoke suites.

These tests need matched external vLLM-Omni/Torch installations, source bundles and fixtures described in their plans. The adapter package targets Python 3.12+; this requirement is separate from the standalone preview client. See [historical reproduction](../docs/reproducibility.md) and [evidence ownership](../evidence/README.md) before choosing a validator. Do not bulk-run historical versions or alter their frozen identities.

Formal model quality and service-performance acceptance remain **INCONCLUSIVE**.
