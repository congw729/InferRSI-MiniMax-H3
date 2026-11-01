# Current validation entry points

The meaningful checks are in `tools/validate_tensors.py` and `tools/validate_runtime.py`, executed with the external InferRSI controller. Fixtures and validator freezes stay in the canonical record tree, not this repository.

The first suite covers official safetensors parsing, independent byte-oracle parity, real upstream PyTorch parameter loading, pinned CUDA staging and fault controls. The second imports the complete unmodified H3 module and attempts a tiny NVFP4 forward check.

These are local L0 contracts; there is no model quality or service acceptance suite yet.
