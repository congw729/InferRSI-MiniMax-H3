# Architecture

`byte_adapter.py` is the preserved bounded TP1 mapping/layout prototype, using an explicit index plus selected NVFP4 sidecars. It retains the fixture file-size guard.

`tensor_loader.py` transfers its transformed bytes into caller-owned real PyTorch parameters. Two pinned host tensor slots and CUDA events protect slot reuse in the tested single-stream scenario. It does not implement model residency, background prefetch or whole-model offload.

The independent validator uses official safetensors, a preserved generic byte oracle and original upstream PyTorch loader methods. The Linux layer imports the full upstream module without import stubs. Its tiny NVFP4 check compares ordinary versus staged loading on identical weights and also checks zero input; it is not an independent quantization-quality oracle.

The next runtime work is full parameter/buffer coverage and construction/lifetime control for text encoder, DiT and decoder. The user's initial baseline direction is approximately three seconds of low-resolution text-to-video. Exact resolution, frame count, model variant and quality gates are still open.
