# License and third-party sources

The project-authored client, adapters, configuration, documentation and experiment tools are licensed under [Apache License 2.0](LICENSE), unless a file carries a different notice. This grant covers this project's contributions; it does not relicense any external engine, model, weight export, dependency or separately obtained source bundle.

The preview connects to a separately installed ComfyUI server. It does not bundle ComfyUI, vLLM-Omni, InferRSI, model implementations, weights or Python environments. Model files remain user-supplied and subject to their own terms. Selected AI-generated comparison previews are included under `docs/assets/showcase/`; their source models and transformations are documented in [the showcase](docs/SHOWCASE.md). The project code license does not override applicable model/output terms.

## Software sources

| Project | Relationship to this repository | Upstream terms |
| --- | --- | --- |
| [InferMatrixLab/InferRSI](https://github.com/InferMatrixLab/InferRSI) | The layout follows its engine delivery template; archived experiments call its external experiment/evidence APIs. The preview client does not require the InferRSI runtime. | [Apache-2.0](https://github.com/InferMatrixLab/InferRSI/blob/main/LICENSE). Copyright and other applicable notices remain with the upstream contributors. |
| [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | External video-generation runtime accessed through its local HTTP API; workflows refer to its native node identifiers. No ComfyUI runtime source is included. | [GNU GPL version 3](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE). Installing or redistributing ComfyUI is governed by its own license and notices. |
| [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) | Experimental adapters import its native classes/loaders. Research attribution, including the source revision used for the checkpoint-layout contract, is recorded in [docs/provenance.md](docs/provenance.md). The engine is not bundled. | [Apache-2.0](https://github.com/vllm-project/vllm-omni/blob/main/LICENSE). |
| [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) | H3 is the model family targeted by the client and adapters. Historical decoder wrappers call a separately supplied original model-source package; that package is not distributed here. | The upstream [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/5d9b308a59ab12e67147f191e184baf704185bd1/LICENSE), rather than this project's Apache license, governs the upstream H3 materials. |
| [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) | Historical optional experiments record this node package's identities and installation references. Its source and weights are not included or required by the Trim preview. | [Apache-2.0 source license at the recorded revision](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/blob/4274783a23afcfdbea3b4876cb79effd6c510785/LICENSE); model weights have separate terms. |

## Model sources and use

Weights must be obtained separately from their publishers. Follow the exact export's model card and any referenced base-model terms; this project's code license grants no model-use or redistribution rights.

- Trim export: [FastVideo/FastVideo-FastH3-Trim-Comfy model card](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-Comfy/blob/f66c13dc17391dd4da99242a407918b92bf7d5fc/README.md).
- H3 component exports: [Comfy-Org/MiniMax-H3 model card](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/README.md).
- Text encoder conversion provenance: [cybermotaz/Qwen3-VL-32B-Instruct-NVFP4 model card](https://huggingface.co/cybermotaz/Qwen3-VL-32B-Instruct-NVFP4/blob/dfcab1395094f56d3c78ba472158284def0fc278/README.md) and [Qwen/Qwen3-VL-32B-Instruct base model](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct). The conversion card and base model show differing license metadata; the conversion card's linked model-repository LICENSE was unavailable when checked. Do not infer that every export has identical terms from the base model name alone.

The H3 Community License includes acceptable-use and geographic conditions. Its stated excluded territories are the European Union, United Kingdom, Republic of Korea and United States; the upstream directs interested users in those territories to obtain separate authorization. It also addresses model derivatives, outputs, distribution and services. Read the [full agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/5d9b308a59ab12e67147f191e184baf704185bd1/LICENSE) before downloading or using the model. Providing this independent client is not a representation that model use is authorized for every user or location.

## Historical material

The `experiments/` and `archive/` trees preserve this project's historical controllers, validators, configuration descriptions, patches, measurements and source-identity references. Their manifests can point to external packages or source snapshots that are not included. Such references are attribution, not a grant to redistribute those external files. Preserve original notices if obtaining or distributing them separately.

Project and model names are used to identify compatibility and provenance; they do not imply endorsement by their owners.
