# Contributing

Keep each change within its declared scope: the maintained ComfyUI preview, experimental vLLM-Omni adapters, or documentation. Preserve historical evidence and frozen validators; new protocols need new evidence directories. Use explicit fixture/source/output paths and an isolated environment for model or tensor checks.

For a quick-start client change, run the existing offline checks from the repository root with Python 3.10+:

```text
python -B -m unittest discover -s quickstart -p test_run_h3.py
```

These checks use the standard library and do not require ComfyUI, InferRSI, model weights or a GPU. Adapter research uses a separate matched Python 3.12/Linux environment and external fixtures; see [validation entry points](tests/README.md). Do not run historical experiment scripts in bulk or install the adapter package just to use the preview client.

Repository smoke is separate from L0–L4 model acceptance. Formal comparisons require an accepted issue and frozen quality/performance gates before candidate search. Keep failures and software identities. A functional clip or offline test does not establish model quality or service performance.

Do not commit personal paths, credentials or raw model weights. Review actual changes before publication. If commits are requested, use the contributor's configured DCO identity rather than inventing an author. Author self-review does not satisfy independent review.
