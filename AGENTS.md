# H3 integration agent guide

This local project owns H3 loading adapters and reproduction tools. InferRSI stays an external evidence controller; upstream vLLM-Omni changes belong in an isolated runtime checkout.

- Follow the user's current authorized scope and the plan-code-test-review sequence.
- Keep historical prototype validators unchanged. New protocols need new evidence directories.
- Store every local run, setup log, failure, fixture and measurement in the user-selected canonical record root. Supply actual paths through CLI arguments; do not embed personal paths.
- Use isolated environments. No automatic reboot, driver replacement, unrelated process termination or model downloads.
- Test tensor dtype, shape and exact bytes before GPU execution. Tiny synthetic tests do not establish model quality, complete loading or performance.
- Campaign verdict remains INCONCLUSIVE until an acceptance issue and complete frozen evidence exist. Do not invent performance gates.
- No remote publication, issue creation, push or public claim without authorization.
- Preserve upstream source licenses and source attribution. Self-review is not independent review.
- Keep CI lightweight if later introduced. No model or GPU jobs by default.
- Standing note-sync authorization (2026-10-23): after saving meaningful experiment progress, automatically commit and push the reviewed, redacted notes to congw729/vllm-omni-draft main. Read SYNC-POLICY.zh-CN.txt in the canonical record root. Adapter/runtime publication remains outside this authorization.
