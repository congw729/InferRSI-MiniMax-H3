# Native DLO shutdown memory gap

Read the canonical evidence iteration `2026-10-22-fullwidth-dlo-capacity` for the frozen protocols, original failures, resource gates and measurements. The integrated tiny functional baseline remains in `2026-10-22-dlo-temporal-chain`.

The current native backend rejects exactly one streamed DiT block. A subsequent synthetic experiment constructs two distinct real-width blocks and maps their tensors to the same verified canonical source values. This is a bounded topology/transport experiment, not a second copy of released model weights.

Durable v3 checkpoints record exact eight-step and final-latent equivalence between full GPU residency and native DLO. The process then exceeds its frozen RSS limit before shutdown and release checks finish. The complete experiment remains **INCONCLUSIVE**.

Native `shutdown()` calls `_disable(restore_allgather_weights=False)`, but rank-local restoration remains enabled. `restore_next_block_to_cpu()` invokes `restore_tensor_storage()`, which makes a new CPU allocation and copies its source. The native test `test_worker_shutdown_skips_allgather_restore` explicitly expects rank-local restoration. Combined with the last durable phase, this identifies shutdown restoration as the next area to isolate and address; allocation-by-allocation attribution is still absent.

An explicit discard policy for a process that will never reuse its model is a possible next change. It must preserve normal disable/restore semantics and test transfer synchronization, partial failure/retry, hook removal, mapping lifetime and rejection of backend reuse after discard. Raising the RSS guard or skipping the remaining checks would not resolve the lifecycle problem.

No upstream runtime patch is applied. The narrow frozen harness that aliases block-one targets to block-zero cache entries is not a general cache reopening API. Generic loader selection, complete pipeline construction and serving CLI integration remain pending.
