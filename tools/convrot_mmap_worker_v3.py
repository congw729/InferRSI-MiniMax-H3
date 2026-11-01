"""Frozen bounded mixed INT8/BF16 original-H3 mmap semantic/lifecycle validator.

All payloads are tiny synthetic fixtures. Full-width construction is meta-only.
The reference uses an independent FP64 Kronecker rotation and native loader.
"""
from __future__ import annotations
import argparse, gc, hashlib, inspect, json, math, os, shutil, traceback, weakref
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from native_dense_worker_v5 import make, state, setup_dist
from convrot_native_worker_v1 import oracle, native_order, forward_compare, hmatrix

MiB = 1024 ** 2
FILE_BUDGET = 64 * MiB
GROUP_BUDGET = 32 * MiB
PROJECTION_BUDGET = 4 * MiB
WORKSPACE_BUDGET = 512 * 1024
EXPECTED_FIXTURE_SHA = "2d8bc41bbc5bca5cacba1cd32d46320b15f804c940b85e6a2a004cc25cc94acf"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--coverage-spec", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = {"campaign_verdict": "INCONCLUSIVE", "device": a.device, "checks": [],
         "scope": "original full DiT meta coverage; tiny mixed INT8 mmap groups and optional 8-token native forward",
         "not_proven": ["real weights", "full-width capacity", "full pipeline", "quality", "performance"]}
    all_loaders = []
    held_errors = []
    raw = expected = reference = candidate = None

    def ck(name, ok, **kw):
        r["checks"].append(dict(name=name, ok=bool(ok), **kw))
        if not ok:
            raise AssertionError(name)

    def reject(name, fn, kinds=(ValueError, RuntimeError, TypeError)):
        caught = None
        try:
            fn()
        except kinds as error:
            caught = error
        ck(name, caught is not None)
        if caught is not None:
            held_errors.append(caught)
        return caught

    def mapped(root):
        return str(root) in Path("/proc/self/maps").read_text()

    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(0.40)
        ck("initial_cuda_free_2GiB", torch.cuda.mem_get_info()[0] >= 2 * 1024**3)
        setup_dist(a.output.parent)
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
        from h3_omni_adapter.convrot import parse_comfy_quant, recover_bf16_projection
        from h3_omni_adapter.native_convrot_mmap import H3NativeConvRotMmapPlan as Plan
        from h3_omni_adapter.native_convrot_mmap import H3NativeConvRotMmapLoader as Loader
        source_record = json.loads((a.research / "source-record.json").read_text())
        ck("pinned_native_source", hashlib.sha256(Path(inspect.getfile(MiniMaxH3DiTModel)).read_bytes()).hexdigest()
           == source_record["files"]["vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py"]["sha256"])
        config = json.loads((a.research / "metadata/FL2VA__transformer__config.json").read_text())
        coverage = json.loads(a.coverage_spec.read_text())
        full = make(config)
        targets = {k: {"shape": tuple(t.shape), "dtype": {torch.bfloat16:"BF16", torch.float32:"F32"}[t.dtype]} for k, t in state(full).items()}
        ck("full_original_535_all_meta", len(targets) == 535 and all(t.is_meta for t in state(full).values()))
        ck("full_meta_matches_archival_targets",
           set(targets) == set(coverage["targets"]) and all(
               list(v["shape"]) == coverage["targets"][k]["shape"]
               and v["dtype"] == coverage["targets"][k]["dtype"]
               for k, v in targets.items()))
        specs = coverage["specs"]
        audited = Plan.audit_headers(coverage["headers"], specs, targets,
                                     qkv_heads=coverage["qkv_heads"], qkv_head_dim=coverage["qkv_head_dim"])
        ck("full_audit_accepts_250_quantized_1035_sources", len(specs) == 250 and len(coverage["headers"]) == 1035)
        ck("full_audit_bindings_match_meta", set(audited["bindings"]) == set(targets) and audited["quantized_count"] == 250 and audited["target_count"] == 535 and audited["source_count"] == 1035)
        ck("full_meta_remains_meta_after_audit", all(t.is_meta for t in state(full).values()))
        r["full_meta"] = {"targets": len(targets), "sources": len(coverage["headers"]), "quantized": len(specs)}
        del audited, specs, targets, full

        # The supported scale domain is deliberately conservative, including q == 0.
        # A 2x margin bounds absolute dot products by BF16max/2 without new tensors.
        import convrot_math_worker_v2 as math_validator
        math_validator.torch = torch
        Trace = math_validator.install_trace()
        r["scale_domain"] = {"expression": "BF16max / (2 * 128 * sqrt(group))",
                             "over_cap_zero_input_rejected": True}
        for group in (64, 256):
            spec = parse_comfy_quant({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group},
                                     max_metadata_bytes=256)
            cap = float(torch.finfo(torch.bfloat16).max) / (2.0 * 128.0 * math.sqrt(group))
            q = torch.full((2, group), 127, dtype=torch.int8)
            s = torch.full((2, 1), cap, dtype=torch.float32)
            output, stat = recover_bf16_projection(q, s, spec, max_output_bytes=PROJECTION_BUDGET,
                                                    max_workspace_bytes=WORKSPACE_BUDGET)
            ck("scale_cap_finite_" + str(group), torch.isfinite(output).all().item())
            ck("scale_cap_bounded_" + str(group), output.float().abs().max().item() <= torch.finfo(torch.bfloat16).max / 2)
            del output
            # Independent H row signs maximize the absolute dot product with
            # INT8 extrema; this is stronger than an all-positive constant row.
            signs = hmatrix(group)[0]
            aligned = torch.where(signs >= 0, 127, -128).to(torch.int8).repeat(2, 1)
            output, _ = recover_bf16_projection(aligned, s, spec, max_output_bytes=PROJECTION_BUDGET,
                                                max_workspace_bytes=WORKSPACE_BUDGET)
            ck("scale_cap_sign_aligned_finite_" + str(group), torch.isfinite(output).all().item())
            ck("scale_cap_sign_aligned_bounded_" + str(group), output.float().abs().max().item() <= torch.finfo(torch.bfloat16).max / 2)
            del output, aligned, signs
            cases = {"negative": -1.0, "nan": float("nan"), "inf": float("inf"),
                     "f32max": torch.finfo(torch.float32).max,
                     "above_cap": torch.nextafter(torch.tensor(cap), torch.tensor(float("inf"))).item()}
            for label, value in cases.items():
                s.fill_(value)
                trace = Trace([q, s])
                with trace:
                    error = reject("scale_rejected_" + str(group) + "_" + label,
                                   lambda: recover_bf16_projection(q, s, spec,
                                       max_output_bytes=PROJECTION_BUDGET, max_workspace_bytes=WORKSPACE_BUDGET))
                gc.collect()
                ck("scale_traceback_no_owned_tensor_" + str(group) + "_" + label, trace.retained() == 0)
                del trace, error
            q.zero_(); s.fill_(torch.finfo(torch.float32).max)
            reject("zero_source_over_cap_rejected_" + str(group), lambda: recover_bf16_projection(
                q, s, spec, max_output_bytes=PROJECTION_BUDGET, max_workspace_bytes=WORKSPACE_BUDGET))
            s.zero_()
            output, _ = recover_bf16_projection(q, s, spec, max_output_bytes=PROJECTION_BUDGET,
                                               max_workspace_bytes=WORKSPACE_BUDGET)
            ck("zero_scale_zero_result_" + str(group), output.count_nonzero().item() == 0)
            del output, q, s, spec

        tiny = dict(config, num_layers=2, token_refiner_num_layers=2, hidden_size=256,
                    num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512, text_dim=128,
                    timestep_input_dim=64, time_embed_hidden_size=256, time_embed_dim=128,
                    adaln_out_features=18*256, final_adaln_out_features=2*256)
        ck("prior_fixture_identity", hashlib.sha256(a.fixture.read_bytes()).hexdigest() == EXPECTED_FIXTURE_SHA)
        ck("prior_fixture_size", a.fixture.stat().st_size < FILE_BUDGET)
        raw = {}
        with safe_open(a.fixture, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                raw[key] = reader.get_tensor(key).clone()
        del reader
        ck("prior_fixture_unmapped_after_clone", not mapped(a.fixture))
        template = make(tiny)
        expected = {}
        quantized = []
        for key in state(template):
            tensor = raw[key]
            if tensor.dtype == torch.int8:
                base = key[:-7]
                metadata = json.loads(bytes(raw[base + ".comfy_quant"].tolist()).decode())
                dense = oracle(tensor, raw[base + ".weight_scale"], metadata["convrot_groupsize"])
                quantized.append(key)
            else:
                dense = tensor.clone()
            expected[key] = native_order(dense) if ".qkv_proj." in key else dense
        del template, tensor, dense
        ck("tiny_ten_quantized", len(quantized) == 10)
        reference = make(tiny, "cpu")
        ck("reference_native_complete", reference.load_weights(iter(expected.items())) == set(state(reference)))
        reference.post_load_weights()

        def build(model, root=a.fixture.parent, checkpoint=a.fixture.name, **kw):
            return Plan.build(model, root, checkpoint=checkpoint,
                              max_total_file_bytes=kw.pop("max_total_file_bytes", FILE_BUDGET), **kw)

        def new_loader(model, plan, **kw):
            loader = Loader(model, plan, device=kw.pop("device", "cpu"),
                            max_group_bytes=kw.pop("max_group_bytes", GROUP_BUDGET),
                            max_projection_bytes=kw.pop("max_projection_bytes", PROJECTION_BUDGET),
                            max_workspace_bytes=kw.pop("max_workspace_bytes", WORKSPACE_BUDGET), **kw)
            all_loaders.append(loader)
            return loader

        candidate = make(tiny)
        plan = build(candidate)
        ck("tiny_exact_target_coverage", set(plan.bindings) == set(state(reference)))
        ck("tiny_exact_source_coverage", set(plan.sources) == set(raw))
        loader = new_loader(candidate, plan)
        reject("resident_required", lambda: loader.load_group("blocks.0"))
        loader.load_group("resident")
        for repeat in range(2):
            for group, names in plan.groups.items():
                if group != "resident":
                    loader.load_group(group)
                ck("parameters_exact_" + str(repeat) + "_" + group,
                   all(torch.equal(state(candidate)[key], state(reference)[key]) for key in names))
                ck("resident_stays_materialized_" + str(repeat) + "_" + group,
                   all(not state(candidate)[key].is_meta for key in plan.groups["resident"]))
                if group != "resident":
                    other = "blocks.1" if group != "blocks.1" else "blocks.0"
                    reject("one_group_" + str(repeat) + "_" + group, lambda: loader.load_group(other))
                    refs = [weakref.ref(state(candidate)[key]) for key in names]
                    loader.release_group(group)
                    gc.collect()
                    ck("group_released_" + str(repeat) + "_" + group,
                       all(state(candidate)[key].is_meta for key in names) and all(ref() is None for ref in refs))
                ck("mmap_released_after_group_" + str(repeat) + "_" + group, not mapped(a.fixture))
        r["repeat_loader_stats"] = dict(loader.stats)
        ck("one_streamed_group_observed", loader.stats["max_active_streamed_groups"] == 1)
        ck("mapping_observed", loader.stats["mmap_observed"])
        loader.close()
        ck("close_all_meta", all(t.is_meta for t in state(candidate).values()) and not loader.active)
        loader.close()
        reject("closed_load", lambda: loader.load_group("resident"))
        reject("closed_release", lambda: loader.release_group("resident"))
        reject("closed_hooks", loader.install_block_hooks)


        # A native load_weights for-loop can retain the preceding loaded_weight,
        # QKV copy, and fc1 chunk aliases while asking the generator for next().
        # Observe both recovery outputs and native QKV copies across that seam.
        import importlib
        mmap_module = importlib.import_module("h3_omni_adapter.native_convrot_mmap")
        native_module = importlib.import_module(MiniMaxH3DiTModel.__module__)
        actual_recover = mmap_module.recover_bf16_projection
        actual_qkv = native_module._reorder_grouped_qkv_to_qkv
        output_refs = []; native_copy_refs = []; recovery_count = [0]
        def observe_recover(*args, **kwargs):
            recovery_count[0] += 1
            ck("cross_yield_previous_recovery_released_" + str(recovery_count[0]),
               all(ref() is None for ref in output_refs))
            ck("cross_yield_native_qkv_copy_released_" + str(recovery_count[0]),
               all(ref() is None for ref in native_copy_refs))
            result = actual_recover(*args, **kwargs)
            output_refs.append(weakref.ref(result[0]))
            return result
        def observe_qkv(*args, **kwargs):
            result = actual_qkv(*args, **kwargs)
            native_copy_refs.append(weakref.ref(result))
            return result
        overlap_loader = new_loader(candidate, plan)
        mmap_module.recover_bf16_projection = observe_recover
        native_module._reorder_grouped_qkv_to_qkv = observe_qkv
        try:
            overlap_loader.load_group("resident")
            for group, names in plan.groups.items():
                if group == "resident":
                    continue
                overlap_loader.load_group(group)
                ck("cross_yield_group_exact_" + group,
                   all(torch.equal(state(candidate)[k], state(reference)[k]) for k in names))
                ck("cross_yield_group_outputs_gone_" + group,
                   all(ref() is None for ref in output_refs + native_copy_refs))
                overlap_loader.release_group(group)
        finally:
            mmap_module.recover_bf16_projection = actual_recover
            native_module._reorder_grouped_qkv_to_qkv = actual_qkv
            overlap_loader.close()
        ck("cross_yield_ten_recoveries_observed", recovery_count[0] == 10)
        r["cross_yield_lifecycle"] = {"recoveries": recovery_count[0],
                                    "native_qkv_copies": len(native_copy_refs)}
        del actual_recover, actual_qkv, overlap_loader

        # Quant recovery/output and materialized group budgets are independent.
        for label, options in (("group", {"max_group_bytes": 1}),
                               ("projection", {"max_projection_bytes": 1}),
                               ("workspace", {"max_workspace_bytes": 1})):
            test = new_loader(candidate, plan, **options)
            if label == "group":
                reject("budget_" + label, lambda: test.load_group("resident"))
            else:
                test.load_group("resident")
                reject("budget_" + label, lambda: test.load_group("blocks.0"))
                ck("budget_block_meta_" + label, all(state(candidate)[k].is_meta for k in plan.groups["blocks.0"]))
            test.close()
            ck("budget_no_mmap_" + label, not mapped(a.fixture))
        reject("file_budget", lambda: build(candidate, max_total_file_bytes=1))
        reject("tp2", lambda: build(candidate, tensor_parallel_size=2))
        reject("changed_arch_shape", lambda: build(make(dict(tiny, ffn_hidden_size=256))))

        # Deterministic cancellation points are measured from a successful load,
        # then exercised at entry, midpoint, and final callback with errors retained.
        calls = [0]
        def count_only():
            calls[0] += 1
            return False
        test = new_loader(candidate, plan, cancelled=count_only)
        test.load_group("resident"); calls[0] = 0
        test.load_group("blocks.0"); total_calls = calls[0]
        test.release_group("blocks.0"); test.close()
        ck("cancel_points_available", total_calls >= 3)
        r["successful_block_callback_calls"] = total_calls
        for label, point in (("entry", 1), ("middle", max(2, total_calls//2)), ("final", total_calls)):
            armed = [False]; calls = [0]
            def cancel():
                if not armed[0]:
                    return False
                calls[0] += 1
                return calls[0] >= point
            test = new_loader(candidate, plan, cancelled=cancel)
            test.load_group("resident"); armed[0] = True
            reject("cancel_" + label, lambda: test.load_group("blocks.0"))
            gc.collect()
            ck("cancel_retained_traceback_meta_" + label, all(state(candidate)[k].is_meta for k in plan.groups["blocks.0"]))
            ck("cancel_retained_traceback_unmapped_" + label, not mapped(a.fixture))
            armed[0] = False
            test.load_group("blocks.0")
            ck("cancel_retry_exact_" + label, all(torch.equal(state(candidate)[k], state(reference)[k]) for k in plan.groups["blocks.0"]))
            test.close()

        # Force cancellation only after one real native weight has committed.
        committed = [0]
        armed = [False]
        original = candidate.load_weights
        def cancelled_after_commit():
            return armed[0] and committed[0] >= 1
        def commit_each(weights):
            loaded = set(); item = None
            try:
                for item in weights:
                    loaded.update(original(iter((item,))))
                    committed[0] += 1
                    item = None
                return loaded
            finally:
                item = None
        test = new_loader(candidate, plan, cancelled=cancelled_after_commit)
        test.load_group("resident")
        candidate.load_weights = commit_each
        armed[0] = True; committed[0] = 0
        try:
            reject("cancel_after_native_commit", lambda: test.load_group("blocks.0"))
        finally:
            candidate.load_weights = original
        ck("cancel_midgroup_really_committed", committed[0] == 1)
        ck("cancel_midgroup_rolls_back_complete_group", all(state(candidate)[k].is_meta for k in plan.groups["blocks.0"]))
        ck("cancel_midgroup_releases_mapping", not mapped(a.fixture))
        armed[0] = False
        test.load_group("blocks.0")
        ck("cancel_midgroup_retry_exact", all(torch.equal(state(candidate)[k], state(reference)[k]) for k in plan.groups["blocks.0"]))
        test.close()
        del original

        test = new_loader(candidate, plan)
        test.load_group("resident")
        original = candidate.load_weights
        fault_refs = []
        def broken(weights):
            name, tensor = next(weights)
            fault_refs.append(weakref.ref(tensor))
            raise RuntimeError("injected native loader failure after first source")
        candidate.load_weights = broken
        try:
            reject("native_loader_failure", lambda: test.load_group("blocks.0"))
        finally:
            candidate.load_weights = original
        gc.collect()
        ck("native_failure_held_traceback_releases_source", all(ref() is None for ref in fault_refs))
        ck("native_failure_held_traceback_unmapped", not mapped(a.fixture))
        ck("native_failure_group_meta", all(state(candidate)[k].is_meta for k in plan.groups["blocks.0"]))
        test.load_group("blocks.0")
        ck("native_failure_retry_exact", all(torch.equal(state(candidate)[k], state(reference)[k]) for k in plan.groups["blocks.0"]))
        test.close()
        del original

        test = new_loader(candidate, plan)
        test.load_group("resident"); test.load_group("blocks.0")
        original_sync = test._sync_device
        def broken_sync():
            raise RuntimeError("injected synchronization failure")
        test._sync_device = broken_sync
        reject("release_sync_failure", lambda: test.release_group("blocks.0"))
        ck("sync_failure_group_meta", all(state(candidate)[k].is_meta for k in plan.groups["blocks.0"]))
        reject("poison_rejects_load", lambda: test.load_group("blocks.1"))
        test._sync_device = original_sync
        test.close()
        ck("poison_close_all_meta", all(t.is_meta for t in state(candidate).values()))
        del original_sync


        # Preserve primitive SystemExit codes while discarding tensor-owning
        # exception frames. A caller must be able to distinguish code 0 and 7.
        for exit_code in (None, 0, 7, "requested stop"):
            test = new_loader(candidate, plan)
            test.load_group("resident")
            original = candidate.load_weights
            def exit_from_native(weights):
                name, tensor = next(weights)
                raise SystemExit(exit_code)
            candidate.load_weights = exit_from_native
            try:
                error = reject("system_exit_" + str(exit_code), lambda: test.load_group("blocks.0"), kinds=(SystemExit,))
                ck("system_exit_code_preserved_" + str(exit_code), error.code == exit_code and type(error.code) is type(exit_code))
            finally:
                candidate.load_weights = original
            ck("system_exit_group_meta_" + str(exit_code), all(state(candidate)[k].is_meta for k in plan.groups["blocks.0"]))
            ck("system_exit_no_mapping_" + str(exit_code), not mapped(a.fixture))
            test.close()
            del original, error

        # A failed close is incomplete while owned groups or hook handles remain.
        # Restore the injected operation and prove a second close really retries.
        test = new_loader(candidate, plan)
        test.load_group("resident"); test.load_group("blocks.0")
        actual_restore = test._restore_meta
        restore_calls = [0]; fail_restore = [True]
        def incomplete_restore(group):
            if group == "blocks.0":
                restore_calls[0] += 1
                if fail_restore[0]:
                    raise RuntimeError("injected restore failure before active removal")
            return actual_restore(group)
        test._restore_meta = incomplete_restore
        reject("close_restore_failure", test.close)
        ck("failed_restore_close_is_retryable", not test.closed and test.poisoned and "blocks.0" in test.active)
        reject("failed_restore_poison_rejects_load", lambda: test.load_group("blocks.1"))
        fail_restore[0] = False
        test.close()
        ck("restore_close_retried", restore_calls[0] == 2)
        ck("restore_close_completed", test.closed and not test.active and all(t.is_meta for t in state(candidate).values()))
        del actual_restore

        test = new_loader(candidate, plan)
        test.load_group("resident"); test.install_block_hooks()
        real_handle = test.handles[0]
        remove_calls = [0]; fail_remove = [True]
        class RetryHandle:
            def remove(self):
                remove_calls[0] += 1
                if fail_remove[0]:
                    raise RuntimeError("injected remove failure before hook removal")
                real_handle.remove()
        wrapped_handle = RetryHandle()
        test.handles[0] = wrapped_handle
        reject("close_handle_failure", test.close)
        ck("failed_handle_close_is_retryable", not test.closed and test.poisoned and wrapped_handle in test.handles)
        reject("failed_handle_poison_rejects_load", lambda: test.load_group("resident"))
        fail_remove[0] = False
        test.close()
        ck("handle_close_retried", remove_calls[0] == 2)
        ck("handle_close_completed", test.closed and not test.handles and not test.active)
        ck("handle_close_no_hooks", all(not candidate.get_submodule(g)._forward_pre_hooks and not candidate.get_submodule(g)._forward_hooks
                                       for g in plan.groups if g != "resident"))
        del real_handle, wrapped_handle

        class InjectedExit(Exception):
            pass
        try:
            with new_loader(candidate, plan) as test:
                test.load_group("resident"); test.load_group("blocks.0")
                raise InjectedExit("owned context exit")
        except InjectedExit as error:
            held_errors.append(error)
        ck("context_exception_all_meta", all(t.is_meta for t in state(candidate).values()))
        ck("context_exception_no_mapping", not mapped(a.fixture))

        # Negative checkpoint copies are tiny and deliberately local. Each prior
        # failed input is described in the results; no real payload is fetched.
        negroot = a.output.parent / "negative-fixtures"; negroot.mkdir(exist_ok=True)
        negfile = negroot / "candidate.safetensors"
        qkey = sorted(quantized)[0]; qbase = qkey[:-7]
        def negative_case(label, transform):
            altered = {k: v.clone() for k, v in raw.items()}
            transform(altered)
            save_file(altered, negfile)
            del altered
            reject("plan_" + label, lambda: build(candidate, negroot, negfile.name))
            ck("plan_" + label + "_no_mapping", not mapped(negroot))
        negative_case("missing_scale", lambda d: d.pop(qbase + ".weight_scale"))
        negative_case("missing_meta", lambda d: d.pop(qbase + ".comfy_quant"))
        negative_case("extra_source", lambda d: d.update({"unknown.weight": torch.zeros(1)}))
        negative_case("scale_shape", lambda d: d.update({qbase + ".weight_scale": d[qbase + ".weight_scale"].flatten()}))
        negative_case("scale_dtype", lambda d: d.update({qbase + ".weight_scale": d[qbase + ".weight_scale"].bfloat16()}))
        negative_case("quant_dtype", lambda d: d.update({qkey: d[qkey].to(torch.uint8)}))
        negative_case("malformed_meta", lambda d: d.update({qbase + ".comfy_quant": torch.tensor(list(b"{}"), dtype=torch.uint8)}))
        duplicate = b'{"format":"int8_tensorwise","convrot":true,"convrot":true,"convrot_groupsize":64}'
        negative_case("duplicate_meta_key", lambda d: d.update({qbase + ".comfy_quant": torch.tensor(list(duplicate), dtype=torch.uint8)}))
        negative_case("oversized_meta", lambda d: d.update({qbase + ".comfy_quant": torch.zeros(4097, dtype=torch.uint8)}))
        save_file(raw, negfile)
        # Explicit positive sidecars: the index covers only the weight shard;
        # scales and metadata arrive only from the named auxiliary shard.
        split_weights = {k: v for k, v in raw.items() if not k.endswith((".weight_scale", ".comfy_quant"))}
        split_aux = {k: v for k, v in raw.items() if k.endswith((".weight_scale", ".comfy_quant"))}
        weight_shard = negroot / "weights.safetensors"
        aux_shard = negroot / "auxiliary.safetensors"
        split_index = negroot / "split.safetensors.index.json"
        save_file(split_weights, weight_shard); save_file(split_aux, aux_shard)
        split_index.write_text(json.dumps({"weight_map": {k: weight_shard.name for k in split_weights}}))
        split_plan = build(candidate, negroot, split_index.name, sidecars=(aux_shard.name,))
        ck("positive_sidecar_exact_sources", set(split_plan.sources) == set(raw))
        split_loader = new_loader(candidate, split_plan)
        split_loader.load_group("resident")
        ck("positive_sidecar_resident_exact", all(torch.equal(state(candidate)[k], state(reference)[k]) for k in split_plan.groups["resident"]))
        split_loader.load_group("blocks.0")
        ck("positive_sidecar_block_exact", all(torch.equal(state(candidate)[k], state(reference)[k]) for k in split_plan.groups["blocks.0"]))
        split_loader.release_group("blocks.0"); split_loader.close()
        ck("positive_sidecar_all_meta", all(t.is_meta for t in state(candidate).values()))
        ck("positive_sidecar_unmapped", not mapped(negroot))
        del split_weights, split_aux, split_loader, split_plan
        sidecar = negroot / "duplicate-scale.safetensors"
        save_file({qbase + ".weight_scale": raw[qbase + ".weight_scale"]}, sidecar)
        reject("duplicate_scale_sidecar", lambda: build(candidate, negroot, negfile.name, sidecars=(sidecar.name,)))
        index = negroot / "model.safetensors.index.json"
        keys = sorted(raw)
        good_index = {"weight_map": {k: negfile.name for k in keys}}
        index.write_text(json.dumps(good_index))
        indexed = build(candidate, negroot, index.name)
        ck("indexed_exact_coverage", set(indexed.bindings) == set(plan.bindings))
        duplicate_index = '{"weight_map":{' + json.dumps(keys[0]) + ':"candidate.safetensors",' + json.dumps(keys[0]) + ':"candidate.safetensors"}}'
        index.write_text(duplicate_index)
        reject("duplicate_index_key", lambda: build(candidate, negroot, index.name))
        bad = {"weight_map": dict(good_index["weight_map"])}; bad["weight_map"].pop(keys[0]); index.write_text(json.dumps(bad))
        reject("index_missing_key", lambda: build(candidate, negroot, index.name))
        bad = {"weight_map": dict(good_index["weight_map"])}; bad["weight_map"][keys[0]] = "../outside.safetensors"; index.write_text(json.dumps(bad))
        reject("index_path_escape", lambda: build(candidate, negroot, index.name))
        bad = {"weight_map": dict(good_index["weight_map"])}; bad["weight_map"]["unknown.weight"] = negfile.name; index.write_text(json.dumps(bad))
        reject("index_extra_key", lambda: build(candidate, negroot, index.name))
        index.write_text(json.dumps(good_index))
        save_file(raw, negfile)
        fresh = build(candidate, negroot, negfile.name)
        stat = negfile.stat(); os.utime(negfile, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        reject("identity_tamper_after_plan", fresh.check_files_unchanged)
        save_file(raw, negfile)
        header = negfile.read_bytes(); negfile.write_bytes(header[:8] + b" " * (len(header)-8))
        reject("malformed_header", lambda: build(candidate, negroot, negfile.name))
        # Recreate valid input, then reject numerical overflow during actual mmap load.
        for label, value in (("overflow", torch.finfo(torch.float32).max), ("nan", float("nan")), ("negative", -1.0)):
            altered = dict(raw); altered[qbase + ".weight_scale"] = torch.full_like(raw[qbase + ".weight_scale"], value)
            save_file(altered, negfile); del altered
            fresh = build(candidate, negroot, negfile.name)
            test = new_loader(candidate, fresh); test.load_group("resident")
            reject("mmap_scale_" + label, lambda: test.load_group("blocks.0"))
            gc.collect()
            ck("mmap_scale_" + label + "_meta", all(state(candidate)[k].is_meta for k in fresh.groups["blocks.0"]))
            ck("mmap_scale_" + label + "_unmapped", not mapped(negroot))
            test.close()

        if a.device == "cuda":
            reference.eval().to("cuda")
            gpu = make(tiny); gpu_plan = build(gpu); gpu_loader = new_loader(gpu, gpu_plan, device="cuda")
            gpu_loader.load_group("resident"); gpu_loader.install_block_hooks()
            reject("duplicate_hook_install", gpu_loader.install_block_hooks)
            gpu.eval()
            r["forward"] = forward_compare(reference, gpu, tiny, ck)
            ck("forward_all_streamed_meta", all(state(gpu)[k].is_meta for g, ns in gpu_plan.groups.items() if g != "resident" for k in ns))
            ck("forward_resident_live", all(not state(gpu)[k].is_meta for k in gpu_plan.groups["resident"]))
            old_forward = gpu.blocks[0].forward
            forward_refs = []
            def fail_forward(*args, **kwargs):
                local_weight = next(gpu.blocks[0].parameters())
                forward_refs.append(weakref.ref(local_weight))
                raise RuntimeError("injected block forward failure")
            gpu.blocks[0].forward = fail_forward
            try:
                reject("forward_exception", lambda: forward_compare(reference, gpu, tiny, ck))
            finally:
                gpu.blocks[0].forward = old_forward
            ck("forward_exception_streamed_meta", all(state(gpu)[k].is_meta for g, ns in gpu_plan.groups.items() if g != "resident" for k in ns))
            gc.collect()
            ck("forward_exception_held_traceback_releases_parameters", all(ref() is None for ref in forward_refs))
            ck("forward_exception_no_mapping", not mapped(a.fixture))
            # Use distinct check labels for a recovery forward after the failure.
            forward_compare(reference, gpu, tiny, lambda n, ok: ck("retry_" + n, ok))
            r["cuda_loader_stats"] = dict(gpu_loader.stats)
            gpu_loader.close()
            ck("cuda_close_all_meta", all(t.is_meta for t in state(gpu).values()))
            del old_forward, gpu, gpu_plan, gpu_loader

        ck("source_fixture_still_unchanged", hashlib.sha256(a.fixture.read_bytes()).hexdigest() == EXPECTED_FIXTURE_SHA)
        r["retained_failure_count"] = len(held_errors)
        r["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        ck("cuda_peak_under_4GiB", r["cuda_peak_allocated_bytes"] < 4 * 1024**3)
    except BaseException as error:
        r["error"] = {"type": type(error).__name__, "traceback": traceback.format_exc()}
    finally:
        for loader in reversed(all_loaders):
            try:
                loader.close()
            except BaseException as error:
                r.setdefault("cleanup_errors", []).append(type(error).__name__)
        all_loaders.clear(); held_errors.clear()
        raw = expected = reference = candidate = None
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as error:
            r.setdefault("cleanup_errors", []).append(type(error).__name__)
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            r["cuda_allocated_after_cleanup"] = torch.cuda.memory_allocated()
    r["checks_total"] = len(r["checks"]); r["checks_ok"] = sum(c["ok"] for c in r["checks"])
    r["exit_code"] = int("error" in r or "cleanup_errors" in r or r["checks_total"] != r["checks_ok"])
    text = json.dumps(r, indent=2)
    for value, label in ((str(a.output.parent), "<TRIAL_ROOT>"), (str(a.research), "<RESEARCH_ROOT>"),
                         (str(a.fixture.parent), "<PRIOR_FIXTURE_ROOT>"), (str(a.coverage_spec.parent), "<COVERAGE_ROOT>"),
                         (str(Path.home()), "<LINUX_HOME>"), (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>")):
        text = text.replace(value, label)
    a.output.write_text(text + "\n")
    print(json.dumps({k: v for k, v in r.items() if k not in ("checks", "error")}))
    return r["exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
