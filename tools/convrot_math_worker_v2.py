"""Frozen CPU-only ConvRot mathematical and Python-visible lifecycle validator.

No model imports or native model execution. An outer controller owns resource
limits. All tensor construction and computation below use synthetic tiny data.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys
import weakref

BUDGET = 8 * 1024 * 1024
CHECKS = []
METRICS = {}
torch = None
recover = None
parse = None
StorageTrace = None


def check(name, condition, detail=None):
    item = {"name": name, "ok": bool(condition)}
    if detail is not None:
        item["detail"] = detail
    CHECKS.append(item)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def meta(group):
    return {"format": "int8_tensorwise", "convrot": True,
            "convrot_groupsize": group}


def spec(group):
    return parse(meta(group), max_metadata_bytes=1024)


def dense_h(group):
    """Independent radix-4 digit-product oracle, not candidate kron code."""
    depth = {64: 3, 256: 4}[group]
    result = []
    for row in range(group):
        values = []
        for col in range(group):
            r, c, sign = row, col, 1
            for _ in range(depth):
                if r % 4 + c % 4 == 3:
                    sign = -sign
                r //= 4
                c //= 4
            values.append(sign / math.sqrt(group))
        result.append(values)
    return result


def oracle(q, scales, group, layout="plain", heads=None, head_dim=None):
    """F32 multiply rounding, explicit dense FP64 sums, then F32/BF16."""
    qrows = q.tolist()
    srows = scales.tolist()
    h = dense_h(group)
    expected = []
    sumabs = []
    for row, scale in zip(qrows, srows):
        rr, aa = [], []
        for start in range(0, len(row), group):
            values = [f32(float(v) * float(scale[0]))
                      for v in row[start:start + group]]
            magnitude = sum(abs(v) / math.sqrt(group) for v in values)
            for j in range(group):
                rr.append(sum(values[k] * h[j][k] for k in range(group)))
                aa.append(magnitude)
        expected.append(rr)
        sumabs.append(aa)
    if layout == "grouped_qkv":
        order = [(t * heads + head) * head_dim + d
                 for head in range(heads)
                 for t in range(3) for d in range(head_dim)]
        expected = [expected[i] for i in order]
        sumabs = [sumabs[i] for i in order]
    exact = torch.tensor(expected, dtype=torch.float64)
    rounded = exact.to(torch.float32).to(torch.bfloat16)
    return exact, rounded, torch.tensor(sumabs, dtype=torch.float64)


def tensor_pattern(rows, cols, non_dyadic=False):
    palette = [-128, -127, -31, -1, 0, 1, 17, 126, 127]
    q = torch.tensor([[palette[(r * 5 + c * 7 + c // 11) % len(palette)]
                       for c in range(cols)] for r in range(rows)],
                     dtype=torch.int8)
    values = ([0.0, 0.0137, 0.1, 0.33333334, 1.7, 3.1415927, 7.01]
              if non_dyadic else
              [0.0, 1 / 64, 3 / 32, 5 / 8, 1.25, 3.5, 16.0, 1 / 1024])
    scales = torch.tensor([[values[r % len(values)]] for r in range(rows)],
                          dtype=torch.float32)
    return q, scales


def storage_id(t):
    return (str(t.device), t.untyped_storage().data_ptr())


def primitive_tree(value):
    if type(value) in (str, int, float, bool, type(None)):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and primitive_tree(v)
                   for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(primitive_tree(v) for v in value)
    return False


def install_trace():
    from torch.utils._python_dispatch import TorchDispatchMode

    class _StorageTrace(TorchDispatchMode):
        """Observes returned Torch storages; never owns a tensor reference."""
        def __init__(self, caller_owned):
            super().__init__()
            self.excluded = {storage_id(t) for t in caller_owned}
            self.refs = []
            self.events = []

        def _register(self, value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    return
                sid = storage_id(value)
                self.refs.append((weakref.ref(value), sid,
                                  value.untyped_storage().nbytes()))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    self._register(item)
            elif isinstance(value, dict):
                for item in value.values():
                    self._register(item)

        def _snapshot(self):
            live = {}
            for ref, sid, size in self.refs:
                tensor = ref()
                if tensor is not None and sid not in self.excluded:
                    live[sid] = max(size, live.get(sid, 0))
                del tensor
            self.events.append(live)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            self._register(result)
            self._snapshot()
            return result

        def workspace_peak(self, output):
            oid = storage_id(output)
            return max((sum(size for sid, size in event.items() if sid != oid)
                        for event in self.events), default=0)

        def retained(self, allowed=()):
            permitted = self.excluded | set(allowed)
            return sum(1 for ref, sid, _ in self.refs
                       if sid not in permitted and ref() is not None)

    return _StorageTrace


def call_success(name, q, scales, group, *, layout="plain", heads=None,
                 head_dim=None, workspace=BUDGET, exact_required=True,
                 use_linear=False, hard_expected=None):
    expected64, expected_bf16, sumabs = oracle(
        q, scales, group, layout, heads, head_dim)
    q_before, s_before = q.clone(), scales.detach().clone()
    trace = StorageTrace([q, scales])
    output_bytes = q.numel() * 2
    with trace:
        output, stats = recover(
            q, scales, spec(group), max_output_bytes=output_bytes,
            max_workspace_bytes=workspace, row_layout=layout,
            qkv_heads=heads, qkv_head_dim=head_dim)
    check(name + ".shape_dtype_device",
          output.shape == q.shape and output.dtype == torch.bfloat16
          and output.device.type == "cpu" and output.is_contiguous())
    check(name + ".private_storage",
          storage_id(output) not in {storage_id(q), storage_id(scales)})
    check(name + ".no_autograd", not output.requires_grad and output.grad_fn is None)
    check(name + ".caller_unchanged",
          torch.equal(q, q_before) and torch.equal(scales.detach(), s_before))
    check(name + ".stats_primitive", isinstance(stats, dict) and primitive_tree(stats))
    check(name + ".output_budget",
          type(stats.get("output_bytes")) is int
          and stats["output_bytes"] == output_bytes)
    observed = trace.workspace_peak(output)
    reported = stats.get("workspace_peak_bytes")
    check(name + ".workspace_budget",
          type(reported) is int and 0 <= observed <= reported <= workspace,
          {"torch_visible_peak": observed, "reported_peak": reported,
           "limit": workspace})
    got64 = output.double()
    finite = bool(torch.isfinite(got64).all())
    exact_match = torch.equal(output, expected_bf16)
    gamma = (group * 2.0 ** -24) / (1.0 - group * 2.0 ** -24)
    bound = (2.0 ** -8 * expected64.abs()
             + (1 + 2.0 ** -8) * gamma * sumabs + 2.0 ** -133)
    error = (got64 - expected64).abs()
    check(name + ".fp64_error_bound", finite and bool((error <= bound).all()))
    if exact_required:
        check(name + ".bf16_exact", exact_match)
    if hard_expected is not None:
        check(name + ".hard_anchor", torch.equal(output, hard_expected))
    if use_linear:
        x = torch.tensor([[(i * 3 + j * 5) % 13 / 8 - 0.75
                           for j in range(q.shape[1])] for i in range(3)],
                         dtype=torch.float32)
        y = torch.nn.functional.linear(x, output.float())
        xd, wd = x.double().tolist(), expected_bf16.double().tolist()
        yd = [[sum(a * b for a, b in zip(row, weight)) for weight in wd]
              for row in xd]
        mag = [[sum(abs(a * b) for a, b in zip(row, weight)) for weight in wd]
               for row in xd]
        reference = torch.tensor(yd, dtype=torch.float64)
        magnitudes = torch.tensor(mag, dtype=torch.float64)
        gamma_k = (q.shape[1] * 2.0 ** -24) / (1 - q.shape[1] * 2.0 ** -24)
        check(name + ".tiny_linear_math",
              bool(((y.double() - reference).abs()
                    <= gamma_k * magnitudes + 1e-30).all()))
        del x, y, xd, wd, yd, mag, reference, magnitudes
    METRICS[name] = {"shape": list(q.shape), "group": group,
                     "bf16_exact_to_dense_oracle": exact_match,
                     "max_abs_error_to_fp64": float(error.max()),
                     "max_bound": float(bound.max()),
                     "observed_torch_workspace_bytes": observed,
                     "reported_workspace_bytes": reported}
    gc.collect()
    check(name + ".temporary_weakrefs_released",
          trace.retained([storage_id(output)]) == 0)
    outref = weakref.ref(output)
    del output, stats
    gc.collect()
    check(name + ".output_and_all_owned_weakrefs_released",
          outref() is None and trace.retained() == 0)


def expect_reject(name, function):
    try:
        result = function()
    except (ValueError, TypeError, RuntimeError):
        check(name, True)
        return
    except Exception as exc:
        check(name, False, {"unexpected_exception": type(exc).__name__})
        return
    check(name, False, {"unexpected_return_type": type(result).__name__})


def parse_tests():
    for group in (64, 256):
        for typ in ("dict", "str", "bytes"):
            value = meta(group)
            if typ != "dict":
                value = json.dumps(value)
            if typ == "bytes":
                value = value.encode("utf8")
            parsed = parse(value, max_metadata_bytes=1024)
            check(f"parse.valid_{group}_{typ}", parsed is not None)
        payload = json.dumps(meta(group))
        check(f"parse.exact_size_{group}",
              parse(payload, max_metadata_bytes=len(payload.encode())) is not None)
        expect_reject(f"parse.too_small_{group}",
                      lambda: parse(payload, max_metadata_bytes=len(payload.encode()) - 1))
    cases = [
        ("format", {**meta(64), "format": "nvfp4"}),
        ("false", {**meta(64), "convrot": False}),
        ("integer_true", {**meta(64), "convrot": 1}),
        ("group_bool", {**meta(64), "convrot_groupsize": True}),
        ("group_float", {**meta(64), "convrot_groupsize": 64.0}),
        ("group_128", {**meta(64), "convrot_groupsize": 128}),
        ("group_zero", {**meta(64), "convrot_groupsize": 0}),
        ("group_alias", {"format": "int8_tensorwise", "convrot": True, "groupsize": 64}),
        ("missing", {"format": "int8_tensorwise", "convrot_groupsize": 64}),
        ("unknown", {**meta(64), "seed": 42}),
        ("list", []), ("null", None), ("bad_json", "{"),
        ("invalid_utf8", b"\xff"),
        ("duplicate_same_key", '{"format":"int8_tensorwise","convrot":true,"convrot":true,"convrot_groupsize":64}'),
    ]
    for name, value in cases:
        expect_reject("parse.reject_" + name,
                      lambda value=value: parse(value, max_metadata_bytes=1024))
    for limit in (0, -1, True, 2.5):
        expect_reject("parse.reject_limit_" + repr(limit),
                      lambda limit=limit: parse(meta(64), max_metadata_bytes=limit))


def argument_tests():
    q, s = tensor_pattern(3, 128)
    good = dict(max_output_bytes=q.numel() * 2, max_workspace_bytes=BUDGET)
    normal = spec(64)
    cases = [
        ("source_dtype", q.float(), s, {}),
        ("source_rank", q.reshape(-1), s, {}),
        ("source_noncontiguous", torch.arange(384, dtype=torch.int8).reshape(128, 3).t(), s, {}),
        ("source_meta_device", torch.empty(q.shape, dtype=torch.int8, device="meta"), s, {}),
        ("columns_not_divisible", q[:, :127].contiguous(), s, {}),
        ("scale_dtype", q, s.double(), {}),
        ("scale_shape", q, s.reshape(-1), {}),
        ("scale_wrong_rows", q, torch.ones(2, 1), {}),
        ("scale_noncontiguous", q, torch.ones(3, 2)[:, :1], {}),
        ("scale_negative", q, torch.tensor([[1.0], [-0.1], [1.0]]), {}),
        ("scale_nan", q, torch.tensor([[1.0], [float("nan")], [1.0]]), {}),
        ("scale_inf", q, torch.tensor([[1.0], [float("inf")], [1.0]]), {}),
        ("scale_meta_device", q, torch.empty(s.shape, device="meta"), {}),
        ("layout_unknown", q, s, {"row_layout": "interleave"}),
        ("qkv_missing", q, s, {"row_layout": "grouped_qkv"}),
        ("qkv_wrong_rows", q, s, {"row_layout": "grouped_qkv", "qkv_heads": 2, "qkv_head_dim": 1}),
        ("qkv_bool_heads", q, s, {"row_layout": "grouped_qkv", "qkv_heads": True, "qkv_head_dim": 1}),
        ("output_short", q, s, {"max_output_bytes": q.numel() * 2 - 1}),
        ("workspace_zero", q, s, {"max_workspace_bytes": 0}),
        ("workspace_one", q, s, {"max_workspace_bytes": 1}),
        ("output_negative", q, s, {"max_output_bytes": -1}),
        ("workspace_negative", q, s, {"max_workspace_bytes": -1}),
        ("output_bool", q, s, {"max_output_bytes": True}),
        ("workspace_float", q, s, {"max_workspace_bytes": 1.5}),
    ]
    for name, weight, scale, extra in cases:
        kwargs = {**good, **extra}
        expect_reject("arguments." + name,
                      lambda weight=weight, scale=scale, kwargs=kwargs:
                      recover(weight, scale, normal, **kwargs))
    grad_s = s.clone().requires_grad_(True)
    try:
        output, stats = recover(q, grad_s, normal, **good)
    except (ValueError, TypeError, RuntimeError):
        check("arguments.grad_scale_rejected_or_inference_only", True)
    else:
        check("arguments.grad_scale_rejected_or_inference_only",
              not output.requires_grad and output.grad_fn is None and grad_s.grad is None)
        del output, stats


def anchors_and_math():
    for group in (64, 256):
        q = torch.full((2, group), 7, dtype=torch.int8)
        s = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
        hard = torch.tensor([[7.0] * group, [3.5] * group], dtype=torch.bfloat16)
        call_success(f"regular_constant_g{group}", q, s, group, hard_expected=hard)
        q = torch.zeros((2, group), dtype=torch.int8)
        q[0, 0], q[1, 3] = 16, -32
        s = torch.ones((2, 1), dtype=torch.float32)
        hard = torch.tensor(
            [[16 * ((-1) ** sum((j // (4 ** k)) % 4 == 3
                               for k in range({64: 3, 256: 4}[group])))
              / math.sqrt(group) for j in range(group)],
             [-32 * ((-1) ** ((j % 4 == 0)
                      + sum((j // (4 ** k)) % 4 == 3
                            for k in range(1, {64: 3, 256: 4}[group]))))
              / math.sqrt(group) for j in range(group)]],
            dtype=torch.bfloat16)
        call_success(f"regular_basis_g{group}", q, s, group, hard_expected=hard)
        for non_dyadic in (False, True):
            q, s = tensor_pattern(9, 2 * group, non_dyadic)
            call_success(f"matrix_g{group}_non_dyadic_{non_dyadic}",
                         q, s, group, exact_required=not non_dyadic,
                         use_linear=not non_dyadic)
        heads, d = (3, 2) if group == 64 else (4, 2)
        rows = 3 * heads * d
        q = torch.tensor([[(r * 11 + c * 7) % 255 - 128
                           for c in range(2 * group)] for r in range(rows)],
                         dtype=torch.int8)
        s = torch.tensor([[(r + 1) / 32] for r in range(rows)], dtype=torch.float32)
        call_success(f"grouped_qkv_g{group}", q, s, group, layout="grouped_qkv",
                     heads=heads, head_dim=d, use_linear=True)
    q, s = tensor_pattern(257, 128)
    call_success("constrained_multi_chunk_g64", q, s, 64,
                 workspace=80 * 1024, use_linear=False)


class InjectedCallbackFailure(Exception):
    pass


def lifecycle_tests():
    q, s = tensor_pattern(257, 128)
    normal = spec(64)
    kwargs = dict(max_output_bytes=q.numel() * 2, max_workspace_bytes=80 * 1024)
    count = [0]

    def count_only():
        count[0] += 1
        return False

    output, stats = recover(q, s, normal, cancelled=count_only, **kwargs)
    total = count[0]
    del output, stats
    check("lifecycle.multichunk_cancel_boundaries", total >= 4, {"callback_calls": total})
    for when in sorted({1, 2, max(total - 1, 1), total}):
        for inject_exception in (False, True):
            before_q, before_s = q.clone(), s.clone()
            n = [0]

            def callback():
                n[0] += 1
                if n[0] == when:
                    if inject_exception:
                        raise InjectedCallbackFailure("frozen injected callback")
                    return True
                return False

            trace = StorageTrace([q, s])
            kept_exception = None
            result = None
            try:
                with trace:
                    result = recover(q, s, normal, cancelled=callback, **kwargs)
            except Exception as exc:
                kept_exception = exc
            label = f"lifecycle.{'exception' if inject_exception else 'cancel'}_{when}"
            expected_type = InjectedCallbackFailure if inject_exception else RuntimeError
            check(label + ".propagates", isinstance(kept_exception, expected_type))
            check(label + ".no_partial_return", result is None)
            check(label + ".caller_unchanged",
                  torch.equal(q, before_q) and torch.equal(s, before_s))
            gc.collect()
            check(label + ".release_while_traceback_retained",
                  kept_exception is not None and kept_exception.__traceback__ is not None
                  and trace.retained() == 0)
            kept_exception = None
            result = None
            del trace, callback, before_q, before_s
            gc.collect()
    # A fresh complete call after all failures proves no poisoned shared state.
    call_success("lifecycle.retry_after_failures", q[:9].contiguous(),
                 s[:9].contiguous(), 64)


def quantization_error_separate():
    group = 64
    original = torch.tensor([[(r * 13 + c * 5) % 61 / 32 - 0.75
                              for c in range(group)] for r in range(4)],
                            dtype=torch.bfloat16)
    dense = dense_h(group)
    original_rows = original.double().tolist()
    rotated = [[sum(row[k] * dense[j][k] for k in range(group))
                for j in range(group)] for row in original_rows]
    scales_values = [f32(max(abs(v) for v in row) / 127) for row in rotated]
    quantized = [[max(-127, min(127, round(v / scale))) if scale else 0
                  for v in row] for row, scale in zip(rotated, scales_values)]
    q = torch.tensor(quantized, dtype=torch.int8)
    scales = torch.tensor([[v] for v in scales_values], dtype=torch.float32)
    call_success("roundtrip_quantized_input", q, scales, group,
                 exact_required=False)
    restored, stats = recover(q, scales, spec(group),
                              max_output_bytes=q.numel() * 2,
                              max_workspace_bytes=BUDGET)
    error = restored.double() - original.double()
    METRICS["original_bf16_quantization_error_only"] = {
        "max_abs": float(error.abs().max()),
        "mean_squared_error": float((error * error).mean()),
        "original_bf16_parity_required": False,
        "quality_threshold": None,
        "interpretation": "Synthetic quantization error, not restoration parity failure or quality evidence",
    }
    check("quantization_error.finite_observation", bool(torch.isfinite(error).all()))
    del restored, stats


def main():
    global torch, recover, parse, StorageTrace
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--candidate-sha256", required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing overwrite")
    candidate = args.project_root / "src/h3_omni_adapter/convrot.py"
    actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual_hash != args.candidate_sha256:
        raise SystemExit("candidate hash mismatch")
    import torch as torch_import
    torch = torch_import
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.project_root / "src"))
    module = importlib.import_module("h3_omni_adapter.convrot")
    if Path(module.__file__).resolve() != candidate.resolve():
        raise SystemExit("candidate import location mismatch")
    parse, recover = module.parse_comfy_quant, module.recover_bf16_projection
    StorageTrace = install_trace()
    fatal = None
    try:
        parse_tests()
        argument_tests()
        anchors_and_math()
        lifecycle_tests()
        quantization_error_separate()
    except Exception as exc:
        fatal = {"type": type(exc).__name__, "message": str(exc)[:500]}
        check("worker.unexpected_exception", False, fatal)
    result = {
        "campaign_verdict": "INCONCLUSIVE",
        "scope": "CPU tiny synthetic mathematical restoration and Python-visible lifecycle only",
        "execution_complete": fatal is None,
        "candidate_sha256": actual_hash,
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "cuda_executed": False,
        "native_model_executed": False,
        "model_weight_payload_downloaded": False,
        "checks": CHECKS,
        "checks_completed": sum(item["ok"] for item in CHECKS),
        "checks_total": len(CHECKS),
        "metrics": METRICS,
        "fatal": fatal,
        "limits": {
            "workspace_stats": "Torch-visible owned storage only; excludes caller input/scale, final BF16 output and opaque BLAS/C++ internal allocations",
            "RSS_CUDA": "Not measured here; outer controller responsible",
            "rounding": "Dyadic/basis anchors exact; non-dyadic bounded against independent explicit FP64 sum after F32 products, not fused CUDA parity",
            "quality_and_end_to_end": "INCONCLUSIVE",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf8")
    print(json.dumps({"execution_complete": result["execution_complete"],
                      "checks_completed": result["checks_completed"],
                      "checks_total": result["checks_total"]}))
    return 0 if fatal is None and all(item["ok"] for item in CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
