"""Official safetensors + upstream PyTorch loader parity, never model inference."""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from dataclasses import dataclass

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from torch import nn
from safetensors import safe_open

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def serialized(tensor):
    return bytes(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist())

def reference_module(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"NVFP4Projection", "_Projection", "_linear_scales", "_swizzled_scales"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("upstream reference definitions changed")
    module = types.ModuleType("h3_isolated_reference")
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, nn=nn, dataclass=dataclass)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module([future, *nodes], type_ignores=[]))
    exec(compile(code, "<isolated-upstream-PyTorch-loader>", "exec"), module.__dict__)
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--fault", choices=["swap-qk", "omit-gate-swap"])
    args = parser.parse_args()
    checks = []
    result = {"scope": "L0 synthetic tensor loading only", "campaign_verdict": "INCONCLUSIVE",
              "full_runtime_imported": False, "model_inference_executed": False,
              "backend": args.device, "fault": args.fault,
              "reference_execution": "isolated original AST definitions with real torch; not full vLLM import",
              "versions": {name: importlib.metadata.version(name)
                           for name in ["torch", "numpy", "safetensors"]},
              "checks": checks}
    def check(name, condition, detail=None):
        checks.append({"name": name, "ok": bool(condition), "detail": detail})
    try:
        from h3_omni_adapter.byte_adapter import build_plan, read_header
        from h3_omni_adapter.tensor_loader import TensorStager
        freeze = json.loads((args.prototype / "plan/freeze.json").read_text(encoding="utf-8"))
        check("historical_freeze_intact", all(sha(args.prototype / p) == h
                                            for p, h in freeze["files"].items()))
        if not checks[-1]["ok"]:
            raise ValueError("historical frozen source changed")
        spec = importlib.util.spec_from_file_location("frozen_generic_reference",
                                                     args.prototype / "validator/reference.py")
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        expected = oracle.expected()
        source = args.prototype / "sources/checkpoint_quantization.py"
        upstream = reference_module(source)
        result["reference_sha256"] = sha(source)
        fixture = args.prototype / "fixtures/valid"
        source_tensors = {}
        official_matches_raw = True
        for filename in ("dense.safetensors", "nvfp4_weights.safetensors"):
            path = fixture / filename
            header = read_header(path)
            raw = path.read_bytes()
            with safe_open(path, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    tensor = reader.get_tensor(key)
                    info = header[key]
                    official_matches_raw &= serialized(tensor) == raw[info["offset"]:info["offset"] + info["length"]]
                    official_matches_raw &= tuple(tensor.shape) == info["shape"]
                    source_tensors[key] = tensor
        check("official_safetensors_matches_frozen_bytes", official_matches_raw)
        check("source_and_target_coverage", len(source_tensors) == 32 and len(expected) == 37)
        device = torch.device(args.device)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            device = torch.device("cuda", torch.cuda.current_device())
            free, total = torch.cuda.mem_get_info()
            result["cuda"] = {"name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                              "architectures": torch.cuda.get_arch_list(), "runtime": torch.version.cuda,
                              "free_before_bytes": free, "total_bytes": total}
            if free < 256 * 1024 * 1024:
                raise RuntimeError("less than 256 MiB free; tiny CUDA probe deferred")
            torch.cuda.reset_peak_memory_stats()
        plan = build_plan(fixture, sidecars=["nvfp4_weights.safetensors"])
        types_by_name = {"U8": torch.uint8, "BF16": torch.bfloat16, "F32": torch.float32}
        runtime_reference = {}
        grouped = {}
        for name, binding in plan.items():
            if binding["field"] == "dense":
                runtime_reference[name] = source_tensors[binding["key"]].to(device)
            else:
                grouped.setdefault(name.rsplit(".", 1)[0], {})[binding["field"]] = binding
        scalar_projection = None
        for prefix, group in grouped.items():
            item = group["weight"]
            tensors = {field: value["key"] for field, value in group.items() if value["default"] is None}
            model_spec = upstream.NVFP4Projection("synthetic-projection", item["rows"], item["columns"],
                                                 tensors, item["swap_halves"])
            with torch.device(device):
                projection = upstream._Projection(model_spec, item["rows"], item["columns"], 0)
            for field, key in tensors.items():
                parameter = getattr(projection, field)
                parameter.weight_loader(parameter, source_tensors[key])
            for field in group:
                runtime_reference[prefix + "." + field] = getattr(projection, field)
            scalar_projection = projection
        reference_ok = all(t.dtype == types_by_name[expected[name][0]]
                           and tuple(t.shape) == expected[name][1]
                           and serialized(t) == expected[name][2] for name, t in runtime_reference.items())
        check("upstream_real_tensors_match_independent_oracle", reference_ok)
        check("upstream_target_set", set(runtime_reference) == set(expected))
        rejected = 0
        for value in (0.0, -1.0, float("nan"), float("inf")):
            try:
                scalar_projection.weight_loader(scalar_projection.alpha, torch.tensor(value, device="cpu"))
            except ValueError:
                rejected += 1
        check("upstream_rejects_invalid_scalars", rejected == 4)
        rejected = 0
        for value in (torch.zeros(1, dtype=torch.uint8),
                      torch.zeros_like(scalar_projection.weight, dtype=torch.float32, device="cpu")):
            try:
                scalar_projection.weight_loader(scalar_projection.weight, value)
            except ValueError:
                rejected += 1
        check("upstream_rejects_wrong_weight_dtype_and_shape", rejected == 2)
        candidate_plan = copy.deepcopy(plan)
        if args.fault == "swap-qk":
            for name in list(candidate_plan):
                if ".qkv_proj.packed.0." in name:
                    other = name.replace(".packed.0.", ".packed.1.")
                    candidate_plan[name], candidate_plan[other] = candidate_plan[other], candidate_plan[name]
        elif args.fault == "omit-gate-swap":
            for binding in candidate_plan.values():
                binding["swap_halves"] = False
        parameters = {name: nn.Parameter(torch.empty(shape, dtype=types_by_name[dtype], device=device),
                                          requires_grad=False)
                      for name, (dtype, shape, _) in expected.items()}
        pointers = {name: p.data_ptr() for name, p in parameters.items()}
        with TensorStager(candidate_plan, device=device) as stager:
            for name, binding in candidate_plan.items():
                stager.load_into(binding, parameters[name])
            stager.synchronize()
            mismatches = [name for name, parameter in parameters.items()
                          if serialized(parameter) != expected[name][2]]
            check("candidate_tensor_bytes_match_oracle", not mismatches, mismatches)
            check("candidate_parameter_storage_preserved",
                  all(p.data_ptr() == pointers[name] for name, p in parameters.items()))
            check("pinned_slots_on_cuda", args.device != "cuda" or all(slot.is_pinned() for slot in stager.slots))
            ids = [slot.data_ptr() for slot in stager.slots]
            for index in range(128):
                name = list(candidate_plan)[index % len(candidate_plan)]
                stager.load_into(candidate_plan[name], parameters[name])
            stager.synchronize()
            check("slots_reused_across_128_loads", ids == [slot.data_ptr() for slot in stager.slots])
            check("reloads_remain_correct",
                  all(serialized(p) == expected[name][2] for name, p in parameters.items()))
            result["staging"] = stager.stats()
        check("staging_closed", stager.closed and not stager.slots and stager.byte_stager.closed)
        try:
            stager.load_into(next(iter(plan.values())), next(iter(parameters.values())))
            check("closed_stager_rejects_load", False)
        except ValueError:
            check("closed_stager_rejects_load", True)
        error_stager = None
        try:
            with TensorStager(plan, device=device) as error_stager:
                binding = next(b for b in plan.values() if b["field"] == "weight")
                error_stager.load_into(binding, nn.Parameter(torch.empty(1, device=device), requires_grad=False))
        except ValueError:
            pass
        check("invalid_target_closes_maps", error_stager is not None and error_stager.closed)
        if args.device == "cuda":
            a = torch.arange(4096, dtype=torch.float32).remainder(7).reshape(64, 64)
            expected_mm = a @ a.T
            actual_mm = a.to(device) @ a.to(device).T
            torch.cuda.synchronize()
            check("cuda_small_matmul_exact_integer_case", torch.equal(actual_mm.cpu(), expected_mm))
            result["cuda"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result["checks_ok"] = sum(item["ok"] for item in checks)
        result["checks_total"] = len(checks)
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
    # Avoid path-bearing traceback capture; messages and roots are redacted before write.
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    for source, target in ((str(args.prototype), "<PROTOTYPE_ROOT>"),
                           (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"),
                           (str(Path.home()), "<USER_HOME>")):
        text = text.replace(source.replace("\\", "\\\\"), target).replace(source, target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"checks_ok": result.get("checks_ok"), "checks_total": len(checks),
                      "execution_error": result.get("execution_error")}))
    return 0 if checks and all(item["ok"] for item in checks) and "execution_error" not in result else 1

if __name__ == "__main__":
    raise SystemExit(main())
