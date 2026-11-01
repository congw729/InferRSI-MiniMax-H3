"""Import the complete upstream module, then reuse the frozen tensor checks.

The reference hook is deliberately replaced by the real imported module.
No imports are stubbed. The tiny FP4 forward check compares identical loaded
weights and checks zero input; it is not a dequantization or model-quality oracle.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
            "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
    os.environ[key] = "1"
os.environ["MAX_JOBS"] = "2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    prototype = Path(remaining[remaining.index("--prototype") + 1])
    device = remaining[remaining.index("--device") + 1]
    project = Path(__file__).resolve().parents[1]
    result = {"campaign_verdict": "INCONCLUSIVE", "model_inference_executed": False}
    code = 1
    try:
        import torch
        import vllm
        import vllm_omni
        upstream = importlib.import_module("vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization")
        origin = Path(upstream.__file__).resolve()
        if not origin.is_relative_to(args.omni_root.resolve()):
            raise ValueError("H3 module import resolved outside selected checkout")
        source_hash = hashlib.sha256(origin.read_bytes()).hexdigest()
        if source_hash != hashlib.sha256((prototype / "sources/checkpoint_quantization.py").read_bytes()).hexdigest():
            raise ValueError("imported source differs from audited reference")
        spec = importlib.util.spec_from_file_location("frozen_tensor_validation",
                                                     project / "tools/validate_tensors.py")
        validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation)
        validation.reference_module = lambda unused: upstream
        sys.argv = [str(project / "tools/validate_tensors.py"), *remaining]
        code = validation.main()
        result = json.loads(output.read_text(encoding="utf-8"))
        result.update(full_runtime_imported=True, reference_execution="full unmodified upstream Python module",
                      versions={**result["versions"], "vllm": vllm.__version__,
                                "vllm-omni": importlib.metadata.version("vllm-omni")},
                      runtime_source_sha256=source_hash,
                      runtime_origin="<OMNI_ROOT>/" + origin.relative_to(args.omni_root).as_posix(),
                      runtime_package_origin="<OMNI_ROOT>/" + Path(vllm_omni.__file__).resolve().relative_to(args.omni_root).as_posix())
        if code == 0 and device == "cuda":
            from safetensors import safe_open
            from h3_omni_adapter.byte_adapter import build_plan
            from h3_omni_adapter.tensor_loader import TensorStager
            plan = build_plan(prototype / "fixtures/valid", sidecars=["nvfp4_weights.safetensors"])
            prefix = "blocks.0.attn.qkv_proj.packed.0."
            selected = {name[len(prefix):]: binding for name, binding in plan.items() if name.startswith(prefix)}
            tensors = {field: binding["key"] for field, binding in selected.items() if binding["default"] is None}
            weight = selected["weight"]
            model_spec = upstream.NVFP4Projection("synthetic-q", weight["rows"], weight["columns"], tensors)
            with torch.device("cuda"):
                ordinary = upstream._Projection(model_spec, weight["rows"], weight["columns"], 0)
                staged = upstream._Projection(model_spec, weight["rows"], weight["columns"], 0)
            with safe_open(prototype / "fixtures/valid/nvfp4_weights.safetensors", framework="pt", device="cpu") as reader:
                for field, key in tensors.items():
                    parameter = getattr(ordinary, field)
                    parameter.weight_loader(parameter, reader.get_tensor(key))
            with TensorStager(plan, device="cuda") as stager:
                for field, binding in selected.items():
                    stager.load_into(binding, getattr(staged, field))
                stager.synchronize()
            x = torch.arange(16 * weight["columns"], dtype=torch.float32, device="cuda").reshape(16, -1)
            x = (x.remainder(5) / 16).to(torch.bfloat16)
            with torch.inference_mode():
                expected = ordinary(x)
                actual = staged(x)
                zero = staged(torch.zeros_like(x))
            torch.cuda.synchronize()
            kernel_checks = [
                {"name": "full_nvfp4_forward_shape", "ok": tuple(actual.shape) == (16, weight["rows"])},
                {"name": "full_nvfp4_forward_finite", "ok": bool(torch.isfinite(actual).all())},
                {"name": "full_nvfp4_forward_loader_parity", "ok": torch.equal(actual, expected)},
                {"name": "full_nvfp4_forward_zero_input", "ok": bool((zero == 0).all())}]
            result["checks"].extend(kernel_checks)
            result["checks_total"] = len(result["checks"])
            result["checks_ok"] = sum(c["ok"] for c in result["checks"])
            result["nvfp4_probe"] = {"implementation": "original _Projection.forward / FlashInfer cutlass",
                                     "shape_m_n_k": [16, weight["rows"], weight["columns"]],
                                     "quality_oracle": "none; exact loader parity and zero-input smoke only"}
            code = 0 if all(c["ok"] for c in result["checks"]) else 1
    except Exception as exc:
        result["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 1
    def sanitize(value):
        if isinstance(value, str):
            for source, label in ((str(prototype), "<PROTOTYPE_ROOT>"), (str(project), "<H3_PROJECT_ROOT>"),
                                  (str(args.omni_root), "<OMNI_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
                value = value.replace(source, label)
            return value
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        return value
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sanitize(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(sanitize({"exit_code": code, "checks_ok": result.get("checks_ok"),
                               "checks_total": result.get("checks_total"),
                               "execution_error": result.get("execution_error")})))
    return code
if __name__ == "__main__":
    raise SystemExit(main())
