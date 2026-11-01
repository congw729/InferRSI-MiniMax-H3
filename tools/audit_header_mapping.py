"""Audit saved headers against real upstream name resolution; no payload reads."""
import argparse
import hashlib
import json
from pathlib import Path
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import _resolve_native_target
    from h3_omni_adapter.byte_adapter import plan_packed
    prefix = "FastVideo__FastH3-Omni-NVFP4-QAD__transformer_ref__"
    dense, packed, files = {}, None, []
    for path in sorted(args.headers.glob(prefix + "*.header.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        raw = path.with_suffix(".raw")
        if hashlib.sha256(raw.read_bytes()).hexdigest() != metadata["header_sha256"]:
            raise ValueError("historical header digest mismatch")
        header = metadata["header"]
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if "nvfp4_weights" in path.name:
            packed = header
        else:
            for key, value in header.items():
                if key != "__metadata__":
                    if key in dense:
                        raise ValueError("duplicate dense source tensor")
                    dense[key] = value
    bindings, unresolved = [], []
    for name, metadata in dense.items():
        module, _, field = name.rpartition(".")
        target = _resolve_native_target(module)
        if target is None or field not in {"weight", "bias", "weight_scale"}:
            unresolved.append(name)
            continue
        bindings.append({"source": name, "target": target[0] + "." + field,
                         "layout": target[1], "source_dtype": metadata["dtype"],
                         "source_shape": metadata["shape"]})
    packed_plan = plan_packed(packed)
    defaults = [name for name, binding in packed_plan.items() if binding["default"] is not None]
    dense_targets = {b["target"] for b in bindings}
    result = {
        "campaign_verdict": "INCONCLUSIVE",
        "scope": "saved-header name mapping audit only; no model constructor or payload read",
        "dense_source_keys": len(dense), "dense_resolved_keys": len(bindings),
        "dense_unique_target_names": len(dense_targets), "unresolved_dense_names": unresolved,
        "packed_source_keys": len(packed) - int("__metadata__" in packed),
        "packed_runtime_names": len(packed_plan), "packed_default_buffers": len(defaults),
        "dense_packed_target_collisions": sorted(dense_targets & set(packed_plan)),
        "extra_runtime_buffer_from_source": "rope.inv_freq reconstructed by load_weights",
        "actual_model_parameter_coverage_verified": False,
        "target_shapes_dtypes_verified_against_model": False,
        "headers": files, "dense_bindings": bindings, "packed_default_names": defaults,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"headers", "dense_bindings", "packed_default_names"}}))
if __name__ == "__main__":
    main()
