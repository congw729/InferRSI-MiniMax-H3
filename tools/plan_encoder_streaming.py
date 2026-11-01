"""Plan text-only encoder streaming from the completed native meta inventory."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    saved = json.loads(a.inventory.read_text())
    groups = defaultdict(int)
    sources = defaultdict(int)
    counts = defaultdict(int)
    for row in saved["inventory"]:
        name = row["name"]
        parts = name.split(".")
        group = ".".join(parts[:3]) if name.startswith(("vision.blocks.", "text_model.layers.")) else (
            "vision.non_blocks" if name.startswith("vision.") else "text_model.non_layers")
        groups[group] += row["logical_bytes"]
        counts[group] += 1
        for source in row["sources"]:
            offsets = source["data_offsets"]
            sources[group] = max(sources[group], offsets[1] - offsets[0])
    text_layers = [v for k, v in groups.items() if k.startswith("text_model.layers.")]
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "static text-only streaming plan, not execution",
        "groups_logical_bytes": dict(groups), "group_targets": dict(counts),
        "group_largest_source_bytes": dict(sources),
        "text_resident_plus_one_layer_bytes": groups["text_model.non_layers"] + max(text_layers),
        "text_resident_bytes": groups["text_model.non_layers"],
        "one_text_layer_bytes": max(text_layers),
        "vision_total_bytes": sum(v for k, v in groups.items() if k.startswith("vision.")),
        "largest_source_bytes": max(sources.values()),
        "plan_constraints": [
            "TP1; text-only IDs; reject image/video placeholders and all pixel inputs",
            "native encoder load_weights requires every parameter each call; cannot reuse it for one-group loading",
            "preflight complete selected source/target set before using individual native parameter.weight_loader",
            "packed reciprocal global scale and default input_scale must use original projection behavior",
            "encode_ids checks first parameter device, which is vision; meta vision breaks this entry guard",
            "text-only wrapper must validate explicit text resident state and preserve native text-path behavior",
            "no wholesale load_to_device or default non-block staging, which moves vision as well",
            "RoPE is created per forward by native text rotary class; inspect its device derivation",
            "preserve layer-50 unnormalized outputs; do not add final norm",
            "all full-model and performance claims remain INCONCLUSIVE"]}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
