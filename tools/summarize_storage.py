"""Static logical storage ledger from actual meta inventories, not a peak measurement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=Path, required=True)
    a = p.parse_args()
    def latest(prefix):
        paths = sorted((a.iteration / "results").glob(prefix + "*/trial/result.json"))
        return json.loads(paths[-1].read_text())
    dit = latest("meta-coverage-v2-")
    groups, max_source = defaultdict(int), defaultdict(int)
    for item in dit["inventory"]:
        name = item["name"]
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "blocks" else ".".join(parts[:3]) if parts[:2] == ["token_refiner", "blocks"] else "resident"
        groups[group] += item["logical_bytes"]
        for source in item.get("sources", []):
            offsets = source["data_offsets"]
            max_source[group] = max(max_source[group], offsets[1] - offsets[0])
    packed = latest("compact-decoder-v3-")
    vae = latest("vae-meta-v3-")
    result = {"campaign_verdict": "INCONCLUSIVE", "measurement_kind": "logical tensor storage only",
        "dit_groups_bytes": dict(groups), "dit_max_source_tensor_bytes_by_group": dict(max_source),
        "dit_total_logical_bytes": dit["logical_bytes"],
        "dit_resident_plus_largest_streamed_group_bytes": groups["resident"] + max(v for k, v in groups.items() if k != "resident"),
        "dit_largest_single_source_tensor_bytes": max(max_source.values()),
        "native_video_vae_fp32_logical_bytes": vae["video"]["logical_bytes"],
        "native_audio_vae_fp32_logical_bytes": vae["audio"]["logical_bytes"],
        "packed_fast8_decoder_logical_bytes": packed["full_decoder"]["logical_bytes"],
        "observed_small_decoder_library_cuda_cache_bytes": packed["release_accounting"]["torch_cuda_allocated_before_cublas_clear"],
        "excluded": ["activations", "attention workspaces", "allocator fragmentation", "driver context",
                     "concurrent CUDA streams", "pinned buffers", "transforms", "filesystem page cache",
                     "future full-shape kernel specializations"],
        "feasibility": "No full-model peak or video quality conclusion. Requires bounded full-shape experiments."}
    target = a.iteration / "results/static-storage-ledger.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
