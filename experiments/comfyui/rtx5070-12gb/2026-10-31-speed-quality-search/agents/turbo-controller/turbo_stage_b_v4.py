"""Q33 bounded speed/quality profiles with native node timing.
This is a new harness derived from frozen Q32; old sources remain unchanged.
Run with the existing portable Python under InferRSI. No model imports here.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import http.client as http_client
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

V1_SHA = "63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35"
STAGE_GATES_SHA = "dd9658a253e1e0c73bd1d1fe9034055622289a9f6748486fb55761bfa195eecf"
STAGE_NODES = {'UNETLoader': 'nodes', 'VAELoader': 'nodes', 'VAEDecode': 'nodes', 'MiniMaxH3SigmaShift': 'comfy_extras.nodes_minimax_h3', 'RandomNoise': 'comfy_extras.nodes_custom_sampler', 'KSamplerSelect': 'comfy_extras.nodes_custom_sampler', 'BasicScheduler': 'comfy_extras.nodes_custom_sampler', 'BasicGuider': 'comfy_extras.nodes_custom_sampler', 'SamplerCustomAdvanced': 'comfy_extras.nodes_custom_sampler', 'VAEDecodeAudio': 'comfy_extras.nodes_audio', 'CreateVideo': 'comfy_extras.nodes_video', 'SaveVideo': 'comfy_extras.nodes_video', 'ConditioningLoader': 'comfy_extras.nodes_cond', 'EmptyMiniMaxH3LatentAV': 'comfy_extras.nodes_minimax_h3'}
NODE_FAMILIES = {'original': {'UNETLoader': 'nodes', 'VAELoader': 'nodes', 'VAEDecode': 'nodes', 'MiniMaxH3SigmaShift': 'comfy_extras.nodes_minimax_h3', 'RandomNoise': 'comfy_extras.nodes_custom_sampler', 'KSamplerSelect': 'comfy_extras.nodes_custom_sampler', 'BasicScheduler': 'comfy_extras.nodes_custom_sampler', 'BasicGuider': 'comfy_extras.nodes_custom_sampler', 'SamplerCustomAdvanced': 'comfy_extras.nodes_custom_sampler', 'VAEDecodeAudio': 'comfy_extras.nodes_audio', 'CreateVideo': 'comfy_extras.nodes_video', 'SaveVideo': 'comfy_extras.nodes_video', 'ConditioningLoader': 'comfy_extras.nodes_cond', 'EmptyMiniMaxH3LatentAV': 'comfy_extras.nodes_minimax_h3'}, 'trim': {'UNETLoader': 'nodes', 'VAELoader': 'nodes', 'VAEDecode': 'nodes', 'MiniMaxH3SigmaShift': 'comfy_extras.nodes_minimax_h3', 'RandomNoise': 'comfy_extras.nodes_custom_sampler', 'KSamplerSelect': 'comfy_extras.nodes_custom_sampler', 'BasicScheduler': 'comfy_extras.nodes_custom_sampler', 'BasicGuider': 'comfy_extras.nodes_custom_sampler', 'SamplerCustomAdvanced': 'comfy_extras.nodes_custom_sampler', 'VAEDecodeAudio': 'comfy_extras.nodes_audio', 'CreateVideo': 'comfy_extras.nodes_video', 'SaveVideo': 'comfy_extras.nodes_video', 'ConditioningLoader': 'comfy_extras.nodes_cond', 'EmptyMiniMaxH3LatentAV': 'comfy_extras.nodes_minimax_h3', 'ModelAttentionBackend': 'comfy_extras.nodes_model_advanced', 'BlockSparseAttention': 'comfy_extras.nodes_sparse_attention'}}
STAGE_SOURCE_SHA256 = {'comfy_extras/nodes_cond.py': '56344e64a20692a0f64521bf4de78c0a9c040e23c3499a00491b5243ca39f898', 'comfy_extras/nodes_minimax_h3.py': '5999ed8c22b13c8251084c0629fd82137d2fb52b55106abfc73bf5cafc73841d', 'comfy_api/latest/_io.py': 'a16cea6dd228468b5fbe3fa2acb5dd86bf9445babcd9d8bfe697b84c449bf5fb', 'folder_paths.py': '045a2ea987c0c7bcb0e47cbc46826ee34e82179562c85123f35ea3ff1c3e7ac4'}
STAGE_SOURCE_SHA256.update({'nodes.py': '776f549a05fc068eca20a9edc1602c6f1b80cfbbf5153bc44ba371a55fbb13be', 'comfy/lora.py': 'ac46c49b26b6d9f6d81d4ae4dcc444a4662e3b291635e2234f9ed7a1d1b16eeb', 'comfy/lora_convert.py': 'cc4401759b37e2208fd6dba8497db1748c1d8ee3231151632349bececf0f4776', 'comfy/weight_adapter/__init__.py': 'd342ef7a6faf4aebecbf206de2811626a017ab4cb2d1cac19109348f53952229', 'comfy/weight_adapter/base.py': 'a5083daaae27a0c0146bdbbaa1b5d06376b01b582fa38cb52eba4c8b10214d66', 'comfy/weight_adapter/boft.py': 'c2efbfa11adfb3c4d54f37df4b8a997de34d0f7aa94e3c669eb3fcda36da6735', 'comfy/weight_adapter/bypass.py': 'd730f23c5d36968e102073051e9677db7761e87228d824470e4034909e3337dd', 'comfy/weight_adapter/glora.py': '443ad93f07cf9c5820df3ccb2d93b8f43452028c7073ca544cc75c79879a6ef3', 'comfy/weight_adapter/loha.py': '89c6750b772017f4b35190b9a75a9161fb2c502ae7cf1377a8ea448a728e7731', 'comfy/weight_adapter/lokr.py': 'ef39581b4f0a5f2eb1175542bf8f65ff74d5e02d5652f598fe96f047ecc7c299', 'comfy/weight_adapter/lora.py': '5e30c8b8a22be6459883cb5d758fa76f725cc4688d4048200145b885567c4cec', 'comfy/weight_adapter/oft.py': 'e6b9c771d96dcde11a6dad62c37c5f062f3f9933452eed8b49c62589e38b9fff'})
SOURCE_MANIFEST_SHA = "bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007"
TIMING_MODULE_SHA = "f0abb1937ca5462d712a73d1a8623b6a9f5df86927dbcf6375903b8a5ad3e614"
EXPECTED_BY_FAMILY = {'original': {'original-dit': {'id': 'original-dit', 'bytes': 34038892334, 'sha256': '7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5', 'relative_path': 'comfy-original/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors'}, 'video-vae': {'id': 'video-vae', 'bytes': 2811065184, 'sha256': '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e', 'relative_path': 'comfy-original/models/vae/minimax_h3_video_vae_int8_convrot.safetensors'}, 'audio-vae': {'id': 'audio-vae', 'bytes': 605254808, 'sha256': '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48', 'relative_path': 'comfy-original/models/vae/minimax_h3_audio_vae_fp32.safetensors'}}, 'trim': {'trim-dit': {'id': 'trim-dit', 'bytes': 11894623166, 'sha256': '748b2bcc9c4b7bf7b23406987b42d4ac40362070c164cf11092482be6a863455', 'relative_path': 'comfy-trim/models/diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors'}, 'video-vae': {'id': 'video-vae', 'bytes': 2811065184, 'sha256': '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e', 'relative_path': 'comfy-trim/models/vae/minimax_h3_video_vae_int8_convrot.safetensors'}, 'audio-vae': {'id': 'audio-vae', 'bytes': 605254808, 'sha256': '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48', 'relative_path': 'comfy-trim/models/vae/minimax_h3_audio_vae_fp32.safetensors'}}}
CANONICAL_SHA = {'original': 'd5937f2505f74e07ce8de746c49c98e8a428535d79ccc878383e8b989807fb29', 'trim': 'a915f62f109cd2e1be615eb7b0c1aa6fab4a4b6ad6e31791fb84488b02a17ad8'}
ASSET_EXPECTED = EXPECTED_BY_FAMILY["original"]
ASSET_IDS = set(ASSET_EXPECTED)

TURBO_FILENAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
TURBO_PROFILE_SHA = "84be14a039cc3955eb475f5fa2cd52a62d92190a2c534ac7ee75f18d9e64bb10"
NODE_FAMILIES["turbo-native"] = {**NODE_FAMILIES["original"], "LoraLoaderModelOnly": "nodes"}
EXPECTED_BY_FAMILY["turbo-native"] = {**EXPECTED_BY_FAMILY["original"],
    "turbo-v4-step600-ema": {"id":"turbo-v4-step600-ema", "bytes":779849816,
        "sha256":"5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
        "relative_path":"comfy-original/models/loras/"+TURBO_FILENAME}}
CANONICAL_SHA["turbo-native"] = CANONICAL_SHA["original"]
NativeWSObserver = None

def load_timing_module(path, iteration_root):
    path = Path(path)
    expected = Path(iteration_root) / "agents/timing-controller/timing_events_v1.py"
    require(path.is_absolute() and path.resolve() == expected.resolve()
            and not path.is_symlink() and digest(path) == TIMING_MODULE_SHA,
            "explicit frozen timing source")
    spec = importlib.util.spec_from_file_location("q33_frozen_native_timing_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), "actual timing import source")
    return module.NativeWSObserver

EXTRA_GATES = {
    "asset_timeout_seconds", "startup_timeout_seconds", "render_timeout_seconds",
    "gpu_poll_seconds", "gpu_query_timeout_seconds", "start_gpu_free_bytes",
    "running_gpu_free_bytes", "reserve_vram_gb", "cuda_device_index",
    "max_output_bytes", "max_total_http_bytes", "max_history_polls",
}


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_dynamic_vram_startup(log_path):
    """Observe only this run's bounded startup-log snapshot before model access."""
    limit = 1048576
    require(log_path.stat().st_size <= limit, "startup log exceeds 1 MiB")
    with log_path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "startup log exceeds 1 MiB")
    require(b"Falling back to legacy ModelPatcher" not in raw,
            "legacy ModelPatcher fallback detected")
    require(b"DynamicVRAM support detected and enabled" in raw,
            "DynamicVRAM enabled marker missing")
    return {"startup_log_bytes": len(raw),
            "startup_log_sha256": hashlib.sha256(raw).hexdigest()}


def conditioning_header(path, unique_json, max_bytes):
    """Stdlib structural validation only; tensor contents/finite values are not checked."""
    before = path.stat()
    require(8 < before.st_size <= max_bytes, "conditioning file bound")
    with path.open("rb") as stream:
        header_bytes = int.from_bytes(stream.read(8), "little")
        require(2 <= header_bytes <= 65536 and header_bytes + 8 < before.st_size,
                "conditioning header bound")
        raw = stream.read(header_bytes)
    header = json.loads(raw, object_pairs_hook=unique_json)
    require(type(header) is dict, "conditioning header object")
    metadata = header.pop("__metadata__", {})
    require(type(metadata) is dict and all(type(k) is str and type(v) is str
            for k, v in metadata.items()), "conditioning metadata type")
    require(set(metadata) <= {"conditioning_options"}, "unexpected conditioning metadata")
    options = json.loads(metadata.get("conditioning_options", "{}"), object_pairs_hook=unique_json)
    require(type(options) is dict and all(type(v) in (str, bool, int, float)
            and (type(v) is not float or math.isfinite(v)) for v in options.values()),
            "conditioning options metadata type")
    required = {"conditioning", "minimax_token_tags"}
    require(required <= set(header) <= required | {"pooled_output"}, "conditioning tensor set")
    require(not set(options) & set(header), "conditioning metadata/tensor collision")
    sizes = {"F16": 2, "BF16": 2, "F32": 4, "I64": 8}
    segments = []
    data_bytes = before.st_size - 8 - header_bytes
    for name, tensor in header.items():
        require(type(tensor) is dict and set(tensor) == {"dtype", "shape", "data_offsets"},
                "conditioning tensor header fields")
        shape, offsets, dtype = tensor["shape"], tensor["data_offsets"], tensor["dtype"]
        require(type(shape) is list and 1 <= len(shape) <= 3
                and all(type(x) is int and 0 < x <= max_bytes for x in shape), "conditioning shape")
        require(type(dtype) is str and dtype in sizes and type(offsets) is list
                and len(offsets) == 2 and all(type(x) is int for x in offsets), "conditioning dtype/offset type")
        begin, end = offsets
        require(0 <= begin < end <= data_bytes
                and end - begin == math.prod(shape) * sizes[dtype], "conditioning tensor byte range")
        segments.append((begin, end))
    cursor = 0
    for begin, end in sorted(segments):
        require(begin == cursor, "conditioning tensor gap/overlap")
        cursor = end
    require(cursor == data_bytes, "conditioning trailing data")
    cond, tags = header["conditioning"], header["minimax_token_tags"]
    require(cond["dtype"] in ("F16", "BF16", "F32") and len(cond["shape"]) == 3
            and cond["shape"][0] == 1 and cond["shape"][2] == 5120, "conditioning dimensions/dtype")
    require(tags["dtype"] == "I64" and tags["shape"] == [cond["shape"][1]],
            "conditioning token tags dimensions/dtype")
    if "pooled_output" in header:
        require(header["pooled_output"]["dtype"] in ("F16", "BF16", "F32"), "pooled output dtype")
    after = path.stat()
    signature = lambda st: (st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino)
    require(signature(before) == signature(after), "conditioning changed during header validation")
    return {"file_signature": signature(after), "header_bytes": header_bytes,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "tensors": header,
            "metadata_keys": sorted(metadata), "option_keys": sorted(options),
            "options_sha256": hashlib.sha256(json.dumps(options, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest(),
            "values_checked": False, "roundtrip_checked": False}



def validate_profile_graph(profile, graph, canonical):
    steps = profile.get("steps")
    require(profile.get("family") == "turbo-native" and type(steps) is int
            and steps in {6,8}, "native Turbo family/steps")
    require(profile.get("id") == "turbo-native-" + str(steps)
            and profile.get("role") == "candidate", "Turbo candidate identifier/role")
    expected = copy.deepcopy(canonical)
    expected["prompt"]["9"]["inputs"]["steps"] = steps
    expected["prompt"]["92"]["inputs"]["filename_prefix"] = "q33-speed/" + profile["id"]
    expected["prompt"]["1100"] = {"class_type":"LoraLoaderModelOnly", "inputs":{
        "model":["6",0], "lora_name":TURBO_FILENAME, "strength_model":1.0}}
    expected["prompt"]["148"]["inputs"]["model"] = ["1100",0]
    require(graph == expected, "graph changes outside fixed native Turbo wiring")
    require(type(graph["prompt"]["1100"]["inputs"]["strength_model"]) is float,
            "LoRA strength type")
    require(profile["canonical_graph"]["sha256"] == CANONICAL_SHA["original"],
            "original full50 canonical identity")

def validate_lora_schema(info, graph):
    required = info["input"].get("required", {})
    optional = info["input"].get("optional", {})
    values = graph["prompt"]["1100"]["inputs"]
    require(set(required) <= set(values) <= set(required) | set(optional)
            and info.get("output") == ["MODEL"], "native LoRA schema")
    choices = required["lora_name"][0]
    require(type(choices) is list and all(type(v) is str for v in choices)
            and values["lora_name"] in choices, "Turbo LoRA not enumerated")


def load_profile(a,v1):
    profile=v1.frozen_json(a.profile,a.profile_sha256,65536)
    require(set(profile)=={"schema","id","family","steps","role","dispatch_ready","dispatch_note",
            "model_subdir","graph","canonical_graph","assets","conditioning"}
            and profile["schema"]=="q33-native-turbo-profile/v1","profile schema")
    require(profile["family"] == "turbo-native","profile family")
    require(type(profile["dispatch_ready"]) is bool and profile["dispatch_ready"],
            "profile not dispatch ready; review required")
    root=Path(__file__).resolve().parent
    def descriptor(key):
        d=profile[key]
        require(type(d) is dict and set(d)=={"file","sha256"},"profile descriptor")
        rel=Path(d["file"]);path=root/rel
        require(not rel.is_absolute() and not rel.drive and ".." not in rel.parts
                and path.resolve().is_relative_to(root) and not path.is_symlink(),"profile relative boundary")
        return path,v1.frozen_json(path,d["sha256"],1048576)
    gp,graph=descriptor("graph");_,canonical=descriptor("canonical_graph")
    validate_profile_graph(profile,graph,canonical)
    ap,_=descriptor("assets");cp,_=descriptor("conditioning")
    require(a.asset_manifest==ap.resolve() and a.asset_manifest_sha256==profile["assets"]["sha256"]
            and a.conditioning_manifest==cp.resolve()
            and a.conditioning_manifest_sha256==profile["conditioning"]["sha256"],"profile metadata bindings")
    require(profile["model_subdir"]=="comfy-original/models","profile model root")
    return profile

def load_stage_b_manifest(a, v1, gates):
    manifest=v1.frozen_json(a.conditioning_manifest,a.conditioning_manifest_sha256,65536)
    require(set(manifest)=={"schema","conditioning","graph"} and
            manifest["schema"]=="q33-speed-conditioning/v1","conditioning manifest schema")
    profile=a.loaded_profile;c=manifest["conditioning"];g=manifest["graph"]
    relative=Path(g["relative_path"]);path=a.iteration_root/relative
    root=Path(__file__).resolve().parent
    require(not relative.is_absolute() and not relative.drive and ".." not in relative.parts and
            path.resolve()==root/profile["graph"]["file"],"profile graph path")
    graph=v1.frozen_json(path,g["sha256"],gates["max_schema_bytes"])
    require(g["sha256"]==profile["graph"]["sha256"],"profile graph identity")
    require(c["basename"]=="h3-conditioning-"+c["sha256"]+".safetensors" and
            graph["prompt"]["1001"]["inputs"]["conditioning_name"]==c["basename"],"conditioning content address")
    require(c["sha256"]=="ad5c2652080729c191db4189ab42a42c5f80476370e82a9e2a99fe008cb303f2"
            and c["bytes"]==594360,"fixed verified text conditioning")
    return manifest,graph

def bounded_notes_json(a, v1, descriptor):
    require(type(descriptor) is dict and {"relative_to_notes", "sha256"} <= set(descriptor),
            "complete metadata descriptor required")
    relative = Path(descriptor["relative_to_notes"])
    require(not relative.is_absolute() and not relative.drive and ".." not in relative.parts
            and relative.parts and relative.parts[0] == "iterations", "metadata relative path boundary")
    path = a.notes_root / relative
    st = path.lstat()
    require(not path.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400)
            and path.resolve().is_relative_to(a.notes_root), "metadata source boundary")
    return v1.frozen_json(path, descriptor["sha256"], 65536)


def load_asset_manifest(a, v1, gates):
    """No payload access: pending receipts reject before service/import/large file hashing."""
    manifest = v1.frozen_json(a.asset_manifest, a.asset_manifest_sha256, gates["max_schema_bytes"])
    require(type(manifest) is dict and set(manifest) == {"schema", "assets"}
            and manifest["schema"] in {"q32-original-asset-set/v1","q33-speed-asset-set/v1","q33-native-turbo-asset-set/v1"}, "Q32 asset manifest schema")
    selected = manifest["assets"]
    require(type(selected) is list and len(selected) == len(ASSET_IDS)
            and all(type(x) is dict for x in selected)
            and {x.get("id") for x in selected} == ASSET_IDS, "fixed family asset set required")
    # This first pass ensures an unfinished original download causes zero payload hashes.
    for asset in selected:
        require(set(asset) == {"id", "bytes", "sha256", "relative_path", "receipt", "transport_plan",
                              "receipt_relative_path", "alias_source_relative_path"}, "asset fields")
        require(all(asset[k] == v for k, v in ASSET_EXPECTED[asset["id"]].items()), "fixed asset identity changed")
        require(type(asset["receipt"]) is dict and type(asset["transport_plan"]) is dict,
                "complete receipt and transport plan required; pending download cannot dispatch")
    for asset in selected:
        descriptor = asset["receipt"]
        require(set(descriptor) == {"relative_to_notes", "sha256", "schema"}
                and descriptor["schema"] in {"q31-asset-receipt/v1", "q32-asset-receipt/v1", "q33-asset-receipt/v1"}, "receipt descriptor schema")
        require(set(asset["transport_plan"]) == {"relative_to_notes", "sha256"}, "transport descriptor schema")
        receipt = bounded_notes_json(a, v1, descriptor)
        plan = bounded_notes_json(a, v1, asset["transport_plan"])
        require(type(receipt) is dict and receipt.get("schema") == descriptor["schema"]
                and receipt.get("complete") is True, "asset receipt incomplete")
        if asset["id"] == "turbo-v4-step600-ema":
            require(descriptor["schema"] == "q33-asset-receipt/v1"
                    and asset["transport_plan"]["sha256"] == TURBO_PROFILE_SHA
                    and type(plan) is dict and plan.get("schema") == "q33-turbo-download-profile/v1",
                    "fixed Turbo receipt/profile source")
            plan = {"schema":"q33-turbo-asset-identity/v1", "id":plan.get("id"),
                    "bytes":plan.get("asset_bytes"), "sha256":plan.get("asset_sha256"),
                    "relative_path":plan.get("relative_path")}
        else:
            require(type(plan) is dict and plan.get("schema") in {"q31-asset/v1", "q32-asset/v1"},
                    "transport asset schema")
        require(receipt.get("plan_sha256") == asset["transport_plan"]["sha256"], "receipt plan hash mismatch")
        for key in ("id", "bytes", "sha256"):
            require(receipt.get(key) == asset[key] and plan.get(key) == asset[key], "receipt/plan identity mismatch")
        require(receipt.get("relative_path") == asset["receipt_relative_path"]
                and plan.get("relative_path") == asset["receipt_relative_path"], "receipt path mismatch")
        is_alias = a.loaded_profile["family"] in {"original","turbo-native"} and asset["id"] in {"video-vae","audio-vae"}
        expected_origin = asset["relative_path"].replace("comfy-original/", "comfy-trim/", 1) if is_alias else asset["relative_path"]
        require(asset["receipt_relative_path"] == expected_origin, "unexpected original receipt path")
        require(asset["alias_source_relative_path"] == (expected_origin if is_alias else None),
                "unexpected alias origin")
    return selected


def parse():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--mode", choices=["speed-video-stage-b"], required=True)
    for name in ("iteration-root", "lab-root", "portable-python", "comfy-root",
                 "run-dir", "stop-marker", "nvidia-smi", "inferrsi-src",
                 "gates", "conditioning-manifest", "notes-root", "q31-root",
                 "model-root", "source-manifest", "asset-manifest", "profile", "timing-module"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--profile-sha256", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--asset-manifest-sha256", required=True)
    p.add_argument("--gates-sha256", required=True)
    p.add_argument("--conditioning-manifest-sha256", required=True)
    p.add_argument("--port", type=int, required=True)
    return p.parse_args()



def cleanup_owned_v2(job, proc, observer, state, gates, event):
    """Release owned service first; no journal write/join can delay Job close."""
    cleanup_start_ns=time.perf_counter_ns()
    diagnostics=state.setdefault("cleanup_diagnostics",[])
    if observer is not None:
        try:
            observer.stop.set()  # Event signal only: no join, network or file write.
        except BaseException as exc:
            diagnostics.append("observer_stop_"+type(exc).__name__)
    state["job_close_requested"]=job is not None
    if job is not None:
        try:
            job.close()
        except BaseException as exc:
            diagnostics.append("job_close_"+type(exc).__name__)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.kill()  # Exact existing handle only; no unrelated process lookup.
            proc.wait(timeout=gates["cleanup_seconds"])
            state["owned_root_exited"]=True;state["owned_returncode"]=proc.returncode
        except BaseException as exc:
            state["owned_root_exited"]=False;state["status"]="cleanup_inconclusive"
            diagnostics.append("owned_process_cleanup_"+type(exc).__name__)
    cleanup_end_ns=time.perf_counter_ns()
    try:
        event("owned_service_cleanup_finished",start_perf_counter_ns=cleanup_start_ns,
              end_perf_counter_ns=cleanup_end_ns,owned_root_exited=state.get("owned_root_exited"))
    except BaseException as exc:
        diagnostics.append("cleanup_log_"+type(exc).__name__)
    if observer is not None:
        try:
            state["node_timing"]=observer.finish()
            state["node_timing_relative_to_run"]="timing/summary.json"
        except BaseException as exc:
            state["node_timing"]={"invalid_reasons":["observer_cleanup_"+type(exc).__name__],"nodes":{}}
    # Errors remain evidence even if the exact root process subsequently exited.
    if any(x.startswith("job_close_") for x in diagnostics):
        state["status"]="cleanup_inconclusive"

def main():
    global STAGE_NODES, ASSET_EXPECTED, ASSET_IDS, NativeWSObserver
    entry_perf_ns=time.perf_counter_ns()
    entry_monotonic=time.monotonic()
    a = parse()
    require(os.name == "nt" and 1024 <= a.port <= 65535, "Windows/port precondition")
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    require(os.environ.get("INFERRSI_RESULT_PATH"), "use InferRSI execute_command")
    result_path = Path(os.environ["INFERRSI_RESULT_PATH"]).resolve()
    require(result_path.is_relative_to(a.iteration_root) and result_path.parent.is_dir()
            and not result_path.exists(), "result must be a new iteration trial file")
    for path in (a.lab_root, a.portable_python, a.comfy_root, a.run_dir):
        require(path.drive.upper() == "E:", "runtime/model/run must resolve to E drive")
    require(a.run_dir.parent.is_dir() and not a.run_dir.exists(), "E run directory must be new")
    require(a.stop_marker.parent == a.run_dir and not a.stop_marker.exists(),
            "stop marker must be this new run's direct child")
    require(a.portable_python.is_file() and a.comfy_root.is_dir() and a.nvidia_smi.is_file(),
            "runtime or board-query executable missing")
    require(not (a.comfy_root / "extra_model_paths.yaml").exists(),
            "automatic extra model paths are forbidden")

    require(a.iteration_root == a.notes_root / "iterations/2026-10-31-speed-quality-search"
            and a.q31_root == a.notes_root / "iterations/2026-10-28-comfy-trim-first-video"
            and Path(__file__).resolve().parent == a.iteration_root / "agents/turbo-controller", "Q33/Q31 roots")
    require(a.portable_python == a.comfy_root.parent / "python_embeded/python.exe", "fixed portable relationship")
    sys.dont_write_bytecode = True
    v1_path = a.q31_root / "tools/comfy_first_video_v1.py"
    require(digest(v1_path) == V1_SHA, "v1 dependency drift")
    spec = importlib.util.spec_from_file_location("q32_frozen_q31_stdlib_helper", v1_path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)  # Imports stdlib only; its main is not called.
    require(Path(sys.executable).resolve()==a.portable_python,"use existing portable Python for native aiohttp")
    NativeWSObserver = load_timing_module(a.timing_module, a.iteration_root)
    a.loaded_profile=load_profile(a,v1)
    STAGE_NODES=NODE_FAMILIES[a.loaded_profile["family"]]
    ASSET_EXPECTED=EXPECTED_BY_FAMILY[a.loaded_profile["family"]];ASSET_IDS=set(ASSET_EXPECTED)
    require(a.model_root==a.lab_root/a.loaded_profile["model_subdir"] and a.model_root.drive.upper()=="E:",
            "profile model root boundary")
    require(a.gates_sha256.lower() == STAGE_GATES_SHA, "Stage-B gates identity mismatch")
    gates = v1.frozen_json(a.gates, a.gates_sha256, 65536)
    require(type(gates) is dict and set(gates) == v1.GATE_KEYS | EXTRA_GATES,
            "explicit resource gate schema mismatch")
    for key, val in gates.items():
        require(type(val) in (int, float) and math.isfinite(val)
                and (val >= 0 if key == "cuda_device_index" else val > 0),
                "invalid numeric resource gate")
        if not key.endswith("_seconds") and key != "reserve_vram_gb":
            require(type(val) is int, "byte/count/device gates must be integers")
    require(gates["start_physical_bytes"] >= gates["running_physical_bytes"]
            and gates["start_commit_bytes"] >= gates["running_commit_bytes"]
            and gates["start_gpu_free_bytes"] >= gates["running_gpu_free_bytes"],
            "startup floors below running floors")
    require(gates["max_schema_bytes"] <= 1024 * 1024
            and gates["max_total_http_bytes"] >= gates["max_schema_bytes"]
            and gates["max_owned_processes"] <= 256,
            "bounded metadata/process limits required")
    require(gates["http_timeout_seconds"] <= gates["poll_seconds"] < gates["timeout_seconds"],
            "invalid polling/deadline relationship")
    manifest, graph = load_stage_b_manifest(a, v1, gates)
    graph_sha = manifest["graph"]["sha256"]
    conditioning = manifest["conditioning"]
    selected = load_asset_manifest(a, v1, gates)  # Null receipt stops here before runtime/monitor/service.
    require(a.source_manifest_sha256.lower() == SOURCE_MANIFEST_SHA, "Q32 source manifest identity")
    source_manifest = v1.frozen_json(a.source_manifest, SOURCE_MANIFEST_SHA, gates["max_schema_bytes"])
    require(source_manifest["schema"] == "q33-speed-runtime-sources/v1", "Q32 source schema")
    source_roots = {"portable": a.comfy_root.parent, "inferrsi": a.inferrsi_src.parent}
    source_start_ns=time.perf_counter_ns()
    for entry in source_manifest["sources"]:
        root = source_roots[entry["root"]]
        path = (root / entry["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file() and path.stat().st_size == entry["bytes"]
                and digest(path) == entry["sha256"], "Q32 frozen runtime source drift")
    source_end_ns=time.perf_counter_ns()
    require({n["class_type"] for n in graph["prompt"].values()} == set(STAGE_NODES),
            "fixed graph node set mismatch")

    sys.path.insert(0, str(a.inferrsi_src))
    from inferrsi.infra import processes
    require(Path(processes.__file__).resolve() == a.inferrsi_src / "inferrsi/infra/processes.py",
            "InferRSI import source mismatch")
    monitor = v1.WinMonitor()
    started = entry_monotonic
    deadline = started + gates["timeout_seconds"]
    proc = job = raw_log = samples = journal = observer = None
    next_sample = next_gpu = 0.0
    gpu = None
    state = {"status": "preflight", "claim": "INCONCLUSIVE", "mode": "speed-video-stage-b",
             "profile_id": a.loaded_profile["id"], "profile_sha256": a.profile_sha256,
             "controller_entry_perf_counter_ns":entry_perf_ns,
             "source_verification_start_ns":source_start_ns,"source_verification_end_ns":source_end_ns,
             "driver_sha256": digest(Path(__file__)), "v1_sha256": V1_SHA,
             "timing_source_relative_to_iteration": str(a.timing_module.relative_to(a.iteration_root)),
             "timing_source_sha256": digest(a.timing_module), "turbo_mode": "native_full_merge",
             "turbo_quality": "INCONCLUSIVE",
             "graph_sha256": graph_sha, "asset_set_sha256": a.asset_manifest_sha256.lower(),
             "source_manifest_sha256": SOURCE_MANIFEST_SHA,
             "conditioning_manifest_sha256": a.conditioning_manifest_sha256.lower(),
             "gates_sha256": a.gates_sha256.lower(), "gates": gates,
             "prompt_posts": 0, "http_bytes": 0, "history_polls": 0,
             "verified_assets": [], "gpu_measurement": "nvidia-smi whole-board MiB",
             "cuda_allocator_metrics": None, "media_validation": "not_performed"}
    asset_stats = {}

    def event(kind, **fields):
        if journal is not None:
            journal.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                                      "event": kind, "perf_counter_ns":time.perf_counter_ns(), **fields}) + "\n")
            journal.flush()

    def sample(force=False, startup=False):
        nonlocal next_sample, next_gpu, gpu
        now = time.monotonic()
        require(not a.stop_marker.exists(), "parent stop marker")
        require(now < deadline, "whole-run deadline")
        if not force and now < next_sample:
            return
        memory = monitor.memory()
        owned = monitor.owned(job, gates["max_owned_processes"]) if job is not None else []
        row = {"elapsed_seconds": now - started, **memory, "owned_processes": owned,
               "owned_rss_sum_bytes": sum(p["rss_bytes"] for p in owned),
               "owned_private_sum_bytes": sum(p["private_bytes"] for p in owned),
               "e_free_bytes": shutil.disk_usage(a.run_dir.parent).free}
        state["last_sample"] = row
        floor = "start_" if startup else "running_"
        require(memory["available_physical_bytes"] >= gates[floor + "physical_bytes"], "physical memory guard")
        require(memory["available_commit_bytes"] >= gates[floor + "commit_bytes"], "commit guard")
        require(row["owned_rss_sum_bytes"] <= gates["max_owned_rss_bytes"], "owned RSS guard")
        require(row["owned_private_sum_bytes"] <= gates["max_owned_private_bytes"], "owned private guard")
        require(row["e_free_bytes"] >= gates["min_free_disk_bytes"], "E free-disk guard")
        if force or now >= next_gpu:
            query = subprocess.run(
                [str(a.nvidia_smi), "--id=" + str(gates["cuda_device_index"]),
                 "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=min(gates["gpu_query_timeout_seconds"], max(0.001, deadline - now)),
                creationflags=subprocess.CREATE_NO_WINDOW, check=True)
            require(len(query.stdout) <= 4096, "board query output bound")
            rows = list(csv.reader(io.StringIO(query.stdout.decode("ascii").strip())))
            require(len(rows) == 1 and len(rows[0]) == 3, "board query schema")
            values = [int(x.strip()) * 1024 * 1024 for x in rows[0]]
            require(all(v >= 0 for v in values) and values[0] > 0, "board query values")
            gpu = dict(zip(("total_bytes", "used_bytes", "free_bytes"), values))
            gpu["sample_elapsed_seconds"] = time.monotonic() - started
            next_gpu = time.monotonic() + gates["gpu_poll_seconds"]
        row["gpu_board"] = gpu
        state["last_sample"] = row
        for key in ("owned_rss_sum_bytes", "owned_private_sum_bytes"):
            state["peak_" + key] = max(state.get("peak_" + key, 0), row[key])
        if samples is not None:
            samples.write(json.dumps(row) + "\n"); samples.flush()
        require(gpu["free_bytes"] >= gates[floor + "gpu_free_bytes"], "board free-VRAM guard")
        next_sample = time.monotonic() + gates["poll_seconds"]

    def guarded_hash(path, phase_deadline):
        h = hashlib.sha256()
        before = path.stat()
        with path.open("rb") as f:
            while True:
                sample()
                require(time.monotonic() < phase_deadline, "file verification deadline")
                block = f.read(4 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
        after = path.stat()
        signature = lambda s: (s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino)
        require(signature(before) == signature(after), "file changed during verification")
        return h.hexdigest(), signature(after)

    def verify_conditioning(path, phase_deadline):
        structure = conditioning_header(path, v1.unique_json, 32 * 1024 * 1024)
        for key in ("tensors", "metadata_keys", "option_keys", "options_sha256"):
            require(structure[key] == conditioning[key], "conditioning structure/options identity mismatch")
        require(path.stat().st_size == conditioning["bytes"], "conditioning size identity mismatch")
        value, signature = guarded_hash(path, phase_deadline)
        require(value == conditioning["sha256"] and signature == structure.pop("file_signature"),
                "conditioning full hash or stable identity mismatch")
        return {"bytes": conditioning["bytes"], "sha256": value, "structure": structure,
                "finite_tensor_values_checked": False}

    def http(method, route, body=None):
        require(proc is not None and proc.poll() is None, "owned Comfy exited")
        require(monitor.owns_listener(proc.pid, a.port), "owned loopback listener not ready")
        allowed_get = {"/object_info/" + name for name in STAGE_NODES} | {
            "/history/" + state["prompt_id"]}
        require((method == "GET" and route in allowed_get) or
                (method == "POST" and route == "/prompt" and state["prompt_posts"] == 1),
                "HTTP operation outside fixed task")
        connection = http_client.HTTPConnection("127.0.0.1", a.port,
                                                 timeout=gates["http_timeout_seconds"])
        try:
            headers = {"Connection": "close"}
            if body is not None:
                require(len(body) <= gates["max_schema_bytes"], "request body bound")
                headers["Content-Type"] = "application/json"
            connection.request(method, route, body=body, headers=headers)
            response = connection.getresponse()
            require(response.getheader("Content-Type", "").split(";")[0] == "application/json",
                    "non-JSON local response")
            parts, count = [], 0
            while not response.isclosed():
                sample()
                chunk = response.read(min(16384, gates["max_schema_bytes"] - count + 1))
                count += len(chunk); state["http_bytes"] += len(chunk)
                require(count <= gates["max_schema_bytes"]
                        and state["http_bytes"] <= gates["max_total_http_bytes"], "HTTP response byte guard")
                if not chunk:
                    break
                parts.append(chunk)
            return response.status, json.loads(b"".join(parts), object_pairs_hook=v1.unique_json)
        finally:
            connection.close()

    try:
        sample(force=True, startup=True)
        for relative, expected in STAGE_SOURCE_SHA256.items():
            require(digest(a.comfy_root / relative) == expected, "Stage-B native source drift")
        state["stage_source_sha256"] = STAGE_SOURCE_SHA256
        # Include server history/UI behavior inspected for this controller.
        state["additional_source_sha256"] = {
            name: digest(a.comfy_root / name)
            for name in ("execution.py", "comfy_api/latest/_ui.py")
        }
        a.run_dir.mkdir()
        samples = (result_path.parent / "resources.jsonl").open("x", encoding="utf-8")
        journal = (result_path.parent / "events.jsonl").open("x", encoding="utf-8")
        event("preflight",source_verification_start_ns=source_start_ns,source_verification_end_ns=source_end_ns,
              source_verification_wall_seconds=(source_end_ns-source_start_ns)/1e9)
        models_root = a.model_root
        require(models_root.drive.upper() == "E:" and models_root.is_dir(), "model root unavailable")
        verify_deadline = min(deadline, time.monotonic() + gates["asset_timeout_seconds"])
        for asset in selected:
            path = (a.lab_root / asset["relative_path"]).resolve()
            require(path.drive.upper() == "E:" and path.is_relative_to(models_root)
                    and path.is_file() and path.stat().st_size == asset["bytes"], "asset file/path/size mismatch")
            if asset["alias_source_relative_path"] is not None:
                origin = (a.lab_root / asset["alias_source_relative_path"]).resolve()
                require(origin.is_relative_to(a.lab_root / "comfy-trim/models") and origin.is_file()
                        and path.samefile(origin), "VAE alias must preserve original receipt file identity")
            event("asset_verification_started",asset_id=asset["id"])
            value, signature = guarded_hash(path, verify_deadline)
            require(value == asset["sha256"], "asset full-file SHA256 mismatch")
            asset_stats[path] = signature
            state["verified_assets"].append({"id": asset["id"], "bytes": asset["bytes"],
                                             "sha256": value, "receipt_sha256": asset["receipt"]["sha256"]})
            event("asset_verified", asset_id=asset["id"])
        embeddings = models_root / "embeddings"
        conditioning_path = embeddings / conditioning["basename"]
        for path in (embeddings, conditioning_path):
            st = path.lstat()
            require(not path.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400),
                    "conditioning input link/reparse boundary")
        require(embeddings.is_dir() and embeddings.resolve() == embeddings
                and conditioning_path.is_file() and conditioning_path.resolve().parent == embeddings,
                "conditioning input native embeddings boundary")
        event("conditioning_verification_started")
        state["conditioning_input"] = verify_conditioning(conditioning_path, verify_deadline)
        state["conditioning_input"]["relative_to_models"] = str(conditioning_path.relative_to(models_root))
        event("conditioning_input_verified", artifact=state["conditioning_input"])
        sample(force=True, startup=True)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            reservation.bind(("127.0.0.1", a.port))
        dirs = {k: a.run_dir / k for k in ("base", "output", "temp", "input", "previews", "user", "cache")}
        for path in dirs.values():
            path.mkdir()
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        for key in ("TEMP", "TMP", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
                    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                    "PYTORCH_KERNEL_CACHE_PATH"):
            path = dirs["cache"] / key.lower(); path.mkdir(); env[key] = str(path)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   DO_NOT_TRACK="1", PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES=str(gates["cuda_device_index"]))
        command = [str(a.portable_python), "-s", "-B", str(a.comfy_root / "main.py"),
                   "--listen", "127.0.0.1", "--port", str(a.port),
                   "--disable-auto-launch", "--disable-all-custom-nodes", "--disable-partner-nodes",
                   "--offline", "--disable-assets", "--disable-pinned-memory", "--disable-async-offload", "--disable-smart-memory", "--cache-none", "--preview-method", "none",
                   "--reserve-vram", str(gates["reserve_vram_gb"]),
                   "--models-directory", str(models_root)]
        for key in ("base", "output", "temp", "input", "previews", "user"):
            command += ["--" + key + "-directory", str(dirs[key])]
        raw_log = (a.run_dir / "comfy.raw.log").open("xb")
        job = processes._WindowsJob()
        state["prompt_id"] = str(uuid.uuid4())  # Known before POST; never retry POST on unknown outcome.
        startup_deadline = min(deadline, time.monotonic() + gates["startup_timeout_seconds"])
        event("service_spawn_started")
        proc = subprocess.Popen(command, cwd=a.comfy_root, env=env, stdin=subprocess.DEVNULL,
                                stdout=raw_log, stderr=subprocess.STDOUT, shell=False,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        job.attach(proc)
        state["owned_pid"] = proc.pid; state["status"] = "starting_service"
        event("service_started", pid=proc.pid)
        while not monitor.owns_listener(proc.pid, a.port):
            sample()
            require(proc.poll() is None, "Comfy exited during startup")
            require(time.monotonic() < startup_deadline, "service startup deadline")
            time.sleep(min(gates["poll_seconds"], max(0, startup_deadline - time.monotonic())))
        state["dynamic_vram_startup"] = verify_dynamic_vram_startup(a.run_dir / "comfy.raw.log")
        event("dynamic_vram_confirmed", **state["dynamic_vram_startup"])
        event("native_schema_verification_started")
        for name, module in STAGE_NODES.items():
            status, info = http("GET", "/object_info/" + name)
            v1.save_new(a.run_dir / (name + ".schema.raw.json"), info)
            require(status == 200 and type(info) is dict and set(info) == {name}
                    and info[name].get("python_module") == module, "Stage-B node source/schema mismatch")
            if name in ("ConditioningLoader", "EmptyMiniMaxH3LatentAV"):
                required = info[name]["input"].get("required", {})
                optional = info[name]["input"].get("optional", {})
                node_id = "1001" if name == "ConditioningLoader" else "1002"
                require(set(required) <= set(graph["prompt"][node_id]["inputs"])
                        and set(graph["prompt"][node_id]["inputs"]) <= set(required) | set(optional),
                        "Stage-B new node input schema mismatch")
                expected_output = ["CONDITIONING"] if name == "ConditioningLoader" else ["LATENT"]
                require(info[name].get("output") == expected_output, "Stage-B new node output schema mismatch")
            if name == "LoraLoaderModelOnly":
                validate_lora_schema(info[name], graph)
            if name in ("UNETLoader", "VAELoader", "ConditioningLoader"):
                field, ids = {"UNETLoader": ("unet_name", ["6"]),
                              "VAELoader": ("vae_name", ["11", "24"]),
                              "ConditioningLoader": ("conditioning_name", ["1001"])}[name]
                if name == "ConditioningLoader":
                    combo = info[name]["input"]["required"][field]
                    require(type(combo) is list and len(combo) == 2 and combo[0] == "COMBO"
                            and type(combo[1]) is dict and combo[1].get("multiselect") is False,
                            "native conditioning COMBO schema mismatch")
                    choices = combo[1].get("options")
                    require(type(choices) is list and all(type(x) is str for x in choices),
                            "native conditioning options must be a string list")
                else:
                    choices = info[name]["input"]["required"][field][0]
                require(all(graph["prompt"][i]["inputs"][field] in choices for i in ids),
                        "model/conditioning file not enumerated")
        event("native_schema_verification_finished")
        require(digest(a.profile)==a.profile_sha256,"profile changed before POST")
        require(digest(a.asset_manifest) == a.asset_manifest_sha256.lower(), "asset manifest changed before POST")
        require(load_asset_manifest(a, v1, gates) == selected, "asset receipt identity changed before POST")
        for path, signature in asset_stats.items():
            st = path.stat()
            require((st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino) == signature, "asset changed before POST")
        state["conditioning_pre_post"] = verify_conditioning(
            conditioning_path, min(deadline, time.monotonic() + gates["asset_timeout_seconds"]))
        require(digest(a.conditioning_manifest) == a.conditioning_manifest_sha256.lower(),
                "conditioning manifest changed before POST")
        graph_path = a.iteration_root / manifest["graph"]["relative_path"]
        require(digest(graph_path) == graph_sha, "Stage-B graph changed before POST")
        sample(force=True, startup=True)
        render_deadline = min(deadline, time.monotonic() + gates["render_timeout_seconds"])
        state["client_id"]=uuid.uuid4().hex
        observer=NativeWSObserver(a.port,state["prompt_id"],state["client_id"],
            {k:n["class_type"] for k,n in graph["prompt"].items()},a.run_dir/"timing",
            a.comfy_root.parent,started)
        require(monitor.owns_listener(proc.pid,a.port),"observer must bind owned listener")
        event("observer_connect_started");observer.start()
        observer_deadline=min(deadline,time.monotonic()+12)
        while not observer.ready.wait(timeout=0.05):
            sample()
            require(not observer.done.is_set(),"observer failed before POST")
            require(time.monotonic()<observer_deadline,"observer readiness deadline")
        require(not observer.done.is_set() and observer.error is None,"observer ready but exited")
        event("observer_ready",client_id=state["client_id"])
        body = json.dumps({"prompt": graph["prompt"], "prompt_id": state["prompt_id"],
                           "client_id":state["client_id"]}).encode("utf-8")
        state["prompt_posts"] = 1; state["status"] = "submitting"
        event("post_attempt", prompt_id=state["prompt_id"], graph_sha256=graph_sha)
        try:
            status, answer = http("POST", "/prompt", body)
            v1.save_new(a.run_dir / "post.response.raw.json", answer)
            require(status == 200 and answer.get("prompt_id") == state["prompt_id"]
                    and not answer.get("node_errors"), "server rejected or changed fixed prompt")
            state["submission_outcome"] = "accepted"
        except (OSError, http_client.HTTPException) as exc:
            state["submission_outcome"] = "unknown"
            event("post_transport_outcome_unknown", error_type=type(exc).__name__)
        state["status"] = "waiting_history"
        while True:
            sample()
            require(proc.poll() is None, "Comfy exited before history completion")
            require(time.monotonic() < render_deadline, "render deadline")
            require(state["history_polls"] < gates["max_history_polls"], "history poll bound")
            state["history_polls"] += 1
            try:
                status, history = http("GET", "/history/" + state["prompt_id"])
                require(status == 200, "history HTTP status")
            except (OSError, http_client.HTTPException) as exc:
                event("history_transport_error", error_type=type(exc).__name__)
                history = {}
            if history:
                v1.save_new(a.run_dir / "history.raw.json", history)
                require(set(history) == {state["prompt_id"]}, "unexpected history prompt identity")
                record = history[state["prompt_id"]]
                require(record.get("status", {}).get("status_str") == "success"
                        and record["status"].get("completed") is True, "prompt execution failed")
                observed = record["prompt"][2]
                require(set(observed) == set(graph["prompt"]) and all(
                    observed[k]["class_type"] == v["class_type"] and observed[k]["inputs"] == v["inputs"]
                    for k, v in graph["prompt"].items()), "executed graph changed")
                videos = record["outputs"]["92"]["images"]
                require(len(videos) == 1 and videos[0].get("type") == "output", "single MP4 artifact not returned")
                item = videos[0]
                relative = Path(item["subfolder"]) / item["filename"]
                require(not relative.is_absolute() and not relative.drive, "absolute output path forbidden")
                output = (dirs["output"] / relative).resolve()
                require(output.is_relative_to(dirs["output"]) and output.suffix.lower() == ".mp4"
                        and output.is_file() and 0 < output.stat().st_size <= gates["max_output_bytes"],
                        "output path/size guard")
                output_hash, _ = guarded_hash(output, render_deadline)
                state["artifact"] = {"relative_to_run": str(output.relative_to(a.run_dir)),
                                     "bytes": output.stat().st_size, "sha256": output_hash}
                state["status"] = "artifact_received_unvalidated"
                event("artifact_received", artifact=state["artifact"])
                break
            time.sleep(min(gates["poll_seconds"], max(0, render_deadline - time.monotonic())))
    except BaseException as exc:
        state["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) or a.stop_marker.exists() else "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc) if type(exc) is RuntimeError else "external operation failed; local raw evidence retained"
        event("stopped", status=state["status"], error_type=type(exc).__name__)
    finally:
        cleanup_owned_v2(job,proc,observer,state,gates,event)
        state["elapsed_seconds"] = time.monotonic() - started
        for stream in (samples, journal, raw_log):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    state.setdefault("cleanup_diagnostics",[]).append("stream_close_"+type(exc).__name__)
        v1.save_new(result_path, state)
    print(json.dumps({"status": state["status"], "claim": state["claim"],
                      "prompt_posts": state["prompt_posts"], "media_validation": "not_performed"}))
    return 0 if state["status"] == "artifact_received_unvalidated" else 1


if __name__ == "__main__":
    raise SystemExit(main())