"""Q33 one-service bounce then hands real TE -> release -> Trim8 with native timing.
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
STAGE_GATES_SHA = "13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd"
STAGE_NODES = {'UNETLoader': 'nodes', 'VAELoader': 'nodes', 'VAEDecode': 'nodes', 'MiniMaxH3SigmaShift': 'comfy_extras.nodes_minimax_h3', 'RandomNoise': 'comfy_extras.nodes_custom_sampler', 'KSamplerSelect': 'comfy_extras.nodes_custom_sampler', 'BasicScheduler': 'comfy_extras.nodes_custom_sampler', 'BasicGuider': 'comfy_extras.nodes_custom_sampler', 'SamplerCustomAdvanced': 'comfy_extras.nodes_custom_sampler', 'VAEDecodeAudio': 'comfy_extras.nodes_audio', 'CreateVideo': 'comfy_extras.nodes_video', 'SaveVideo': 'comfy_extras.nodes_video', 'ConditioningLoader': 'comfy_extras.nodes_cond', 'EmptyMiniMaxH3LatentAV': 'comfy_extras.nodes_minimax_h3'}
NODE_FAMILIES = {'original': {'UNETLoader': 'nodes', 'VAELoader': 'nodes', 'VAEDecode': 'nodes', 'MiniMaxH3SigmaShift': 'comfy_extras.nodes_minimax_h3', 'RandomNoise': 'comfy_extras.nodes_custom_sampler', 'KSamplerSelect': 'comfy_extras.nodes_custom_sampler', 'BasicScheduler': 'comfy_extras.nodes_custom_sampler', 'BasicGuider': 'comfy_extras.nodes_custom_sampler', 'SamplerCustomAdvanced': 'comfy_extras.nodes_custom_sampler', 'VAEDecodeAudio': 'comfy_extras.nodes_audio', 'CreateVideo': 'comfy_extras.nodes_video', 'SaveVideo': 'comfy_extras.nodes_video', 'ConditioningLoader': 'comfy_extras.nodes_cond', 'EmptyMiniMaxH3LatentAV': 'comfy_extras.nodes_minimax_h3'}, 'trim': {'UNETLoader': 'nodes', 'VAELoader': 'nodes', 'VAEDecode': 'nodes', 'MiniMaxH3SigmaShift': 'comfy_extras.nodes_minimax_h3', 'RandomNoise': 'comfy_extras.nodes_custom_sampler', 'KSamplerSelect': 'comfy_extras.nodes_custom_sampler', 'BasicScheduler': 'comfy_extras.nodes_custom_sampler', 'BasicGuider': 'comfy_extras.nodes_custom_sampler', 'SamplerCustomAdvanced': 'comfy_extras.nodes_custom_sampler', 'VAEDecodeAudio': 'comfy_extras.nodes_audio', 'CreateVideo': 'comfy_extras.nodes_video', 'SaveVideo': 'comfy_extras.nodes_video', 'ConditioningLoader': 'comfy_extras.nodes_cond', 'EmptyMiniMaxH3LatentAV': 'comfy_extras.nodes_minimax_h3', 'ModelAttentionBackend': 'comfy_extras.nodes_model_advanced', 'BlockSparseAttention': 'comfy_extras.nodes_sparse_attention'}}
STAGE_SOURCE_SHA256 = {'comfy_extras/nodes_cond.py': '56344e64a20692a0f64521bf4de78c0a9c040e23c3499a00491b5243ca39f898', 'comfy_extras/nodes_minimax_h3.py': '5999ed8c22b13c8251084c0629fd82137d2fb52b55106abfc73bf5cafc73841d', 'comfy_api/latest/_io.py': 'a16cea6dd228468b5fbe3fa2acb5dd86bf9445babcd9d8bfe697b84c449bf5fb', 'folder_paths.py': '045a2ea987c0c7bcb0e47cbc46826ee34e82179562c85123f35ea3ff1c3e7ac4'}
SOURCE_MANIFEST_SHA = "bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007"
TIMING_MODULE_SHA = "f0abb1937ca5462d712a73d1a8623b6a9f5df86927dbcf6375903b8a5ad3e614"
EXPECTED_BY_FAMILY = {'original': {'original-dit': {'id': 'original-dit', 'bytes': 34038892334, 'sha256': '7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5', 'relative_path': 'comfy-original/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors'}, 'video-vae': {'id': 'video-vae', 'bytes': 2811065184, 'sha256': '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e', 'relative_path': 'comfy-original/models/vae/minimax_h3_video_vae_int8_convrot.safetensors'}, 'audio-vae': {'id': 'audio-vae', 'bytes': 605254808, 'sha256': '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48', 'relative_path': 'comfy-original/models/vae/minimax_h3_audio_vae_fp32.safetensors'}}, 'trim': {'trim-dit': {'id': 'trim-dit', 'bytes': 11894623166, 'sha256': '748b2bcc9c4b7bf7b23406987b42d4ac40362070c164cf11092482be6a863455', 'relative_path': 'comfy-trim/models/diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors'}, 'video-vae': {'id': 'video-vae', 'bytes': 2811065184, 'sha256': '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e', 'relative_path': 'comfy-trim/models/vae/minimax_h3_video_vae_int8_convrot.safetensors'}, 'audio-vae': {'id': 'audio-vae', 'bytes': 605254808, 'sha256': '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48', 'relative_path': 'comfy-trim/models/vae/minimax_h3_audio_vae_fp32.safetensors'}}}
CANONICAL_SHA = {'original': 'd5937f2505f74e07ce8de746c49c98e8a428535d79ccc878383e8b989807fb29', 'trim': 'a915f62f109cd2e1be615eb7b0c1aa6fab4a4b6ad6e31791fb84488b02a17ad8'}
ASSET_EXPECTED = EXPECTED_BY_FAMILY["original"]
ASSET_IDS = set(ASSET_EXPECTED)

# Portable Python isolates script-directory imports; load only the frozen sibling.
_timing_path = Path(__file__).resolve().parents[1] / "timing-controller/timing_events_v1.py"
if hashlib.sha256(_timing_path.read_bytes()).hexdigest() != TIMING_MODULE_SHA:
    raise RuntimeError("frozen timing sibling hash mismatch")
_timing_spec = importlib.util.spec_from_file_location("q33_frozen_native_timing_v1", _timing_path)
_timing_module = importlib.util.module_from_spec(_timing_spec)
_timing_spec.loader.exec_module(_timing_module)
if Path(_timing_module.__file__).resolve() != _timing_path.resolve():
    raise RuntimeError("actual timing import source")
NativeWSObserver = _timing_module.NativeWSObserver
E2E_ASSETS_SHA = "fcaa021e7aa7ecc50ecd14f51e81448a758dc6c8352b65d5ea7aa657d1fe31a8"
STAGE_A_GRAPHS = {'bounce-audio': {'file': 'STAGE-A-bounce-audio-v3.json', 'sha256': 'e94c58afb0181f3a79f85c76b6e2d6441ae4ba49809bd30f40081f5f41da2caf'}, 'hands-face-mug': {'file': 'STAGE-A-hands-face-mug-v3.json', 'sha256': '41d4fc0abcf22f10cf79afcbd439e769f7a5edaefb82376523056827dda4c2fb'}}
E2E_HELPER_SHA = "da1f2273e93a18ba78e58f2ccae1f4cc52559e89a9285aeb91110e0f15aaf4d0"
_e2e_path = Path(__file__).resolve().parent / "e2e_helpers_v3.py"
if hashlib.sha256(_e2e_path.read_bytes()).hexdigest() != E2E_HELPER_SHA: raise RuntimeError("frozen lifecycle helper drift")
_e2e_spec = importlib.util.spec_from_file_location("q33_e2e_helpers_v1", _e2e_path)
e2e = importlib.util.module_from_spec(_e2e_spec); _e2e_spec.loader.exec_module(e2e)
if Path(e2e.__file__).resolve() != _e2e_path.resolve(): raise RuntimeError("actual lifecycle helper source")
STAGE_A_NODES = {'CLIPLoader':'nodes','VAELoader':'nodes','MiniMaxH3ImageToVideo':'comfy_extras.nodes_minimax_h3','SaveConditioning':'comfy_extras.nodes_cond'}
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
    require(b"Using async weight offloading with 2 streams" in raw, "native two-stream async marker missing")
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
    family=profile.get("family");steps=profile.get("steps")
    require(family in {"original","trim"} and type(steps) is int and
            steps in ({"original":{50,30,20},"trim":{8,6,4}}[family]),"profile family/steps")
    require(profile.get("id")==family+"-"+str(steps),"profile identifier")
    require(profile.get("role")==("reference" if family=="original" and steps==50 else "candidate"),
            "only original50 is reference")
    expected=copy.deepcopy(canonical)
    expected["prompt"]["9"]["inputs"]["steps"]=steps
    expected["prompt"]["92"]["inputs"]["filename_prefix"]="q33-speed/"+profile["id"]
    require(graph==expected,"graph changes outside frozen family steps/output prefix")
    require(profile["canonical_graph"]["sha256"]==CANONICAL_SHA[family],"canonical family identity")

def load_profile(a,v1):
    profile=v1.frozen_json(a.profile,a.profile_sha256,65536)
    require(set(profile)=={"schema","id","family","steps","role","dispatch_ready","dispatch_note",
            "model_subdir","graph","canonical_graph","assets","conditioning"}
            and profile["schema"]=="q33-speed-profile/v1","profile schema")
    require(profile["family"] in EXPECTED_BY_FAMILY,"profile family")
    require(type(profile["dispatch_ready"]) is bool and profile["dispatch_ready"],
            "profile not dispatch ready; review required")
    root=a.iteration_root / "agents/timing-controller"
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
    require(a.asset_manifest==Path(__file__).resolve().parent / "ASSETS-v1.json" and a.asset_manifest_sha256==E2E_ASSETS_SHA
            and a.conditioning_manifest==cp.resolve()
            and a.conditioning_manifest_sha256==profile["conditioning"]["sha256"],"profile metadata bindings")
    require(profile["model_subdir"]=="comfy-"+profile["family"]+"/models","profile model root")
    return profile

def load_stage_b_manifest(a, v1, gates):
    manifest=v1.frozen_json(a.conditioning_manifest,a.conditioning_manifest_sha256,65536)
    require(set(manifest)=={"schema","conditioning","graph"} and
            manifest["schema"]=="q33-speed-conditioning/v1","conditioning manifest schema")
    profile=a.loaded_profile;c=manifest["conditioning"];g=manifest["graph"]
    relative=Path(g["relative_path"]);path=a.iteration_root/relative
    root=a.iteration_root / "agents/timing-controller"
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
            and manifest["schema"] in {"q32-original-asset-set/v1","q33-speed-asset-set/v1"}, "Q32 asset manifest schema")
    selected = manifest["assets"]
    require(type(selected) is list and len(selected) == 4
            and all(type(x) is dict for x in selected)
            and {x.get("id") for x in selected} == ASSET_IDS, "four fixed new-prompt assets required")
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
                and descriptor["schema"] in {"q31-asset-receipt/v1", "q32-asset-receipt/v1"}, "receipt descriptor schema")
        require(set(asset["transport_plan"]) == {"relative_to_notes", "sha256"}, "transport descriptor schema")
        receipt = bounded_notes_json(a, v1, descriptor)
        plan = bounded_notes_json(a, v1, asset["transport_plan"])
        require(type(receipt) is dict and receipt.get("schema") == descriptor["schema"]
                and receipt.get("complete") is True, "asset receipt incomplete")
        require(type(plan) is dict and plan.get("schema") in {"q31-asset/v1", "q32-asset/v1"},
                "transport asset schema")
        require(receipt.get("plan_sha256") == asset["transport_plan"]["sha256"], "receipt plan hash mismatch")
        for key in ("id", "bytes", "sha256"):
            require(receipt.get(key) == asset[key] and plan.get(key) == asset[key], "receipt/plan identity mismatch")
        require(receipt.get("relative_path") == asset["receipt_relative_path"]
                and plan.get("relative_path") == asset["receipt_relative_path"], "receipt path mismatch")
        is_alias = a.loaded_profile["family"] == "original" and asset["id"] != "original-dit"
        expected_origin = asset["relative_path"].replace("comfy-original/", "comfy-trim/", 1) if is_alias else asset["relative_path"]
        require(asset["receipt_relative_path"] == expected_origin, "unexpected original receipt path")
        require(asset["alias_source_relative_path"] == (expected_origin if is_alias else None),
                "unexpected alias origin")
    return selected


def parse():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--mode", choices=["two-new-prompts-trim8"], required=True)
    for name in ("iteration-root", "lab-root", "portable-python", "comfy-root",
                 "run-dir", "stop-marker", "nvidia-smi", "inferrsi-src",
                 "gates", "conditioning-manifest", "notes-root", "q31-root",
                 "model-root", "source-manifest", "asset-manifest", "profile"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--profile-sha256", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--asset-manifest-sha256", required=True)
    p.add_argument("--gates-sha256", required=True)
    p.add_argument("--conditioning-manifest-sha256", required=True)
    p.add_argument("--memory-policy", choices=["baseline"], default="baseline")
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

def memory_command(command, policy):
    require(policy in {"baseline", "smart"}, "memory policy")
    require(command.count("--disable-smart-memory") == 1 and command.count("--cache-none") == 1, "fixed memory/cache baseline")
    return list(command) if policy == "baseline" else [x for x in command if x != "--disable-smart-memory"]


def begin_request(state, request_index, prompt_id):
    require(type(request_index) is int and request_index in (1, 2, 3, 4), "exact four requests")
    require(len(state["requests"]) == request_index - 1 and state["prompt_posts"] == request_index - 1, "request/POST sequence")
    require(not state["requests"] or state["requests"][-1]["status"] == "complete", "previous request incomplete")
    require(type(prompt_id) is str and prompt_id and all(x["prompt_id"] != prompt_id for x in state["requests"]), "unique prompt identity")
    state["prompt_id"] = prompt_id
    state["current_request_index"] = request_index
    current = {"request_index": request_index, "prompt_id": prompt_id, "status": "preparing", "post_attempted": False}
    state["requests"].append(current)
    return current


def begin_post(state):
    current = state["requests"][-1]
    require(current["post_attempted"] is False and state["prompt_posts"] == current["request_index"] - 1, "POST attempted once")
    require(current["request_index"] in (1, 2, 3, 4), "POST total bound")
    current["post_attempted"] = True
    state["prompt_posts"] += 1


def request_metrics(timing, record, prompt_id):
    require(timing.get("prompt_id") == prompt_id and timing.get("ready") is True
            and timing.get("terminal") == "execution_success" and not timing.get("invalid_reasons")
            and timing.get("error_type") is None and timing.get("observer_thread_exited") is True, "request timing incomplete")
    is_text="104" in record["prompt"][2]
    sampler = timing.get("nodes", {}).get("104" if is_text else "14", {})
    require(sampler.get("cached") is False,"request node must be uncached")
    if not is_text:require(sampler.get("progress_value")==8 and sampler.get("progress_max")==8,"request must execute eight uncached steps")
    wall = sampler.get("observed_wall_seconds")
    require(type(wall) in (float, int) and math.isfinite(wall) and wall > 0, "sampler timing missing")
    endpoints = {}
    for kind, body in record["status"]["messages"]:
        if kind in {"execution_start", "execution_success"}:
            require(kind not in endpoints and body.get("prompt_id") == prompt_id
                    and type(body.get("timestamp")) is int, "native timestamp identity")
            endpoints[kind] = body["timestamp"]
    require(set(endpoints) == {"execution_start", "execution_success"}
            and endpoints["execution_success"] > endpoints["execution_start"], "native prompt interval missing")
    return {"native_prompt_wall_seconds": (endpoints["execution_success"] - endpoints["execution_start"]) / 1000,
            ("text_encoder_observed_wall_seconds" if is_text else "sampler_observed_wall_seconds"): wall, "gpu_compute_only_seconds": None}


def main():
    global STAGE_NODES, ASSET_EXPECTED, ASSET_IDS
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
            and Path(__file__).resolve().parent == a.iteration_root / "agents/end-to-end-preparation", "Q33/Q31 roots")
    require(a.portable_python == a.comfy_root.parent / "python_embeded/python.exe", "fixed portable relationship")
    sys.dont_write_bytecode = True
    v1_path = a.q31_root / "tools/comfy_first_video_v1.py"
    require(digest(v1_path) == V1_SHA, "v1 dependency drift")
    spec = importlib.util.spec_from_file_location("q32_frozen_q31_stdlib_helper", v1_path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)  # Imports stdlib only; its main is not called.
    require(Path(sys.executable).resolve()==a.portable_python,"use existing portable Python for native aiohttp")
    require(digest(_timing_path)==TIMING_MODULE_SHA,"timing module drift")
    require(a.profile == a.iteration_root / "agents/timing-controller/profiles/trim-8.json", "fixed Trim8 profile path")
    a.loaded_profile=load_profile(a,v1)
    require(a.loaded_profile["id"] == "trim-8" and a.loaded_profile["family"] == "trim" and a.loaded_profile["steps"] == 8, "fixed Trim8 recipe")
    STAGE_NODES={**NODE_FAMILIES["trim"],**STAGE_A_NODES}
    ASSET_EXPECTED=copy.deepcopy(EXPECTED_BY_FAMILY["trim"])
    ASSET_EXPECTED["text-encoder"]={'id':'text-encoder','bytes':15687142551,'sha256':'35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6','relative_path':'comfy-trim/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'}
    ASSET_IDS=set(ASSET_EXPECTED)
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
    conditioning = None
    stage_a_graphs={}
    for case, descriptor in STAGE_A_GRAPHS.items():
        path=Path(__file__).resolve().parent/descriptor['file']
        item=v1.frozen_json(path,descriptor['sha256'],gates['max_schema_bytes'])
        require(set(item)=={'prompt'} and {x['class_type'] for x in item['prompt'].values()}==set(STAGE_A_NODES),'StageA fixed nodes')
        stage_a_graphs[case]=item
    stage_a_graph=stage_a_graphs['bounce-audio']  # Shared CLIP/VAE enumeration only.
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
    require({n["class_type"] for n in graph["prompt"].values()} == set(NODE_FAMILIES["trim"]),
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
    state = {"status": "preflight", "claim": "INCONCLUSIVE", "mode": "two-new-prompts-trim8",
             "profile_id": a.loaded_profile["id"], "profile_sha256": a.profile_sha256,
             "controller_entry_perf_counter_ns":entry_perf_ns,
             "source_verification_start_ns":source_start_ns,"source_verification_end_ns":source_end_ns,
             "requested_aimdo_vram_headroom_gib": 1.0,
             "driver_sha256": digest(Path(__file__)), "v1_sha256": V1_SHA,
             "graph_sha256": graph_sha, "asset_set_sha256": a.asset_manifest_sha256.lower(),
             "source_manifest_sha256": SOURCE_MANIFEST_SHA,
             "conditioning_manifest_sha256": a.conditioning_manifest_sha256.lower(),
             "gates_sha256": a.gates_sha256.lower(), "gates": gates,
             "prompt_posts": 0, "http_bytes": 0, "history_polls": 0, "free_posts":0,
             "stage_a_graphs":STAGE_A_GRAPHS,"fixed_case_order":["bounce-audio","hands-face-mug"],"releases":[],"videos":[],
             "fresh_between_stages":{"physical_bytes":12*2**30,"commit_policy":"remaining_owned_private_growth_plus_original_running_floor","gpu_free_bytes":6*2**30,"pre_post_available_minimum":True},
             "requests": [], "completed_requests": 0, "memory_policy": a.memory_policy,
             "cache_policy": "none", "total_output_bytes": 0, "total_ws_bytes": 0,
             "verified_assets": [], "gpu_measurement": "nvidia-smi whole-board MiB",
             "cuda_allocator_metrics": None, "media_validation": "not_performed"}
    asset_stats = {}

    def event(kind, **fields):
        if journal is not None:
            journal.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                                      "event": kind, "perf_counter_ns":time.perf_counter_ns(),
                                       "request_index": state.get("current_request_index"), **fields}) + "\n")
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
        row = {"elapsed_seconds": now - started, "request_index": state.get("current_request_index"),
               **memory, "owned_processes": owned,
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
        if state["requests"]:
            current = state["requests"][-1]
            for key in ("owned_rss_sum_bytes", "owned_private_sum_bytes"):
                current["peak_" + key] = max(current.get("peak_" + key, 0), row[key])
            current["min_gpu_board_free_bytes"] = min(current.get("min_gpu_board_free_bytes", gpu["free_bytes"]), gpu["free_bytes"])
            current["max_gpu_board_used_bytes"] = max(current.get("max_gpu_board_used_bytes", 0), gpu["used_bytes"])
        if samples is not None:
            samples.write(json.dumps(row) + "\n"); samples.flush()
        require(gpu["free_bytes"] >= gates[floor + "gpu_free_bytes"], "board free-VRAM guard")
        next_sample = time.monotonic() + gates["poll_seconds"]

    def warm_admit():
        after=monitor.memory()
        decision=e2e.warm_admission(state["last_sample"],after,gates)
        decision["post_read_perf_counter_ns"]=time.perf_counter_ns()
        state["last_warm_admission"]=decision
        return decision["admitted"]

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
                "finite_tensor_values_checked_in_stage_a": True}

    def http(method, route, body=None):
        require(proc is not None and proc.poll() is None, "owned Comfy exited")
        require(monitor.owns_listener(proc.pid, a.port), "owned loopback listener not ready")
        allowed_get = {"/object_info/" + name for name in STAGE_NODES} | {
            "/history/" + state["prompt_id"]}
        require((method == "GET" and route in allowed_get) or
                (method == "POST" and route == "/prompt" and 1 <= state["prompt_posts"] == len(state["requests"]) <= 4 and state["requests"][-1].get("post_attempted") is True) or
                (method=="POST" and route=="/free" and 1<=state["free_posts"]==state["completed_requests"]==state["prompt_posts"]<=3 and state["requests"][-1]["status"]=="complete" and body==b'{"unload_models":true,"free_memory":true}'), "HTTP operation outside fixed task")
        connection = http_client.HTTPConnection("127.0.0.1", a.port,
                                                 timeout=gates["http_timeout_seconds"])
        try:
            headers = {"Connection": "close"}
            if body is not None:
                require(len(body) <= gates["max_schema_bytes"], "request body bound")
                headers["Content-Type"] = "application/json"
            connection.request(method, route, body=body, headers=headers)
            response = connection.getresponse()
            if route!="/free":
                require(response.getheader("Content-Type", "").split(";")[0] == "application/json", "non-JSON local response")
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
            if route=="/free":
                require(response.status==200 and b"".join(parts)==b"", "native free empty-200 response")
                return response.status, {}
            return response.status, json.loads(b"".join(parts), object_pairs_hook=v1.unique_json)
        finally:
            connection.close()

    try:
        sample(force=True, startup=True)
        require(e2e.fresh_ready(state["last_sample"]), "initial external admission 12/16.5/6")
        for relative, expected in STAGE_SOURCE_SHA256.items():
            require(digest(a.comfy_root / relative) == expected, "Stage-B native source drift")
        state["stage_source_sha256"] = STAGE_SOURCE_SHA256
        release_sources={'server.py':'52618f91c663dc6524a028162212e0434fc9239618ddc0efbf7c605027081c8c','main.py':'baaa16d69ee78e08c490eeec20bf27c56b746c070f76cc05b4aa13cbc616bcef','execution.py':'5312ba107f69f2bc70f4009c5d52524b3865dd54fceafdc8649b22c98d6c612f','comfy/model_management.py':'63610e41cf06fccb5eeaa153c5fd3799e01a0c645facba844763781d2036989a','comfy/model_patcher.py':'4f8ccf957a7227942c0e167384ed4e128f038245ad09b9bb547102281be13612','comfy_execution/caching.py':'f71d43f6196330e73674e13795a9d392a95750ad55be2c9d96cf54355794b862'}
        for name,expected in release_sources.items():require(digest(a.comfy_root/name)==expected,"native release source drift")
        state["release_source_sha256"]=release_sources
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
        e2e.no_links(embeddings)
        require(embeddings.is_dir() and embeddings.resolve()==embeddings,"native embeddings root")
        conditioning_path=None
        sample(force=True,startup=True)

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
                   "--offline", "--disable-assets", "--disable-pinned-memory", "--disable-smart-memory", "--cache-none", "--preview-method", "none",
                   "--vram-headroom", "1.0",
                   "--reserve-vram", str(gates["reserve_vram_gb"]),
                   "--models-directory", str(models_root)]
        command = memory_command(command, a.memory_policy)
        state["disable_smart_memory_requested"] = "--disable-smart-memory" in command
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
            if name in ("UNETLoader", "VAELoader", "CLIPLoader"):
                field, ids = {"UNETLoader":("unet_name",["6"]),"VAELoader":("vae_name",["11","24"]),"CLIPLoader":("clip_name",["13"])}[name]
                choices=info[name]["input"]["required"][field][0]
                selected_graph=stage_a_graph if name=="CLIPLoader" else graph
                require(type(choices) is list and all(selected_graph["prompt"][i]["inputs"][field] in choices for i in ids),"model file not enumerated")
        event("native_schema_verification_finished")
        def run_one(request_index):
            nonlocal observer, conditioning, conditioning_path
            require(observer is None, "previous observer not finalized")
            current = begin_request(state, request_index, str(uuid.uuid4()))
            state.pop("node_timing",None)  # Do not attach a previous request timing on early failure.
            request_started = time.monotonic()
            case='bounce-audio' if request_index<=2 else 'hands-face-mug'
            is_text=request_index in (1,3)
            reference=e2e.CASES[case]
            current['case']=case
            if is_text:
                conditioning=None;conditioning_path=None
                current['new_prompt_start_elapsed_seconds']=request_started-started
            else:
                require(conditioning is not None and conditioning_path is not None and state['requests'][-2]['case']==case,'matching preceding TE required')
            if request_index>1:
                require(len(state['releases'])==request_index-1 and state['releases'][-1]['after_request_index']==request_index-1 and state['releases'][-1]['fresh_admitted'],'prior transition release/admission required')
            request_dir = a.run_dir / ("request-" + str(request_index))
            request_dir.mkdir()
            current_graph = copy.deepcopy(stage_a_graphs[case]) if is_text else e2e.bind_stage_b(graph,conditioning_path.name,case)
            current["stage"]="text-encoder" if is_text else "trim-video"
            current["graph_sha256"] = hashlib.sha256(json.dumps(current_graph,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()
            current["history_relative_to_run"] = str((request_dir / "history.raw.json").relative_to(a.run_dir))
            current["timing_relative_to_run"] = str((request_dir / "timing/summary.json").relative_to(a.run_dir))
            v1.save_new(request_dir / "graph.raw.json", current_graph)
            event("request_preparation_started")
            require(digest(a.profile)==a.profile_sha256,"profile changed before POST")
            require(digest(a.asset_manifest) == a.asset_manifest_sha256.lower(), "asset manifest changed before POST")
            require(load_asset_manifest(a, v1, gates) == selected, "asset receipt identity changed before POST")
            for path, signature in asset_stats.items():
                st = path.stat()
                require((st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino) == signature, "asset changed before POST")
            require(digest(Path(__file__).resolve().parent/STAGE_A_GRAPHS[case]["file"])==STAGE_A_GRAPHS[case]["sha256"],"StageA graph drift before POST")
            if not is_text:
                state["conditioning_pre_post"] = verify_conditioning(conditioning_path,min(deadline,time.monotonic()+60))
                status,info=http("GET","/object_info/ConditioningLoader")
                v1.save_new(request_dir/"ConditioningLoader.schema.raw.json",info)
                require(status==200 and set(info)=={"ConditioningLoader"} and info["ConditioningLoader"]["python_module"]=="comfy_extras.nodes_cond","refreshed conditioning schema source")
                combo=info["ConditioningLoader"]["input"]["required"]["conditioning_name"]
                require(type(combo) is list and len(combo)==2 and combo[0]=="COMBO" and combo[1].get("multiselect") is False and conditioning_path.name in combo[1].get("options",[]),"new condition must be enumerated")
            require(digest(a.conditioning_manifest)==a.conditioning_manifest_sha256,"baseline graph manifest drift")
            require(digest(a.iteration_root/manifest["graph"]["relative_path"])==graph_sha,"baseline graph drift")
            sample(force=True,startup=request_index==1)
            if request_index>1:require(warm_admit(),"warm admission before next POST")
            render_deadline = min(deadline, time.monotonic() + gates["render_timeout_seconds"])
            state["client_id"]=uuid.uuid4().hex
            current["client_id"] = state["client_id"]
            observer=NativeWSObserver(a.port,state["prompt_id"],state["client_id"],
                {k:n["class_type"] for k,n in current_graph["prompt"].items()},request_dir/"timing",
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
            body = json.dumps({"prompt": current_graph["prompt"], "prompt_id": state["prompt_id"],
                               "client_id":state["client_id"]}).encode("utf-8")
            if request_index>1:
                sample(force=True)
                require(warm_admit(),"warm admission immediately before next POST")
                current['immediately_pre_post_warm_admission']=copy.deepcopy(state['last_warm_admission'])
            begin_post(state); state["status"] = "submitting"
            event("post_attempt", prompt_id=state["prompt_id"], graph_sha256=current["graph_sha256"])
            try:
                status, answer = http("POST", "/prompt", body)
                v1.save_new(request_dir / "post.response.raw.json", answer)
                require(status == 200 and answer.get("prompt_id") == state["prompt_id"]
                        and not answer.get("node_errors"), "server rejected or changed fixed prompt")
                current["submission_outcome"] = "accepted"
            except (OSError, http_client.HTTPException) as exc:
                current["submission_outcome"] = "unknown"
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
                    v1.save_new(request_dir / "history.raw.json", history)
                    require(set(history) == {state["prompt_id"]}, "unexpected history prompt identity")
                    record = history[state["prompt_id"]]
                    require(record.get("status", {}).get("status_str") == "success"
                            and record["status"].get("completed") is True, "prompt execution failed")
                    observed = record["prompt"][2]
                    require(set(observed) == set(current_graph["prompt"]) and all(
                        observed[k]["class_type"] == v["class_type"] and observed[k]["inputs"] == v["inputs"]
                        for k, v in current_graph["prompt"].items()), "executed graph changed")
                    if is_text:
                        event("conditioning_verification_started")
                        condition_root=dirs['output']/'q33-e2e-two'/case/'conditioning'
                        e2e.no_links(condition_root)
                        output=e2e.find_conditioning_artifact(condition_root,32*2**20,sample)
                        structure=conditioning_header(output,v1.unique_json,32*2**20)
                        value_scan=e2e.scan_conditioning(output,structure,sample,min(deadline,time.monotonic()+60),case)
                        output_hash,signature=guarded_hash(output,render_deadline)
                        require(signature==structure.pop("file_signature"),"conditioning changed during finite/hash scan")
                        require(output_hash==reference["sha256"] and output.stat().st_size==reference["bytes"],"case conditioning differs from frozen split-process reference")
                        conditioning={"bytes":output.stat().st_size,"sha256":output_hash,**{k:structure[k] for k in ("tensors","metadata_keys","option_keys","options_sha256")}}
                        current["artifact"]={"relative_to_run":str(output.relative_to(a.run_dir)),"bytes":output.stat().st_size,"sha256":output_hash,"structure":structure}
                        current["conditioning_value_validation"]=value_scan
                        basename="h3-e2e-"+case+"-"+a.run_dir.name.rsplit("-",1)[-1]+"-"+output_hash+".safetensors"
                        conditioning_path=embeddings/basename
                        require(not conditioning_path.exists() and not conditioning_path.is_symlink(),"new condition alias already exists")
                        os.link(output,conditioning_path)
                        e2e.no_links(conditioning_path)
                        require(conditioning_path.samefile(output),"new conditioning alias identity")
                        current["conditioning_alias"]={"relative_to_models":str(conditioning_path.relative_to(models_root)),"sha256":output_hash,"samefile":True,"created":True}
                        state["conditioning_input"]=verify_conditioning(conditioning_path,min(deadline,time.monotonic()+60))
                        state["conditioning_input"]["relative_to_models"]=str(conditioning_path.relative_to(models_root))
                        current["conditioning_input"]=copy.deepcopy(state["conditioning_input"])
                        event("conditioning_verified_and_registered",artifact=current["artifact"],alias=current["conditioning_alias"],values=value_scan)
                    else:
                        videos = record["outputs"]["92"]["images"]
                        require(len(videos) == 1 and videos[0].get("type") == "output", "single MP4 artifact not returned")
                        item = videos[0]
                        relative = Path(item["subfolder"]) / item["filename"]
                        require(not relative.is_absolute() and not relative.drive, "absolute output path forbidden")
                        output = (dirs["output"] / relative).resolve()
                        require(output.is_relative_to(dirs["output"] / "q33-e2e-two" / case) and output.name.startswith("video_") and output.suffix.lower() == ".mp4"
                                and output.is_file() and 0 < output.stat().st_size <= gates["max_output_bytes"],
                                "output path/size guard")
                        output_hash, _ = guarded_hash(output, render_deadline)
                        current["artifact"] = {"relative_to_run": str(output.relative_to(a.run_dir)),
                                             "bytes": output.stat().st_size, "sha256": output_hash}
                    if not is_text:
                        current['video_artifact_wall_seconds']=time.monotonic()-started
                        state['end_to_end_artifact_wall_seconds']=current['video_artifact_wall_seconds']
                        state['videos'].append({'case':case,'request_index':request_index,'artifact':copy.deepcopy(current['artifact']),'artifact_wall_seconds':current['video_artifact_wall_seconds']})
                    current["status"] = "artifact_received_unvalidated"
                    event("artifact_received", artifact=current["artifact"])
                    break
                time.sleep(min(gates["poll_seconds"], max(0, render_deadline - time.monotonic())))
            observer_deadline = min(deadline, time.monotonic() + 12)
            while not observer.done.wait(timeout=0.05):
                sample()
                require(time.monotonic() < observer_deadline, "request observer terminal deadline")
            timing = observer.finish()
            observer = None
            current["node_timing"] = timing
            current["metrics"] = request_metrics(timing, record, state["prompt_id"])
            if not is_text:
                state["artifact"]=current["artifact"]
                state["node_timing"]=timing

            state["total_ws_bytes"] += timing["raw_received_bytes"]
            state["total_output_bytes"] += current["artifact"]["bytes"]
            require(state["total_ws_bytes"] <= gates["max_total_http_bytes"], "whole-session WS budget")
            require(state["total_output_bytes"] <= gates["max_output_bytes"], "whole-session output budget")
            current["request_elapsed_seconds"] = time.monotonic() - request_started
            current["status"] = "complete"
            state["completed_requests"] += 1
            sample(force=True)
            current["after_request_sample"] = copy.deepcopy(state["last_sample"])
            event("request_complete", metrics=current["metrics"], artifact=current["artifact"])
        def release_transition(after_request_index):
            event("release_started")
            release_started=time.perf_counter_ns()
            log_path=a.run_dir/"comfy.raw.log";log_offset=log_path.stat().st_size
            require(after_request_index in (1,2,3) and state["free_posts"]==after_request_index-1 and state["completed_requests"]==state["prompt_posts"]==after_request_index,"one free per completed transition")
            state["free_posts"]+=1
            state["release"]={"after_request_index":after_request_index,"log_offset":log_offset,"post_attempted":True,"reset_path_entered":False,"fresh_admitted":False,"te_object_absence_proven":False,"start_perf_counter_ns":release_started}
            status,_=http("POST","/free",b'{"unload_models":true,"free_memory":true}')
            state["release"]["http_status"]=status
            release_deadline=min(deadline,time.monotonic()+30)
            while True:
                sample(force=True)
                require(proc.poll() is None and time.monotonic()<release_deadline,"release/admission deadline or service exit")
                entered=e2e.reset_path_entered(log_path,log_offset)
                if entered and not state["release"]["reset_path_entered"]:
                    state["release"]["reset_path_entered"]=True
                    state["release"]["marker_perf_counter_ns"]=time.perf_counter_ns()
                    event("reset_path_entered_not_completion_ack")
                if entered and warm_admit():
                    state["release"]["fresh_admitted"]=True
                    state["release"]["fresh_sample"]=copy.deepcopy(state["last_sample"])
                    state["release"]["warm_admission"]=copy.deepcopy(state["last_warm_admission"])
                    state["release"]["admitted_perf_counter_ns"]=time.perf_counter_ns()
                    break
                time.sleep(gates["poll_seconds"])
            state["release"]["elapsed_seconds"]=(time.perf_counter_ns()-release_started)/1e9
            event("release_fresh_admission_finished",release=state["release"])
            state["releases"].append(copy.deepcopy(state["release"]))
        for request_index in (1,2,3,4):
            if request_index>1:release_transition(request_index-1)
            run_one(request_index)
        require(state['completed_requests']==state['prompt_posts']==4 and state['free_posts']==3 and len(state['videos'])==2,'four completed requests and three transitions required')
        state['status']='two_new_prompt_videos_received_unvalidated'
    except BaseException as exc:
        state["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) or a.stop_marker.exists() else "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc) if type(exc) is RuntimeError else "external operation failed; local raw evidence retained"
        event("stopped", status=state["status"], error_type=type(exc).__name__)
    finally:
        cleanup_owned_v2(job,proc,observer,state,gates,event)
        if state["requests"] and "node_timing" in state:
            state["requests"][-1]["node_timing"] = state["node_timing"]
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
    return 0 if state["status"] == "two_new_prompt_videos_received_unvalidated" else 1


if __name__ == "__main__":
    raise SystemExit(main())