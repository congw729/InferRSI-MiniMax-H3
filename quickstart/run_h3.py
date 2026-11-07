#!/usr/bin/env python3
"""Small standard-library client for a dedicated local ComfyUI H3 instance.

No model downloads, service startup, hidden experimental imports or global interrupt.
The supplied preset uses native ComfyUI nodes; Python 3.10+ is sufficient here.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

GIB = 1024**3
MAX_JSON = 32 * 1024**2
MAX_CONDITIONING = 32 * 1024**2
MAX_VIDEO = 1024**3
MIN_RAM = 12 * GIB
MIN_COMMIT = int(16.5 * GIB)
MIN_GPU = 6 * GIB


class PreviewError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PreviewError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_url(url):
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"},
            "Use a dedicated local ComfyUI HTTP instance (127.0.0.1, localhost or ::1).")
    require(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
            and parsed.path in {"", "/"}, "Server URL must not contain credentials, paths or queries.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PreviewError("Invalid server port.") from exc
    require(port is not None and 1 <= port <= 65535, "Specify the ComfyUI server port.")
    return url.rstrip("/")


class API:
    def __init__(self, url):
        self.url = validate_url(url)
        # Local-only requests must not inherit an HTTP/VPN proxy.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, message, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(self, method, route, value=None):
        payload = None if value is None else json.dumps(value).encode("utf-8")
        request = urllib.request.Request(self.url + route, data=payload, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=15) as response:
                data = response.read(MAX_JSON + 1)
        except urllib.error.HTTPError as exc:
            raise PreviewError(f"ComfyUI returned HTTP {exc.code} for {route}. Check its console and node/model versions.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PreviewError(f"Cannot reach ComfyUI at {self.url}; start the dedicated server from README.") from exc
        require(len(data) <= MAX_JSON, "ComfyUI response exceeds the client size limit.")
        return json.loads(data) if data else {}


def queue_ids(queue):
    running = {row[1] for row in queue.get("queue_running", [])}
    pending = {row[1] for row in queue.get("queue_pending", [])}
    return running, pending


def ensure_idle(api):
    running, pending = queue_ids(api.call("GET", "/queue"))
    require(not running and not pending,
            "ComfyUI already has work queued. Use an idle, dedicated instance for this preview.")


def windows_memory():
    if os.name != "nt":
        return {}
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_pagefile", ctypes.c_ulonglong), ("available_commit", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)), "Cannot query Windows memory.")
    return {"ram_free": status.available_physical, "commit_free": status.available_commit}


def compact_stats(stats):
    devices = stats.get("devices", [])
    require(devices and devices[0].get("type") == "cuda", "ComfyUI must use a CUDA GPU.")
    gpu = devices[0]
    system = stats["system"]
    result = {"ram_free": system["ram_free"], "vram_free": gpu["vram_free"],
              "torch_vram_reserved": gpu["torch_vram_total"],
              "comfyui_version": system.get("comfyui_version"),
              "pytorch_version": system.get("pytorch_version"), "gpu_name": gpu["name"]}
    result.update(windows_memory())
    return result


def check_headroom(stats, *, initial=True):
    ram_floor = MIN_RAM if initial else 2 * GIB
    commit_floor = MIN_COMMIT if initial else 3 * GIB
    require(stats["ram_free"] >= ram_floor,
            f"Need at least {ram_floor/GIB:g} GiB free system RAM; currently {stats['ram_free']/GIB:.1f} GiB. Close unused applications.")
    if "commit_free" in stats:
        require(stats["commit_free"] >= commit_floor,
                f"Need {commit_floor/GIB:g} GiB available commit memory; currently {stats['commit_free']/GIB:.1f} GiB.")
    require(stats["vram_free"] >= MIN_GPU,
            f"Need at least 6 GiB free VRAM; currently {stats['vram_free']/GIB:.1f} GiB. Close unused GPU applications.")


def combo_options(node, name):
    spec = node["input"]["required"][name]
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return []


def inspect_server(api, preset, *, check_resources=True):
    ensure_idle(api)
    raw = api.call("GET", "/system_stats")
    stats = compact_stats(raw)
    require(stats["comfyui_version"] == "0.43.0",
            "This preview targets ComfyUI 0.43.0; use the pinned environment described in README.")
    require("--cache-none" in raw["system"].get("argv", []),
            "Restart this dedicated ComfyUI instance with --cache-none (see README).")
    info = api.call("GET", "/object_info")
    types = {node["class_type"] for key in ("stage_text", "stage_video") for node in preset[key].values()}
    missing = sorted(types - set(info))
    require(not missing, "Missing native ComfyUI nodes: " + ", ".join(missing))
    for key in ("stage_text", "stage_video"):
        for node in preset[key].values():
            model_field = {"CLIPLoader": "clip_name", "UNETLoader": "unet_name", "VAELoader": "vae_name"}.get(node["class_type"])
            if model_field:
                name = node["inputs"][model_field]
                require(name in combo_options(info[node["class_type"]], model_field),
                        f"Model not found by ComfyUI: {name}. See quickstart/MODELS.md and restart the server after installing models.")
    if check_resources:
        check_headroom(stats)
    return stats


def build_graphs(preset, prompt, seed, run_id, conditioning_name):
    require(isinstance(prompt, str) and bool(prompt.strip()) and len(prompt) <= 8000,
            "Supply a nonempty prompt of at most 8000 characters.")
    require(type(seed) is int and 0 <= seed < 2**64, "Seed must be an unsigned 64-bit integer.")
    require(len(run_id) == 32 and all(c in "0123456789abcdef" for c in run_id), "Invalid run id.")
    text = copy.deepcopy(preset["stage_text"])
    video = copy.deepcopy(preset["stage_video"])
    text["104"]["inputs"]["prompt"] = prompt
    text["1001"]["inputs"]["filename_prefix"] = f"h3-preview/{run_id}/conditioning/stage-a"
    video["15"]["inputs"]["noise_seed"] = seed
    video["1001"]["inputs"]["conditioning_name"] = conditioning_name
    video["92"]["inputs"]["filename_prefix"] = f"h3-preview/{run_id}/video/clip"
    return text, video


def native_seconds(history):
    stamps = {}
    for kind, details in history.get("status", {}).get("messages", []):
        if kind in {"execution_start", "execution_success"} and isinstance(details.get("timestamp"), (int, float)):
            stamps[kind] = details["timestamp"]
    if len(stamps) == 2:
        return max(0, (stamps["execution_success"] - stamps["execution_start"]) / 1000)
    return None


def run_stage(api, graph, label, timeout, record):
    ensure_idle(api)
    prompt_id = str(uuid.uuid4())
    item = {"stage": label, "prompt_id": prompt_id, "status": "submitting"}
    record["stages"].append(item)
    start = time.monotonic()
    result = api.call("POST", "/prompt", {"prompt": graph, "prompt_id": prompt_id, "client_id": record["run_id"]})
    require(result.get("prompt_id") == prompt_id and not result.get("node_errors"),
            "ComfyUI rejected the workflow; check model and node compatibility in its console.")
    item["status"] = "running"
    deadline = start + timeout
    next_sample = start
    while time.monotonic() < deadline:
        all_history = api.call("GET", "/history/" + prompt_id)
        if prompt_id in all_history:
            history = all_history[prompt_id]
            status = history.get("status", {})
            item["wall_seconds"] = round(time.monotonic() - start, 6)
            item["native_execution_seconds"] = native_seconds(history)
            item["status"] = status.get("status_str", "unknown")
            require(status.get("completed") is True and item["status"] == "success",
                    f"{label} failed. Check this prompt in the ComfyUI console/history: {prompt_id}")
            return history
        if time.monotonic() >= next_sample:
            stats = compact_stats(api.call("GET", "/system_stats"))
            item["sampled_min_ram_free"] = min(item.get("sampled_min_ram_free", stats["ram_free"]), stats["ram_free"])
            item["sampled_min_vram_free"] = min(item.get("sampled_min_vram_free", stats["vram_free"]), stats["vram_free"])
            item["sampled_max_torch_reserved"] = max(item.get("sampled_max_torch_reserved", 0), stats["torch_vram_reserved"])
            require(stats["ram_free"] >= 2 * GIB and stats.get("commit_free", 3 * GIB) >= 3 * GIB,
                    "System memory reserve exhausted; cancelling this client's prompt.")
            require(stats["vram_free"] >= 256 * 1024**2, "GPU free memory fell below 256 MiB; cancelling this client's prompt.")
            next_sample = time.monotonic() + 1
        time.sleep(0.25)
    raise PreviewError(f"{label} exceeded {timeout:g} seconds; stopping only this client's prompt.")


def cancel_owned(api, record):
    failures = []
    for item in record.get("stages", []):
        if item["status"] not in {"submitting", "running"}:
            continue
        prompt_id = item["prompt_id"]
        try:
            # Server 0.43.0 implements prompt_id-scoped interruption.
            api.call("POST", "/queue", {"delete": [prompt_id]})
            api.call("POST", "/interrupt", {"prompt_id": prompt_id})
            item["status"] = "cancellation_requested"
        except PreviewError:
            failures.append(prompt_id)
    return failures


def cleanup_failed_run(api, record, baseline):
    """Observe cancellation; release only after the dedicated queue is empty."""
    owned = {item["prompt_id"] for item in record.get("stages", [])}
    deadline = time.monotonic() + 30
    try:
        while True:
            running, pending = queue_ids(api.call("GET", "/queue"))
            if not owned & (running | pending):
                for item in record.get("stages", []):
                    if item["status"] == "cancellation_requested":
                        item["status"] = "no_longer_queued"
                record["owned_prompts_no_longer_queued"] = True
                if running or pending:
                    record["cleanup_status"] = "skipped_other_work_queued"
                elif baseline is not None and owned:
                    record["after_failure_release"] = release_models(api, baseline)
                    record["cleanup_status"] = "completed"
                else:
                    record["cleanup_status"] = "not_needed"
                return
            if time.monotonic() >= deadline:
                record["owned_prompts_no_longer_queued"] = False
                record["cleanup_status"] = "pending_owned_work"
                return
            time.sleep(0.5)
    except (PreviewError, OSError, ValueError, KeyError, TypeError) as exc:
        record["cleanup_status"] = "pending_inspection"
        record["cleanup_error"] = str(exc)


def release_models(api, baseline):
    ensure_idle(api)
    api.call("POST", "/free", {"unload_models": True, "free_memory": True})
    start = time.monotonic()
    deadline = start + 45
    while time.monotonic() < deadline:
        ensure_idle(api)
        stats = compact_stats(api.call("GET", "/system_stats"))
        if time.monotonic() - start >= 1 and stats["torch_vram_reserved"] <= baseline["torch_vram_reserved"] + 64 * 1024**2:
            check_headroom(stats, initial=False)
            return stats
        time.sleep(0.25)
    raise PreviewError(f"ComfyUI model release did not reach the idle GPU budget in 45 seconds: torch reserved {stats['torch_vram_reserved']/1024**2:.0f} MiB, baseline {baseline['torch_vram_reserved']/1024**2:.0f} MiB. Stop here and inspect the server.")


def contained_file(root, relative, expected_prefix):
    require(isinstance(relative, str) and "\\" not in relative, "Invalid artifact path.")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and all(p not in {"..", "."} and ":" not in p for p in parts.parts), "Unsafe artifact path.")
    require(relative.startswith(expected_prefix + "/"), "Artifact does not belong to this run.")
    root = root.resolve()
    path = root.joinpath(*parts.parts)
    for part in [path, *path.parents]:
        if part == root:
            break
        require(not part.is_symlink(), "Artifact symlinks are not supported.")
    require(path.resolve().is_relative_to(root), "Artifact escapes ComfyUI output directory.")
    require(path.is_file(), "ComfyUI did not write the expected artifact; verify --comfy-output matches the server.")
    return path


def find_video(history, run_id, output_root):
    outputs = history.get("outputs", {}).get("92", {})
    matches = []
    for key in ("images", "videos", "gifs"):
        for item in outputs.get(key, []):
            if item.get("type") == "output" and str(item.get("filename", "")).endswith(".mp4"):
                filename = item["filename"]
                require(Path(filename).name == filename and "/" not in filename and "\\" not in filename,
                        "Invalid video filename in ComfyUI history.")
                relative = str(PurePosixPath(item.get("subfolder", "")) / filename)
                matches.append(contained_file(output_root, relative, f"h3-preview/{run_id}/video"))
    require(len(matches) == 1, "Expected one MP4 in this prompt's SaveVideo history.")
    require(0 < matches[0].stat().st_size <= MAX_VIDEO, "Video is empty or larger than 1 GiB.")
    return matches[0]


def verify_weights(model_root, manifest):
    result = []
    for asset in load_json(manifest)["assets"]:
        path = (model_root / asset["relative_path"]).resolve()
        require(path.is_relative_to(model_root.resolve()), "Model manifest contains an unsafe path.")
        require(path.is_file() and path.stat().st_size == asset["bytes"], f"Missing/wrong-sized model: {asset['filename']}")
        print("Checking SHA256:", asset["filename"], flush=True)
        require(sha256(path) == asset["sha256"], f"Model hash mismatch: {asset['filename']}")
        result.append(asset["filename"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "generate"))
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, required=True, help="The ComfyUI folder containing main.py, not its portable parent.")
    parser.add_argument("--comfy-output", type=Path, help="Only needed if the server has a custom --output-directory.")
    parser.add_argument("--embeddings-dir", type=Path, help="Defaults to ComfyUI/models/embeddings; must be enumerated by that server.")
    parser.add_argument("--preset", type=Path, default=Path(__file__).with_name("trim8.json"))
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output", type=Path, help="New output MP4 path; an adjacent .run.json records timings.")
    parser.add_argument("--timeout", type=float, default=600, help="Maximum seconds per submitted stage.")
    parser.add_argument("--verify-weights", action="store_true", help="Optional full SHA256 read; useful once after downloading.")
    parser.add_argument("--model-root", type=Path, help="For --verify-weights only; defaults to ComfyUI/models.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models.json"))
    args = parser.parse_args(argv)
    record = {"schema": "h3-preview-run/v1", "run_id": uuid.uuid4().hex, "stages": [], "status": "preflight"}
    api = None
    baseline = None
    summary = None
    start = time.monotonic()
    try:
        require(1 <= args.timeout <= 3600, "--timeout must be between 1 and 3600 seconds.")
        require((args.comfy_root / "main.py").is_file(), "--comfy-root must point to the ComfyUI directory containing main.py.")
        preset = load_json(args.preset)
        require(preset.get("schema") == "h3-preview-preset/v1", "Unsupported preset schema.")
        api = API(args.url)
        baseline = inspect_server(api, preset)
        record.update(preset=preset["id"], environment=baseline,
                      output_geometry={key: preset[key] for key in ("width", "height", "frames", "fps", "steps")})
        if args.verify_weights:
            record["verified_models"] = verify_weights(args.model_root or args.comfy_root / "models", args.manifest)
        if args.command == "doctor":
            print(json.dumps({"ready": True, **record}, ensure_ascii=False, indent=2))
            return 0
        require(args.output is not None and args.output.suffix.lower() == ".mp4", "generate requires --output ending in .mp4.")
        require(args.prompt is not None, "generate requires --prompt.")
        require(not args.output.exists(), "Output already exists; choose a new filename.")
        summary = args.output.with_suffix(".run.json")
        require(not summary.exists(), "Run summary already exists; choose a new output filename.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(args.output.parent).free >= GIB, "Need at least 1 GiB free output disk space.")
        output_root = (args.comfy_output or args.comfy_root / "output").resolve()
        require(output_root.is_dir(), "ComfyUI output directory is missing; check --comfy-output.")
        run_root = output_root / "h3-preview" / record["run_id"]
        run_root.mkdir(parents=True, exist_ok=False)
        embedding_root = (args.embeddings_dir or args.comfy_root / "models/embeddings").resolve()
        embedding_root.mkdir(parents=True, exist_ok=True)
        embedding_name = "h3-preview-" + record["run_id"] + ".safetensors"
        text, video = build_graphs(preset, args.prompt, args.seed, record["run_id"], embedding_name)
        record.update(prompt=args.prompt, seed=args.seed, service_startup_included=False,
                      weight_sha256_validation_included=bool(args.verify_weights),
                      preset_sha256=sha256(args.preset), runner_sha256=sha256(Path(__file__)),
                      node_timings="Not measured by this minimal HTTP client; stage request and native execution totals are recorded.")
        print("1/3 Text encoding...", flush=True)
        run_stage(api, text, "text_encoding", args.timeout, record)
        candidates = list((run_root / "conditioning").glob("*.safetensors"))
        require(len(candidates) == 1, "Expected exactly one saved conditioning. Check --comfy-output and server settings.")
        cond = contained_file(output_root, candidates[0].relative_to(output_root).as_posix(),
                              f"h3-preview/{record['run_id']}/conditioning")
        require(8 < cond.stat().st_size <= MAX_CONDITIONING, "Conditioning exceeds the 32 MiB limit.")
        with cond.open("rb") as source, (embedding_root / embedding_name).open("xb") as dest:
            shutil.copyfileobj(source, dest, 1024**2)
        record["conditioning_sha256"] = sha256(cond)
        print("2/3 Releasing text encoder...", flush=True)
        release_start = time.monotonic()
        record["after_text_release"] = release_models(api, baseline)
        record["release_seconds"] = time.monotonic() - release_start
        info = api.call("GET", "/object_info/ConditioningLoader")
        require(embedding_name in combo_options(info["ConditioningLoader"], "conditioning_name"),
                "ComfyUI cannot see the generated conditioning; set --embeddings-dir to one of its configured embeddings folders.")
        print("3/3 Sampling, decoding and saving video...", flush=True)
        history = run_stage(api, video, "video_generation_with_audio", args.timeout, record)
        artifact = find_video(history, record["run_id"], output_root)
        with artifact.open("rb") as source, args.output.open("xb") as dest:
            shutil.copyfileobj(source, dest, 1024**2)
        record.update(status="completed", output_filename=args.output.name, output_bytes=args.output.stat().st_size,
                      output_sha256=sha256(args.output), elapsed_seconds=time.monotonic() - start)
        print("Releasing models for the next request...", flush=True)
        cleanup_error = None
        cleanup_start = time.monotonic()
        try:
            record["after_video_release"] = release_models(api, baseline)
            record["cleanup_status"] = "completed"
        except PreviewError as exc:
            cleanup_error = str(exc)
            record["cleanup_status"] = "failed"
            record["cleanup_error"] = cleanup_error
        record["cleanup_seconds"] = time.monotonic() - cleanup_start
        record["total_with_cleanup_seconds"] = time.monotonic() - start
        save_json(summary, record)
        summary = None
        print("Saved:", args.output)
        print("Timing record:", args.output.with_suffix(".run.json"))
        if cleanup_error:
            print("Video saved, but model cleanup needs attention:", cleanup_error, file=sys.stderr)
            return 2
        return 0
    except (PreviewError, OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as exc:
        record["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = "Interrupted by user." if isinstance(exc, KeyboardInterrupt) else str(exc)
        if api is not None:
            record["cancellation_request_failures"] = cancel_owned(api, record)
            cleanup_failed_run(api, record, baseline)
        record["elapsed_seconds"] = time.monotonic() - start
        if summary is not None and not summary.exists():
            save_json(summary, record)
        print("Error:", record["error"], file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
