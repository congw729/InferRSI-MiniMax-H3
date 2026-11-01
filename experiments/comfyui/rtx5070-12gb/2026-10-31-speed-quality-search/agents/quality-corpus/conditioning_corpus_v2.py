"""Q33 fixed quality-corpus Stage A; one selected prompt, one service, one POST.

Invoke under InferRSI execute_command. This controller imports no inference
runtime. It uses v1's verified Windows ownership/monitor code and never downloads,
installs, schedules another run, changes global settings, or deletes files.
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
BASELINE_GRAPH_SHA = "bf1578a3753e527f782f4701cde2f7cd2547dab41ab7f729445f1b6a16cb6e1d"
OLD_GRAPH_SHA = '1da6a1e7184e13d7e783918d09f2a3bbec750b53ec3fd66c7c3e8c57a103cb88'
STAGE_GATES_SHA = '4d5114f7abc9041f34778b3f615034cc3da797150e750b4a6e7a4c684ad5a9be'
STAGE_NODES = {'CLIPLoader': 'nodes', 'VAELoader': 'nodes', 'MiniMaxH3ImageToVideo': 'comfy_extras.nodes_minimax_h3', 'SaveConditioning': 'comfy_extras.nodes_cond'}
STAGE_SOURCE_SHA256 = {'comfy_extras/nodes_cond.py': '56344e64a20692a0f64521bf4de78c0a9c040e23c3499a00491b5243ca39f898', 'comfy/text_encoders/minimax.py': 'cff2303063502b63afce83545c7443a48b339e42ee821fb716279a6aa2db4430', 'comfy/sd.py': 'b5dcb5d1156a1f4e5fb9b32ef684105cca27298416f8abe494eafac75a4d2863'}
ASSET_SET_SHA = "1a24d5245740f570c4169e86b5528036c26ad134ec784f8aec7cfc079234305d"
SCHEMA_SHA = "ba8a9b8be9fcf808bc55224d338437cef039a00e2754419ff4edb8187871d672"
ASSET_IDS = {"trim-dit", "text-encoder", "video-vae", "audio-vae"}
CORPUS_MANIFEST_SHA = "870756bbdae953f6c52b40a05a6ac35005d8818402170f1aa9d964bed715ef02"
TIMING_CONTROLLER_SHA = "524bf48df04c3ecdb251115b4db08c6482ed36eeb33405db34799b3a5669f66e"
SOURCE_MANIFEST_SHA = "bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007"
SCHEMA_RESULT_SHA = "60a916c5ae597eca9abb8901f788691b0759aa9204df4b8168065d68b4933788"
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


def find_conditioning_artifact(root, max_bytes, sample):
    """Only this new output tree; reject links/reparse points and excessive entries."""
    root = root.resolve()
    files = []
    count = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        sample()
        parent = Path(directory)
        require(parent.resolve().is_relative_to(root) and len(parent.relative_to(root).parts) <= 8,
                "conditioning output traversal boundary")
        for name in dirs + names:
            count += 1
            require(count <= 128, "conditioning output entry bound")
            path = parent / name
            st = path.lstat()
            require(not path.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400)
                    and path.resolve().is_relative_to(root), "conditioning output link/reparse boundary")
            if name in names:
                require(path.is_file() and path.suffix.lower() == ".safetensors",
                        "unexpected conditioning output file")
                require(8 < st.st_size <= max_bytes, "conditioning output size guard")
                files.append(path)
                require(len(files) <= 1, "multiple conditioning artifacts")
    require(len(files) == 1, "conditioning artifact missing")
    return files[0]


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
            "values_checked": False, "roundtrip_checked": False}



def validate_corpus_graph(corpus_id,case,graph,canonical):
    require(corpus_id in {"bounce-audio","hands-face-mug"},"fixed corpus ID")
    require(case["expected_conditioning_sha256"] is None and case["stageB_ready"] is False,
            "future real conditioning must not be manufactured")
    require(hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()==case["prompt_sha256"],
            "corpus prompt identity")
    expected=copy.deepcopy(canonical)
    expected["prompt"]["104"]["inputs"]["prompt"]=case["prompt"]
    expected["prompt"]["1001"]["inputs"]["filename_prefix"]="h3-quality-corpus/"+corpus_id
    require(graph==expected,"corpus graph changed outside prompt/prefix")

def load_corpus(a,v1,gates):
    root=Path(__file__).resolve().parent
    require(a.corpus_manifest==root/"CORPUS-MANIFEST-v1.json" and
            a.corpus_manifest_sha256==CORPUS_MANIFEST_SHA,"fixed corpus manifest")
    mf=v1.frozen_json(a.corpus_manifest,CORPUS_MANIFEST_SHA,gates["max_schema_bytes"])
    require(mf["schema"]=="q33-quality-corpus-manifest/v1" and set(mf["cases"])=={"bounce-audio","hands-face-mug"},
            "corpus schema")
    case=mf["cases"][a.corpus];gp=root/case["graph"]["file"]
    require(gp.resolve().is_relative_to(root) and not gp.is_symlink(),"corpus graph boundary")
    graph=v1.frozen_json(gp,case["graph"]["sha256"],gates["max_schema_bytes"])
    cp=root/mf["canonical"]["file"]
    canonical=v1.frozen_json(cp,OLD_GRAPH_SHA,gates["max_schema_bytes"])
    validate_corpus_graph(a.corpus,case,graph,canonical)
    protocol_path=a.notes_root/mf["quality_protocol"]["relative_to_notes"]
    require(protocol_path.resolve().is_relative_to(a.notes_root),"protocol path boundary")
    protocol=v1.frozen_json(protocol_path,mf["quality_protocol"]["sha256"],gates["max_schema_bytes"])
    require(next(x["prompt"] for x in protocol["suggested_next_prompts_not_executed"] if x["id"]==a.corpus)==case["prompt"],
            "quality protocol text mismatch")
    return case,graph

def parse():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--mode", choices=["quality-corpus-stage-a"], required=True)
    for name in ("iteration-root", "lab-root", "portable-python", "comfy-root",
                 "run-dir", "stop-marker", "nvidia-smi", "inferrsi-src",
                 "schema-result", "gates", "notes-root", "q31-root", "corpus-manifest",
                 "timing-controller", "source-manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--corpus",choices=["bounce-audio","hands-face-mug"],required=True)
    p.add_argument("--corpus-manifest-sha256",required=True)
    p.add_argument("--source-manifest-sha256",required=True)
    p.add_argument("--schema-result-sha256", required=True)
    p.add_argument("--gates-sha256", required=True)
    p.add_argument("--port", type=int, required=True)
    return p.parse_args()


def main():
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

    require(a.iteration_root==a.notes_root/"iterations/2026-10-31-speed-quality-search" and
            a.q31_root==a.notes_root/"iterations/2026-10-28-comfy-trim-first-video" and
            Path(__file__).resolve().parent==a.iteration_root/"agents/quality-corpus","Q33/Q31 fixed roots")
    require(Path(sys.executable).resolve()==a.portable_python,"use existing portable Python")
    require(a.timing_controller==a.iteration_root/"agents/timing-controller/speed_stage_b_v3.py" and
            digest(a.timing_controller)==TIMING_CONTROLLER_SHA,"reviewed timing controller identity")
    timing_spec=importlib.util.spec_from_file_location("q33_reviewed_timing_helper",a.timing_controller)
    timing=importlib.util.module_from_spec(timing_spec);timing_spec.loader.exec_module(timing)
    sys.dont_write_bytecode = True
    v1_path = a.q31_root / "tools/comfy_first_video_v1.py"
    require(digest(v1_path) == V1_SHA, "v1 dependency drift")
    spec = importlib.util.spec_from_file_location("q31_object_info_v1", v1_path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)  # Imports stdlib only; its main is not called.
    require(a.gates_sha256.lower() == STAGE_GATES_SHA, "Stage-A gates identity mismatch")
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
    case,graph=load_corpus(a,v1,gates)
    graph_sha=case["graph"]["sha256"]
    assets_path = a.q31_root / "plan/ASSET-SET-v1.json"
    asset_set = v1.frozen_json(assets_path, ASSET_SET_SHA, gates["max_schema_bytes"])
    require(a.schema_result_sha256==SCHEMA_RESULT_SHA,"fixed successful schema source")
    baseline = v1.frozen_json(a.schema_result, a.schema_result_sha256, gates["max_schema_bytes"])
    require(a.source_manifest_sha256==SOURCE_MANIFEST_SHA,"Q33 source identity")
    sources=v1.frozen_json(a.source_manifest,SOURCE_MANIFEST_SHA,gates["max_schema_bytes"])
    source_start_ns=time.perf_counter_ns()
    roots={"portable":a.comfy_root.parent,"inferrsi":a.inferrsi_src.parent}
    for row in sources["sources"]:
        root=roots[row["root"]];path=(root/row["path"]).resolve()
        require(path.is_relative_to(root) and path.stat().st_size==row["bytes"] and digest(path)==row["sha256"],
                "frozen runtime source changed")
    source_end_ns=time.perf_counter_ns()
    require(baseline["status"] == "schema_collected_unreviewed"
            and baseline["candidate_sha256"] == BASELINE_GRAPH_SHA
            and baseline["schema_sha256"] == SCHEMA_SHA, "unexpected schema baseline")
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
    state = {"status": "preflight", "claim": "INCONCLUSIVE", "mode": "quality-corpus-stage-a",
             "corpus_id":a.corpus,"prompt_sha256":case["prompt_sha256"],"stageB_ready":False,
             "corpus_manifest_sha256":CORPUS_MANIFEST_SHA,
             "timing_controller_sha256":TIMING_CONTROLLER_SHA,
             "runtime_sources_sha256":SOURCE_MANIFEST_SHA,
             "driver_sha256": digest(Path(__file__)), "v1_sha256": V1_SHA,
             "graph_sha256": graph_sha, "asset_set_sha256": ASSET_SET_SHA,
             "schema_result_sha256": a.schema_result_sha256.lower(),
             "gates_sha256": a.gates_sha256.lower(), "gates": gates,
             "prompt_posts": 0, "http_bytes": 0, "history_polls": 0,
             "verified_assets": [], "gpu_measurement": "nvidia-smi whole-board MiB",
             "cuda_allocator_metrics": None, "conditioning_validation": "not_performed"}
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
        for key, expected in baseline["source_sha256"].items():
            path = a.comfy_root / key.removeprefix("comfy/") if key.startswith("comfy/") else a.inferrsi_src / key
            require(digest(path) == expected, "source changed since CPU schema run")
        for relative, expected in STAGE_SOURCE_SHA256.items():
            require(digest(a.comfy_root / relative) == expected, "Stage-A native source drift")
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
        models_root = (a.lab_root / "comfy-trim/models").resolve()
        require(models_root.drive.upper() == "E:" and models_root.is_dir(), "model root unavailable")
        selected = [x for x in asset_set["assets"] if x["id"] in ASSET_IDS]
        require(len(selected) == 4 and {x["id"] for x in selected} == ASSET_IDS, "four fixed assets required")
        require(all((a.q31_root / ("state/transport/assets/" + x["id"] + ".receipt.json")).is_file()
                    for x in selected), "one or more complete receipts absent; no file hashing started")
        verify_deadline = min(deadline, time.monotonic() + gates["asset_timeout_seconds"])
        for asset in selected:
            plan_path = a.q31_root / ("plan/asset-" + asset["id"] + "-v1.json")
            receipt_path = a.q31_root / ("state/transport/assets/" + asset["id"] + ".receipt.json")
            require(plan_path.is_file() and receipt_path.is_file(), "complete asset plan/receipt missing")
            require(plan_path.stat().st_size <= 65536 and receipt_path.stat().st_size <= 65536, "asset metadata bound")
            plan = json.loads(plan_path.read_bytes(), object_pairs_hook=v1.unique_json)
            receipt = json.loads(receipt_path.read_bytes(), object_pairs_hook=v1.unique_json)
            require(plan == asset and receipt.get("schema") == "q31-asset-receipt/v1"
                    and receipt.get("complete") is True, "asset plan/receipt incomplete")
            for key in ("id", "bytes", "sha256", "relative_path"):
                require(receipt.get(key) == asset[key], "receipt identity mismatch")
            transport_plan_path = a.q31_root / ("state/transport/assets/" + asset["id"] + ".json")
            require(transport_plan_path.is_file() and transport_plan_path.stat().st_size <= 65536,
                    "transport asset plan missing or oversized")
            transport_plan = json.loads(transport_plan_path.read_bytes(), object_pairs_hook=v1.unique_json)
            require(transport_plan == asset, "transport asset plan content mismatch")
            require(receipt.get("plan_sha256") == digest(transport_plan_path), "receipt transport plan hash mismatch")
            path = (a.lab_root / asset["relative_path"]).resolve()
            require(path.drive.upper() == "E:" and path.is_relative_to(models_root)
                    and path.is_file() and path.stat().st_size == asset["bytes"], "asset file/path/size mismatch")
            event("asset_verification_started",asset_id=asset["id"])
            value, signature = guarded_hash(path, verify_deadline)
            require(value == asset["sha256"], "asset full-file SHA256 mismatch")
            asset_stats[path] = signature
            state["verified_assets"].append({"id": asset["id"], "bytes": asset["bytes"],
                                             "sha256": value, "receipt_sha256": digest(receipt_path)})
            event("asset_verified", asset_id=asset["id"])
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
                    and info[name].get("python_module") == module, "Stage-A node source/schema mismatch")
            required = info[name]["input"].get("required", {})
            optional = info[name]["input"].get("optional", {})
            for node in graph["prompt"].values():
                if node["class_type"] == name:
                    require(set(required) <= set(node["inputs"])
                            and set(node["inputs"]) <= set(required) | set(optional),
                            "Stage-A fixed input schema mismatch")
            if name == "SaveConditioning":
                require(info[name].get("output_node") is True
                        and info[name].get("output") == ["CONDITIONING"], "native saver contract mismatch")
            if name in ("CLIPLoader", "VAELoader"):
                field, node_id = ("clip_name", "13") if name == "CLIPLoader" else ("vae_name", "11")
                choices = required[field][0]
                require(graph["prompt"][node_id]["inputs"][field] in choices, "model file not enumerated")
        event("native_schema_verification_finished")
        for path, signature in asset_stats.items():
            st = path.stat()
            require((st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino) == signature, "asset changed before POST")
        sample(force=True, startup=True)
        render_deadline = min(deadline, time.monotonic() + gates["render_timeout_seconds"])
        require(digest(a.corpus_manifest)==CORPUS_MANIFEST_SHA,"corpus manifest changed before POST")
        require(digest(Path(__file__).resolve().parent/case["graph"]["file"])==graph_sha,"corpus graph changed before POST")
        state["client_id"]=uuid.uuid4().hex
        observer=timing.NativeWSObserver(a.port,state["prompt_id"],state["client_id"],
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
                output = find_conditioning_artifact(dirs["output"], gates["max_output_bytes"], sample)
                structure = conditioning_header(output, v1.unique_json, gates["max_output_bytes"])
                output_hash, signature = guarded_hash(output, render_deadline)
                require(signature == structure.pop("file_signature"), "conditioning changed after header check")
                state["artifact"] = {"relative_to_run": str(output.relative_to(a.run_dir)),
                                     "bytes": output.stat().st_size, "sha256": output_hash,
                                     "structure": structure}
                state["conditioning_validation"] = "header_only_values_and_roundtrip_not_checked"
                state["status"] = "conditioning_artifact_received_unvalidated"
                event("conditioning_artifact_received", artifact=state["artifact"])
                break
            time.sleep(min(gates["poll_seconds"], max(0, render_deadline - time.monotonic())))
    except BaseException as exc:
        state["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) or a.stop_marker.exists() else "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc) if type(exc) is RuntimeError else "external operation failed; local raw evidence retained"
        event("stopped", status=state["status"], error_type=type(exc).__name__)
    finally:
        timing.cleanup_owned_v2(job,proc,observer,state,gates,event)
        state["elapsed_seconds"] = time.monotonic() - started
        for stream in (samples, journal, raw_log):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    state.setdefault("cleanup_diagnostics",[]).append("stream_close_"+type(exc).__name__)
        v1.save_new(result_path, state)
    print(json.dumps({"status": state["status"], "claim": state["claim"],
                      "prompt_posts": state["prompt_posts"], "conditioning_validation": "not_performed"}))
    return 0 if state["status"] == "conditioning_artifact_received_unvalidated" else 1


if __name__ == "__main__":
    raise SystemExit(main())