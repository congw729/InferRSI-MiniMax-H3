"""Q31 fixed first-clip CPU media inspection; prepared only, never a quality PASS.

Run with the existing E-drive portable Python. The parent process supervises one
owned decoder Job; PyAV/NumPy/Pillow are imported only in the decoder child.
No source image edits: first/middle/last are unscaled decoded RGB24 frames.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

EXPECTED = {"width": 576, "height": 384, "frames": 73, "fps": 24}
MONITOR_SHA = "63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35"
PROCESS_SHA = "3b96ebc7a50913e78e86ff3dcba74c8ee8d03d547d9591ca6488f390ac1e6189"
GATE_KEYS = {
    "max_input_bytes", "max_video_frames", "max_audio_frames",
    "max_audio_samples_per_channel", "max_audio_samples_per_frame",
    "max_audio_sample_rate", "max_audio_channels", "max_packets",
    "max_png_bytes", "max_report_bytes", "timeout_seconds", "poll_seconds",
    "cleanup_seconds", "start_physical_bytes", "start_commit_bytes",
    "running_physical_bytes", "running_commit_bytes", "max_owned_rss_bytes",
    "max_owned_private_bytes", "min_free_disk_bytes", "max_owned_processes",
}


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, data, limit):
    raw = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= limit, "report byte cap")
    with path.open("xb") as f:
        f.write(raw)


def signature(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino


def parse():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("input", "run-root", "output-dir", "gates", "monitor-source", "inferrsi-src"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--expected-sha256", required=True,
                   help="MP4 SHA256 from the completed driver artifact record")
    p.add_argument("--gates-sha256", required=True)
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def inputs(a):
    require(os.name == "nt", "Windows only")
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    require(Path(sys.executable).resolve().drive.upper() == "E:", "use E portable Python")
    require(a.run_root.drive.upper() == "E:" and a.run_root.is_dir(), "E run root required")
    require(a.input.is_relative_to(a.run_root) and a.input.drive.upper() == "E:"
            and a.input.suffix.lower() == ".mp4" and a.input.is_file(), "MP4 outside this E run")
    require(a.output_dir.drive.upper() == "E:" and a.output_dir.is_relative_to(a.run_root)
            and a.output_dir != a.run_root and a.output_dir.parent.is_dir(),
            "output must be a new child of this E run")
    require(re.fullmatch("[0-9a-fA-F]{64}", a.expected_sha256), "expected artifact SHA required")
    require(a.gates.is_file() and a.gates.stat().st_size <= 65536, "gate file cap")
    raw = a.gates.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == a.gates_sha256.lower(), "gate identity")
    g = json.loads(raw, object_pairs_hook=unique)
    require(type(g) is dict and set(g) == GATE_KEYS, "explicit gate schema")
    for k, v in g.items():
        require(type(v) in (int, float) and math.isfinite(v) and v > 0, "invalid gate")
        if not k.endswith("_seconds"):
            require(type(v) is int, "integer count/byte gate required")
    require(g["max_video_frames"] == EXPECTED["frames"], "fixed 73-frame cap")
    require(g["poll_seconds"] <= 1 and g["poll_seconds"] < g["timeout_seconds"],
            "poll interval bound")
    require(g["max_owned_processes"] <= 16, "process count cap")
    require(g["start_physical_bytes"] >= g["running_physical_bytes"]
            and g["start_commit_bytes"] >= g["running_commit_bytes"], "startup floors")
    require(0 < a.input.stat().st_size <= g["max_input_bytes"], "input byte cap")
    return g


class CappedPNG:
    """Pillow writes via this object; no fileno is exposed to bypass the cap."""
    def __init__(self, f, limit):
        self.f, self.limit, self.count = f, limit, 0

    def write(self, data):
        require(self.count + len(data) <= self.limit, "PNG byte cap")
        written = self.f.write(data)
        self.count += written
        return written

    def flush(self):
        self.f.flush()

    def tell(self):
        return self.count


def decode(a, g):
    start = time.monotonic()
    deadline = start + g["timeout_seconds"]
    state = {"status": "starting", "claim": "INCONCLUSIVE", "semantic_quality": "not_assessed",
             "expected": EXPECTED, "expected_input_sha256": a.expected_sha256.lower(),
             "source_sha256": sha(Path(__file__)), "gates_sha256": a.gates_sha256.lower(),
             "video_frames": [], "audio_frames": [], "pngs": [], "full_demux_eof": False,
             "container_closed": False, "input_unchanged": False}
    stage, container, handle = "input_hash", None, None

    def check():
        require(time.monotonic() < deadline, "decoder deadline")

    def rational(value):
        return str(value) if value is not None else None

    try:
        before = signature(a.input)
        handle = a.input.open("rb")
        h = hashlib.sha256()
        hashed_bytes = 0
        while True:
            check()
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            hashed_bytes += len(block)
            require(hashed_bytes <= g["max_input_bytes"], "input grew beyond byte cap")
            h.update(block)
        require(h.hexdigest() == a.expected_sha256.lower(), "driver artifact SHA mismatch")
        require(signature(a.input) == before, "input changed during hashing")
        state["input_bytes"], state["input_sha256"] = before[0], h.hexdigest()
        handle.seek(0)
        stage = "imports"
        import av
        import numpy as np
        from PIL import Image
        state["versions"] = {"av": av.__version__, "numpy": np.__version__}
        stage = "open"
        # File object + fixed MOV/MP4 demuxer; external MOV data references disabled.
        container = av.open(handle, mode="r", format="mov",
                            options={"enable_drefs": "0", "use_absolute_path": "0"})
        require(len(container.streams) == 2 and len(container.streams.video) == 1
                and len(container.streams.audio) == 1, "exactly one video and one audio stream required")
        vs, aus = container.streams.video[0], container.streams.audio[0]
        for stream in (vs, aus):
            stream.codec_context.thread_count = 1
        require((vs.codec_context.width, vs.codec_context.height) == (576, 384),
                "coded geometry differs from fixed graph")
        require(vs.average_rate == Fraction(24, 1), "reported average rate differs from 24 fps")
        state["streams"] = {
            "video": {"codec": vs.codec_context.name, "time_base": rational(vs.time_base),
                      "average_rate": rational(vs.average_rate), "duration_ticks": vs.duration,
                      "start_time_ticks": vs.start_time, "declared_frames": vs.frames},
            "audio": {"codec": aus.codec_context.name, "time_base": rational(aus.time_base),
                      "duration_ticks": aus.duration, "start_time_ticks": aus.start_time},
        }
        state["container_duration_seconds"] = (
            container.duration / av.time_base if container.duration is not None else None)
        stage = "decode"
        previous = None
        previous_time = None
        audio_last_time = None
        audio_rate = audio_channels = None
        audio_samples = packets = 0
        video_min, video_max, audio_min, audio_max = 255, 0, None, None
        adjacent = []
        audio_square_sum, audio_values = 0.0, 0
        audio_timeline_start = audio_timeline_end = video_start = video_end = None
        for packet in container.demux(vs, aus):
            check()
            packets += 1
            require(packets <= g["max_packets"], "packet count cap")
            require(not packet.is_corrupt, "corrupt packet")
            for frame in packet.decode():
                check()
                require(not frame.is_corrupt, "corrupt decoded frame")
                require(frame.pts is not None and frame.time_base is not None, "missing frame PTS/timebase")
                t = Fraction(frame.pts) * frame.time_base
                if packet.stream.type == "video":
                    n = len(state["video_frames"])
                    require(n < g["max_video_frames"], "video frame cap")
                    require((frame.width, frame.height) == (576, 384), "decoded geometry")
                    require(previous_time is None or t - previous_time == Fraction(1, 24),
                            "decoded PTS cadence differs from fixed 24 fps")
                    rgb = frame.to_ndarray(format="rgb24")
                    require(rgb.shape == (384, 576, 3) and rgb.dtype == np.uint8, "RGB frame shape/dtype")
                    require(bool(np.isfinite(rgb).all()), "nonfinite video values")
                    mn, mx = int(rgb.min()), int(rgb.max())
                    video_min, video_max = min(video_min, mn), max(video_max, mx)
                    row = {"index": n, "pts": frame.pts, "time_base": rational(frame.time_base),
                           "time_seconds": float(t), "duration_ticks": frame.duration,
                           "width": frame.width, "height": frame.height,
                           "decoded_format": frame.format.name, "rgb_min": mn, "rgb_max": mx,
                           "spatially_constant_rgb": bool(np.all(rgb == rgb[0, 0, :]))}
                    if previous is not None:
                        diff = np.abs(rgb.astype(np.int16) - previous.astype(np.int16))
                        adjacent.append({"to_frame": n, "mean_abs_rgb": float(diff.mean()),
                                         "max_abs_rgb": int(diff.max()),
                                         "changed_pixel_fraction": float(np.any(diff != 0, axis=2).mean())})
                        del diff
                    if n in (0, 36, 72):
                        name = {0: "first.png", 36: "middle.png", 72: "last.png"}[n]
                        with (a.output_dir / name).open("xb") as f:
                            Image.fromarray(rgb).save(CappedPNG(f, g["max_png_bytes"]), format="PNG")
                        state["pngs"].append({"name": name, "frame": n,
                                             "bytes": (a.output_dir / name).stat().st_size,
                                             "sha256": sha(a.output_dir / name),
                                             "operation": "decoded RGB24, no resize/crop/enhancement"})
                    state["video_frames"].append(row)
                    previous, previous_time = rgb, t
                    if video_start is None:
                        video_start = t
                    video_end = t + Fraction(1, 24)
                else:
                    n = len(state["audio_frames"])
                    require(n < g["max_audio_frames"], "audio frame cap")
                    rate, channels = frame.sample_rate, len(frame.layout.channels)
                    require(0 < rate <= g["max_audio_sample_rate"]
                            and 0 < channels <= g["max_audio_channels"], "audio format cap")
                    require(0 < frame.samples <= g["max_audio_samples_per_frame"], "audio frame sample cap")
                    require(audio_rate is None or (rate, channels) == (audio_rate, audio_channels),
                            "audio rate/channel changes")
                    require(audio_last_time is None or t > audio_last_time, "non-increasing audio PTS")
                    audio_samples += frame.samples
                    require(audio_samples <= g["max_audio_samples_per_channel"], "audio total sample cap")
                    values = frame.to_ndarray()
                    require(values.size == frame.samples * channels, "audio decoded array sample count")
                    require(bool(np.isfinite(values).all()), "nonfinite audio values")
                    mn, mx = float(values.min()), float(values.max())
                    audio_min = mn if audio_min is None else min(audio_min, mn)
                    audio_max = mx if audio_max is None else max(audio_max, mx)
                    dv = values.astype(np.float64)
                    audio_square_sum += float(np.square(dv).sum())
                    audio_values += dv.size
                    require(math.isfinite(audio_square_sum), "nonfinite audio statistic")
                    state["audio_frames"].append({
                        "index": n, "pts": frame.pts, "time_base": rational(frame.time_base),
                        "time_seconds": float(t), "samples_per_channel": frame.samples,
                        "sample_rate": rate, "channels": channels, "layout": frame.layout.name,
                        "sample_format": frame.format.name})
                    if audio_timeline_start is None:
                        audio_timeline_start = t
                    audio_timeline_end = t + Fraction(frame.samples, rate)
                    audio_last_time, audio_rate, audio_channels = t, rate, channels
                    del values, dv
        state["full_demux_eof"] = True
        require(len(state["video_frames"]) == 73 and audio_samples > 0, "missing expected video/audio content")
        require(len(state["pngs"]) == 3, "three inspection frames required")
        state["video"] = {
            "decoded_frames": 73, "width": 576, "height": 384, "pts_rate": "24",
            "pts_start_seconds": float(video_start), "pts_end_with_one_frame_seconds": float(video_end),
            "decoded_duration_seconds": float(video_end - video_start),
            "all_values_finite": True, "not_all_black": video_max > 0,
            "not_spatially_constant_somewhere": any(not x["spatially_constant_rgb"] for x in state["video_frames"]),
            "not_temporally_constant": any(x["max_abs_rgb"] > 0 for x in adjacent),
            "adjacent_frame_differences": adjacent,
            "rgb_min": video_min, "rgb_max": video_max,
        }
        state["audio"] = {
            "sample_rate": audio_rate, "channels": audio_channels,
            "samples_per_channel": audio_samples,
            "sample_count_duration_seconds": audio_samples / audio_rate,
            "pts_start_seconds": float(audio_timeline_start),
            "pts_end_seconds": float(audio_timeline_end),
            "pts_span_seconds": float(audio_timeline_end - audio_timeline_start),
            "all_values_finite": True, "native_value_min": audio_min, "native_value_max": audio_max,
            "native_value_rms": math.sqrt(audio_square_sum / audio_values),
            "not_all_zero_native_samples": audio_min != 0 or audio_max != 0,
            "not_constant": audio_min != audio_max,
        }
        state["av_start_delta_seconds"] = float(audio_timeline_start - video_start)
        state["av_end_delta_seconds"] = float(audio_timeline_end - video_end)
        state["av_sync_quality"] = "reported offsets only; not perceptual or synchronization PASS"
        require(signature(a.input) == before, "input changed during decoding")
        state["input_unchanged"] = True
        state["packets_demuxed"] = packets
        state["status"] = "decoded_structure_checked_human_review_pending"
    except BaseException as exc:
        state["status"] = "failed"
        state["failure"] = {"stage": stage, "type": type(exc).__name__,
                            "reason": str(exc) if type(exc) is RuntimeError else "external decoder operation failed"}
        if type(getattr(exc, "errno", None)) is int:
            state["failure"]["errno"] = exc.errno
    finally:
        if container is not None:
            try:
                container.close()
                state["container_closed"] = True
            except BaseException as exc:
                state["status"] = "failed"
                state["close_error_type"] = type(exc).__name__
        if handle is not None:
            handle.close()
        state["elapsed_seconds"] = time.monotonic() - start
        save_new(a.output_dir / "decode-result.json", state, g["max_report_bytes"])
    return 0 if state["status"] == "decoded_structure_checked_human_review_pending" else 1


def supervise(a, g):
    require(not a.output_dir.exists(), "refuse existing output directory")
    require(sha(a.monitor_source) == MONITOR_SHA, "monitor source drift")
    process_path = a.inferrsi_src / "inferrsi/infra/processes.py"
    require(sha(process_path) == PROCESS_SHA, "ownership source drift")
    spec = importlib.util.spec_from_file_location("q31_media_monitor", a.monitor_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(a.inferrsi_src))
    from inferrsi.infra import processes
    require(Path(processes.__file__).resolve() == process_path, "ownership import drift")
    mon = module.WinMonitor()
    import shutil
    start = time.monotonic()
    deadline = start + g["timeout_seconds"]
    state = {"claim": "INCONCLUSIVE", "status": "preflight", "semantic_quality": "not_assessed",
             "source_sha256": sha(Path(__file__)), "gates_sha256": a.gates_sha256.lower(),
             "expected_input_sha256": a.expected_sha256.lower(), "owned_root_exited": False}
    job = proc = None
    samples = None

    def sample(startup=False):
        require(time.monotonic() < deadline, "hard wall-clock deadline")
        m = mon.memory()
        prefix = "start_" if startup else "running_"
        require(m["available_physical_bytes"] >= g[prefix + "physical_bytes"], "physical floor")
        require(m["available_commit_bytes"] >= g[prefix + "commit_bytes"], "commit floor")
        require(shutil.disk_usage(a.output_dir.parent).free >= g["min_free_disk_bytes"], "E disk floor")
        owned = mon.owned(job, g["max_owned_processes"]) if job else []
        rss, private = sum(x["rss_bytes"] for x in owned), sum(x["private_bytes"] for x in owned)
        require(rss <= g["max_owned_rss_bytes"] and private <= g["max_owned_private_bytes"], "owned memory cap")
        state["sampled_peak_owned_rss_bytes"] = max(state.get("sampled_peak_owned_rss_bytes", 0), rss)
        state["sampled_peak_owned_private_bytes"] = max(state.get("sampled_peak_owned_private_bytes", 0), private)
        if samples is not None:
            samples.write(json.dumps({"elapsed_seconds": time.monotonic() - start, **m,
                                      "owned_rss_bytes": rss, "owned_private_bytes": private}) + "\n")
            samples.flush()

    try:
        sample(startup=True)
        a.output_dir.mkdir()
        samples = (a.output_dir / "resources.jsonl").open("x", encoding="utf-8")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        cache = a.output_dir / "cache"
        cache.mkdir()
        for key in ("TEMP", "TMP", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            env[key] = str(cache)
        env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        job = processes._WindowsJob()
        command = [sys.executable, "-s", "-B", str(Path(__file__).resolve())]
        for key in ("input", "run_root", "output_dir", "gates", "monitor_source", "inferrsi_src",
                    "expected_sha256", "gates_sha256"):
            command += ["--" + key.replace("_", "-"), str(getattr(a, key))]
        command += ["--_worker"]
        proc = subprocess.Popen(command, cwd=a.output_dir, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
        job.attach(proc)
        state["status"] = "decoding"
        while proc.poll() is None:
            sample()
            time.sleep(min(g["poll_seconds"], max(0, deadline - time.monotonic())))
        state["worker_returncode"] = proc.returncode
        p = a.output_dir / "decode-result.json"
        require(p.is_file() and p.stat().st_size <= g["max_report_bytes"], "decoder result missing/oversized")
        result = json.loads(p.read_bytes(), object_pairs_hook=unique)
        require(proc.returncode == 0
                and result["status"] == "decoded_structure_checked_human_review_pending"
                and result["source_sha256"] == state["source_sha256"]
                and result["gates_sha256"] == state["gates_sha256"]
                and result["input_sha256"] == a.expected_sha256.lower()
                and result["full_demux_eof"] and result["container_closed"]
                and result["input_unchanged"], "decoder result failed or identity mismatch")
        state["status"] = "inspection_ready_human_review_pending"
        state["decoder_result_sha256"] = sha(p)
    except BaseException as exc:
        state["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc) if type(exc) is RuntimeError else "external supervision operation failed"
    finally:
        if job is not None:
            job.close()
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=g["cleanup_seconds"])
                state["owned_root_exited"] = True
            except (OSError, subprocess.TimeoutExpired):
                state["status"] = "cleanup_inconclusive"
        state["job_close_requested"] = job is not None
        state["elapsed_seconds"] = time.monotonic() - start
        if samples is not None:
            samples.close()
        # Preflight failure before mkdir leaves no existing path modified.
        if a.output_dir.is_dir():
            save_new(a.output_dir / "validation-summary.json", state, g["max_report_bytes"])
    print(json.dumps({"status": state["status"], "claim": "INCONCLUSIVE",
                      "semantic_quality": "not_assessed"}))
    return 0 if state["status"] == "inspection_ready_human_review_pending" else 1


def main():
    a = parse()
    g = inputs(a)
    return decode(a, g) if a._worker else supervise(a, g)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "preflight_failed", "claim": "INCONCLUSIVE",
                          "error_type": type(exc).__name__}))
        raise SystemExit(1)
