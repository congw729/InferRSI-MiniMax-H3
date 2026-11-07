"""Offline contract checks. No ComfyUI, model loading or GPU is used."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import run_h3 as app


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.preset = app.load_json(Path(__file__).with_name("trim8.json"))

    def test_graph_uses_new_prompt_seed_and_unique_owned_paths(self):
        before = copy.deepcopy(self.preset)
        token = "a" * 32
        text, video = app.build_graphs(self.preset, "A blue bird flies.", 17, token, "owned.safetensors")
        self.assertEqual(text["104"]["inputs"]["prompt"], "A blue bird flies.")
        self.assertEqual(video["15"]["inputs"]["noise_seed"], 17)
        self.assertEqual(video["1001"]["inputs"]["conditioning_name"], "owned.safetensors")
        self.assertIn(token, text["1001"]["inputs"]["filename_prefix"])
        self.assertIn(token, video["92"]["inputs"]["filename_prefix"])
        self.assertEqual(self.preset, before)
        self.assertEqual(video["9"]["inputs"]["steps"], 8)
        self.assertEqual(video["128"]["inputs"]["attention"], "comfy kitchen attention")

    def test_reject_empty_prompt_and_invalid_seed(self):
        for prompt, seed in [("", 0), (" ", 0), ("ok", -1), ("ok", 2**64)]:
            with self.subTest(prompt=prompt, seed=seed), self.assertRaises(app.PreviewError):
                app.build_graphs(self.preset, prompt, seed, "a" * 32, "owned.safetensors")

    def test_only_local_server_urls(self):
        for value in ["http://example.com:8188", "http://127.0.0.1:8188/extra", "http://a:b@localhost:8188", "file:///tmp/test", "http://localhost", "http://localhost:abc"]:
            with self.subTest(value=value), self.assertRaises(app.PreviewError):
                app.validate_url(value)
        self.assertEqual(app.validate_url("http://127.0.0.1:8188/"), "http://127.0.0.1:8188")

    def test_queue_busy_blocks_work(self):
        class Busy:
            def call(self, *args):
                return {"queue_running": [[1, "someone-else"]], "queue_pending": []}
        with self.assertRaises(app.PreviewError):
            app.ensure_idle(Busy())

    def test_prompt_id_is_canonical_and_history_is_owned(self):
        class Fake:
            def call(self, method, route, value=None):
                if route == "/queue":
                    return {}
                if route == "/prompt":
                    self.prompt_id = value["prompt_id"]
                    if str(uuid.UUID(self.prompt_id)) != self.prompt_id:
                        raise AssertionError("ComfyUI rejects non-canonical UUID")
                    return {"prompt_id": self.prompt_id, "node_errors": {}}
                if route == "/history/" + self.prompt_id:
                    return {self.prompt_id: {"status": {"status_str": "success", "completed": True,
                            "messages": [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 2500}]]}}}
                raise AssertionError(route)
        record = {"run_id": "a" * 32, "stages": []}
        app.run_stage(Fake(), {}, "test", 1, record)
        self.assertEqual(record["stages"][0]["native_execution_seconds"], 1.5)
        self.assertEqual(record["stages"][0]["status"], "success")

    def test_failed_comfy_execution_is_not_success(self):
        class Failed:
            def call(self, method, route, value=None):
                if route == "/queue":
                    return {}
                if route == "/prompt":
                    self.prompt_id = value["prompt_id"]
                    return {"prompt_id": self.prompt_id}
                return {self.prompt_id: {"status": {"completed": False, "status_str": "error"}}}
        record = {"run_id": "a" * 32, "stages": []}
        with self.assertRaises(app.PreviewError):
            app.run_stage(Failed(), {}, "test", 1, record)
        self.assertEqual(record["stages"][0]["status"], "error")

    def test_cancellation_is_prompt_scoped(self):
        calls = []
        class Fake:
            def call(self, method, route, value=None):
                calls.append((method, route, value))
                return {}
        record = {"stages": [{"prompt_id": "owned", "status": "running"},
                              {"prompt_id": "completed", "status": "success"}]}
        self.assertEqual(app.cancel_owned(Fake(), record), [])
        self.assertEqual(calls, [("POST", "/queue", {"delete": ["owned"]}),
                                 ("POST", "/interrupt", {"prompt_id": "owned"})])

    def test_cleanup_does_not_release_other_queued_work(self):
        class Fake:
            def call(self, method, route, value=None):
                self.asserted_route = route
                if route != "/queue":
                    raise AssertionError("must not free another client's work")
                return {"queue_running": [[1, "someone-else"]]}
        record = {"stages": [{"prompt_id": "owned", "status": "cancellation_requested"}]}
        app.cleanup_failed_run(Fake(), record, {})
        self.assertTrue(record["owned_prompts_no_longer_queued"])
        self.assertEqual(record["cleanup_status"], "skipped_other_work_queued")
        self.assertEqual(record["stages"][0]["status"], "no_longer_queued")

    def test_cleanup_observes_own_request_exit_before_release(self):
        class Fake:
            def call(self, method, route, value=None):
                return {}
        record = {"stages": [{"prompt_id": "owned", "status": "cancellation_requested"}]}
        with patch.object(app, "release_models", return_value={"checked": True}) as release:
            app.cleanup_failed_run(Fake(), record, {"idle": True})
        release.assert_called_once()
        self.assertEqual(record["cleanup_status"], "completed")

    def test_warm_floor_does_not_require_full_cold_ram(self):
        stats = {"ram_free": 4 * app.GIB, "commit_free": 8 * app.GIB, "vram_free": 8 * app.GIB}
        app.check_headroom(stats, initial=False)
        with self.assertRaises(app.PreviewError):
            app.check_headroom(stats)

    def test_artifact_cannot_escape_owned_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ["../private.mp4", "/etc/private", "C:/private.mp4", "other-run/file.mp4", "owned/../private.mp4"]:
                with self.subTest(relative=relative), self.assertRaises(app.PreviewError):
                    app.contained_file(root, relative, "owned")
            (root / "owned").mkdir()
            (root / "owned/file.mp4").write_bytes(b"video")
            self.assertEqual(app.contained_file(root, "owned/file.mp4", "owned"), root / "owned/file.mp4")

    def test_video_must_come_from_matching_save_node_and_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            token = "b" * 32
            target = root / f"h3-preview/{token}/video/clip.mp4"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"video")
            history = {"outputs": {"92": {"images": [{"type": "output", "filename": "clip.mp4",
                         "subfolder": f"h3-preview/{token}/video"}]}}}
            self.assertEqual(app.find_video(history, token, root), target)
            with self.assertRaises(app.PreviewError):
                app.find_video(history, "c" * 32, root)

    def test_windows_video_subfolder_is_normalized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            token = "b" * 32
            target = root / f"h3-preview/{token}/video/clip.mp4"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"video")
            history = {"outputs": {"92": {"images": [{"type": "output", "filename": "clip.mp4",
                         "subfolder": f"h3-preview\\{token}\\video"}]}}}
            self.assertEqual(app.find_video(history, token, root), target)

    def test_windows_video_subfolder_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            token = "b" * 32
            for subfolder in [f"h3-preview\\{token}\\video\\..\\private", "C:\\private",
                              "\\\\server\\share", "\\private", "..\\private"]:
                history = {"outputs": {"92": {"images": [{"type": "output", "filename": "clip.mp4",
                             "subfolder": subfolder}]}}}
                with self.subTest(subfolder=subfolder), self.assertRaises(app.PreviewError):
                    app.find_video(history, token, root)
            for filename in ["..\\clip.mp4", "../clip.mp4", "folder\\clip.mp4", "folder/clip.mp4"]:
                history = {"outputs": {"92": {"images": [{"type": "output", "filename": filename,
                             "subfolder": f"h3-preview\\{token}\\video"}]}}}
                with self.subTest(filename=filename), self.assertRaises(app.PreviewError):
                    app.find_video(history, token, root)

    def test_models_hash_validation_rejects_wrong_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.txt").write_bytes(b"abcd")
            data = {"assets": [{"filename": "model.txt", "relative_path": "model.txt", "bytes": 4,
                               "sha256": hashlib.sha256(b"abcd").hexdigest()}]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(data))
            self.assertEqual(app.verify_weights(root, manifest), ["model.txt"])
            (root / "model.txt").write_bytes(b"abce")
            with self.assertRaises(app.PreviewError):
                app.verify_weights(root, manifest)

    def test_combo_formats(self):
        for spec in [[["a", "b"], {}], ["COMBO", {"options": ["a", "b"]}]]:
            self.assertEqual(app.combo_options({"input": {"required": {"file": spec}}}, "file"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
