"""Frozen CPU API audit: selected FastH3 request and text presentation contracts."""
import argparse
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--source-record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result = {"campaign_verdict": "INCONCLUSIVE", "checks": [], "cases": [],
              "scope": "CPU native request gates; tokenizer spy, no vocabulary or model weights",
              "weights_loaded": False, "real_tokenizer_validated": False}
    def check(name, ok, **detail):
        result["checks"].append({"name": name, "ok": bool(ok), **detail})
        if not ok:
            raise AssertionError(name)
    def rejected(name, fn, words):
        try:
            fn()
        except Exception as exc:
            message = str(exc)
            result["cases"].append({"name": name, "status": "rejected",
                                    "type": type(exc).__name__, "message": message})
            check(name, isinstance(exc, OmniClientError) and all(w in message for w in words))
        else:
            check(name, False)
    def sampling(task):
        return SimpleNamespace(fps=24, height=256, width=256, num_frames=96,
            extra_args={"task": task, "duration_seconds": 4, "aspect_ratio": "1:1"},
            num_inference_steps=8, timesteps=None, sigmas=None, lora_request=None,
            guidance_scale=1.0)
    try:
        import torch
        from PIL import Image
        from vllm_omni.errors import OmniClientError
        from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import (
            FastH3CheckpointSpec, FastH3OmniCheckpointSpec)
        from vllm_omni.model_executor.models.minimax_h3.encoder_processing import prepare_encoder_inputs
        from vllm_omni.model_executor.models.minimax_h3.preprocessing import build_minimax_h3_presentation

        source_expected = {}
        for folder in (a.source_record, a.source_record / "extra"):
            record = json.loads((folder / "source-audit.json").read_text())
            source_expected.update({k:v["sha256"] for k,v in record["files"].items()})
        for obj in (FastH3CheckpointSpec, prepare_encoder_inputs, build_minimax_h3_presentation):
            file = Path(inspect.getfile(obj))
            rel = "vllm_omni/" + file.as_posix().split("/vllm_omni/", 1)[1]
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            check("source_" + file.stem, source_expected.get(rel) == digest, sha256=digest)

        metadata = json.loads((a.research / "config-fastvideo_inference.json").read_text())
        spec = FastH3CheckpointSpec.from_metadata(metadata["content"])
        check("selected_metadata_is_omni_ref2va", type(spec) is FastH3OmniCheckpointSpec and spec.partition == "ref2va")
        rejected("selected_checkpoint_rejects_t2va",
                 lambda: spec.check_request(sampling("t2va")), ["Ref2VA", "only"])
        spec.check_request(sampling("ref2va"))
        check("selected_checkpoint_accepts_ref2va_sampling", True)
        prompt = "A small paper boat floating on a quiet pond."
        rejected("native_ref2va_rejects_prompt_without_reference",
                 lambda: prepare_encoder_inputs(prompt, sampling("ref2va"), task="ref2va"),
                 ["at least one image or video reference"])
        prepared = prepare_encoder_inputs(prompt, sampling("t2va"), task="t2va")
        media = prepared.media
        geometry = {n: getattr(media, n) for n in ("task", "height", "width", "num_frames", "latent_t", "audio_t")}
        result["t2va_geometry_only"] = geometry
        check("base_t2va_geometry_does_not_qualify_checkpoint",
              geometry == dict(task="t2va", height=256, width=256, num_frames=107, latent_t=32, audio_t=178))
        picture = Image.new("RGB", (256,256), color=(128,128,128))
        image_prepared = prepare_encoder_inputs(
            {"prompt": prompt, "multi_modal_data":{"image":picture}}, sampling("ref2va"), task="ref2va")
        check("native_ref2va_accepts_one_synthetic_reference",
              image_prepared.media.task == "ref2va" and len(image_prepared.images) == 1
              and image_prepared.condition_labels == [("image",1)])
        class TokenizerSpy:
            def __init__(self):
                self.calls = []
            def __call__(self, text, **kwargs):
                self.calls.append({"text":text, "kwargs":kwargs})
                return {"input_ids":[ord(c)+100 for c in text]}
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("generic chat template must not run")
        for i, text in enumerate(("Boat.", "船在水上缓慢漂浮。" * 12)):
            tokenizer = TokenizerSpy()
            ids, tags = build_minimax_h3_presentation(tokenizer, prompt=text, task="t2va",
                condition_labels=[], image_grid_thw=None, video_grid_thw=None,
                video_timestamps=None, merge_size=2)
            check("verbatim_no_special_or_chat_" + str(i),
                  tokenizer.calls == [{"text":text, "kwargs":{"add_special_tokens":False}}])
            check("dynamic_full_token_length_" + str(i),
                  ids.tolist() == [ord(c)+100 for c in text]
                  and ids.numel() == len(text) and tags.tolist() == [1]*len(text))
        tokenizer = TokenizerSpy()
        ids, tags = build_minimax_h3_presentation(tokenizer, prompt=prompt, task="ref2va",
            condition_labels=[], image_grid_thw=None, video_grid_thw=None,
            video_timestamps=None, merge_size=2)
        check("low_level_ref_presentation_accepts_empty_references_but_request_gate_does_not",
              ids.numel() == len(prompt) and tags.tolist() == [1]*len(prompt))
        bad = sampling("ref2va")
        bad.num_inference_steps = 4
        rejected("selected_release_rejects_four_steps", lambda:spec.check_request(bad), ["requires num_inference_steps=8"])
        check("no_cuda_context_initialized", not torch.cuda.is_initialized())
        result["goal_request_status"] = "BLOCKED_BY_SELECTED_CHECKPOINT_NATIVE_REQUEST_CONTRACT"
        result["implication"] = "Do not treat a pure-text synthetic component chain as native prompt-only support."
    except BaseException as exc:
        trace = traceback.format_exc()
        for source, label in ((str(a.source_record),"<ITERATION_ROOT>"),
                              (str(a.research),"<RESEARCH_ROOT>"),
                              (str(Path.home()),"<LINUX_HOME>"),
                              (str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
            trace = trace.replace(source,label)
        result["error"] = {"type":type(exc).__name__, "traceback":trace}
    result["checks_total"] = len(result["checks"])
    result["checks_ok"] = sum(x["ok"] for x in result["checks"])
    result["exit_code"] = int("error" in result or result["checks_ok"] != result["checks_total"])
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2) + "\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","error","cases"}},ensure_ascii=False))
    return result["exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
