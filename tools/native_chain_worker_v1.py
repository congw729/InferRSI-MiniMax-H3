"""Frozen v1: actual tiny native models, one reference chain or one streamed stage."""
import argparse
import gc
import json
import resource
import sys
import time
import traceback
import weakref
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("prepare", "reference", "staged"), required=True)
    p.add_argument("--stage", choices=("encode", "denoise", "decode"), required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--research", type=Path)
    p.add_argument("--text-fixture", type=Path)
    p.add_argument("--decoder-fixture", type=Path)
    p.add_argument("--fail", action="store_true")
    p.add_argument("--delay", type=float, default=0)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    report = {"campaign_verdict": "INCONCLUSIVE", "mode": a.mode, "stage": a.stage,
              "model_video_generated": False}
    code = 1
    loader = None
    try:
        if a.delay:
            time.sleep(a.delay)
        if a.fail:
            raise RuntimeError("intentional failure before stage execution")
        sys.path.insert(0, str(a.native_source / "Ref2VA"))
        import native_chain_common_v1 as c
        import torch
        from validate_meta_coverage_v2 import setup_dist

        from h3_omni_adapter.stage_handoff import read_payload, write_payload
        report["initial_cuda_allocated"] = torch.cuda.memory_allocated()
        torch.cuda.set_per_process_memory_fraction(0.25)
        setup_dist(a.work)
        if a.mode == "prepare":
            c.prepare(a.research, a.text_fixture, a.decoder_fixture, a.native_source, a.checkpoint)
            output = {"kind": "prepared_synthetic_fixtures", "real_weights": False}
        elif a.mode == "reference":
            text, _ = c.text_model(a.checkpoint / "text", False)
            hidden = c.text_hidden(text, False)
            dit, _, config = c.dit_model(a.checkpoint / "dit", False)
            sampled = c.denoise(dit, config, hidden)
            decoder = c.decoder_model(a.checkpoint, False)
            decoded = c.decode(decoder, a.checkpoint, sampled, False)
            output = {"hidden": hidden, "sampled": sampled, "decoded": decoded}
            report["attention_backend"] = dit.blocks[0].attn.attention.attn_backend.get_name()
        elif a.stage == "encode":
            model, loader = c.text_model(a.checkpoint / "text", True)
            hidden = c.text_hidden(model, True)
            report["loader_stats"] = c.close_loader(model, loader)
            report["all_meta_after_close"] = all(t.is_meta for t in c.state(model).values())
            output = {"kind": "native_text_hidden", "hidden": hidden}
            del model, loader
            loader = None
        elif a.stage == "denoise":
            incoming = read_payload(a.input)
            if incoming.get("kind") != "native_text_hidden" or tuple(incoming["hidden"].shape) != (2, 128):
                raise ValueError("invalid native text handoff")
            model, loader, config = c.dit_model(a.checkpoint / "dit", True)
            output = c.denoise(model, config, incoming["hidden"])
            report["attention_backend"] = model.blocks[0].attn.attention.attn_backend.get_name()
            report["loader_stats"] = c.close_loader(model, loader)
            report["all_meta_after_close"] = all(t.is_meta for t in c.state(model).values())
            del model, loader
            loader = None
        else:
            incoming = read_payload(a.input)
            model = c.decoder_model(a.checkpoint, True)
            refs = [weakref.ref(t) for t in c.state(model).values()]
            output = c.decode(model, a.checkpoint, incoming, True)
            del model
            gc.collect()
            report["all_decoder_tensor_objects_released"] = all(r() is None for r in refs)
        gc.collect()
        report["no_checkpoint_mmaps"] = str(a.checkpoint) not in Path("/proc/self/maps").read_text()
        write_payload(a.work / "handoff", output)
        torch.cuda.synchronize()
        report["cuda_peak_allocated"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved"] = torch.cuda.max_memory_reserved()
        report["completed"] = True
        code = 0
    except Exception as exc:
        report["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            if loader is not None:
                loader.close()
            import torch
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            if torch.distributed.is_initialized():
                destroy_model_parallel()
                torch.distributed.destroy_process_group()
        except Exception as exc:
            report["cleanup_error"] = str(exc)
            code = 1
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    text = json.dumps(report, indent=2)
    for source, label in ((str(a.work), "<STAGE_WORK>"), (str(a.checkpoint), "<SYNTHETIC_CHECKPOINT>"),
            (str(a.native_source), "<NATIVE_SOURCE>"), (str(a.research), "<RESEARCH_ROOT>"),
            (str(a.text_fixture), "<TEXT_FIXTURE>"), (str(a.decoder_fixture), "<DECODER_FIXTURE>"),
            (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        if source != "None":
            text = text.replace(source, label)
    (a.work / "worker-result.json").write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
