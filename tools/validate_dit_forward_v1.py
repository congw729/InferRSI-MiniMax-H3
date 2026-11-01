"""Frozen v1: complete tiny real DiT forward with on-demand block hooks."""
import argparse
import gc
import json
from pathlib import Path
import resource
import traceback

import torch
from validate_mmap_v2 import fixture, native_load, state
from validate_meta_coverage_v2 import make_model, setup_dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "complete reduced-dimension H3 DiT forward, synthetic input and weights",
              "real_model_quality": False, "service_performance": False, "checks": []}
    code = 1
    loader = None
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)
    try:
        setup_dist(a.output.parent)
        root = a.output.parent / "fixture"
        config, _, _, _ = fixture(a.research, root)
        from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
        from vllm_omni.diffusion.forward_context import set_forward_context
        from h3_omni_adapter.mmap_loader import H3MmapPlan, H3MmapLoader
        quant = FastH3NVFP4Config.from_dit(root / "nvfp4_weights.safetensors", rank=0)
        reference = make_model(config, quant, device="cpu")
        native_load(reference, root)
        reference.eval().to("cuda")
        candidate = make_model(config, quant).eval()
        plan = H3MmapPlan.build(candidate, root, sidecars=["nvfp4_weights.safetensors"])
        loader = H3MmapLoader(candidate, plan, device="cuda", max_group_bytes=32 * 1024**2)
        loader.load_group("resident")
        loader.install_block_hooks()
        od = OmniDiffusionConfig(model=None, model_class_name="MiniMaxH3Pipeline",
            tf_model_config=TransformerConfig.from_dict(config),
            cache_config={"minimax_h3_adaln_cache": False, "minimax_h3_adaln_offload": False})
        torch.manual_seed(19)
        def ids(values):
            return {"position_ids": torch.tensor(values, device="cuda")}
        kwargs = {
            "x": torch.randn(1, 8, reference.video_patch_proj.input_size, device="cuda"),
            "audio_x": torch.randn(1, 8, reference.audio_patch_proj.input_size, device="cuda"),
            "img_position_ids": torch.zeros(1, 8, 3, device="cuda"),
            "unique_timesteps": torch.tensor([0.5], device="cuda"),
            "inverse_indices": torch.zeros(8, dtype=torch.long, device="cuda"),
            "update_mask": torch.tensor([1, 1, 1, 0], device="cuda"),
            "token_tags": torch.tensor([0, 0, 1, 1, 1, 1, 2, 2], device="cuda"),
            "prompt_embeds": torch.randn(2, 128, dtype=torch.bfloat16, device="cuda"),
            "img_pos_info": ids([2, 3, 4, 5]), "audio_pos_info": ids([6, 7]),
            "text_pos_info": ids([0, 1]), "img_pos_for_infer_output_info": ids([2, 3, 4, 5]),
            "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
                                  "max_seqlen_q": 8, "num_requests": 1},
            "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
                                          "max_seqlen_q": 2, "num_requests": 1},
        }
        result["attention_backend"] = reference.blocks[0].attn.attention.attn_backend.get_name()
        with torch.inference_mode(), set_forward_context(omni_diffusion_config=od):
            for step in (0, 1, 0):
                expected = reference(pdd_step=step, **kwargs)
                actual = candidate(pdd_step=step, **kwargs)
                torch.cuda.synchronize()
                check(f"video_audio_exact_step_{step}", all(torch.equal(x, y) for x, y in zip(expected, actual)))
                check(f"finite_outputs_step_{step}", all(torch.isfinite(x).all().item() for x in actual))
                check(f"condition_mask_step_{step}", torch.count_nonzero(actual[0][-1]).item() == 0)
                check(f"only_resident_group_after_step_{step}", loader.active == {"resident"})
                check(f"all_streamed_groups_meta_step_{step}", all(
                    state(candidate)[n].is_meta for group, names in plan.groups.items()
                    if group != "resident" for n in names))
            result["output_shapes"] = [list(x.shape) for x in actual]
        result["loader_stats"] = dict(loader.stats)
        check("all expected groups used each forward", loader.stats["groups_loaded"] == 10)
        loader.close()
        check("close_all_meta", all(t.is_meta for t in state(candidate).values()))
        check("no mapped checkpoint remains", str(root) not in Path("/proc/self/maps").read_text())
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        code = 0
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if loader is not None:
            loader.close()
        try:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            code = 1
        gc.collect()
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["checks_passed"] = sum(c["ok"] for c in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["exit_code"] = code
    text = json.dumps(result, indent=2)
    for path, label in ((str(a.output.parent), "<RUN_ROOT>"), (str(a.research), "<RESEARCH_ROOT>"),
                        (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(path, label)
    a.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
