"""Frozen v1: complete tiny real DiT forward with on-demand block hooks."""
import argparse
import gc
import json
from pathlib import Path
import resource
import traceback

import torch
from native_dense_worker_v5 import make, state, setup_dist
from safetensors import safe_open
import hashlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture-trial",type=Path,required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict": "INCONCLUSIVE", "scope": "original dense H3 2 blocks + 2 refiners, 3 independent 8-token velocity forwards; no PDD or sampling/video",
              "real_model_quality": False, "service_performance": False, "checks": []}
    code = 1
    loader = None
    def check(name, ok):
        result["checks"].append({"name": name, "ok": bool(ok)})
        if not ok:
            raise AssertionError(name)
    try:
        torch.cuda.set_per_process_memory_fraction(0.40)
        if torch.cuda.mem_get_info()[0]<2*1024**3:
            raise RuntimeError("insufficient initial CUDA free memory")
        setup_dist(a.output.parent)
        root=a.fixture_trial/"fixture"
        manifest=json.loads((a.fixture_trial/"fixture-manifest.json").read_text())
        check("reused_fixture_hashes",all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in manifest.items()))
        config=json.loads((a.research/"metadata/FL2VA__transformer__config.json").read_text())
        config.update(num_layers=2,token_refiner_num_layers=2,hidden_size=256,
          num_attention_heads=2,attention_head_dim=128,ffn_hidden_size=512,text_dim=128,
          timestep_input_dim=64,time_embed_hidden_size=256,time_embed_dim=128,
          adaln_out_features=18*256,final_adaln_out_features=2*256)
        from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
        from vllm_omni.diffusion.forward_context import set_forward_context
        from h3_omni_adapter.native_mmap import H3NativeMmapPlan,H3NativeMmapLoader
        reference=make(config,"cpu")
        def weights():
            for name in ("a.safetensors","b.safetensors"):
                with safe_open(root/name,framework="pt",device="cpu") as f:
                    for k in f.keys():yield k,f.get_tensor(k)
        check("native_cpu_reference_all_loaded",reference.load_weights(weights())==set(state(reference)))
        reference.post_load_weights()
        reference.eval().to("cuda")
        candidate = make(config).eval()
        plan = H3NativeMmapPlan.build(candidate,root,max_total_file_bytes=64*1024**2)
        loader = H3NativeMmapLoader(candidate, plan, device="cuda", max_group_bytes=32 * 1024**2)
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
            for step in (0.1, 0.5, 0.9):
                kwargs["unique_timesteps"].fill_(step)
                expected = reference(**kwargs)
                actual = candidate(**kwargs)
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
        check("all expected groups used each forward", loader.stats["groups_loaded"] == 13)
        loader.close()
        check("close_all_meta", all(t.is_meta for t in state(candidate).values()))
        check("no mapped checkpoint remains", str(root) not in Path("/proc/self/maps").read_text())
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        check("cuda_peak_under_resource_bound",torch.cuda.max_memory_allocated()<4*1024**3)
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
    for path, label in ((str(a.fixture_trial), "<FIXTURE_TRIAL>"), (str(a.output.parent), "<RUN_ROOT>"), (str(a.research), "<RESEARCH_ROOT>"),
                        (str(Path(__file__).resolve().parents[1]), "<H3_PROJECT_ROOT>"), (str(Path.home()), "<LINUX_HOME>")):
        text = text.replace(path, label)
    a.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
