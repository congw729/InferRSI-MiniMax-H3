"""Frozen follow-up CPU audit: unmodified metadata of a prompt-only candidate."""
import argparse, hashlib, inspect, json, os, traceback
from pathlib import Path
from types import SimpleNamespace

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--research",type=Path,required=True)
    p.add_argument("--source-record",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    result={"campaign_verdict":"INCONCLUSIVE","checks":[],"weights_loaded":False}
    def check(name,ok,**details):
        result["checks"].append({"name":name,"ok":bool(ok),**details})
        if not ok:
            raise AssertionError(name)
    try:
        import torch
        from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3CheckpointSpec
        file=Path(inspect.getfile(FastH3CheckpointSpec))
        audited=json.loads((a.source_record/"extra/source-audit.json").read_text())
        expected=audited["files"]["vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py"]["sha256"]
        check("native_contract_source_unchanged",hashlib.sha256(file.read_bytes()).hexdigest()==expected)
        candidate=a.source_record/"candidate-metadata"
        receipt=json.loads((candidate/"fetch-record.json").read_text())
        check("candidate_metadata_integrity",all(
            hashlib.sha256((candidate/name).read_bytes()).hexdigest()==info["sha256"]
            for name,info in receipt["files"].items()))
        metadata=json.loads((candidate/"fastvideo_inference.json").read_text())
        try:
            FastH3CheckpointSpec.from_metadata(metadata)
        except ValueError as exc:
            check("unmodified_qad_v2_metadata_rejected_by_model_identity",
                  "model_id" in str(exc), native_error=str(exc))
        else:
            check("unmodified_qad_v2_metadata_rejected_by_model_identity",False)
        sampling=SimpleNamespace(extra_args={"task":"t2va"}, num_inference_steps=8,
            timesteps=None,sigmas=None,lora_request=None,guidance_scale=1.0)
        FastH3CheckpointSpec().check_request(sampling)
        check("native_v2_class_allows_t2va_eight_steps",True)
        old=json.loads((a.research/"config-text_encoder__config.json").read_text())["content"]
        new=json.loads((candidate/"text_encoder__config.json").read_text())
        check("encoder_config_equal_not_weight_identity",old==new)
        old=json.loads((a.research/"config-transformer_ref__config.json").read_text())["content"]
        new=json.loads((candidate/"transformer__config.json").read_text())
        difference={k:{"omni":old.get(k),"qad_v2":new.get(k)} for k in old.keys()|new.keys()
                    if old.get(k)!=new.get(k)}
        check("transformer_config_delta_pdd_head_contract",difference=={"pdd_steps":{"omni":32,"qad_v2":None}},
              config_difference=difference)
        check("candidate_has_distinct_schedule_and_vsa",metadata["video_scheduler_shift"]==10.0
              and metadata["vsa_sparsity"]==0.8 and metadata["vsa_tile_size"]==64
              and metadata["transformer_forwards"]==8)
        check("no_cuda_context_initialized",not torch.cuda.is_initialized())
        result.update(repository=receipt["repository"],revision=receipt["revision"],
            adoption_status="NOT_ADOPTED: native metadata identity, headers, scheduler and decoder need qualification",
            note="Do not edit downloaded metadata to bypass model_id rejection. Config equality is not weight equivalence.")
    except BaseException as exc:
        trace=traceback.format_exc()
        for source,label in ((str(a.source_record),"<ITERATION_ROOT>"),(str(a.research),"<RESEARCH_ROOT>"),
                             (str(Path.home()),"<LINUX_HOME>"),(str(Path(__file__).resolve().parents[1]),"<H3_PROJECT_ROOT>")):
            trace=trace.replace(source,label)
        result["error"]={"type":type(exc).__name__,"traceback":trace}
    result["checks_total"]=len(result["checks"])
    result["checks_ok"]=sum(c["ok"] for c in result["checks"])
    result["exit_code"]=int("error" in result or result["checks_ok"]!=result["checks_total"])
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"checks","error"}}))
    return result["exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
