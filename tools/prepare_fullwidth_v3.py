from pathlib import Path
import sys
p=Path(sys.argv[1])
s=(p/"tools/dlo_fullwidth_worker_v2.py").read_text(encoding="utf-8")
s=s.replace('"""Frozen v2:','"""Frozen v3 (same gates, durable phase receipts):')
s=s.replace('    def check(name,ok):','''    def checkpoint(phase):
        import torch
        data={"phase":phase,"checks":result["checks"],
              "cuda_allocated":torch.cuda.memory_allocated(),
              "cuda_peak":torch.cuda.max_memory_allocated(),
              "ru_maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        with (a.work/"live-phases.jsonl").open("a") as stream:
            stream.write(json.dumps(data)+"\\n")
            stream.flush()
    def check(name,ok):''')
s=s.replace('        if not ok:\n            raise AssertionError(name)','        checkpoint(name)\n        if not ok:\n            raise AssertionError(name)')
s=s.replace('        model.cuda()','        checkpoint("reference_before_cuda")\n        model.cuda()\n        checkpoint("reference_after_cuda")')
s=s.replace('        baseline=sample(model,config,hidden,g)','        checkpoint("reference_before_sampling")\n        baseline=sample(model,config,hidden,g)')
s=s.replace('        backend.enable(pipeline)','        checkpoint("candidate_before_enable")\n        backend.enable(pipeline)\n        checkpoint("candidate_after_enable")')
s=s.replace('        actual=sample(model,config,hidden,g)','        checkpoint("candidate_before_sampling")\n        actual=sample(model,config,hidden,g)')
(p/"tools/dlo_fullwidth_worker_v3.py").write_text(s,encoding="utf-8")
s=(p/"tools/validate_dlo_fullwidth_v2.py").read_text(encoding="utf-8")
s=s.replace("Frozen v2","Frozen v3").replace("dlo_fullwidth_worker_v2.py","dlo_fullwidth_worker_v3.py").replace('"two-blocks"','"two-blocks-instrumented"')
(p/"tools/validate_dlo_fullwidth_v3.py").write_text(s,encoding="utf-8")
