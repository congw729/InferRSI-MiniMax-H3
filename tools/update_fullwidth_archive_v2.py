from pathlib import Path
import sys
p=Path(sys.argv[1])
path=p/"tools/checkpoint_fullwidth_iteration.py"
s=path.read_text(encoding="utf-8")
s=s.replace('    intact={}\n', '    frozen2=json.loads((iteration/"plan/freeze-v2.json").read_text())\n    intact={}\n')
s=s.replace('    for name,digest in frozen["files"].items():','    for name,digest in {**frozen["files"],**frozen2["files"]}.items():')
s=s.replace('    fixture=json.loads', '    intact["gates_v2"]=sha(iteration/"plan/TWO-BLOCK-GATES.zh-CN.txt")==frozen2["gates_sha256"]\n    fixture=json.loads')
s=s.replace('        required=frozen["required_"+value["mode"]+"_checks"]','        required=frozen2["required_checks"] if value["mode"]=="two-blocks" else frozen["required_"+value["mode"]+"_checks"]')
s=s.replace('"actual_slots":worker.get("actual_slots"),','"actual_slots":worker.get("actual_slots"),"reference_cuda_peak":worker.get("reference_cuda_peak"),')
s=s.replace('"real-width one-block native DLO conversion, transport and repeated sampling capacity"','"real-width canonical conversion, preserved one-block rejection and two-block native DLO capacity"')
path.write_text(s,encoding="utf-8")
