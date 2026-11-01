import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

p=Path(__file__).resolve().parents[1]
i=Path(sys.argv[1])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
names=["dlo_fullwidth_worker_v1.py","validate_dlo_fullwidth_v1.py","validate_dlo_temporal_chain_v1.py",
       "dlo_temporal_common_v1.py","validate_capacity_v1.py","validate_meta_coverage_v2.py"]
checks={}
for node in ast.parse((p/"tools/dlo_fullwidth_worker_v1.py").read_text()).body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id.endswith("_CHECKS"):
        checks[node.targets[0].id]=ast.literal_eval(node.value)
target=i/"validator"
target.mkdir(exist_ok=True)
for name in names:
    shutil.copyfile(p/"tools"/name,target/name)
freeze={"campaign_verdict":"INCONCLUSIVE","plane":"infra","files":{n:sha(p/"tools"/n) for n in names},
 "runtime":{x.name:sha(x) for x in (p/"src/h3_omni_adapter").glob("*.py")},
 "required_budget_checks":checks["BUDGET_CHECKS"],
 "required_execute_checks":checks["BUDGET_CHECKS"]+checks["EXEC_CHECKS"],
 "gates_sha256":sha(i/"plan/FROZEN-GATES.zh-CN.txt")}
path=i/"plan/freeze-v1.json"
with path.open("x",encoding="utf-8") as stream:
    json.dump(freeze,stream,indent=2)
print(json.dumps({"budget":len(freeze["required_budget_checks"]),"execute":len(freeze["required_execute_checks"]),
                 "files":freeze["files"]}))
