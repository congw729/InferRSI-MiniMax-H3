import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
p=Path(__file__).resolve().parents[1]
i=Path(sys.argv[1])
cache=Path(sys.argv[2])
def sha(x):
    return hashlib.sha256(x.read_bytes()).hexdigest()
names=["dlo_fullwidth_worker_v2.py","validate_dlo_fullwidth_v2.py"]
node=next(n for n in ast.parse((p/"tools/dlo_fullwidth_worker_v2.py").read_text()).body
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="REQUIRED")
for n in names:
    shutil.copyfile(p/"tools"/n,i/"validator"/n)
freeze={"campaign_verdict":"INCONCLUSIVE","files":{n:sha(p/"tools"/n) for n in names},
    "required_checks":ast.literal_eval(node.value),"gates_sha256":sha(i/"plan/TWO-BLOCK-GATES.zh-CN.txt"),
    "cache_manifest_sha256":sha(cache/"manifest.json")}
with (i/"plan/freeze-v2.json").open("x") as stream:
    json.dump(freeze,stream,indent=2)
print(json.dumps(freeze))
