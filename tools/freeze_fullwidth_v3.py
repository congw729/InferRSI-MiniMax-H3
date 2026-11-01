import ast,hashlib,json,shutil,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]
i=Path(sys.argv[1])
base=json.loads((i/"plan/freeze-v2.json").read_text(encoding="utf-8"))
names=["dlo_fullwidth_worker_v3.py","validate_dlo_fullwidth_v3.py"]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
node=next(n for n in ast.parse((p/"tools/dlo_fullwidth_worker_v3.py").read_text()).body
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="REQUIRED")
assert ast.literal_eval(node.value)==base["required_checks"]
for n in names: shutil.copyfile(p/"tools"/n,i/"validator"/n)
base["files"]={n:sha(p/"tools"/n) for n in names}
base["instrumentation_sha256"]=sha(i/"plan/V3-INSTRUMENTATION.zh-CN.txt")
with (i/"plan/freeze-v3.json").open("x") as stream: json.dump(base,stream,indent=2)
print(json.dumps(base["files"]))
