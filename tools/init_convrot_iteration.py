import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--iteration",type=Path,required=True);p.add_argument("--project",type=Path,required=True);a=p.parse_args()
a.iteration.mkdir(exist_ok=False)
for d in ("plan","validator","results","patch","sources","agents/implementation","agents/validation"):
 (a.iteration/d).mkdir(parents=True)
before={x.relative_to(a.project).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((a.project/"src").rglob("*.py"))}
(a.iteration/"plan/runtime-before.json").write_text(json.dumps(before,indent=2)+"\n",encoding="utf-8")
