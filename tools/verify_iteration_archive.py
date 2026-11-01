"""Read-only streaming integrity and path-redaction check for one archive."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--iteration",type=Path,required=True)
    p.add_argument("--forbidden",action="append",default=[])
    a=p.parse_args()
    manifest=json.loads((a.iteration/"artifact-manifest.json").read_text(encoding="utf-8"))["files"]
    bad=[n for n,v in manifest.items() if not (a.iteration/n).is_file()
         or (a.iteration/n).stat().st_size!=v["bytes"] or sha(a.iteration/n)!=v["sha256"]]
    actual={x.relative_to(a.iteration).as_posix() for x in a.iteration.rglob("*")
            if x.is_file() and x.name!="artifact-manifest.json"}
    missing=sorted(actual-set(manifest))
    hits=[]
    for path in a.iteration.rglob("*"):
        if path.is_file() and path.suffix in {".py",".md",".txt",".json",".jsonl",".log",".toml",".yml",".yaml"}:
            text=path.read_text(encoding="utf-8")
            if any(value in text for value in a.forbidden):
                hits.append(path.relative_to(a.iteration).as_posix())
    print(json.dumps({"files":len(manifest),"hash_mismatches":bad,"unlisted":missing,"privacy_hits":hits}))
    return int(bool(bad or missing or hits))

if __name__=="__main__":
    raise SystemExit(main())
