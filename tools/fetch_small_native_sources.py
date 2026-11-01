"""Fetch only small pinned public source/config files listed in an existing manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    selected = [item for item in manifest["files"]
                if item["path"].startswith(("Ref2VA/video_vae/", "Ref2VA/audio_vae/"))
                and Path(item["path"]).suffix in {".py", ".json", ".yaml"}]
    if any(item["bytes"] > 512 * 1024 for item in selected) or sum(x["bytes"] for x in selected) > 2 * 1024**2:
        raise ValueError("source download exceeds explicit small-file budget")
    if a.destination.exists():
        raise FileExistsError("source bundle already exists")
    a.destination.mkdir(parents=True)
    out = []
    for item in selected:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid manifest path")
        url = "https://huggingface.co/" + manifest["repository"] + "/resolve/" + manifest["revision"] + "/" + item["path"]
        with urlopen(url, timeout=30) as response:
            data = response.read(512 * 1024 + 1)
        if len(data) != item["bytes"]:
            raise ValueError("download size differs from saved metadata")
        target = a.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        out.append({"path": item["path"], "source_url": url, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})
    (a.destination / "source-manifest.json").write_text(json.dumps({
        "repository": manifest["repository"], "revision": manifest["revision"],
        "scope": "source/config only; no weight payload", "files": out}, indent=2) + "\n")
    print(json.dumps({"files": len(out), "bytes": sum(x["bytes"] for x in out), "weight_payload_downloaded": False}))
if __name__ == "__main__":
    main()
