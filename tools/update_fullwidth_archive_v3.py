from pathlib import Path
import sys
p=Path(sys.argv[1])
path=p/"tools/checkpoint_fullwidth_iteration.py"
s=path.read_text(encoding="utf-8")
s=s.replace('    frozen2=json.loads((iteration/"plan/freeze-v2.json").read_text())',
'''    versions={int(x.stem.rsplit("v",1)[1]):json.loads(x.read_text()) for x in (iteration/"plan").glob("freeze-v*.json")}
    frozen2=versions[2]
    version_files={n:h for v in versions.values() for n,h in v["files"].items()}''')
s=s.replace('{**frozen["files"],**frozen2["files"]}.items()','version_files.items()')
s=s.replace('    fixture=json.loads', '''    for number,v in versions.items():
        if "instrumentation_sha256" in v:
            intact[f"instrumentation_v{number}"]=sha(iteration/"plan/V3-INSTRUMENTATION.zh-CN.txt")==v["instrumentation_sha256"]
    fixture=json.loads''')
s=s.replace('        required=frozen2["required_checks"] if value["mode"]=="two-blocks" else frozen["required_"+value["mode"]+"_checks"]',
'''        version=3 if "two-blocks-v3-" in path.parent.parent.name else 2
        required=versions[version]["required_checks"] if value["mode"].startswith("two-blocks") else frozen["required_"+value["mode"]+"_checks"]
        phase_path=path.parent/"worker/live-phases.jsonl"
        last_phase=json.loads(phase_path.read_text().strip().splitlines()[-1]) if phase_path.exists() and phase_path.stat().st_size else None''')
s=s.replace('"mode":value["mode"],"geometry":worker.get("geometry")','"mode":value["mode"],"last_durable_phase":last_phase,"geometry":worker.get("geometry")')
path.write_text(s,encoding="utf-8")
