"""Frozen CPU worker: no real weights, no GPU test selection."""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(args.omni_root)
    sys.path.insert(0, str(args.omni_root))
    import pytest
    from validate_discard_cpu_v2 import NATIVE_SELECTION
    tests = Path(__file__).resolve().parents[1] / "tests/test_stage_dlo_discard_v2.py"
    results = []
    for name, selection in [
        ("native", ["tests/diffusion/offloader/test_distributed_layerwise_backend.py", "-k", NATIVE_SELECTION]),
        ("adapter", [str(tests)]),
    ]:
        junit = args.work / (name + ".xml")
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--override-ini=addopts=",
                            "--basetemp", str(args.work / (name + "-temp")),
                            "--junitxml", str(junit), *selection])
        doc = ET.parse(junit).getroot()
        cases = list(doc.iter("testcase"))
        failures = sum(len(case.findall("failure")) + len(case.findall("error")) for case in cases)
        skips = sum(len(case.findall("skipped")) for case in cases)
        results.append({"selection": name, "exit_code": int(code), "cases": len(cases),
                        "failures": failures, "skipped": skips,
                        "names": [case.attrib.get("name") for case in cases]})
        if code:
            break
    code = int(len(results) != 2 or any(r["exit_code"] or r["failures"] or r["skipped"] or not r["cases"]
                                      for r in results))
    report = {"campaign_verdict": "INCONCLUSIVE", "cpu_only_selection": True,
              "suites": results, "exit_code": code}
    (args.work / "worker-result.json").write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
