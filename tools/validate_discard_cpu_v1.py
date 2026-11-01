"""Frozen bounded CPU validation controller, using existing native tests."""
import argparse
import json
from pathlib import Path
import sys
from validate_dlo_temporal_chain_v1 import run_owned

NATIVE_SELECTION = (
    "test_worker_shutdown_skips_allgather_restore or "
    "test_disable_runs_collectives_before_best_effort_local_cleanup or "
    "test_rank_local_disable_drains_pending_transfers or "
    "test_rank_local_mmap_retains_source_and_uses_staging or "
    "test_rank_local_mmap_staging_is_bounded_by_largest_block"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.parent
    worker = Path(__file__).with_name("discard_cpu_worker_v1.py")
    record = run_owned([sys.executable, "-B", str(worker), "--omni-root", str(args.omni_root)],
                       root / "worker")
    path = root / "worker/worker-result.json"
    result = {"campaign_verdict": "INCONCLUSIVE", "monitor": record}
    if path.is_file():
        result["worker"] = json.loads(path.read_text())
    code = int(record.get("exit_code") != 0 or bool(record.get("guard_stop_reason"))
               or not record.get("process_reaped"))
    result["exit_code"] = code
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exit_code": code, "worker": result.get("worker"), "monitor": record}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
