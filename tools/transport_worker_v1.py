"""Bounded transport diagnostics; no model import or performance acceptance claim."""
import argparse, gc, hashlib, json, mmap, os, shutil, sys, time, uuid
from pathlib import Path
from h3_omni_adapter.stage_process import _termination_guard

MIB = 1024**2
GIB = 1024**3
TOTAL = 512*MIB

def digest_prefix(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        left = TOTAL
        while left:
            data = f.read(min(8*MIB, left))
            if not data:
                raise RuntimeError("short prefix")
            h.update(data)
            left -= len(data)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--staging", type=Path, required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"campaign_verdict":"INCONCLUSIVE", "scope":"bounded synthetic transport L1 diagnostics",
              "cache_policy":"No global cache clearing; first-observed and repeated reads are not cold-disk measurements",
              "bytes_per_storage_pass":TOTAL, "checks":[], "storage":[], "cpu_copy":[], "h2d":[], "staged_h2d":[]}
    checks = result["checks"]
    def check(name, ok):
        checks.append({"name":name, "ok":bool(ok)})
        if not ok:
            raise RuntimeError(name)
    def save():
        pending=a.output.with_suffix(".pending")
        pending.write_text(json.dumps(result, indent=2)+"\n")
        os.replace(pending,a.output)
    temp = None
    try:
        with _termination_guard():
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            result["torch_version"] = torch.__version__
            result["torch_threads"] = torch.get_num_threads()
            result["measurement_order"] = "D then E then ext4; order effects are not controlled"
            result["transfer_stream"] = "CUDA default stream; synchronize before reusing staging slot"
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            free, total = torch.cuda.mem_get_info()
            result["cuda_before"] = {"free_bytes":free,"total_bytes":total}
            check("gpu_start_headroom", free >= 2*GIB)
            torch.cuda.set_per_process_memory_fraction(0.40)
            torch.cuda.reset_peak_memory_stats()
            check("source_prefix_exists", a.source.stat().st_size >= TOTAL)
            check("staging_exact_size", a.staging.stat().st_size == TOTAL)
            expected = digest_prefix(a.source)
            check("e_copy_digest", digest_prefix(a.staging) == expected)
            result["prefix_sha256"] = expected
            a.scratch_root.mkdir(parents=True, exist_ok=True)
            check("scratch_disk_free", shutil.disk_usage(a.scratch_root).free >= 8*GIB)
            temp = a.scratch_root/("h3-io-"+uuid.uuid4().hex)
            if temp.exists(): raise FileExistsError("temporary collision")
            result["temporary_directory_name"] = temp.name
            save()
            temp.mkdir()
            ext = temp/"prefix.bin"
            with a.source.open("rb", buffering=0) as src, ext.open("xb", buffering=0) as dst:
                left = TOTAL
                while left:
                    data = src.read(min(8*MIB,left))
                    if not data: raise RuntimeError("short staging copy")
                    if dst.write(data) != len(data): raise RuntimeError("short staging write")
                    left -= len(data)
                os.fsync(dst.fileno())
            check("ext4_copy_digest", digest_prefix(ext) == expected)
            paths = [("D_drvfs",a.source), ("E_drvfs",a.staging), ("Linux_ext4",ext)]
            result["temporary_bytes"] = TOTAL
            # Storage timings have no allocation or digest calculation in their loop.
            buf = bytearray(8*MIB)
            for label,path in paths:
                for mode in ("readinto","mmap_copy"):
                    rows=[]
                    for repeat in range(3):
                        with path.open("rb",buffering=0) as f:
                            mm = mmap.mmap(f.fileno(), TOTAL, access=mmap.ACCESS_READ) if mode=="mmap_copy" else None
                            started = time.perf_counter()
                            for offset in range(0,TOTAL,len(buf)):
                                if mm is None:
                                    got = f.readinto(buf)
                                    if got != len(buf): raise RuntimeError("short measured read")
                                else:
                                    view = memoryview(mm)[offset:offset+len(buf)]
                                    memoryview(buf)[:] = view
                                    view.release()
                            elapsed=time.perf_counter()-started
                            if mm is not None: mm.close()
                        rows.append({"repeat":repeat, "seconds":elapsed, "GiB_per_second":TOTAL/GIB/elapsed})
                    # Separate validation pass; does not assert that each measured byte was independently checked.
                    check(label+"_"+mode+"_post_digest",digest_prefix(path)==expected)
                    result["storage"].append({"location":label,"mode":mode,"samples":rows})
                    save()
            del buf
            # End-to-end wall time includes dispatch and synchronization; event time is separately retained.
            for n in (8*MIB,64*MIB,256*MIB):
                pageable=torch.empty(n,dtype=torch.uint8)
                pageable.fill_(37)
                pinned=torch.empty(n,dtype=torch.uint8,pin_memory=True)
                pinned.fill_(37)
                check("pinned_allocation_"+str(n),pinned.is_pinned())
                device=torch.empty(n,dtype=torch.uint8,device="cuda")
                event_start=torch.cuda.Event(enable_timing=True)
                event_end=torch.cuda.Event(enable_timing=True)
                def run_copy(src):
                    torch.cuda.synchronize()
                    begin=time.perf_counter()
                    event_start.record()
                    for _ in range(TOTAL//n):
                        device.copy_(src,non_blocking=True)
                    event_end.record()
                    event_end.synchronize()
                    return {"wall_seconds":time.perf_counter()-begin,
                            "cuda_event_seconds":event_start.elapsed_time(event_end)/1000}
                cpu_samples=[]
                for _ in range(TOTAL//n): pinned.copy_(pageable)
                for _ in range(5):
                    start=time.perf_counter()
                    for _ in range(TOTAL//n): pinned.copy_(pageable)
                    cpu_samples.append(time.perf_counter()-start)
                check("cpu_copy_bytes_"+str(n),torch.equal(pinned,pageable))
                result["cpu_copy"].append({"block_bytes":n,"bytes_per_sample":TOTAL,"copies_per_sample":TOTAL//n,"warmups":1,"seconds":cpu_samples,
                    "meaning":"pageable-to-pinned memory copy on one torch CPU thread"})
                for kind,host in (("pageable",pageable),("pinned",pinned)):
                    run_copy(host)
                    samples=[run_copy(host) for _ in range(5)]
                    check(kind+"_h2d_bytes_"+str(n),torch.equal(device.cpu(),host))
                    result["h2d"].append({"block_bytes":n,"bytes_per_sample":TOTAL,"copies_per_sample":TOTAL//n,"host":kind,"warmups":1,"samples":samples})
                del pageable,pinned,device,event_start,event_end,host,run_copy
                gc.collect()
                torch.cuda.empty_cache()
                save()
            # Same prefix and same 64MiB slots: mmap -> pinned -> GPU.
            n=64*MIB
            pinned=torch.empty(n,dtype=torch.uint8,pin_memory=True)
            device=torch.empty(n,dtype=torch.uint8,device="cuda")
            check("staged_pinned_allocation",pinned.is_pinned())
            host_view=memoryview(pinned.numpy())
            for label,path in paths:
                rows=[]
                with path.open("rb",buffering=0) as f, mmap.mmap(f.fileno(),TOTAL,access=mmap.ACCESS_READ) as mm:
                    for repeat in range(4):
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        for offset in range(0,TOTAL,n):
                            view=memoryview(mm)[offset:offset+n]
                            host_view[:] = view
                            view.release()
                            device.copy_(pinned,non_blocking=True)
                            torch.cuda.synchronize()
                        elapsed=time.perf_counter()-start
                        if repeat:
                            rows.append({"repeat":repeat-1,"seconds":elapsed,"GiB_per_second":TOTAL/GIB/elapsed})
                    # Last block exact check; whole-file digest checked separately above.
                    check(label+"_staged_last_block",torch.equal(device.cpu(),pinned))
                    transferred=hashlib.sha256()
                    for offset in range(0,TOTAL,n):
                        view=memoryview(mm)[offset:offset+n]
                        host_view[:]=view
                        view.release()
                        device.copy_(pinned,non_blocking=True)
                        torch.cuda.synchronize()
                        returned=device.cpu()
                        transferred.update(memoryview(returned.numpy()))
                        del returned
                    check(label+"_staged_full_prefix_digest",transferred.hexdigest()==expected)
                result["staged_h2d"].append({"location":label,"bytes":TOTAL,"block_bytes":n,
                    "warmups":1,"overlap":False,"samples":rows})
                save()
            host_view.release()
            del host_view,pinned,device
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_after,total_after=torch.cuda.mem_get_info()
            result["cuda_after"]={"free_bytes":free_after,"total_bytes":total_after}
            result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
            result["cuda_allocated_after_cleanup_bytes"]=torch.cuda.memory_allocated()
            check("cuda_peak_bound",result["cuda_peak_allocated_bytes"]<4*GIB)
            check("cuda_objects_released",result["cuda_allocated_after_cleanup_bytes"]==0)
            result["execution_complete"]=True
    except BaseException as exc:
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
    finally:
        if temp is not None:
            try:
                if temp.exists():
                    (temp/"prefix.bin").unlink(missing_ok=True)
                    temp.rmdir()
                result["ext4_temporary_removed"]=not temp.exists()
            except BaseException as exc:
                result["cleanup_error"]={"type":type(exc).__name__,"message":str(exc)}
                result["execution_complete"]=False
        save()
    return int(not result.get("execution_complete",False))

if __name__=="__main__":
    raise SystemExit(main())
