"""Windows direct-read prefix probe; creates only one bounded owned staging copy."""
import argparse, ctypes, hashlib, json, os, shutil, time
from ctypes import wintypes as W
from pathlib import Path
MIB=1024**2
GIB=1024**3
TOTAL=512*MIB
BLOCK=8*MIB

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--staging",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result={"campaign_verdict":"INCONCLUSIVE","scope":"512MiB prefix direct-read diagnostics",
            "cache_policy":"FILE_FLAG_NO_BUFFERING bypasses Windows filesystem cache, not SSD/device cache; no global flush",
            "bytes_per_pass":TOTAL,"block_bytes":BLOCK,"reads":[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    k.CreateFileW.argtypes=[W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE]
    k.CreateFileW.restype=W.HANDLE
    k.ReadFile.argtypes=[W.HANDLE,ctypes.c_void_p,W.DWORD,ctypes.POINTER(W.DWORD),ctypes.c_void_p]
    k.ReadFile.restype=W.BOOL
    k.CloseHandle.argtypes=[W.HANDLE]
    k.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,W.DWORD,W.DWORD]
    k.VirtualAlloc.restype=ctypes.c_void_p
    k.VirtualFree.argtypes=[ctypes.c_void_p,ctypes.c_size_t,W.DWORD]
    k.GetFileAttributesW.argtypes=[W.LPCWSTR]
    k.GetFileAttributesW.restype=W.DWORD
    k.GetFileInformationByHandleEx.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD]
    k.GetFileInformationByHandleEx.restype=W.BOOL
    k.GetCompressedFileSizeW.argtypes=[W.LPCWSTR,ctypes.POINTER(W.DWORD)]
    k.GetCompressedFileSizeW.restype=W.DWORD
    class STORAGEINFO(ctypes.Structure):
        _fields_=[(name,W.DWORD) for name in ("logical","physical_atomic","physical_performance",
                  "effective_physical","flags","sector_offset","partition_offset")]
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_=[("length",W.DWORD),("load",W.DWORD)]+[(x,ctypes.c_ulonglong) for x in
            ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
    def budget():
        mem=MEMORYSTATUSEX()
        mem.length=ctypes.sizeof(mem)
        if not k.GlobalMemoryStatusEx(ctypes.byref(mem)): raise ctypes.WinError(ctypes.get_last_error())
        if mem.avail_phys<2*GIB: raise RuntimeError("host physical memory guard")
        if time.monotonic()-started>180: raise RuntimeError("probe deadline")
    ptr=None
    try:
        budget()
        if a.source.stat().st_size<TOTAL: raise RuntimeError("source too small")
        if shutil.disk_usage(a.staging.parent).free<8*GIB: raise RuntimeError("staging disk guard")
        sha=hashlib.sha256()
        with a.source.open("rb",buffering=0) as src,a.staging.open("xb",buffering=0) as dst:
            left=TOTAL
            while left:
                budget()
                data=src.read(min(BLOCK,left))
                if not data: raise RuntimeError("short source")
                if dst.write(data)!=len(data): raise RuntimeError("short write")
                sha.update(data)
                left-=len(data)
            os.fsync(dst.fileno())
        result["prefix_sha256"]=sha.hexdigest()
        result["staging_created_bytes"]=a.staging.stat().st_size
        ptr=k.VirtualAlloc(None,BLOCK,0x3000,0x04)
        if not ptr: raise ctypes.WinError(ctypes.get_last_error())
        result["buffer_aligned_4096"]=(ptr%4096==0)
        for label,path in (("D_native",a.source),("E_native",a.staging)):
            attr=k.GetFileAttributesW(str(path))
            if attr==0xFFFFFFFF: raise ctypes.WinError(ctypes.get_last_error())
            row={"location":label,"file_attributes":attr,"compressed":bool(attr&0x800),"sparse":bool(attr&0x200),"encrypted":bool(attr&0x4000),"samples":[]}
            hi=W.DWORD()
            ctypes.set_last_error(0)
            lo=k.GetCompressedFileSizeW(str(path),ctypes.byref(hi))
            last_error=ctypes.get_last_error()
            if lo==0xFFFFFFFF and last_error:
                row["GetCompressedFileSizeW_bytes"]=None
                row["allocation_query_error"]=last_error
            else:
                row["GetCompressedFileSizeW_bytes"]=int(lo)+(int(hi.value)<<32)
            for repeat in range(3):
                h=k.CreateFileW(str(path),0x80000000,7,None,3,0x20000000|0x08000000,None)
                if h==ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
                try:
                    storage=STORAGEINFO()
                    if not k.GetFileInformationByHandleEx(h,16,ctypes.byref(storage),ctypes.sizeof(storage)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    row["storage_alignment"]={name:int(getattr(storage,name)) for name,_ in storage._fields_}
                    align=max(storage.logical,storage.physical_atomic,storage.physical_performance,storage.effective_physical)
                    if align<1 or ptr%align or BLOCK%align or TOTAL%align:
                        raise RuntimeError("buffer/length not storage aligned")
                    got=W.DWORD()
                    begin=time.perf_counter()
                    for offset in range(0,TOTAL,BLOCK):
                        budget()
                        if not k.ReadFile(h,ptr,BLOCK,ctypes.byref(got),None): raise ctypes.WinError(ctypes.get_last_error())
                        if got.value!=BLOCK: raise RuntimeError("short direct read")
                    elapsed=time.perf_counter()-begin
                finally:
                    k.CloseHandle(h)
                row["samples"].append({"repeat":repeat,"seconds":elapsed,"GiB_per_second":TOTAL/GIB/elapsed})
            # Independent full-prefix direct-read digest after timed passes.
            digest=hashlib.sha256()
            h=k.CreateFileW(str(path),0x80000000,7,None,3,0x20000000|0x08000000,None)
            if h==ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
            try:
                for offset in range(0,TOTAL,BLOCK):
                    budget()
                    if not k.ReadFile(h,ptr,BLOCK,ctypes.byref(got),None): raise ctypes.WinError(ctypes.get_last_error())
                    if got.value!=BLOCK: raise RuntimeError("short validation read")
                    digest.update(ctypes.string_at(ptr,BLOCK))
            finally: k.CloseHandle(h)
            row["direct_digest_match"]=digest.hexdigest()==result["prefix_sha256"]
            if not row["direct_digest_match"]: raise RuntimeError("direct read byte mismatch")
            result["reads"].append(row)
        result["execution_complete"]=True
    except BaseException as exc:
        result["error"]={"type":type(exc).__name__,"message":str(exc)}
    finally:
        if ptr: k.VirtualFree(ptr,0,0x8000)
        result["elapsed_seconds"]=time.monotonic()-started
        a.output.write_text(json.dumps(result,indent=2)+"\n")
    return int(not result.get("execution_complete",False))
if __name__=="__main__": raise SystemExit(main())
