"""Q33 fixed-corpus pure-text conditioning integrity validator; frozen v1."""
import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import time
from pathlib import Path, PureWindowsPath

MAX_FILE=32*1024**2
MAX_HEADER=65536
CHUNK=512*1024
DEADLINE_SECONDS=60
class Rejected(Exception):pass

def require(ok,code):
    if not ok:raise Rejected(code)

def unique(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate_json_key')
        out[k]=v
    return out

def invalid_constant(value):raise Rejected('nonfinite_json_constant')
def parse_json(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid_constant)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
def signature(st):return identity(st)+(st.st_ctime_ns,)
def tick(deadline):require(time.monotonic()<deadline,'deadline')
def no_links(path):
    require(path.is_absolute(),'absolute_path_required')
    for entry in [*reversed(path.parents),path]:
        st=entry.lstat()
        require(not stat.S_ISLNK(st.st_mode) and not getattr(st,'st_file_attributes',0)&0x400,'link_or_reparse')
    return path.resolve(strict=True)
def read_small(path,limit):
    no_links(path);before=path.stat();require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=limit,'metadata_size')
    with path.open('rb') as f:
        handle_before=os.fstat(f.fileno())
        require(identity(handle_before)==identity(before),'metadata_open_identity')
        raw=f.read(limit+1)
        require(len(raw)==before.st_size,'metadata_read_length')
        require(signature(os.fstat(f.fileno()))==signature(handle_before),'metadata_read_identity')
    require(signature(path.stat())==signature(before),'metadata_changed')
    return raw,signature(before)
def file_hash(stream,deadline,expected_bytes):
    require(8<expected_bytes<=MAX_FILE,'hash_size_bound')
    h=hashlib.sha256();stream.seek(0);remaining=expected_bytes
    while remaining:
        tick(deadline);size=min(CHUNK,remaining);data=stream.read(size)
        require(len(data)==size,'hash_short_file');h.update(data);remaining-=size
    require(stream.read(1)==b'','hash_file_grew')
    return h.hexdigest()
def args():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    for n in ['qroot','owned-runs-root','run-dir','input','receipt','freeze','result']:ap.add_argument('--'+n,type=Path,required=True)
    for n in ['expected-sha256','receipt-sha256','freeze-sha256']:ap.add_argument('--'+n,required=True)
    ap.add_argument('--expected-bytes',type=int,required=True)
    ap.add_argument('--corpus',choices=['bounce-audio','hands-face-mug'],required=True)
    supplied=[x for x in sys.argv[1:] if x.startswith('--')]
    require(len(supplied)==len(set(supplied)),'duplicate_cli_option')
    a=ap.parse_args()
    for n in ['expected_sha256','receipt_sha256','freeze_sha256']:require(re.fullmatch('[0-9a-f]{64}',getattr(a,n)) is not None,'digest_argument')
    return a

def validate_corpus_receipt(receipt,frozen,corpus_id,check):
    check('fixed_corpus',corpus_id in {'bounce-audio','hands-face-mug'})
    case=frozen['cases'][corpus_id]
    check('receipt_stage_success',receipt.get('mode')=='quality-corpus-stage-a' and receipt.get('status')=='conditioning_artifact_received_unvalidated')
    check('receipt_corpus_identity',receipt.get('corpus_id')==corpus_id and receipt.get('prompt_sha256')==case['prompt_sha256'])
    check('receipt_native_identity',receipt.get('driver_sha256')==frozen['stage_a_driver_sha256'] and receipt.get('graph_sha256')==case['graph_sha256'])

def main():
    a=args();start=time.monotonic();deadline=start+DEADLINE_SECONDS
    out={'schema':'q33-corpus-conditioning-values/v1','corpus_id':a.corpus,'complete':False,'claim':'INCONCLUSIVE','integrity_verdict':'INCONCLUSIVE','quality_verdict':'INCONCLUSIVE','roundtrip_equivalence_checked':False,'model_dispatches':0,'torch_imported':False,'network_requests':0,'checks':[]}
    def check(name,ok):
        out['checks'].append({'name':name,'ok':bool(ok)});require(ok,name)
    q=no_links(a.qroot);rp=a.result.absolute()
    require(rp.parent.is_relative_to(q/'results') and rp.suffix=='.json' and not rp.exists(),'result_location_or_exists')
    no_links(rp.parent)
    try:
        frozen_raw,_=read_small(a.freeze,65536);check('freeze_hash',sha(frozen_raw)==a.freeze_sha256)
        frozen=parse_json(frozen_raw);check('freeze_schema',frozen['schema']=='q33-corpus-conditioning-values-freeze/v1')
        for row in frozen['sources']:
            source=no_links(q/row['path']);check('source_'+row['path'],source.is_relative_to(q) and source.stat().st_size==row['bytes'] and sha(source.read_bytes())==row['sha256'])
        check('validator_self_identity',sha(Path(__file__).read_bytes())==frozen['validator_sha256'])
        runs=no_links(a.owned_runs_root);run=no_links(a.run_dir);target=no_links(a.input)
        check('owned_E_output',runs.drive.upper()=='E:' and run.parent==runs and target.is_relative_to(run/'output'))
        check('file_extension',target.suffix=='.safetensors')
        check('expected_size_bound',8<a.expected_bytes<=MAX_FILE)
        receipt_path=no_links(a.receipt);check('receipt_under_results',receipt_path.is_relative_to(q/'results'))
        raw_receipt,receipt_sig=read_small(receipt_path,1024**2);check('receipt_hash',sha(raw_receipt)==a.receipt_sha256)
        receipt=parse_json(raw_receipt)
        validate_corpus_receipt(receipt,frozen,a.corpus,check)
        check('receipt_cleanup_and_single_prompt',receipt.get('owned_root_exited') is True and receipt.get('job_close_requested') is True and receipt.get('prompt_posts')==1)
        artifact=receipt['artifact'];relative=PureWindowsPath(artifact['relative_to_run'])
        check('receipt_relative_path',not relative.is_absolute() and relative.parts[0]=='output' and all(x not in ('.','..') for x in relative.parts) and target==run.joinpath(*relative.parts).resolve(strict=True))
        check('receipt_artifact_identity',type(artifact['bytes']) is int and artifact['bytes']==a.expected_bytes and artifact['sha256']==a.expected_sha256)
        before=target.stat();check('actual_file_size',stat.S_ISREG(before.st_mode) and before.st_size==a.expected_bytes)
        with target.open('rb') as stream:
            handle_before=os.fstat(stream.fileno())
            check('open_identity',identity(handle_before)==identity(before))
            check('full_sha_before',file_hash(stream,deadline,a.expected_bytes)==a.expected_sha256)
            stream.seek(0);prefix=stream.read(8);require(len(prefix)==8,'header_prefix_short')
            n=int.from_bytes(prefix,'little');check('header_bound',2<=n<=MAX_HEADER and 8+n<before.st_size)
            raw=stream.read(n);require(len(raw)==n,'header_short');header=parse_json(raw)
            check('header_object',type(header) is dict)
            metadata=header.pop('__metadata__',{})
            check('metadata_allowlist',type(metadata) is dict and set(metadata)<={'conditioning_options'} and all(type(v) is str for v in metadata.values()))
            options=parse_json(metadata.get('conditioning_options','{}'));check('pure_text_options_empty',type(options) is dict and options=={})
            check('exact_tensor_allowlist',set(header)=={'conditioning','minimax_token_tags'})
            sizes={'F16':2,'BF16':2,'F32':4,'I64':8};segments=[];data_bytes=before.st_size-8-n
            for name,t in header.items():
                require(type(t) is dict and set(t)=={'dtype','shape','data_offsets'},'tensor_fields')
                sh,off,dt=t['shape'],t['data_offsets'],t['dtype']
                require(type(sh) is list and 1<=len(sh)<=3 and all(type(x) is int and 0<x<=MAX_FILE for x in sh),'shape_bound')
                require(type(dt) is str and dt in sizes and type(off) is list and len(off)==2 and all(type(x) is int for x in off),'dtype_offset_types')
                begin,end=off;require(0<=begin<end<=data_bytes and end-begin==math.prod(sh)*sizes[dt],'tensor_byte_range');segments.append((begin,end))
            cursor=0
            for begin,end in sorted(segments):require(begin==cursor,'tensor_gap_overlap');cursor=end
            check('exact_data_coverage',cursor==data_bytes)
            cond,tags=header['conditioning'],header['minimax_token_tags']
            check('conditioning_shape_dtype',cond['dtype'] in ('F16','BF16','F32') and len(cond['shape'])==3 and cond['shape'][0]==1 and cond['shape'][2]==5120)
            check('tags_shape_dtype',tags['dtype']=='I64' and tags['shape']==[cond['shape'][1]])
            structure={'header_bytes':n,'header_sha256':sha(raw),'tensors':header,'metadata_keys':sorted(metadata),'option_keys':[],'values_checked':False,'roundtrip_checked':False}
            check('receipt_header_exact',artifact['structure']==structure)
            info={}
            for name,t in header.items():
                begin,end=t['data_offsets'];stream.seek(8+n+begin);remaining=end-begin;count=nonzero=0
                dt=t['dtype'];width=sizes[dt];fmt='<q' if dt=='I64' else ('<I' if dt=='F32' else '<H')
                mask=0x7f800000 if dt=='F32' else (0x7f80 if dt=='BF16' else 0x7c00)
                magnitude=0x7fffffff if dt=='F32' else 0x7fff
                while remaining:
                    tick(deadline);buf=stream.read(min(CHUNK,remaining));require(len(buf)==min(CHUNK,remaining) and len(buf)%width==0,'tensor_chunk_short')
                    for (bits,) in struct.iter_unpack(fmt,buf):
                        if dt=='I64':require(bits==1,'token_tags_not_all_one')
                        else:
                            require(bits&mask!=mask,'floating_nan_or_infinity')
                            nonzero+=int(bool(bits&magnitude))
                        count+=1
                    remaining-=len(buf)
                check(name+'_element_count',count==math.prod(t['shape']))
                if name=='conditioning':check('conditioning_not_all_zero',nonzero>0)
                info[name]={'elements_checked':count,'finite_checked':dt!='I64','all_tags_one_checked':dt=='I64','nonzero_present':bool(nonzero) if dt!='I64' else None}
            check('full_sha_after',file_hash(stream,deadline,a.expected_bytes)==a.expected_sha256)
            check('file_handle_stable',signature(os.fstat(stream.fileno()))==signature(handle_before))
        check('file_path_stable',signature(target.stat())==signature(before));no_links(target)
        receipt_after,receipt_after_sig=read_small(receipt_path,1024**2)
        check('receipt_stable',sha(receipt_after)==a.receipt_sha256 and receipt_after_sig==receipt_sig)
        check('no_torch_import','torch' not in sys.modules)
        tick(deadline)
        out.update(complete=True,integrity_verdict='PASS',artifact={'bytes':a.expected_bytes,'sha256':a.expected_sha256,'tensors':header,'metadata_keys':sorted(metadata),'option_keys':[],'options_sha256':sha(b'{}'),'header_bytes':n,'header_sha256':sha(raw)},scan=info,stage_a_receipt_sha256=a.receipt_sha256,prompt_sha256=frozen['cases'][a.corpus]['prompt_sha256'],stage_a_graph_sha256=frozen['cases'][a.corpus]['graph_sha256'],stage_a_driver_sha256=frozen['stage_a_driver_sha256'],validator_sha256=frozen['validator_sha256'],freeze_sha256=a.freeze_sha256)
    except BaseException as exc:
        out.update(integrity_verdict='FAIL',error_type=type(exc).__name__,reason=str(exc) if type(exc) is Rejected else 'unexpected_operation_failure_no_private_details')
    out.update(elapsed_seconds=time.monotonic()-start,check_count=len(out['checks']))
    encoded=json.dumps(out,indent=2,allow_nan=False).encode('utf-8')+b'\n';require(len(encoded)<=65536,'result_size')
    with rp.open('xb') as f:f.write(encoded)
    print(json.dumps({'complete':out['complete'],'integrity_verdict':out['integrity_verdict'],'checks':out['check_count'],'claim':'INCONCLUSIVE'}))
    return 0 if out['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
