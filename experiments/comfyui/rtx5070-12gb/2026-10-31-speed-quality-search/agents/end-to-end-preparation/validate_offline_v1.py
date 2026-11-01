"""Targeted CPU-only metadata/lifecycle checks for the new two-stage harness."""
import argparse,ast,copy,hashlib,importlib.util,json,math,struct,time,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--qroot',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args();d=Path(__file__).resolve().parent;q=a.qroot.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(d/'CANDIDATE-FREEZE-v1.json')==a.freeze_sha256
f=json.loads((d/'CANDIDATE-FREEZE-v1.json').read_text())
for x in f['files']:assert (d/x['path']).stat().st_size==x['bytes'] and sha(d/x['path'])==x['sha256']
spec=importlib.util.spec_from_file_location('e2e_cpu_candidate',d/'end_to_end_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);h=m.e2e
out={'schema':'q33-e2e-targeted-offline/v1','complete':False,'model_dispatches':0,'network_requests':0,'checks':[]};start=time.monotonic();work=d/('offline-'+uuid.uuid4().hex);work.mkdir()
def check(name,ok):out['checks'].append({'name':name,'ok':bool(ok)});assert ok,name
def reject(name,fn):
 try:fn()
 except RuntimeError:check(name,True)
 else:check(name,False)
try:
 def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 old=functions(q/'agents/persistent-preparation/persistent_stage_b_v1.py');new=functions(d/'end_to_end_v1.py')
 for name in ['sample','guarded_hash','cleanup_owned_v2','begin_request','begin_post','conditioning_header','verify_dynamic_vram_startup']:
  check('inherited_'+name,old[name]==new[name])
 row={'available_physical_bytes':12*2**30,'available_commit_bytes':int(16.5*2**30),'gpu_board':{'free_bytes':6*2**30}}
 check('fresh_exact_floor',h.fresh_ready(row))
 for key in ['available_physical_bytes','available_commit_bytes','gpu_board']:
  bad=copy.deepcopy(row)
  if key=='gpu_board':bad[key]['free_bytes']-=1
  else:bad[key]-=1
  check('fresh_reject_'+key,not h.fresh_ready(bad))
 log=work/'log.txt';log.write_bytes(h.RESET_MARKER+b'\n');offset=log.stat().st_size;check('startup_marker_not_ack',not h.reset_path_entered(log,offset))
 with log.open('ab') as fd:fd.write(h.RESET_MARKER+b'\n')
 check('new_marker_enters_reset_path',h.reset_path_entered(log,offset));reject('truncated_log_rejected',lambda:h.reset_path_entered(log,log.stat().st_size+1))
 state={'requests':[],'prompt_posts':0};c=m.begin_request(state,1,'one');m.begin_post(state);reject('repeat_post_rejected',lambda:m.begin_post(state));c['status']='complete';m.begin_request(state,2,'two');m.begin_post(state);reject('third_request_rejected',lambda:m.begin_request(state,3,'three'))
 canonical=json.loads((q/'agents/timing-controller/graphs/trim-8.json').read_text());basename='h3-e2e-bounce-test-'+h.REFERENCE_SHA+'.safetensors';bound=h.bind_stage_b(canonical,basename);expected=copy.deepcopy(canonical);expected['prompt']['1001']['inputs']['conditioning_name']=basename;expected['prompt']['92']['inputs']['filename_prefix']='q33-e2e/bounce-video';check('only_condition_prefix_delta',bound==expected)
 reject('path_alias_rejected',lambda:h.bind_stage_b(canonical,'../bad.safetensors'))
 count=41*5120;payload=struct.pack('<I',0x3f800000)*count+struct.pack('<q',1)*41
 header={'__metadata__':{'conditioning_options':'{}'},'conditioning':{'dtype':'F32','shape':[1,41,5120],'data_offsets':[0,count*4]},'minimax_token_tags':{'dtype':'I64','shape':[41],'data_offsets':[count*4,len(payload)]}}
 raw=json.dumps(header,separators=(',',':')).encode();good=work/'finite.safetensors';good.write_bytes(len(raw).to_bytes(8,'little')+raw+payload)
 def unique(pairs):
  out={}
  for k,v in pairs:assert k not in out;out[k]=v
  return out
 def scan(path):return h.scan_conditioning(path,m.conditioning_header(path,unique,32*2**20),lambda:None,time.monotonic()+5)
 check('finite_and_tags_valid',scan(good)['elements_checked']==count+41)
 for name,at,replace in [('nan',8+len(raw),struct.pack('<I',0x7fc00000)),('badtag',8+len(raw)+count*4,struct.pack('<q',0))]:
  path=work/(name+'.safetensors');data=bytearray(good.read_bytes());data[at:at+len(replace)]=replace;path.write_bytes(data);reject(name+'_rejected',lambda:scan(path))
 bad=copy.deepcopy(m.conditioning_header(good,unique,32*2**20));bad['tensors']['conditioning']['shape']=[1,40,5120];reject('shape_rejected',lambda:h.scan_conditioning(good,bad,lambda:None,time.monotonic()+5))
 tree=ast.parse((d/'end_to_end_v1.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_one'];check('exact_stage_calls',[ast.literal_eval(x.args[0]) for x in calls]==[1,2])
 check('no_torch_import','torch' not in __import__('sys').modules)
 out['complete']=True
except BaseException as exc:out['error_type']=type(exc).__name__;out['reason']=str(exc)
out['elapsed_seconds']=time.monotonic()-start;out['fixture_directory']=work.name;out['check_count']=len(out['checks'])
with (d/'OFFLINE-RESULT-v1.json').open('x',encoding='utf-8') as fd:json.dump(out,fd,indent=2);fd.write('\n')
print(json.dumps({'complete':out['complete'],'checks':len(out['checks']),'reason':out.get('reason')}));raise SystemExit(0 if out['complete'] else 1)
