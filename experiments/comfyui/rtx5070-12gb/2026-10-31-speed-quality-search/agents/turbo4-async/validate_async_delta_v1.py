"""Frozen narrow Turbo4 async delta/metadata checks. No model/torch/GPU."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,hashlib,importlib.util,json,sys
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--notes-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.notes_root;Q=R/'iterations/2026-10-31-speed-quality-search';D=Q/'agents/turbo4-async';B=Q/'agents/turbo4-corpus';A=Q/'agents/offload-preparation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def fn(src,name):
 n=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name);return '\n'.join(src.splitlines()[n.lineno-1:n.end_lineno])
checks=[]
def ck(name,val):checks.append({'name':name,'ok':bool(val)});assert val,name
old=(B/'turbo4_corpus_stage_b_v1.py').read_text(encoding='utf8');new=(D/'turbo4_async_stage_b_v1.py').read_text(encoding='utf8');asyncs=(A/'async_persistent_stage_b_v1.py').read_text(encoding='utf8')
state='        state["disable_async_offload_requested"] = False\n        state["disable_pinned_memory_requested"] = True\n        state["requested_async_offload_streams"] = 2\n'
reverse=new.replace('"agents/turbo4-async"','"agents/turbo4-corpus"').replace('"turbo4-async-video-stage-b"','"turbo4-corpus-video-stage-b"').replace('"--disable-pinned-memory", "--disable-smart-memory"','"--disable-pinned-memory", "--disable-async-offload", "--disable-smart-memory"').replace('    require(profile["corpus_id"] == "bounce-audio", "single bounce experiment")\n','').replace(state,'').replace(fn(asyncs,'verify_dynamic_vram_startup'),fn(old,'verify_dynamic_vram_startup'))
ck('reverse declared changes exactly old controller',reverse==old and sha(B/'turbo4_corpus_stage_b_v1.py')=='a94784c03a22de99eef9ebb379c0d7f741d0cb5923fffba3fae1cf5c0ad79aaf')
def cmd(src):return next(n.value for n in ast.walk(ast.parse(src)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets))
co=copy.deepcopy(cmd(old));co.elts=[x for x in co.elts if not(isinstance(x,ast.Constant) and x.value=='--disable-async-offload')];ck('only native argv flag removed',ast.dump(co)==ast.dump(cmd(new)))
m=load(D/'turbo4_async_stage_b_v1.py','meta_async');h=load(R/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py','meta_monitor');pp=D/'profiles/bounce-audio.json';pf=read(pp);gates=read(Q/'agents/timing-controller/RESOURCE-GATES-RESERVE3-v1.json');b=SimpleNamespace(iteration_root=Q,notes_root=R,profile=pp,profile_sha256=sha(pp),asset_manifest=D/pf['assets']['file'],asset_manifest_sha256=pf['assets']['sha256'],conditioning_manifest=D/pf['conditioning']['file'],conditioning_manifest_sha256=pf['conditioning']['sha256']);b.loaded_profile=m.load_profile(b,h);m.ASSET_EXPECTED=m.EXPECTED_BY_FAMILY['turbo-native'];m.ASSET_IDS=set(m.ASSET_EXPECTED);assets=m.load_asset_manifest(b,h,gates);cm,graph=m.load_stage_b_manifest(b,h,gates)
ck('real4asset receipts and41token provenance',len(assets)==4 and cm['conditioning']['tensors']['conditioning']['shape']==[1,41,5120])
ck('graph canonical assets byte unchanged',all((D/x).read_bytes()==(B/x).read_bytes() for x in ['graphs/bounce-audio.json','canonical/original.json','assets/turbo-native.ready.json']))
bcm=read(B/'conditioning/bounce-audio.json');ck('condition payload provenance unchanged',cm['conditioning']==bcm['conditioning'] and cm['provenance']==bcm['provenance'])
unit=D/'unit-v1';unit.mkdir();wrong=copy.deepcopy(pf);wrong['corpus_id']='hands-face-mug';wp=unit/'wrong-case.json';wp.write_text(json.dumps(wrong),encoding='utf8');bad=copy.copy(b);bad.profile=wp;bad.profile_sha256=sha(wp)
try:m.load_profile(bad,h);ok=False
except RuntimeError as e:ok=str(e)=='single bounce experiment'
ck('reject hands before dispatch',ok)
log=unit/'startup.log';log.write_bytes(b'DynamicVRAM support detected and enabled\nUsing async weight offloading with 2 streams\n');ck('exact observed2streams',m.verify_dynamic_vram_startup(log)['observed_async_offload_streams']==2)
for label,body in [('one',b'Using async weight offloading with 1 streams'),('missing',b'')]:
 log.write_bytes(b'DynamicVRAM support detected and enabled\n'+body)
 try:m.verify_dynamic_vram_startup(log);ok=False
 except RuntimeError:ok=True
 ck('reject_'+label+'_marker',ok)
ck('noTorch','torch' not in sys.modules)
with a.output.open('x',encoding='utf8') as f:json.dump({'schema':'q33-turbo4-async-checks/v1','checks':checks,'count':len(checks),'complete':True,'model_gpu_runs':0,'claim':'INCONCLUSIVE'},f,indent=2)
print(json.dumps({'complete':True,'checks':len(checks),'model_gpu_runs':0}))
