"""Narrow offline admission and fixed graph isolation; no models/runtime imports."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,tempfile
D=Path(__file__).resolve().parent
old=ast.parse((D/'end_to_end_v2.py').read_text());new=ast.parse((D/'end_to_end_v3.py').read_text())
count=0
def check(ok,label):
 global count
 assert ok,label;count+=1
by=lambda t:{n.name:n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=by(old),by(new)
for name in ('sample','cleanup_owned_v2','guarded_hash','warm_admit','load_asset_manifest','verify_conditioning','request_metrics'):
 check(ast.dump(a[name])==ast.dump(b[name]),'protected '+name)
spec=importlib.util.spec_from_file_location('two_prompt_helpers',D/'e2e_helpers_v3.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
check(ast.dump(by(ast.parse((D/'e2e_helpers_v2.py').read_text()))['warm_admission'])==ast.dump(by(ast.parse((D/'e2e_helpers_v3.py').read_text()))['warm_admission']),'warm formula unchanged')
ns={};exec(compile(ast.Module(body=[b[n] for n in ('require','begin_request','begin_post')],type_ignores=[]),'sequence','exec'),ns)
s={'requests':[],'prompt_posts':0}
for i in (1,2,3,4):
 c=ns['begin_request'](s,i,str(i));ns['begin_post'](s);c['status']='complete'
check(s['prompt_posts']==4,'four bounded POST')
for fn,args in ((ns['begin_request'],(s,5,'5')),(ns['begin_post'],(s,))):
 try:fn(*args)
 except RuntimeError:pass
 else:raise AssertionError('extra/repeat POST accepted')
check(True,'fifth/repeat POST rejected')
t={'requests':[],'prompt_posts':0};ns['begin_request'](t,1,'1');ns['begin_post'](t)
try:ns['begin_request'](t,2,'2')
except RuntimeError:pass
else:raise AssertionError('failed prior accepted')
check(True,'incomplete request rejected')
for case in h.CASES:
 original=json.loads((D.parent/'quality-corpus/graphs'/ (case+'.json')).read_text());graph=json.loads((D/('STAGE-A-'+case+'-v3.json')).read_text())
 expected=copy.deepcopy(original);expected['prompt']['1001']['inputs']['filename_prefix']='q33-e2e-two/'+case+'/conditioning/stage-a'
 check(graph==expected,'StageA only output prefix '+case)
 graph={'prompt':{'9':{'inputs':{'steps':8}},'15':{'inputs':{'noise_seed':424242}},'1001':{'inputs':{'conditioning_name':'old'}},'92':{'inputs':{'filename_prefix':'old'}}}}
 out=h.bind_stage_b(graph,'h3-e2e-'+case+'-nonce-hash.safetensors',case)
 expected=copy.deepcopy(graph);expected['prompt']['1001']['inputs']['conditioning_name']='h3-e2e-'+case+'-nonce-hash.safetensors';expected['prompt']['92']['inputs']['filename_prefix']='q33-e2e-two/'+case+'/video'
 check(out==expected and graph['prompt']['1001']['inputs']['conditioning_name']=='old','StageB only condition/prefix '+case)
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'old.mp4').write_bytes(b'old');(p/'old.safetensors').write_bytes(b'old');sub=p/'hands-face-mug'/'conditioning';sub.mkdir(parents=True);f=sub/'stage-a.safetensors';f.write_bytes(b'x'*16)
 check(h.find_conditioning_artifact(sub,32*2**20,lambda:None)==f,'scan only new subtree preserves prior outputs')
check(h.CASES['bounce-audio']['tokens']==41 and h.CASES['hands-face-mug']['tokens']==49,'two fixed real reference shapes')
check('if request_index>1:' in (D/'end_to_end_v3.py').read_text() and 'for request_index in (1,2,3,4):' in (D/'end_to_end_v3.py').read_text(),'all transitions warm guarded')
print(json.dumps({'checks':count,'passed':count,'gpu_used':False,'new_model_dispatches':0}))
