import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent
T=D.parent/'timing-controller'
R=D.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def functions(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
m=module(D/'attention_stage_b_v1.py','attention_candidate')
profile=json.loads((D/'profiles/trim-6-pytorch.json').read_text())
graph=json.loads((D/'graphs/trim-6-pytorch.json').read_text())
canonical=json.loads((D/'canonical/trim.json').read_text())
checks=[]
def checked(name,fn):
 fn();checks.append({'name':name,'completed':True})
def rejects(name,p,g):
 try:m.validate_profile_graph(p,g,canonical)
 except RuntimeError:checks.append({'name':name,'completed':True});return
 raise AssertionError(name)
checked('exact graph accepted',lambda:m.validate_profile_graph(profile,graph,canonical))
g=copy.deepcopy(graph);g['prompt']['128']['inputs']['attention']='comfy kitchen attention';rejects('kitchen rejected for this candidate',profile,g)
p=copy.deepcopy(profile);p['steps']=8;rejects('steps8 rejected',p,graph)
g=copy.deepcopy(graph);g['prompt']['127']['inputs']['min_tokens']=0;rejects('min_tokens change rejected',profile,g)
g=copy.deepcopy(graph);g['prompt']['15']['inputs']['noise_seed']=7;rejects('seed change rejected',profile,g)
g=copy.deepcopy(graph);g['prompt']['148']['inputs']['shift_video']=11;rejects('unrelated graph change rejected',profile,g)
f1=functions(T/'speed_stage_b_v6.py');f2=functions(D/'attention_stage_b_v1.py')
assert f1.keys()==f2.keys()
assert all(f1[n]==f2[n] for n in f1 if n!='validate_profile_graph')
checks.append({'name':'all other functions AST identical','completed':True})
h=module(R/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py','frozen_helper')
a=SimpleNamespace(profile=D/'profiles/trim-6-pytorch.json',profile_sha256=sha(D/'profiles/trim-6-pytorch.json'),asset_manifest=(D/profile['assets']['file']).resolve(),asset_manifest_sha256=profile['assets']['sha256'],conditioning_manifest=(D/profile['conditioning']['file']).resolve(),conditioning_manifest_sha256=profile['conditioning']['sha256'],iteration_root=D.parents[1])
a.loaded_profile=m.load_profile(a,h)
m.load_stage_b_manifest(a,h,{'max_schema_bytes':1048576})
checks.append({'name':'real profile and conditioning metadata binding','completed':True})
assert 'torch' not in sys.modules
result={'schema':'q33-attention-targeted-check/v1','checks':checks,'completed':len(checks),'candidate_sha256':sha(D/'attention_stage_b_v1.py'),'no_torch_model_gpu_network':True}
with (D/'OFFLINE-RESULT-v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))