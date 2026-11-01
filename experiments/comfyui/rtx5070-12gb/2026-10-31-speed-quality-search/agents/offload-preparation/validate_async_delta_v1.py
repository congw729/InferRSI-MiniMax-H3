"""Small offline delta gates only; no Torch, models, GPU query or network."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,types
D=Path(__file__).resolve().parent; B=D.parent/'persistent-preparation';Q=D.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text(encoding='utf8'))) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
checks=[]
def check(name,value):
 checks.append({'name':name,'complete':bool(value)});assert value,name
def rejects(fn):
 try:fn()
 except RuntimeError:return True
 return False
base=B/'persistent_stage_b_v2.py';candidate=D/'async_persistent_stage_b_v1.py'
check('frozen baseline',sha(base)=='f27dff8a8792a99c47d9ba71f5ddd005a95bebd950e8f365b51cb4685b4c31b8')
b=load(base,'offline_async_baseline');c=load(candidate,'offline_async_candidate');m=load(D/'validate_async_persistent_media_v1.py','offline_async_media')
changed={'verify_dynamic_vram_startup','parse','cache_command','main'}
bf,cf=funcs(base),funcs(candidate)
check('all nonchanged function AST incl nested monitor/hash/http/loop/cleanup',set(bf)==set(cf) and all(v==cf[k] for k,v in bf.items() if k not in changed))
def command(path):
 nodes=[n for n in ast.walk(ast.parse(path.read_text(encoding='utf8'))) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets) and isinstance(n.value,ast.List)]
 assert len(nodes)==1
 return eval(compile(ast.Expression(nodes[0].value),'<frozen-command-list>','eval'),{'a':types.SimpleNamespace(portable_python=Path('E:/owned/python.exe'),comfy_root=Path('E:/owned/ComfyUI'),port=8193),'gates':{'reserve_vram_gb':3},'models_root':Path('E:/owned/models'),'str':str})
bc=b.cache_command(command(base),'classic');cc=c.cache_command(command(candidate),'classic')
check('actual CLI only removes disable async',bc.count('--disable-async-offload')==1 and cc==[x for x in bc if x!='--disable-async-offload'])
check('no pin/headroom/reserve/classic maintained','--disable-pinned-memory' in cc and '--cache-classic' in cc and '--cache-none' not in cc and cc[cc.index('--vram-headroom')+1]=='1.0' and cc[cc.index('--reserve-vram')+1]=='3')
check('reject none cache',rejects(lambda:c.cache_command(command(candidate),'none')))
g=json.loads((Q/'agents/timing-controller/graphs/trim-8.json').read_text(encoding='utf8'))
check('two graphs unchanged with intended seeds',all(c.request_graph(g,i)==b.request_graph(g,i) and c.request_graph(g,i)['prompt']['15']['inputs']['noise_seed']==424241+i for i in (1,2)))
logs=D/'offline-log-fixtures-v1';logs.mkdir()
for name,raw in [('valid',b'DynamicVRAM support detected and enabled\nUsing async weight offloading with 2 streams\n'),('missing',b'DynamicVRAM support detected and enabled\n'),('legacy',b'DynamicVRAM support detected and enabled\nUsing async weight offloading with 2 streams\nFalling back to legacy ModelPatcher\n')]:
 with (logs/(name+'.txt')).open('xb') as f:f.write(raw)
check('actual startup marker accepted',c.verify_dynamic_vram_startup(logs/'valid.txt')['observed_async_offload_streams']==2)
check('missing async marker rejected',rejects(lambda:c.verify_dynamic_vram_startup(logs/'missing.txt')))
check('legacy rejected',rejects(lambda:c.verify_dynamic_vram_startup(logs/'legacy.txt')))
check('media source exact filename/hash',m.CONTROLLER_SHA==sha(candidate) and "sha(D/'async_persistent_stage_b_v1.py')==CONTROLLER_SHA" in (D/'validate_async_persistent_media_v1.py').read_text(encoding='utf8'))
r={'status':'two_artifacts_received_unvalidated','mode':'async-persistent-video-stage-b','profile_id':'trim-8','driver_sha256':sha(candidate),'gates_sha256':m.CONTROLLER_GATES_SHA,'source_manifest_sha256':m.SOURCES_SHA,'owned_root_exited':True,'job_close_requested':True,'cleanup_diagnostics':[],'memory_policy':'smart','cache_policy':'classic','disable_smart_memory_requested':False,'request_seeds':[424242,424243],'disable_async_offload_requested':False,'disable_pinned_memory_requested':True,'dynamic_vram_startup':{'observed_async_offload_streams':2},'completed_requests':2,'prompt_posts':2,'requests':[{'request_index':i,'seed':424241+i,'status':'complete','submission_outcome':'accepted','post_attempted':True,'prompt_id':str(i),'artifact':{'relative_to_run':f'output/{i}.mp4','bytes':1,'sha256':'0'*64}} for i in (1,2)]}
check('synthetic metadata new identity accepted',len(m.validate_session(r))==2)
for key,val in [('mode','persistent-video-stage-b'),('cache_policy','none'),('disable_async_offload_requested',True),('disable_pinned_memory_requested',False),('owned_root_exited',False)]:
 x=copy.deepcopy(r);x[key]=val;check('reject metadata '+key,rejects(lambda:m.validate_session(x)))
x=copy.deepcopy(r);x['dynamic_vram_startup']={};check('reject missing observed 2 streams',rejects(lambda:m.validate_session(x)))
check('roots/mode fixed',"agents/offload-preparation" in candidate.read_text(encoding='utf8') and 'choices=["async-persistent-video-stage-b"]' in candidate.read_text(encoding='utf8') and 'choices=["classic"]' in candidate.read_text(encoding='utf8'))
result={'schema':'q33-async-offline-delta/v1','status':'complete','checks':checks,'check_count':len(checks),'gpu_runs':0,'model_dispatches':0,'limitations':'Only metadata/AST/command/startup-log checks; synthetic media record is not a real receipt or decoded media. GPU and end-to-end performance INCONCLUSIVE.','candidate_sha256':sha(candidate),'validator_sha256':sha(Path(__file__))}
with (D/'OFFLINE-RESULT-v1.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'status':result['status'],'checks':len(checks),'gpu_runs':0}))
