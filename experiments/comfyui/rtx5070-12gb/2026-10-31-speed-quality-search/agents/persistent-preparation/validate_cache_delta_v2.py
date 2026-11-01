import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
T=D.parent/'timing-controller'
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def rejected(fn):
 try:fn()
 except RuntimeError:return True
 return False
m=load(D/'persistent_stage_b_v2.py','persistent_cache_candidate');v=load(D/'validate_persistent_media_v2.py','persistent_cache_media')
checks=[]
def ck(n,b):assert b,n;checks.append({'name':n,'completed':True})
base=['python','main.py','--disable-smart-memory','--cache-none','--vram-headroom','1.0','--reserve-vram','3']
a=m.cache_command(base,'none');b=m.cache_command(base,'classic')
ck('both_smart_and_only_cache_flag_diff','--disable-smart-memory' not in a+b and a==[x for x in base if x!='--disable-smart-memory'] and b==['--cache-classic' if x=='--cache-none' else x for x in a])
ck('unknown_policy_or_missing_baseline_rejected',rejected(lambda:m.cache_command(base,'lru')) and rejected(lambda:m.cache_command(a,'none')))
g=json.loads((T/'graphs/trim-8.json').read_text());g1=m.request_graph(g,1);g2=m.request_graph(g,2)
ck('two_request_seeds_and_original_unchanged',g1['prompt']['15']['inputs']['noise_seed']==424242 and g2['prompt']['15']['inputs']['noise_seed']==424243 and g['prompt']['15']['inputs']['noise_seed']==424242)
for i,z in enumerate((g1,g2),1):
 z=copy.deepcopy(z);z['prompt']['15']['inputs']['noise_seed']=424242;z['prompt']['92']['inputs']['filename_prefix']=g['prompt']['92']['inputs']['filename_prefix'];assert z==g
ck('graph_only_seed_and_output_prefix_change',True)
ck('third_request_rejected',rejected(lambda:m.request_graph(g,3)))
# Do not rerun old 15 cases: the sampler requirement and all resource/cleanup helpers must be unchanged.
def fs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef)}
x=fs(D/'persistent_stage_b_v1.py');y=fs(D/'persistent_stage_b_v2.py');assert (set(x)-{'memory_command'})==(set(y)-{'cache_command'})
ck('other_functions_unchanged',all(x[k]==y[k] for k in x if k not in {'memory_command','request_graph','parse','main'}))
assert 'progress_value' in m.request_metrics.__code__.co_consts and 'cached' in m.request_metrics.__code__.co_consts
r={'status':'two_artifacts_received_unvalidated','mode':'persistent-video-stage-b','profile_id':'trim-8','driver_sha256':v.CONTROLLER_SHA,'gates_sha256':v.CONTROLLER_GATES_SHA,'source_manifest_sha256':v.SOURCES_SHA,'memory_policy':'smart','cache_policy':'none','request_seeds':[424242,424243],'disable_smart_memory_requested':False,'owned_root_exited':True,'job_close_requested':True,'cleanup_diagnostics':[],'completed_requests':2,'prompt_posts':2,'requests':[{'request_index':i,'seed':424241+i,'prompt_id':'p'+str(i),'status':'complete','submission_outcome':'accepted','post_attempted':True,'artifact':{'relative_to_run':'output/request-'+str(i)+'.mp4','sha256':'a'*64,'bytes':100}} for i in (1,2)]}
for policy in ('none','classic'):
 r['cache_policy']=policy;assert len(v.validate_session(r))==2
ck('two_cache_policies_media_metadata_accepted',True)
for key,value in [('memory_policy','baseline'),('disable_smart_memory_requested',True),('request_seeds',[424242,424242]),('cache_policy','lru')]:
 z=copy.deepcopy(r);z[key]=value;assert rejected(lambda:v.validate_session(z)),key
z=copy.deepcopy(r);z['requests'][1]['seed']=424242;assert rejected(lambda:v.validate_session(z))
ck('wrong_policy_or_seed_media_metadata_rejected',True)
ck('no_torch_model_dispatch','torch' not in sys.modules)
p=D/'OFFLINE-RESULT-v2.json'
with p.open('x',encoding='utf-8') as f:json.dump({'schema':'q33-persistent-cache-delta-check/v2','checks':checks,'completed':len(checks),'scope':'Synthetic graph/CLI/metadata and unchanged helper AST; no actual service/cache hit/model/decoder','controller_sha256':hashlib.sha256((D/'persistent_stage_b_v2.py').read_bytes()).hexdigest()},f,indent=2)
print(json.dumps({'completed':len(checks),'model_gpu_decode_dispatches':0}))
