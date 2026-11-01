"""Only new warm-admission arithmetic and source-delta checks; no GPU/model."""
import argparse,ast,copy,hashlib,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--freeze-sha256',required=True);a=p.parse_args();d=Path(__file__).resolve().parent
assert hashlib.sha256((d/'WARM-CANDIDATE-FREEZE-v2.json').read_bytes()).hexdigest()==a.freeze_sha256
f=json.loads((d/'WARM-CANDIDATE-FREEZE-v2.json').read_text())
for x in f['files']:assert (d/x['path']).stat().st_size==x['bytes'] and hashlib.sha256((d/x['path']).read_bytes()).hexdigest()==x['sha256']
spec=importlib.util.spec_from_file_location('warm_helper',d/'e2e_helpers_v2.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
checks=[]
def check(name,ok):checks.append({'name':name,'ok':bool(ok)});assert ok,name
def reject(name,fn):
 try:fn()
 except RuntimeError:check(name,True)
 else:check(name,False)
g={'max_owned_private_bytes':16*2**30,'running_commit_bytes':2*2**30,'running_physical_bytes':2*2**30};r={'owned_private_sum_bytes':4*2**30,'available_physical_bytes':12*2**30,'available_commit_bytes':14*2**30,'gpu_board':{'free_bytes':6*2**30}};after={k:r[k] for k in ['available_physical_bytes','available_commit_bytes']}
check('exact_remaining_budget_floor',h.warm_admission(r,after,g)['admitted'])
low=copy.deepcopy(after);low['available_commit_bytes']-=1;check('post_commit_drop_rejects',not h.warm_admission(r,low,g)['admitted'])
low=copy.deepcopy(after);low['available_physical_bytes']-=1;check('post_physical_drop_rejects',not h.warm_admission(r,low,g)['admitted'])
low=copy.deepcopy(r);low['available_commit_bytes']-=1;check('earlier_low_sample_retained',not h.warm_admission(low,after,g)['admitted'])
low=copy.deepcopy(r);low['gpu_board']['free_bytes']-=1;check('GPU_floor_unchanged',not h.warm_admission(low,after,g)['admitted'])
for owned in [0,16*2**30]:
 row=copy.deepcopy(r);row['owned_private_sum_bytes']=owned;row['available_commit_bytes']=g['max_owned_private_bytes']-owned+g['running_commit_bytes'];post={k:row[k] for k in after};check('owned_endpoint_'+str(owned),h.warm_admission(row,post,g)['admitted'])
for owned in [-1,16*2**30+1,4.0*2**30]:
 row=copy.deepcopy(r);row['owned_private_sum_bytes']=owned;reject('owned_domain_'+str(owned),lambda:h.warm_admission(row,after,g))
low=copy.deepcopy(after);low['available_commit_bytes']=2*2**30-1;reject('post_read_running_floor',lambda:h.warm_admission(r,low,g))
def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
old=funcs(d/'end_to_end_v1.py');new=funcs(d/'end_to_end_v2.py');check('unchanged_all_old_named_functions_except_main',all(old[k]==new[k] for k in old if k not in ['main']))
h1=funcs(d/'e2e_helpers_v1.py');h2=funcs(d/'e2e_helpers_v2.py');check('unchanged_all_old_helpers',all(h1[k]==h2[k] for k in h1))
s=(d/'end_to_end_v2.py').read_text();check('cold_initial_admission_still_fixed',s.count('require(e2e.fresh_ready(state["last_sample"]), "initial external admission 12/16.5/6")')==1);check('warm_only_three_sites',s.count('warm_admit()')==4)
out={'schema':'q33-e2e-warm-offline/v2','complete':True,'checks':checks,'check_count':len(checks),'model_dispatches':0,'network_requests':0,'claim':'INCONCLUSIVE'}
with (d/'WARM-OFFLINE-RESULT-v2.json').open('x',encoding='utf-8') as fd:json.dump(out,fd,indent=2);fd.write('\n')
print(json.dumps({'complete':True,'checks':len(checks)}))
