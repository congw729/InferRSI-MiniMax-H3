"""Frozen offline adapter gates. Synthetic metadata only, no service/network/GPU."""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--infra-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
s=importlib.util.spec_from_file_location('q33_pair_candidate',a.candidate);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
checks=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError,TypeError):check(name,True)
 else:raise AssertionError(name)
def docs():
 r={'status':'artifact_received_unvalidated','profile_id':'trim-8','prompt_id':'fixture-prompt','prompt_posts':1,'submission_outcome':'accepted','owned_root_exited':True,'job_close_requested':True,'cleanup_diagnostics':[], 'artifact':{'sha256':'a'*64},'node_timing':{'prompt_id':'fixture-prompt','terminal':'execution_success','invalid_reasons':[],'observer_thread_exited':True,'error_type':None,'nodes':{'14':{'observed_wall_seconds':8.0,'cached':False}}}}
 h={'fixture-prompt':{'status':{'completed':True,'status_str':'success','messages':[['execution_start',{'prompt_id':'fixture-prompt','timestamp':1000}],['execution_success',{'prompt_id':'fixture-prompt','timestamp':11000}]]}}}
 media={'status':'inspection_ready_human_review_pending','owned_root_exited':True,'job_close_requested':True,'worker_returncode':0,'expected_input_sha256':'a'*64,'source_sha256':m.MEDIA_SOURCE_SHA,'gates_sha256':m.MEDIA_GATES_SHA,'decoder_result_sha256':'b'*64}
 return r,h,media
r,h,media=docs();x=m.extract_metrics(r,h,media,'trim-8');check('native_interval',x['prompt_execution_seconds']==10.0);check('sampler_interval',x['sampler_wall_seconds']==8.0)
for name,mut in [
 ('inner_failure',lambda r,h,v:r.update(status='failed')),('extra_post',lambda r,h,v:r.update(prompt_posts=2)),('root_alive',lambda r,h,v:r.update(owned_root_exited=False)),('cleanup_error',lambda r,h,v:r.update(cleanup_diagnostics=['x'])),('wrong_profile',lambda r,h,v:r.update(profile_id='trim-4')),('invalid_timing',lambda r,h,v:r['node_timing'].update(invalid_reasons=['disconnect'])),('observer_alive',lambda r,h,v:r['node_timing'].update(observer_thread_exited=False)),('cached_sampler',lambda r,h,v:r['node_timing']['nodes']['14'].update(cached=True)),('missing_sampler_metric',lambda r,h,v:r['node_timing']['nodes']['14'].update(observed_wall_seconds=None)),('negative_interval',lambda r,h,v:h['fixture-prompt']['status']['messages'][1][1].update(timestamp=999)),('wrong_history_prompt',lambda r,h,v:h['fixture-prompt']['status']['messages'][1][1].update(prompt_id='other')),('missing_terminal',lambda r,h,v:h['fixture-prompt']['status']['messages'].pop()),('duplicate_terminal',lambda r,h,v:h['fixture-prompt']['status']['messages'].append(copy.deepcopy(h['fixture-prompt']['status']['messages'][1]))),('media_incomplete',lambda r,h,v:v.update(status='failed')),('media_identity',lambda r,h,v:v.update(expected_input_sha256='c'*64)),('media_source',lambda r,h,v:v.update(source_sha256='c'*64)),('media_worker',lambda r,h,v:v.update(worker_returncode=1))]:
 r,h,v=docs();mut(r,h,v);reject(name,lambda:m.extract_metrics(r,h,v,'trim-8'))
source=a.out/'source';source.mkdir();f=source/'a.txt';f.write_bytes(b'unchanged')
manifest={'schema':'q33-rsi-source-bundle/v1','files':[{'root':'fixture','path':'a.txt','bytes':9,'sha256':hashlib.sha256(b'unchanged').hexdigest()}]}
mp=a.out/'manifest.json';mp.write_text(json.dumps(manifest),encoding='utf8');mh=m.sha(mp)
identity=m.source_snapshot(mp,mh,{'fixture':source});check('real_source_hash',identity['available'] and identity['identity_kind']=='sha256_file_manifest')
f.write_bytes(b'changed!!');check('source_tamper_unavailable',not m.source_snapshot(mp,mh,{'fixture':source})['available'])
reject('unrecognized_profile',lambda:m.profile_id('original-50'))
check('profile4',m.profile_id('trim-4-screen-v1')=='trim-4')
sys.path.insert(0,str(a.infra_root/'src'))
from inferrsi.harness.runner import run_experiment
from inferrsi.runspec import RunSpec
repo=a.out/'non_git_source';repo.mkdir();probe=repo/'identity';probe.write_text('frozen',encoding='utf8')
order=[]
def snap(path,exclude):return {'available':True,'snapshot_hash':hashlib.sha256(probe.read_bytes()).hexdigest(),'identity_kind':'synthetic_real_file_hash'}
def execute(command,cwd,trial,deadline,key):
 trial.mkdir(parents=True);order.append(command[0]);value=10.0 if command[0]=='A' else 8.0
 (trial/'result.json').write_text(json.dumps({'metrics':{'seconds':value}}),encoding='utf8')
 return {'status':'completed','returncode':0,'metric':value,'result_path':str(trial/'result.json'),'elapsed_seconds':0.0}
spec=RunSpec(repo=str(repo),baseline_command=('A',),candidate_command=('B',),repetitions=2,validation_command=None,metric_key='metrics.seconds',metric_unit='s',output_dir=str(a.out/'runner'),max_minutes=1)
run=run_experiment(spec,{},execute=execute,snapshot=snap);ev=json.loads((run/'evaluation.json').read_text())
check('native_abba_order',order==['A','B','B','A']);check('unreviewed_quality_inconclusive',ev['verdict']=='INCONCLUSIVE');check('descriptive_metrics_retained',ev['baseline']['median']==10 and ev['candidate']['median']==8)
result={'schema':'q33-rsi-offline/v1','checks':checks,'count':len(checks),'all_observed':True,'candidate_sha256':m.sha(a.candidate),'validator_sha256':m.sha(Path(__file__)),'model_dispatches':0,'network_requests':0,'scope':'Synthetic records and actual existing runner with synthetic executor; no real process/GPU/model/media run'}
(a.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'checks':len(checks),'model_dispatches':0,'all_observed':True}))