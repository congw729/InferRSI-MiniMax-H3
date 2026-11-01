"""Q33 minimal paired execution using unchanged InferRSI. No GPU work at import.
All external sources are bound to a caller-reviewed SHA256 bundle, not fake Git.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
PROFILES={'trim-8':'trim-8','trim-6-screen-v1':'trim-6','trim-4-screen-v1':'trim-4'}
MEDIA_SOURCE_SHA='7fa12bd773e5933f2ce0e71c5be2f4124fc6c35284fbe5d3c624ab1b7012ca1a'
MEDIA_GATES_SHA='3e880606a5744fc290be9ca4011169be7dac65140e66ac3525e7b58b65a82e60'
GATES_SHA='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd'
METRIC='metrics.prompt_execution_seconds'
def require(ok,code):
 if not ok:raise ValueError(code)
def unique(pairs):
 out={}
 for k,v in pairs:require(k not in out,'duplicate_key');out[k]=v
 return out
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(512*1024):h.update(b)
 return h.hexdigest()
def read_json(path):
 p=Path(path);require(p.stat().st_size<=2**20,'json_bound');return json.loads(p.read_bytes(),object_pairs_hook=unique)
def save(path,obj):
 with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def profile_id(value):
 require(value in PROFILES,'profile_not_allowed');return PROFILES[value]
def positive(value):
 require(type(value) in (float,int) and math.isfinite(value) and value>0,'metric_invalid');return float(value)
def extract_metrics(r,history,media,profile):
 require(r.get('status')=='artifact_received_unvalidated' and r.get('profile_id')==profile,'controller_incomplete_or_profile')
 require(r.get('prompt_posts')==1 and r.get('submission_outcome')=='accepted','single_accepted_post')
 require(r.get('owned_root_exited') is True and r.get('job_close_requested') is True and not r.get('cleanup_diagnostics'),'owned_cleanup')
 t=r['node_timing'];pid=r['prompt_id']
 require(t.get('prompt_id')==pid and t.get('terminal')=='execution_success' and not t.get('invalid_reasons') and t.get('error_type') is None and t.get('observer_thread_exited') is True,'timing_incomplete')
 sampler=t['nodes']['14'];require(sampler.get('cached') is False,'cached_sampler');s=positive(sampler.get('observed_wall_seconds'))
 require(set(history)=={pid},'history_identity');status=history[pid]['status'];require(status.get('completed') is True and status.get('status_str')=='success','history_incomplete')
 endpoints={}
 for kind,body in status['messages']:
  if kind in ('execution_start','execution_success'):
   require(kind not in endpoints and body.get('prompt_id')==pid and type(body.get('timestamp')) is int,'history_endpoint_identity')
   endpoints[kind]=body['timestamp']
 require(set(endpoints)=={'execution_start','execution_success'},'history_endpoints_missing')
 seconds=positive((endpoints['execution_success']-endpoints['execution_start'])/1000)
 require(media.get('status')=='inspection_ready_human_review_pending' and media.get('owned_root_exited') is True and media.get('job_close_requested') is True and media.get('worker_returncode')==0,'media_incomplete')
 require(media.get('expected_input_sha256')==r['artifact']['sha256'] and media.get('source_sha256')==MEDIA_SOURCE_SHA and media.get('gates_sha256')==MEDIA_GATES_SHA,'media_identity')
 return {'prompt_execution_seconds':seconds,'sampler_wall_seconds':s}
def source_snapshot(bundle_path,expected_sha,roots):
 rows=[];available=True
 try:
  require(sha(bundle_path)==expected_sha,'bundle_sha');bundle=read_json(bundle_path);require(bundle.get('schema')=='q33-rsi-source-bundle/v1','bundle_schema')
  require(type(bundle.get('files')) is list and 0<len(bundle['files'])<=400,'bundle_count');seen=set()
  for item in bundle['files']:
   key=(item['root'],item['path']);require(key not in seen,'duplicate_source');seen.add(key)
   rel=Path(item['path']);root=Path(roots[item['root']]).resolve();path=root/rel
   require(not rel.is_absolute() and not rel.drive and '..' not in rel.parts and path.resolve().is_relative_to(root),'source_boundary')
   current={'root':item['root'],'path':item['path'],'expected_sha256':item['sha256']}
   try:
    before=path.stat();require(not path.is_symlink() and not (getattr(path.lstat(),'st_file_attributes',0)&0x400),'source_link')
    current['bytes']=before.st_size;current['sha256']=sha(path);after=path.stat()
    current['matches']=before.st_size==item['bytes'] and current['sha256']==item['sha256'] and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
   except (OSError,ValueError):current['matches']=False;current['error']='source_unavailable_or_changed'
   available=available and current['matches'];rows.append(current)
 except (OSError,ValueError,KeyError,TypeError):available=False
 return {'available':bool(available),'snapshot_hash':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'identity_kind':'sha256_file_manifest','bundle_sha256':expected_sha,'files':rows,'model_payload_scope':'Receipts/manifests here; original controller performs full model SHA per trial'}
def run_relative(run,runs_parent):
 run=Path(run).resolve();runs_parent=Path(runs_parent).resolve()
 require(run.parent==runs_parent,'explicit_runs_parent_boundary')
 return run.relative_to(runs_parent).as_posix()
def require_selected_pair(bundle,baseline,candidate):
 require(bundle.get('baseline_profile')==baseline=='trim-8' and bundle.get('candidate_profile')==candidate=='trim-6-screen-v1','pair_not_selected')
def main(argv=None):
 p=argparse.ArgumentParser(allow_abbrev=False)
 for n in ('notes-root','infra-root','portable-root','model-root','runs-parent','nvidia-smi','source-bundle','stop-file'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source-bundle-sha256',required=True);p.add_argument('--baseline',choices=tuple(PROFILES),required=True);p.add_argument('--candidate',choices=tuple(PROFILES),required=True)
 p.add_argument('--remaining-percent',type=int,required=True);p.add_argument('--max-minutes',type=float,required=True);p.add_argument('--execute',action='store_true')
 a=p.parse_args(argv)
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 require(a.baseline!=a.candidate,'same_profile');require(20<a.remaining_percent<=100,'quota_stop');require(math.isfinite(a.max_minutes) and 0<a.max_minutes<=30,'bounded_pair_minutes')
 Q=a.notes_root/'iterations/2026-10-31-speed-quality-search';C=Q/'agents/timing-controller';HERE=Path(__file__).resolve().parent
 require(HERE==Q/'agents/rsi-pair' and a.source_bundle.parent==HERE,'adapter_bundle_boundary')
 require(a.stop_file.is_relative_to(Q/'results') and a.stop_file.parent.is_dir(),'stop_boundary')
 require(a.runs_parent.drive.upper()=='E:' and a.runs_parent.is_dir(),'run_E_boundary')
 roots={'adapter':HERE,'notes':a.notes_root,'q33':Q,'infra':a.infra_root,'portable':a.portable_root}
 def snapshot(repo=None,excluded=None):return source_snapshot(a.source_bundle,a.source_bundle_sha256,roots)
 initial=snapshot();require(initial['available'],'frozen_source_mismatch');bundle=read_json(a.source_bundle)
 require_selected_pair(bundle,a.baseline,a.candidate)
 require(bundle.get('controller_name')=='speed_stage_b_v6.py' and bundle.get('reserve_gib')==3 and bundle.get('native_headroom_gib')==1.0,'reviewed_resource_recipe')
 require(bundle.get('launcher')=='tools/run_profile_v3.py' and bundle.get('media_wrapper')=='tools/validate_profile_media_v1.py','fixed_wrappers')
 require(any(x['root']=='adapter' and x['path']==Path(__file__).name and x['sha256']==sha(__file__) for x in bundle['files']),'adapter_self_not_frozen')
 require(not a.stop_file.exists(),'parent_stop')
 sys.path.insert(0,str(a.infra_root/'src'))
 from inferrsi.harness.runner import run_experiment
 from inferrsi.harness.scenarios.external import execute_trial
 from inferrsi.runspec import RunSpec
 from inferrsi.model.protocols.metrics import read_metric
 import inferrsi.harness.runner as runner_module
 require(Path(runner_module.__file__).resolve()==a.infra_root/'src/inferrsi/harness/runner.py','IR_import_source')
 ppy=a.portable_root/'python_embeded/python.exe';require(Path(sys.executable).resolve()==ppy,'portable_interpreter')
 def command(profile):
  return tuple(map(str,[ppy,'-s','-B','-X','utf8',Q/bundle['launcher'],'--notes-root',a.notes_root,'--infra-root',a.infra_root,'--portable-root',a.portable_root,'--model-root',a.model_root,'--runs-parent',a.runs_parent,'--nvidia-smi',a.nvidia_smi,'--profile',profile,'--controller-name',bundle['controller_name'],'--reserve-gib','3','--freeze-name',bundle['controller_freeze_name'],'--freeze-sha256',bundle['controller_freeze_sha256'],'--remaining-percent',a.remaining_percent]))
 commands={command(a.baseline):a.baseline,command(a.candidate):a.candidate}
 def execute(command_argv,cwd,trial_dir,deadline,metric_key):
  trial_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic();record={'status':'error','argv':list(command_argv),'cwd':str(cwd),'returncode':None,'result_path':str(trial_dir/'result.json'),'metric':None,'metric_error':None}
  raw={'schema':'q33-rsi-trial/v1','claim':'INCONCLUSIVE','perceptual_quality':'pending'}
  try:
   require(command_argv in commands,'unexpected_command');selected=commands[command_argv];raw['profile']=selected
   require(not a.stop_file.exists() and time.monotonic()<deadline,'parent_stop_or_deadline');require(snapshot()['available'],'frozen_source_mismatch_before_dispatch')
   child=execute_trial(command_argv,cwd,trial_dir/'launcher',deadline,None);raw['launcher_execution']=child
   if child['status']!='completed':record['status']=child['status'];raise ValueError('launcher_incomplete')
   out=Path(child['stdout_path']);require(out.stat().st_size<=65536,'launcher_stdout_bound')
   messages=[json.loads(line,object_pairs_hook=unique) for line in out.read_text(encoding='utf8').splitlines() if line.startswith('{')]
   dispatch=[x for x in messages if x.get('event')=='dispatch'];done=[x for x in messages if x.get('event')=='completed']
   require(len(dispatch)==len(done)==1,'unique_launcher_dispatch');d=dispatch[0]
   require(d.get('profile')==selected and done[0].get('status')=='completed' and done[0].get('returncode')==0,'inner_controller_incomplete')
   run=Path(d['run']).resolve();actual=Path(d['trial']).resolve();require(run.parent==a.runs_parent and actual.parent==Q/'results','launched_path_boundary')
   require(done[0]['run']==d['run'] and done[0]['trial']==d['trial'],'completion_identity')
   raw['run_relative_to_runs_parent']=run_relative(run,a.runs_parent);raw['controller_trial_relative_to_q33']=actual.relative_to(Q).as_posix()
   result=read_json(actual/'result.json');require(result.get('driver_sha256')==bundle['controller_sha256'] and result.get('gates_sha256')==GATES_SHA,'actual_controller_identity')
   require(result.get('owned_root_exited') is True and result.get('job_close_requested') is True,'controller_cleanup_before_media')
   require(not a.stop_file.exists() and time.monotonic()<deadline,'parent_stop_or_deadline');require(snapshot()['available'],'source_mismatch_before_media')
   media_command=tuple(map(str,[ppy,'-s','-B','-X','utf8',Q/bundle['media_wrapper'],'--notes-root',a.notes_root,'--infra-root',a.infra_root,'--portable-root',a.portable_root,'--run',run,'--trial',actual]))
   media_execution=execute_trial(media_command,cwd,trial_dir/'media-wrapper',min(deadline,time.monotonic()+145),None);raw['media_execution']=media_execution
   require(media_execution['status']=='completed','media_wrapper_incomplete')
   media=read_json(run/'media-inspection-v1/validation-summary.json');require(sha(run/'media-inspection-v1/decode-result.json')==media['decoder_result_sha256'],'media_decode_identity')
   history=read_json(run/'history.raw.json');metrics=extract_metrics(result,history,media,profile_id(selected))
   pf=read_json(C/'profiles'/f'{selected}.json');graph=read_json(C/pf['graph']['file'])
   require(history[result['prompt_id']]['prompt'][2]==graph['prompt'],'history_profile_graph')
   require(result['node_timing'].get('ready') is True,'observer_not_ready')
   for key,node in [('video_decode_wall_seconds','10'),('audio_decode_wall_seconds','23'),('save_video_wall_seconds','92')]:metrics[key]=positive(result['node_timing']['nodes'][node]['observed_wall_seconds'])
   metrics['controller_wall_seconds']=positive(result['elapsed_seconds']);raw['metrics']=metrics
   raw['references']=[{'root':'controller_trial','path':'result.json','sha256':sha(actual/'result.json')},{'root':'run','path':'history.raw.json','sha256':sha(run/'history.raw.json')},{'root':'run','path':'media-inspection-v1/validation-summary.json','sha256':sha(run/'media-inspection-v1/validation-summary.json')}]
   raw['media_structural_validation']='complete_human_perceptual_review_pending';record.update(status='completed',returncode=0,metric=read_metric(raw,metric_key))
  except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
   raw['error_type']=type(exc).__name__;raw['reason']=str(exc) if type(exc)is ValueError else 'external_or_evidence_error_details_omitted'
   record['error']=raw['reason'];record['metric_error']='trial_incomplete'
  record['elapsed_seconds']=time.monotonic()-started;save(trial_dir/'result.json',raw);save(trial_dir/'execution.json',record);return record
 asset=read_json(C/'assets/trim.json');modelsha=sha(C/'assets/trim.json')
 environment={'schema':'q33-rsi-pair-environment/v1','source_identity_kind':'sha256_file_manifest','source_bundle_sha256':a.source_bundle_sha256,'source_integrity_reason_note':'InferRSI runner legacy reason text says Git; actual adapter uses these verified file identities, not Git commits','model_assets':asset,'warm_policy':'Each trial new service + cache-none; reuse fixed conditioning; no extra warmup','reserve_vram_gib':3,'native_headroom_gib':1.0,'media_validation':'Frozen complete decoder per trial','perceptual_quality':'pending','acceptance_issue':None,'claim':'INCONCLUSIVE','parent_remaining_percent_at_launch':a.remaining_percent,'pair_stop':'checked before each owned child; parent controls active-job interruption'}
 spec=RunSpec(repo=str(HERE),candidate_repo=str(HERE),repo_type='comfyui',backend='cuda',scenario='h3.trim_step_screen',mode='semantic',baseline_command=command(a.baseline),candidate_command=command(a.candidate),validation_command=None,metric_key=METRIC,metric_unit='s',comparison_scope='native_comfy_graph_wall_reused_conditioning_cold_service',direction='minimize',repetitions=2,max_minutes=a.max_minutes,model='FastH3-Trim-NVFP4',model_revision='sha256:'+modelsha,workload_id=bundle['workload_sha256'],revision='sha256:'+a.source_bundle_sha256,snapshot_id=a.source_bundle_sha256,device_ids=('0',),device='RTX5070',executor='local',output_dir=str(Q/'results/rsi-pairs'))
 spec.validate()
 if not a.execute:print(json.dumps({'preview':spec.to_dict(),'identity_available':True,'dispatches':0},indent=2));return 0
 require(bundle.get('dispatch_ready') is True,'bundle_not_dispatch_ready')
 directory=run_experiment(spec,environment,execute=execute,snapshot=snapshot)
 evaluation=read_json(directory/'evaluation.json');print(json.dumps({'run_directory':str(directory),'verdict':evaluation['verdict'],'scope':'descriptive paired screening; human quality pending'}))
 require(evaluation['verdict']=='INCONCLUSIVE','unexpected_quality_verdict');return 2
if __name__=='__main__':raise SystemExit(main())