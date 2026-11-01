"""H3-only warm second-request ABBA adapter; unchanged InferRSI and native controllers."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys,time
BASE_SHA='4830c462866ea832481a1a97f0989ed01481911126998b61d5f6c1fe728a8b6c'
_base_path=Path(__file__).resolve().parents[1]/'rsi-pair/rsi_pair_v2.py'
if hashlib.sha256(_base_path.read_bytes()).hexdigest()!=BASE_SHA:raise RuntimeError('frozen base helper drift')
_bs=importlib.util.spec_from_file_location('frozen_rsi_pair_helpers',_base_path);base=importlib.util.module_from_spec(_bs);_bs.loader.exec_module(base)
if Path(base.__file__).resolve()!=_base_path.resolve():raise RuntimeError('actual base import source')
require=base.require;sha=base.sha;read_json=base.read_json;save=base.save;positive=base.positive
SIDES={'A': {'directory': 'agents/offload-preparation', 'headroom_gib': 1.0, 'mode': 'async-persistent-video-stage-b', 'controller': 'async_persistent_stage_b_v1.py', 'controller_sha256': 'be5ed4634a212e0d68d9b014f04ac2ca8e8547ade62c93967e73ccdecb9a8bfe', 'launcher': 'run_async_persistent_v1.py', 'launcher_sha256': '618638a30e2092deb676981657cfd106b0960c843d3139f5026709452217a6f8', 'media_helper': 'run_async_persistent_media_v1.py', 'media_helper_sha256': '460e98178bc28649a822286acb54137fb298fed150305228695efe5206183617', 'media_validator_sha256': '27d8a29e51d9e18da93e803e16eaa8276445e103665c2eeef7dc1e6ee746b84e'}, 'B': {'directory': 'agents/headroom-preparation', 'headroom_gib': 0.75, 'mode': 'headroom-persistent-video-stage-b', 'controller': 'headroom_persistent_stage_b_v1.py', 'controller_sha256': '2455f4c923639103a294fc53145ff6e3ef4eb2c6fa7dfd36014a5a50905981e1', 'launcher': 'run_headroom_persistent_v1.py', 'launcher_sha256': '83788ccb1ccc370a413ef01340a447edc324faed915e7c5e14c1e13692813741', 'media_helper': 'run_headroom_persistent_media_v1.py', 'media_helper_sha256': '28f82690c0630e69ea4256eec2c9cdb184ddd804686e25f151e1c9ef755ba351', 'media_validator_sha256': '10fbe161061af3b38b8fd8d926771be1940d2fb4b7e379c646784da3470e4367'}}
GATES_SHA='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd'
SEEDS=(424242,424243)
GRAPHS=('2c262a7d706ee09111d961f4f0bdad504cd6f33c89eee013e8c335cce212b6f5','c6b6f4a436b4899495a25009e860ac917fda38baf55920e35d76b0bcc71bee48')
OUTPUTS=((240122,'9c0ec62b2694f56299a15872e2b4259777dbe91428c7bea4a85a45e15b91d226'),(312427,'899f475cf381e417873470707b119c3e5bb5ff65386cb0858f5ab8fa8887c501'))
METRIC='metrics.warm_second_native_history_seconds'
def extract_warm_metrics(r,histories,media,side):
 cfg=SIDES[side]
 require(r.get('status')=='two_artifacts_received_unvalidated' and r.get('profile_id')=='trim-8' and r.get('mode')==cfg['mode'],'complete_recipe')
 require(r.get('driver_sha256')==cfg['controller_sha256'] and r.get('gates_sha256')==GATES_SHA and r.get('requested_aimdo_vram_headroom_gib')==cfg['headroom_gib'],'controller_gates_headroom')
 require(r.get('prompt_posts')==r.get('completed_requests')==2 and len(r['requests'])==len(histories)==2,'two_requests')
 require(r.get('owned_root_exited') is True and r.get('job_close_requested') is True and r.get('cleanup_diagnostics')==[],'owned_cleanup')
 require(r.get('cache_policy')=='classic' and r.get('memory_policy')=='smart' and r.get('request_seeds')==list(SEEDS) and r.get('requested_async_offload_streams')==2 and r.get('disable_async_offload_requested') is False and r.get('disable_pinned_memory_requested') is True,'same_infra_except_headroom')
 require(media.get('status')=='two_media_inspections_ready_human_review_pending' and media.get('source_controller_sha256')==cfg['controller_sha256'] and media.get('driver_sha256')==cfg['media_validator_sha256'] and media.get('request_seeds')==list(SEEDS) and len(media['requests'])==2,'two_complete_media')
 seconds=[];samplers=[]
 for i,(x,h,m) in enumerate(zip(r['requests'],histories,media['requests'])):
  require(x.get('request_index')==i+1 and x.get('status')=='complete' and x.get('post_attempted') is True and x.get('submission_outcome')=='accepted' and x.get('seed')==SEEDS[i] and x.get('graph_sha256')==GRAPHS[i],'request_recipe')
  art=x['artifact'];require((art['bytes'],art['sha256'])==OUTPUTS[i],'whole_mp4_expected_identity')
  require(m.get('request_index')==i+1 and m.get('prompt_id')==x['prompt_id'] and m.get('status')=='inspection_ready_human_review_pending' and m.get('returncode')==0 and m.get('execution_status')=='completed' and m.get('artifact')==art,'per_request_media')
  ti=x['node_timing'];require(ti.get('prompt_id')==x['prompt_id'] and ti.get('ready') is True and ti.get('terminal')=='execution_success' and ti.get('invalid_reasons')==[] and ti.get('error_type') is None and ti.get('observer_thread_exited') is True,'timing_complete')
  sn=ti['nodes']['14'];require(sn.get('cached') is False and sn.get('progress_value')==sn.get('progress_max')==8,'eight_uncached_steps');samplers.append(positive(sn['observed_wall_seconds']))
  pid=x['prompt_id'];require(set(h)=={pid} and h[pid]['status'].get('completed') is True and h[pid]['status'].get('status_str')=='success','history_complete')
  ep={}
  for kind,body in h[pid]['status']['messages']:
   if kind in ('execution_start','execution_success'):
    require(kind not in ep and body.get('prompt_id')==pid and type(body.get('timestamp')) is int,'history_endpoints');ep[kind]=body['timestamp']
  require(set(ep)=={'execution_start','execution_success'},'history_endpoint_missing');seconds.append(positive((ep['execution_success']-ep['execution_start'])/1000))
  require(seconds[-1]==x['metrics']['native_prompt_wall_seconds'],'history_metric_mismatch')
 return {'warm_second_native_history_seconds':seconds[1],'priming_first_native_history_seconds':seconds[0],'warm_second_sampler_observed_seconds':samplers[1],'priming_first_sampler_observed_seconds':samplers[0],'controller_wall_seconds':positive(r['elapsed_seconds'])}
def child_path(root,relative):
 rel=Path(relative);root=Path(root).resolve();path=root/rel
 require(not rel.is_absolute() and not rel.drive and '..' not in rel.parts and path.resolve().is_relative_to(root),'child_path_boundary')
 for node in [root,*[root/Path(*rel.parts[:i]) for i in range(1,len(rel.parts)+1)]]:
  st=node.lstat();require(not node.is_symlink() and not(getattr(st,'st_file_attributes',0)&0x400),'child_link_reparse')
 return path

def main(argv=None):
 p=argparse.ArgumentParser(allow_abbrev=False)
 for n in ('notes-root','infra-root','portable-root','model-root','runs-parent','nvidia-smi','source-bundle','stop-file'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source-bundle-sha256',required=True);p.add_argument('--remaining-percent',type=int,required=True);p.add_argument('--max-minutes',type=float,default=10);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 require(20<a.remaining_percent<=100 and math.isfinite(a.max_minutes) and 0<a.max_minutes<=10,'quota_or_deadline')
 Q=a.notes_root/'iterations/2026-10-31-speed-quality-search';HERE=Path(__file__).resolve().parent
 require(HERE==Q/'agents/headroom-rsi-pair' and a.source_bundle.parent==HERE,'adapter_bundle_boundary');require(a.runs_parent.drive.upper()=='E:' and a.runs_parent.is_dir(),'runs_E_boundary')
 require(a.stop_file.is_relative_to(Q/'results') and a.stop_file.parent.is_dir() and not a.stop_file.exists(),'stop_boundary_or_requested')
 roots={'adapter':HERE,'rsi_base':Q/'agents/rsi-pair','notes':a.notes_root,'q33':Q,'infra':a.infra_root,'portable':a.portable_root}
 def snapshot(repo=None,excluded=None):return base.source_snapshot(a.source_bundle,a.source_bundle_sha256,roots)
 initial=snapshot();require(initial['available'],'frozen_sources_unavailable');bundle=read_json(a.source_bundle)
 require(bundle.get('purpose')=='q33-headroom-warm-pair/v1' and bundle.get('sides')==SIDES and bundle.get('schedule')==['A','B','B','A'] and bundle.get('repetitions')==2,'bundle_recipe')
 require(any(x['root']=='adapter' and x['path']==Path(__file__).name and x['sha256']==sha(__file__) for x in bundle['files']),'self_not_frozen')
 ppy=a.portable_root/'python_embeded/python.exe';require(Path(sys.executable).resolve()==ppy,'portable_interpreter')
 sys.path.insert(0,str(a.infra_root/'src'))
 from inferrsi.harness.runner import run_experiment
 from inferrsi.harness.scenarios.external import execute_trial
 from inferrsi.runspec import RunSpec
 from inferrsi.model.protocols.metrics import read_metric
 import inferrsi.harness.runner as irm
 import inferrsi.harness.scenarios.external as iem
 import inferrsi.infra.executors.local as ilm
 import inferrsi.infra.processes as ipm
 for module,rel in [(irm,'harness/runner.py'),(iem,'harness/scenarios/external.py'),(ilm,'infra/executors/local.py'),(ipm,'infra/processes.py')]:require(Path(module.__file__).resolve()==a.infra_root/'src/inferrsi'/rel,'actual_IR_import_source')
 def command(side):
  cfg=SIDES[side];return tuple(map(str,[ppy,'-s','-B','-X','utf8',Q/cfg['directory']/cfg['launcher'],'--notes-root',a.notes_root,'--infra-root',a.infra_root,'--portable-root',a.portable_root,'--model-root',a.model_root,'--runs-parent',a.runs_parent,'--nvidia-smi',a.nvidia_smi,'--cache-policy','classic','--remaining-percent',a.remaining_percent]))
 commands={command(side):side for side in SIDES}
 def still_allowed(deadline):require(not a.stop_file.exists() and time.monotonic()<deadline,'parent_stop_or_deadline');require(snapshot()['available'],'source_drift_before_dispatch')
 def messages(path):
  require(path.stat().st_size<=65536,'stdout_bound');return [json.loads(s,object_pairs_hook=base.unique) for s in path.read_text(encoding='utf8').splitlines() if s.startswith('{')]
 def execute(command_argv,cwd,trial_dir,deadline,metric_key):
  trial_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic();rec={'status':'error','argv':list(command_argv),'cwd':str(cwd),'returncode':None,'result_path':str(trial_dir/'result.json'),'metric':None,'metric_error':None};out={'schema':'q33-headroom-rsi-member/v1','claim':'INCONCLUSIVE','quality':'exact_fixed_MP4_identity_only'}
  try:
   require(command_argv in commands,'unexpected_command');side=commands[command_argv];cfg=SIDES[side];out['side']=side;still_allowed(deadline)
   launched=execute_trial(command_argv,cwd,trial_dir/'launcher',deadline,None);out['launcher_execution']=launched;require(launched['status']=='completed','launcher_failed')
   msgs=messages(Path(launched['stdout_path']));dispatch=[x for x in msgs if x.get('event')=='dispatch'];done=[x for x in msgs if x.get('event')=='completed'];require(len(dispatch)==len(done)==1,'unique_dispatch_completion');d=dispatch[0];f=done[0]
   require(f.get('status')=='completed' and f.get('returncode')==0 and all(f[k]==d[k] for k in ('run','trial')),'controller_failed')
   run=Path(d['run']).resolve();actual=Path(d['trial']).resolve();require(run.parent==a.runs_parent and actual.parent==Q/'results','launched_boundary');out['run_relative_to_runs_parent']=run.relative_to(a.runs_parent).as_posix();out['controller_trial_relative_to_q33']=actual.relative_to(Q).as_posix()
   r=read_json(actual/'result.json');ce=read_json(actual/'execution.json');require(r.get('owned_root_exited') is True and r.get('job_close_requested') is True and r.get('cleanup_diagnostics')==[] and ce['returncode']==0,'cleanup_before_media')
   require(r['driver_sha256']==cfg['controller_sha256'],'side_controller');still_allowed(deadline)
   mc=tuple(map(str,[ppy,'-s','-B','-X','utf8',Q/cfg['directory']/cfg['media_helper'],'--notes-root',a.notes_root,'--infra-root',a.infra_root,'--portable-root',a.portable_root,'--run',run,'--trial',actual]))
   medexec=execute_trial(mc,cwd,trial_dir/'media-wrapper',min(deadline,time.monotonic()+340),None);out['media_execution']=medexec;require(medexec['status']=='completed','media_wrapper_failed');mm=messages(Path(medexec['stdout_path']));require(len(mm)==1 and mm[0].get('status')=='completed' and mm[0].get('returncode')==0,'media_completion');mt=Path(mm[0]['trial']).resolve();require(mt.parent==Q/'results','media_trial_boundary');media=read_json(mt/'result.json');require(media['source_result_sha256']==sha(actual/'result.json'),'media_source_result');out['media_trial_relative_to_q33']=mt.relative_to(Q).as_posix()
   histories=[];identities=[]
   for i,x in enumerate(r['requests']):
    art=x['artifact'];file=child_path(run,art['relative_to_run']);before=file.stat();require(0<before.st_size<=2**20 and before.st_size==art['bytes'] and sha(file)==art['sha256'],'actual_MP4_hash');after=file.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'MP4_changed')
    hp=child_path(run,x['history_relative_to_run']);h=read_json(hp);histories.append(h);graph=read_json(hp.parent/'graph.raw.json');require(hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==x['graph_sha256'] and graph['prompt']==h[x['prompt_id']]['prompt'][2],'saved_and_actual_graph')
    require(read_json(child_path(run,x['timing_relative_to_run']))==x['node_timing'],'request_timing_file')
    mr=media['requests'][i];md=child_path(run,mr['output_relative_to_run']);su=read_json(md/'validation-summary.json');de=read_json(md/'decode-result.json');require(sha(md/'validation-summary.json')==mr['summary_sha256'] and sha(md/'decode-result.json')==mr['decoder_result_sha256'],'media_report_hashes')
    require(su['status']=='inspection_ready_human_review_pending' and su['owned_root_exited'] is True and su['job_close_requested'] is True and su['worker_returncode']==0 and su['expected_input_sha256']==art['sha256'] and su['source_sha256']==base.MEDIA_SOURCE_SHA and su['gates_sha256']==base.MEDIA_GATES_SHA,'media_owned_identity')
    require(de['full_demux_eof'] is True and de['container_closed'] is True and de['input_unchanged'] is True and de['input_sha256']==art['sha256'] and de['video']['decoded_frames']==73 and de['video']['width']==384 and de['video']['height']==256 and de['audio']['samples_per_channel']==97334,'complete_media_structure')
    identities.append({'seed':x['seed'],'artifact':art,'whole_MP4_rehashed':True,'summary_sha256':mr['summary_sha256'],'decoder_result_sha256':mr['decoder_result_sha256']})
   metrics=extract_warm_metrics(r,histories,media,side);metrics['controller_outer_seconds']=positive(ce['elapsed_seconds']);metrics['launcher_wall_seconds']=positive(launched['elapsed_seconds']);out['metrics']=metrics;out['artifacts']=identities
   rp=actual/'resources.jsonl';require(rp.stat().st_size<=4*2**20,'resource_log_bound');rr=[json.loads(x,object_pairs_hook=base.unique) for x in rp.read_text(encoding='utf8').splitlines()];require(rr,'resource_rows_missing');out['sampled_resources']={'whole_board_min_free_bytes':min(x['gpu_board']['free_bytes'] for x in rr if x.get('gpu_board')),'owned_private_peak_bytes':max(x['owned_private_sum_bytes'] for x in rr),'owned_rss_peak_bytes':max(x['owned_rss_sum_bytes'] for x in rr),'host_commit_min_bytes':min(x['available_commit_bytes'] for x in rr),'measurement':'sampled whole board/owned processes, not allocator or full peak capture'}
   out['references']=[{'relative_to_q33':p.relative_to(Q).as_posix(),'sha256':sha(p)} for p in (actual/'result.json',actual/'execution.json',rp,mt/'result.json')];rec.update(status='completed',returncode=0,metric=read_metric(out,metric_key))
  except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
   out['error_type']=type(exc).__name__;out['reason']=str(exc) if type(exc)is ValueError else 'external_or_evidence_error_details_omitted';rec['metric_error']='member_incomplete';rec['error']=out['reason']
  rec['elapsed_seconds']=time.monotonic()-started;save(trial_dir/'result.json',out);save(trial_dir/'execution.json',rec);return rec
 assets=read_json(Q/'agents/timing-controller/assets/trim.json');env={'schema':'q33-headroom-warm-environment/v1','source_identity_kind':'sha256_file_manifest','source_bundle_sha256':a.source_bundle_sha256,'legacy_source_reason_note':'IR source reason says Git but snapshot is the verified file manifest; no Git claim','sides':SIDES,'warm_policy':'Each member fresh service; first seed priming+validated; second seed primary metric; both media checked','fixed_seeds':list(SEEDS),'fixed_output_identities':OUTPUTS,'acceptance_issue':None,'quality_scope':'exact bytes for two fixed videos only; no general quality validator','claim':'INCONCLUSIVE','remaining_percent_at_parent_launch':a.remaining_percent,'stop_scope':'between children; active-child deadline and owned Job cleanup, parent controls immediate stop'}
 spec=RunSpec(repo=str(HERE),candidate_repo=str(HERE),repo_type='comfyui',backend='cuda',scenario='h3.headroom_warm_pair',mode='semantic',baseline_command=command('A'),candidate_command=command('B'),validation_command=None,metric_key=METRIC,metric_unit='s',comparison_scope='native_history_warm_second_request_reused_conditioning_excludes_TE_startup_asset_hash',direction='minimize',repetitions=2,max_minutes=a.max_minutes,model='FastH3-Trim-NVFP4',model_revision='sha256:'+sha(Q/'agents/timing-controller/assets/trim.json'),workload_id=bundle['workload_sha256'],revision='sha256:'+a.source_bundle_sha256,snapshot_id=a.source_bundle_sha256,device_ids=('0',),device='RTX5070',executor='local',output_dir=str(Q/'results/rsi-headroom-pairs'));spec.validate()
 if not a.execute:print(json.dumps({'preview':spec.to_dict(),'verified_source_files':len(bundle['files']),'actual_imports_checked':True,'dispatches':0},indent=2));return 0
 require(bundle.get('dispatch_ready') is True,'bundle_not_ready');directory=run_experiment(spec,env,execute=execute,snapshot=snapshot);evaluation=read_json(directory/'evaluation.json');require(evaluation['verdict']=='INCONCLUSIVE','unexpected_formal_verdict');print(json.dumps({'run_directory':str(directory),'verdict':evaluation['verdict'],'scope':'ABBA warm-second descriptive screen; quality limited to fixed MP4 identity'}));return 2
if __name__=='__main__':raise SystemExit(main())
