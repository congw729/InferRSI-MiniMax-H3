"""Two actual persistent artifacts -> unchanged CPU media validator, after GPU cleanup."""
from pathlib import Path
import argparse,hashlib,json,os,re,sys,time,uuid
CONTROLLER_SHA='f27dff8a8792a99c47d9ba71f5ddd005a95bebd950e8f365b51cb4685b4c31b8'
CONTROLLER_GATES_SHA='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd'
SOURCES_SHA='bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007'
MEDIA_SHA='7fa12bd773e5933f2ce0e71c5be2f4124fc6c35284fbe5d3c624ab1b7012ca1a'
MEDIA_GATES_SHA='3e880606a5744fc290be9ca4011169be7dac65140e66ac3525e7b58b65a82e60'
MONITOR_SHA='63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35'
def require(ok,why):
 if not ok:raise RuntimeError(why)
def unique(pairs):
 d={}
 for k,v in pairs:
  require(k not in d,'duplicate JSON key');d[k]=v
 return d
def read(p,limit=2**20):
 require(p.is_file() and 0<p.stat().st_size<=limit,'bounded metadata file')
 return json.loads(p.read_bytes(),object_pairs_hook=unique)
def save(p,value):
 with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def signature(p):
 s=p.stat();return s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino
def sha(p):
 h=hashlib.sha256();before=signature(p)
 with p.open('rb') as f:
  for block in iter(lambda:f.read(512*1024),b''):h.update(block)
 require(signature(p)==before,'file identity changed while hashing');return h.hexdigest()
def safe_relative(value):
 require(type(value) is str,'relative output string');p=Path(value)
 require(not p.is_absolute() and not p.drive and '..' not in p.parts and p.suffix.lower()=='.mp4','relative MP4 boundary')
 return p
def validate_session(r):
 require(r.get('status')=='two_artifacts_received_unvalidated' and r.get('mode')=='persistent-video-stage-b'
  and r.get('profile_id')=='trim-8' and r.get('driver_sha256')==CONTROLLER_SHA,'actual persistent controller complete')
 require(r.get('gates_sha256')==CONTROLLER_GATES_SHA and r.get('source_manifest_sha256')==SOURCES_SHA,'source/gates identity')
 require(r.get('owned_root_exited') is True and r.get('job_close_requested') is True and not r.get('cleanup_diagnostics'),'GPU service cleanup required')
 require(r.get('memory_policy')=='smart' and r.get('cache_policy') in ('none','classic')
  and r.get('disable_smart_memory_requested') is False and r.get('request_seeds')==[424242,424243],'recorded cache configuration and seed schedule')
 require(r.get('completed_requests')==r.get('prompt_posts')==2 and type(r.get('requests')) is list and len(r['requests'])==2,'two actual requests required')
 pids=set();paths=set();total=0
 for i,x in enumerate(r['requests'],1):
  require(x.get('request_index')==i and x.get('seed')==424241+i and x.get('status')=='complete' and x.get('submission_outcome')=='accepted' and x.get('post_attempted') is True,'request dispatch identity')
  require(type(x.get('prompt_id')) is str and x['prompt_id'] and x['prompt_id'] not in pids,'distinct prompt ids');pids.add(x['prompt_id'])
  f=x['artifact'];rel=safe_relative(f['relative_to_run']);require(rel not in paths,'distinct outputs');paths.add(rel)
  require(type(f['bytes']) is int and 0<f['bytes']<=128*1024**2 and type(f['sha256']) is str and re.fullmatch('[0-9a-f]{64}',f['sha256']),'artifact size/hash')
  total+=f['bytes']
 require(total<=128*1024**2,'aggregate media bound')
 return r['requests']
def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for n in ('notes-root','infra-root','portable-root','run','trial','stop-marker'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--result-sha256',required=True);a=p.parse_args()
 for k,v in vars(a).items():
  if isinstance(v,Path):setattr(a,k,v.resolve())
 R=a.notes_root;Q=R/'iterations/2026-10-31-speed-quality-search';Q31=R/'iterations/2026-10-28-comfy-trim-first-video';D=Q/'agents/persistent-preparation';IR=a.infra_root
 require(Path(__file__).resolve().parent==D and a.run.drive.upper()=='E:' and a.run.is_dir(),'owned notes/run roots')
 require(a.trial.parent==Q/'results' and a.trial.is_dir() and a.stop_marker.parent==a.run,'trial/stop boundary')
 require(os.environ.get('INFERRSI_RESULT_PATH'),'use InferRSI execute_command')
 result_path=Path(os.environ['INFERRSI_RESULT_PATH']).resolve();require(result_path.parent.parent==Q/'results' and result_path.parent.is_dir() and not result_path.exists(),'new wrapper result boundary')
 require(Path(sys.executable).resolve()==a.portable_root/'python_embeded/python.exe','installed portable interpreter')
 started=time.monotonic();deadline=started+300;state={'schema':'q33-persistent-two-media/v2','status':'preflight','claim':'INCONCLUSIVE','quality':'not_assessed','source_result_sha256':a.result_sha256,'driver_sha256':sha(Path(__file__)),'requests':[]}
 def boundary():require(not a.stop_marker.exists() and time.monotonic()<deadline,'parent stop or aggregate deadline')
 try:
  boundary();rp=a.trial/'result.json';require(sha(rp)==a.result_sha256,'actual aggregate result identity');r=read(rp);requests=validate_session(r)
  state['memory_policy']=r['memory_policy'];state['cache_policy']=r['cache_policy'];state['request_seeds']=r['request_seeds'];state['source_controller_sha256']=r['driver_sha256'];state['source_trial_relative_to_iteration']=a.trial.relative_to(Q).as_posix()
  require(sha(D/'persistent_stage_b_v1.py')==CONTROLLER_SHA,'controller source drift')
  script=Q31/'tools/validate_first_video_media_v1.py';gates=Q31/'plan/MEDIA-VALIDATION-GATES-v1.json';monitor=Q31/'tools/comfy_first_video_v1.py'
  require(sha(script)==MEDIA_SHA and sha(gates)==MEDIA_GATES_SHA and sha(monitor)==MONITOR_SHA,'original media tools identity')
  source=Q/'agents/timing-controller/SOURCES-v1.json';require(sha(source)==SOURCES_SHA,'runtime source manifest drift')
  roots={'portable':a.portable_root,'inferrsi':IR};manifest=read(source)
  for item in manifest['sources']:
   path=roots[item['root']]/item['path'];require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'selected runtime source drift')
  videos=[]
  for x in requests:
   boundary();f=x['artifact'];rel=safe_relative(f['relative_to_run']);video=a.run/rel
   require(video.resolve().is_relative_to(a.run) and video.is_file() and not video.is_symlink() and not(getattr(video.lstat(),'st_file_attributes',0)&0x400),'actual video path boundary')
   require(video.stat().st_size==f['bytes'] and sha(video)==f['sha256'],'actual MP4 size/fullSHA')
   out=a.run/('media-inspection-request-'+str(x['request_index'])+'-v2');require(not out.exists(),'fresh per-request media output');videos.append((x,video,out))
  sys.path.insert(0,str(IR/'src'));import inferrsi.infra.executors.local as local
  require(Path(local.__file__).resolve()==IR/'src/inferrsi/infra/executors/local.py','actual executor source')
  for x,video,out in videos:
   boundary();require(sha(rp)==a.result_sha256,'aggregate result changed');f=x['artifact'];nonce=uuid.uuid4().hex
   trial=Q/'results'/('persistent-cache-media-'+r['cache_policy']+'-request-'+str(x['request_index'])+'-'+nonce)
   argv=[str(a.portable_root/'python_embeded/python.exe'),'-s','-B','-X','utf8',str(script),'--input',str(video),'--run-root',str(a.run),'--output-dir',str(out),'--gates',str(gates),'--monitor-source',str(monitor),'--inferrsi-src',str(IR/'src'),'--expected-sha256',f['sha256'],'--gates-sha256',MEDIA_GATES_SHA]
   entry={'request_index':x['request_index'],'prompt_id':x['prompt_id'],'artifact':f,'trial_relative_to_iteration':trial.relative_to(Q).as_posix(),'output_relative_to_run':out.relative_to(a.run).as_posix(),'status':'executing'};state['requests'].append(entry)
   rec=local.execute_command(tuple(argv),IR,trial,min(deadline,time.monotonic()+140));save(trial/'execution.json',rec)
   entry.update(execution_status=rec['status'],returncode=rec['returncode'],elapsed_seconds=rec['elapsed_seconds'],execution_sha256=sha(trial/'execution.json'))
   require(rec['status']=='completed' and rec['returncode']==0,'media worker incomplete')
   summary=read(out/'validation-summary.json',2**21);decoded=read(out/'decode-result.json',2**21)
   require(summary.get('status')=='inspection_ready_human_review_pending' and summary.get('owned_root_exited') is True and summary.get('job_close_requested') is True and summary.get('expected_input_sha256')==f['sha256'],'media summary cleanup/input')
   require(summary.get('source_sha256')==MEDIA_SHA and summary.get('gates_sha256')==MEDIA_GATES_SHA and summary.get('decoder_result_sha256')==sha(out/'decode-result.json'),'media source/result identity')
   require(decoded.get('status')=='decoded_structure_checked_human_review_pending' and decoded.get('input_sha256')==f['sha256'] and decoded.get('full_demux_eof') is True and decoded.get('container_closed') is True and decoded.get('input_unchanged') is True,'full media decode identity')
   entry.update(status='inspection_ready_human_review_pending',summary_sha256=sha(out/'validation-summary.json'),decoder_result_sha256=sha(out/'decode-result.json'),video=decoded['video'],audio=decoded['audio'])
  require(len(state['requests'])==2 and sha(rp)==a.result_sha256,'two completed media inputs');state['status']='two_media_inspections_ready_human_review_pending'
 except BaseException as exc:
  state['status']='cancelled' if isinstance(exc,KeyboardInterrupt) or a.stop_marker.exists() else 'failed';state['error_type']=type(exc).__name__;state['error']=str(exc) if type(exc) is RuntimeError else 'external operation failed; raw evidence retained'
 finally:
  state['elapsed_seconds']=time.monotonic()-started;save(result_path,state)
 print(json.dumps({'status':state['status'],'memory_policy':state.get('memory_policy'),'media_attempts':len(state['requests']),'claim':'INCONCLUSIVE'}))
 return 0 if state['status']=='two_media_inspections_ready_human_review_pending' else 1
if __name__=='__main__':raise SystemExit(main())
