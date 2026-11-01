"""Validate the sole actual StageB MP4 after one-service new-prompt cleanup."""
import argparse,hashlib,json,sys,time,uuid
from pathlib import Path
CONTROLLER_SHA='378b3830cb85edaaec9c9b6a1539e084c3a2a04ccf74ed088c527ccd6b8ff903'
p=argparse.ArgumentParser(allow_abbrev=False)
for n in ('notes-root','infra-root','portable-root','run','trial'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--result-sha256',required=True);a=p.parse_args()
R=a.notes_root.resolve();Q=R/'iterations/2026-10-31-speed-quality-search';Q31=R/'iterations/2026-10-28-comfy-trim-first-video';D=Q/'agents/end-to-end-preparation';IR=a.infra_root.resolve();run=a.run.resolve();prior=a.trial.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def redact(text):
 for path,label in [(R,'R'),(IR,'INFERRSI'),(a.portable_root.resolve(),'PORTABLE'),(run,'OWNED_RUN')]:
  text=text.replace(str(path).replace('\\','\\\\'),label).replace(str(path).replace('\\','/'),label)
 return text
assert Path(__file__).resolve().parent==D and run.drive.upper()=='E:' and prior.parent==Q/'results'
assert sha(D/'end_to_end_v1.py')==CONTROLLER_SHA
rp=prior/'result.json';assert rp.stat().st_size<=2**20 and sha(rp)==a.result_sha256;r=json.loads(rp.read_text(encoding='utf-8'))
assert r['status']=='new_prompt_video_received_unvalidated' and r['mode']=='new-prompt-trim8' and r['driver_sha256']==CONTROLLER_SHA
assert r['owned_root_exited'] and r['job_close_requested'] and not r.get('cleanup_diagnostics')
assert r['completed_requests']==r['prompt_posts']==2 and r['free_posts']==1 and r['release']['fresh_admitted'] and r['release']['reset_path_entered']
assert r['gates_sha256']=='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd' and r['source_manifest_sha256']=='bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007'
assert r['requests'][0]['artifact']['sha256']=='70a19cfeb46fe9411ac6f3fbfea63c253056d28cffb9bb28f269da4c2134d170' and r['requests'][0]['conditioning_value_validation']['finite_values_checked']
assert all(x['status']=='complete' for x in r['requests']) and r['artifact']==r['requests'][1]['artifact']
f=r['artifact'];rel=Path(f['relative_to_run']);assert not rel.is_absolute() and not rel.drive and '..' not in rel.parts and rel.parts[0]=='output'
video=(run/rel).resolve();assert video.is_relative_to(run/'output') and video.suffix=='.mp4' and video.stat().st_size==f['bytes']
script=Q31/'tools/validate_first_video_media_v1.py';gates=Q31/'plan/MEDIA-VALIDATION-GATES-v1.json';monitor=Q31/'tools/comfy_first_video_v1.py'
assert sha(script)=='7fa12bd773e5933f2ce0e71c5be2f4124fc6c35284fbe5d3c624ab1b7012ca1a' and sha(gates)=='3e880606a5744fc290be9ca4011169be7dac65140e66ac3525e7b58b65a82e60' and sha(monitor)=='63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35'
out=run/'media-inspection-v1';assert not out.exists();trial=Q/'results'/('media-e2e-'+uuid.uuid4().hex)
argv=[str(a.portable_root/'python_embeded/python.exe'),'-s','-B','-X','utf8',str(script),'--input',str(video),'--run-root',str(run),'--output-dir',str(out),'--gates',str(gates),'--monitor-source',str(monitor),'--inferrsi-src',str(IR/'src'),'--expected-sha256',f['sha256'],'--gates-sha256','3e880606a5744fc290be9ca4011169be7dac65140e66ac3525e7b58b65a82e60']
sys.path.insert(0,str(IR/'src'));from inferrsi.infra.executors.local import execute_command
rec=execute_command(tuple(argv),IR,trial,time.monotonic()+140)
with (trial/'execution.json').open('x',encoding='utf-8') as stream:stream.write(redact(json.dumps(rec,indent=2)))
with (trial/'SOURCE-v1.json').open('x',encoding='utf-8') as stream:json.dump({'generation_result_sha256':a.result_sha256,'prior_trial_relative_to_iteration':prior.relative_to(Q).as_posix(),'controller_sha256':CONTROLLER_SHA,'media_source_sha256':sha(script),'artifact':f,'claim':'INCONCLUSIVE'},stream,indent=2)
print(json.dumps({'status':rec['status'],'returncode':rec['returncode'],'seconds':rec['elapsed_seconds'],'trial_relative_to_iteration':trial.relative_to(Q).as_posix()}))
