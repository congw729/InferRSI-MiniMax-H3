"""Reuse the unchanged Q31 CPU full-media validator after owned GPU cleanup."""
import argparse,hashlib,json,sys,time,uuid
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('notes-root','infra-root','portable-root','run','trial'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();R=a.notes_root.resolve();Q=R/'iterations/2026-10-31-speed-quality-search';Q31=R/'iterations/2026-10-28-comfy-trim-first-video';IR=a.infra_root.resolve()
assert a.run.resolve().drive.upper()=='E:' and a.trial.resolve().is_relative_to(Q/'results')
r=json.loads((a.trial/'result.json').read_text(encoding='utf-8'))
assert r['status']=='artifact_received_unvalidated' and r['owned_root_exited'] and r['job_close_requested']
f=r['artifact'];video=a.run/f['relative_to_run'];assert video.resolve().is_relative_to(a.run.resolve())
script=Q31/'tools/validate_first_video_media_v1.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='7fa12bd773e5933f2ce0e71c5be2f4124fc6c35284fbe5d3c624ab1b7012ca1a'
gates=Q31/'plan/MEDIA-VALIDATION-GATES-v1.json'
assert hashlib.sha256(gates.read_bytes()).hexdigest()=='3e880606a5744fc290be9ca4011169be7dac65140e66ac3525e7b58b65a82e60'
out=a.run/'media-inspection-v1';trial=Q/'results'/('media-'+r['profile_id']+'-'+uuid.uuid4().hex)
argv=[str(a.portable_root/'python_embeded/python.exe'),'-s','-B','-X','utf8',str(script),'--input',str(video),'--run-root',str(a.run),'--output-dir',str(out),'--gates',str(gates),'--monitor-source',str(Q31/'tools/comfy_first_video_v1.py'),'--inferrsi-src',str(IR/'src'),'--expected-sha256',f['sha256'],'--gates-sha256','3e880606a5744fc290be9ca4011169be7dac65140e66ac3525e7b58b65a82e60']
sys.path.insert(0,str(IR/'src'))
from inferrsi.infra.executors.local import execute_command
rec=execute_command(tuple(argv),IR,trial,time.monotonic()+140)
(trial/'execution.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
print(json.dumps({'status':rec['status'],'returncode':rec['returncode'],'seconds':rec['elapsed_seconds'],'trial':str(trial),'output':str(out)}))
print((trial/'stdout.log').read_text(encoding='utf-8')[-5000:])
if (trial/'stderr.log').stat().st_size:print((trial/'stderr.log').read_text(encoding='utf-8')[-2000:])
