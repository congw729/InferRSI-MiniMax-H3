from pathlib import Path
import json,hashlib,sys,time,uuid,argparse
p=argparse.ArgumentParser(allow_abbrev=False)
for k in ['notes-root','infra-root','portable-root','run','trial']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args()
r=a.notes_root;q=r/'iterations/2026-10-31-speed-quality-search';pp=q/'agents/offload-preparation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(__file__).resolve().parent==pp.resolve()
fr=pp/'EXECUTION-FREEZE-v1.json'
assert sha(fr)=='a996fe37e5f693bab17c2d7ec3b120ac4d4c12ff717562d6e1f9b181fc493a8a'
f=json.loads(fr.read_text())
for group,base,key in [('files',pp,'path'),('external_notes_files',r,'relative_to_notes'),('portable_files',a.portable_root,'path')]:
 for row in f[group]:
  file=(base/row[key]).resolve();assert file.is_relative_to(base.resolve())
  assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
script=pp/'validate_async_persistent_media_v1.py'
assert sha(script)=='27d8a29e51d9e18da93e803e16eaa8276445e103665c2eeef7dc1e6ee746b84e'
result=a.trial/'result.json';actual=json.loads(result.read_text());assert actual['owned_root_exited'] and not actual['cleanup_diagnostics']
assert actual['status']=='two_artifacts_received_unvalidated' and actual['mode']=='async-persistent-video-stage-b' and actual['cache_policy']=='classic'
assert actual['driver_sha256']=='be5ed4634a212e0d68d9b014f04ac2ca8e8547ade62c93967e73ccdecb9a8bfe'
trial=q/'results'/('async-persistent-media-classic-'+uuid.uuid4().hex)
argv=[str(a.portable_root/'python_embeded/python.exe'),'-s','-B','-X','utf8',str(script),'--notes-root',str(r),'--infra-root',str(a.infra_root),'--portable-root',str(a.portable_root),'--run',str(a.run),'--trial',str(a.trial),'--stop-marker',str(a.run/'stop.signal'),'--result-sha256',sha(result)]
sys.path.insert(0,str(a.infra_root/'src'))
from inferrsi.infra.executors.local import execute_command
rec=execute_command(tuple(argv),a.infra_root,trial,time.monotonic()+330)
with (trial/'execution.json').open('x',encoding='utf-8') as o:json.dump(rec,o,indent=2)
print(json.dumps({'trial':str(trial),'status':rec['status'],'returncode':rec['returncode'],'seconds':rec['elapsed_seconds']}))
