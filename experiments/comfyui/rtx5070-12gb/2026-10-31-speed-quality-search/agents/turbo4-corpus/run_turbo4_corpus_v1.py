"""Parent launch adapter: existing InferRSI executor, one frozen profile per invocation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time,uuid
from datetime import datetime,timezone
p=argparse.ArgumentParser(allow_abbrev=False)
for name in ('notes-root','infra-root','portable-root','model-root','runs-parent','nvidia-smi'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--corpus-id',choices=('bounce-audio','hands-face-mug'),required=True)
p.add_argument('--remaining-percent',type=int,required=True)
a=p.parse_args()
a.profile=a.corpus_id;a.controller_name="turbo4_corpus_stage_b_v1.py";a.reserve_gib=3
a.freeze_name="EXECUTION-FREEZE-v1.json";a.freeze_sha256="2eb3836bf6850b5d1172f1e76a520c922bf6255e204911c7a26f8bf865c1a822"
assert a.remaining_percent>20
R=a.notes_root.resolve();IR=a.infra_root.resolve();P=a.portable_root.resolve();M=a.model_root.resolve()
Q=R/'iterations/2026-10-31-speed-quality-search';Q31=R/'iterations/2026-10-28-comfy-trim-first-video';Q32=R/'iterations/2026-10-30-original-h3-first-video';C=Q/'agents/turbo4-corpus';T=Q/'agents/timing-controller'
assert Path(__file__).resolve().parent==C
assert a.runs_parent.resolve().drive.upper()=='E:' and a.runs_parent.is_dir()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
freeze=C/a.freeze_name
assert sha(freeze)==a.freeze_sha256
items=json.loads(freeze.read_text(encoding='utf-8'))['files']
for row in items:
    target=(C/row['path']).resolve();assert target.is_relative_to(C)
    assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],row['path']
assert any(x['path']==a.controller_name for x in items)
for row in json.loads((C/'CANDIDATE-FREEZE-v1.json').read_text(encoding='utf-8'))['files']:
    target=(C/row['path']).resolve();assert target.is_relative_to(C)
    assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],row['path']
for row in json.loads(freeze.read_text(encoding='utf-8'))['external_notes_files']:
    target=(R/row['relative_to_notes']).resolve();assert target.is_relative_to(R)
    assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],row['relative_to_notes']
assert sha(C/a.controller_name)=='a94784c03a22de99eef9ebb379c0d7f741d0cb5923fffba3fae1cf5c0ad79aaf'
helper=Q31/'tools/comfy_first_video_v1.py'
assert sha(helper)=='63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35'
spec=importlib.util.spec_from_file_location('resource_helper',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
mem=h.WinMonitor().memory()
gpu_raw=subprocess.check_output([str(a.nvidia_smi),'--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
gpu=[int(v.strip()) for v in gpu_raw.split(',')]
preflight={'utc':datetime.now(timezone.utc).isoformat(),'memory':mem,'gpu_MiB':gpu,'remaining_percent_at_parent_check':a.remaining_percent,'external_admission_revision':'same-profile-repeat-v1','physical_min_GiB':12,'commit_min_GiB':16.5,'gpu_free_min_GiB':6,'runtime_stop_gates_unchanged':True}
nonce=uuid.uuid4().hex
save(Q/'results'/('PREFLIGHT-'+nonce+'.json'),preflight)
assert mem['available_physical_bytes']>=12*2**30 and mem['available_commit_bytes']>=16.5*2**30 and gpu[2]>=6*1024,'external resource admission'
profile=C/'profiles'/(a.profile+'.json');pf=json.loads(profile.read_text(encoding='utf-8'))
run=a.runs_parent/('q33-turbo4-corpus-'+a.corpus_id+'-'+nonce);trial=Q/'results'/('inferrsi-turbo4-corpus-'+a.corpus_id+'-'+nonce)
ppy=P/'python_embeded/python.exe'
gates_path=T/'RESOURCE-GATES-RESERVE3-v1.json' if a.reserve_gib==3 else Q32/'agents/native-graph/ORIGINAL-RESOURCE-GATES-v1.json'
gates_sha='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd' if a.reserve_gib==3 else 'dd9658a253e1e0c73bd1d1fe9034055622289a9f6748486fb55761bfa195eecf'
assert sha(gates_path)==gates_sha
argv=[str(ppy),'-s','-B','-X','utf8',str(C/a.controller_name),'--mode','turbo4-corpus-video-stage-b','--timing-module',str(T/'timing_events_v1.py'),'--iteration-root',str(Q),'--notes-root',str(R),'--q31-root',str(Q31),'--lab-root',str(M),'--model-root',str(M/pf['model_subdir']),'--portable-python',str(ppy),'--comfy-root',str(P/'ComfyUI'),'--run-dir',str(run),'--stop-marker',str(run/'stop.signal'),'--nvidia-smi',str(a.nvidia_smi),'--inferrsi-src',str(IR/'src'),'--profile',str(profile),'--profile-sha256',sha(profile),'--gates',str(gates_path),'--gates-sha256',gates_sha,'--source-manifest',str(T/'SOURCES-v1.json'),'--source-manifest-sha256','bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007','--conditioning-manifest',str(C/pf['conditioning']['file']),'--conditioning-manifest-sha256',pf['conditioning']['sha256'],'--asset-manifest',str(C/pf['assets']['file']),'--asset-manifest-sha256',pf['assets']['sha256'],'--port','8193']
snapshot={'schema':'q33-parent-runspec/v1','argv':argv,'trial':str(trial),'run':str(run),'profile':pf['id'],'corpus_id':a.corpus_id,'reserve_vram_GiB':a.reserve_gib,'launcher_sha256':sha(Path(__file__)),'controller_sha256':sha(C/a.controller_name),'freeze_sha256':a.freeze_sha256,'scenario':{'width':384,'height':256,'frames':73,'fps':24,'seed':424242},'preflight':preflight,'claim':'INCONCLUSIVE','timestamp':datetime.now(timezone.utc).isoformat()}
save(Q/'results'/('RUNSPEC-'+nonce+'.json'),snapshot)
print(json.dumps({'event':'dispatch','trial':str(trial),'run':str(run),'profile':pf['id'],'corpus_id':a.corpus_id}),flush=True)
sys.path.insert(0,str(IR/'src'))
from inferrsi.infra.executors.local import execute_command
record=execute_command(tuple(argv),IR,trial,time.monotonic()+2730)
save(trial/'execution.json',record)
print(json.dumps({'event':'completed','status':record['status'],'returncode':record['returncode'],'seconds':record['elapsed_seconds'],'trial':str(trial),'run':str(run)}),flush=True)
