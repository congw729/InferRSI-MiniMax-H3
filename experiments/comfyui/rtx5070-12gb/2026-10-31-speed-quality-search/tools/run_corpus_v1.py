"""One fixed quality-corpus Stage A under the existing InferRSI executor."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time,uuid
p=argparse.ArgumentParser(allow_abbrev=False)
for n in ('notes-root','infra-root','portable-root','model-root','runs-parent','nvidia-smi'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--corpus',choices=('bounce-audio','hands-face-mug'),required=True);p.add_argument('--remaining-percent',type=int,required=True)
a=p.parse_args();assert a.remaining_percent>20
R=a.notes_root.resolve();IR=a.infra_root.resolve();P=a.portable_root.resolve();Q=R/'iterations/2026-10-31-speed-quality-search';Q31=R/'iterations/2026-10-28-comfy-trim-first-video';C=Q/'agents/quality-corpus';T=Q/'agents/timing-controller'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
freeze=C/'EXECUTION-FREEZE-v3-headroom.json';assert sha(freeze)=='b516c28f90bd2174097b1ab20a5200c888555a8edadfdab20a10f8042e8e7011'
f=json.loads(freeze.read_text())
for root,key,name in ((C,'files','path'),(R,'external_sources','relative_to_notes')):
 for v in f[key]:
  target=(root/v[name]).resolve();assert target.is_relative_to(root) and target.stat().st_size==v['bytes'] and sha(target)==v['sha256']
helper=Q31/'tools/comfy_first_video_v1.py';assert sha(helper)=='63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35'
sp=importlib.util.spec_from_file_location('corpus_resource',helper);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
mem=h.WinMonitor().memory();gpu=[int(x.strip())for x in subprocess.check_output([str(a.nvidia_smi),'--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')]
nonce=uuid.uuid4().hex
pre={'memory':mem,'gpu_MiB':gpu,'remaining_percent':a.remaining_percent,'external_admission_GiB':{'physical':12,'commit':17.5,'gpu_free':6},'basis':'Prior realTE sampled private13.78GiB; reserve3.72GiB at external commit floor. Internal stops unchanged.'}
save(Q/'results'/('CORPUS-PREFLIGHT-'+nonce+'.json'),pre)
assert mem['available_physical_bytes']>=12*2**30 and mem['available_commit_bytes']>=17.5*2**30 and gpu[2]>=6144,'resource admission'
assert a.runs_parent.resolve().drive.upper()=='E:' and a.runs_parent.is_dir()
run=a.runs_parent/('q33-condition-'+a.corpus+'-'+nonce);trial=Q/'results'/('corpus-'+a.corpus+'-'+nonce);ppy=P/'python_embeded/python.exe'
argv=[str(ppy),'-s','-B','-X','utf8',str(C/'conditioning_corpus_v3.py'),'--mode','quality-corpus-stage-a','--corpus',a.corpus,'--iteration-root',str(Q),'--notes-root',str(R),'--q31-root',str(Q31),'--lab-root',str(a.model_root),'--portable-python',str(ppy),'--comfy-root',str(P/'ComfyUI'),'--run-dir',str(run),'--stop-marker',str(run/'stop.signal'),'--nvidia-smi',str(a.nvidia_smi),'--inferrsi-src',str(IR/'src'),'--corpus-manifest',str(C/'CORPUS-MANIFEST-v1.json'),'--corpus-manifest-sha256','870756bbdae953f6c52b40a05a6ac35005d8818402170f1aa9d964bed715ef02','--timing-controller',str(T/'speed_stage_b_v3.py'),'--source-manifest',str(T/'SOURCES-v1.json'),'--source-manifest-sha256','bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007','--schema-result',str(Q31/'results/object-info-v1-20261028T131953602235Z/result.json'),'--schema-result-sha256','60a916c5ae597eca9abb8901f788691b0759aa9204df4b8168065d68b4933788','--gates',str(Q31/'plan/CONDITIONING-STAGE-RESOURCE-GATES-v1.json'),'--gates-sha256','4d5114f7abc9041f34778b3f615034cc3da797150e750b4a6e7a4c684ad5a9be','--port','8193']
save(Q/'results'/('CORPUS-RUNSPEC-'+nonce+'.json'),{'argv':argv,'trial':str(trial),'run':str(run),'corpus':a.corpus,'preflight':pre,'launcher_sha256':sha(Path(__file__)),'freeze_sha256':sha(freeze),'quality':'INCONCLUSIVE'})
print(json.dumps({'event':'dispatch','trial':str(trial),'run':str(run)}),flush=True)
sys.path.insert(0,str(IR/'src'))
from inferrsi.infra.executors.local import execute_command
r=execute_command(tuple(argv),IR,trial,time.monotonic()+2730);save(trial/'execution.json',r)
print(json.dumps({k:r[k]for k in ('status','returncode','elapsed_seconds')}),flush=True)
