"""Parent-only launch: one service, fixed bounce and hands real TE, three releases, two Trim8 videos."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time,uuid
from datetime import datetime,timezone
FREEZE_SHA='9de3448531bcd7a0067b8b0f1cd823e9a851d65452c4264fde8a589ef1525dca'
CONTROLLER_SHA='8b787a016c311184955f98eccb760d058d266c2fc994dfa7bd094b34fd79da85'
GATES_SHA='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd'
p=argparse.ArgumentParser(allow_abbrev=False)
for name in ('notes-root','infra-root','portable-root','model-root','runs-parent','nvidia-smi'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--remaining-percent',type=int,required=True);a=p.parse_args();assert a.remaining_percent>20
R=a.notes_root.resolve();IR=a.infra_root.resolve();P=a.portable_root.resolve();M=a.model_root.resolve();RUNS=a.runs_parent.resolve()
Q=R/'iterations/2026-10-31-speed-quality-search';Q31=R/'iterations/2026-10-28-comfy-trim-first-video';T=Q/'agents/timing-controller';D=Q/'agents/end-to-end-preparation'
assert Path(__file__).resolve().parent==D and RUNS.drive.upper()=='E:' and RUNS.is_dir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def redact(text):
 for path,label in [(R,'R'),(IR,'INFERRSI'),(P,'PORTABLE'),(M,'MODEL_ROOT'),(RUNS,'RUNS_PARENT'),(a.nvidia_smi.resolve(),'NVIDIA_SMI')]:text=text.replace(str(path).replace('\\','\\\\'),label).replace(str(path).replace('\\','/'),label)
 return text
def save(path,obj):
 with path.open('x',encoding='utf-8') as f:f.write(redact(json.dumps(obj,indent=2,allow_nan=False))+'\n')
fpath=D/'EXECUTION-FREEZE-v6.json';assert sha(fpath)==FREEZE_SHA;frozen=json.loads(fpath.read_text())
for item in frozen['files']:
 target=(D/item['path']).resolve();assert target.is_relative_to(D) and target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
for item in frozen['external_notes_files']:
 target=(R/item['relative_to_notes']).resolve();assert target.is_relative_to(R) and target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
for item in frozen['runtime_sources']:
 root={'portable':P,'inferrsi':IR}[item['root']];target=(root/item['path']).resolve();assert target.is_relative_to(root) and target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
assert sha(D/'end_to_end_v6.py')==CONTROLLER_SHA
helper=Q31/'tools/comfy_first_video_v1.py';assert sha(helper)=='63eb319e9fe294209e65c188a447a101987738e5ebc49bae879b65d36bb20d35'
spec=importlib.util.spec_from_file_location('e2e_parent_resource_helper',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
mem=h.WinMonitor().memory();gpu=[int(v.strip()) for v in subprocess.check_output([str(a.nvidia_smi),'--id=0','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')]
nonce=uuid.uuid4().hex;pre={'utc':datetime.now(timezone.utc).isoformat(),'memory':mem,'gpu_MiB':gpu,'remaining_percent_at_parent_check':a.remaining_percent,'physical_min_GiB':12,'commit_min_GiB':16.5,'gpu_free_min_GiB':6,'runtime_stop_gates_unchanged':True}
save(Q/'results'/('E2E-PREFLIGHT-'+nonce+'.json'),pre)
assert mem['available_physical_bytes']>=12*2**30 and mem['available_commit_bytes']>=16.5*2**30 and gpu[2]>=6*1024,'external resource admission'
profile=T/'profiles/trim-8.json';pf=json.loads(profile.read_text());gates=T/'RESOURCE-GATES-RESERVE3-v1.json';assert sha(gates)==GATES_SHA
run=RUNS/('q33-e2e-two-async-wake-trim8-'+nonce);trial=Q/'results'/('inferrsi-e2e-two-async-wake-trim8-'+nonce);ppy=P/'python_embeded/python.exe'
argv=[str(ppy),'-s','-B','-X','utf8',str(D/'end_to_end_v6.py'),'--mode','two-new-prompts-trim8','--memory-policy','baseline','--iteration-root',str(Q),'--notes-root',str(R),'--q31-root',str(Q31),'--lab-root',str(M),'--model-root',str(M/'comfy-trim/models'),'--portable-python',str(ppy),'--comfy-root',str(P/'ComfyUI'),'--run-dir',str(run),'--stop-marker',str(run/'stop.signal'),'--nvidia-smi',str(a.nvidia_smi),'--inferrsi-src',str(IR/'src'),'--profile',str(profile),'--profile-sha256',sha(profile),'--gates',str(gates),'--gates-sha256',GATES_SHA,'--source-manifest',str(T/'SOURCES-v1.json'),'--source-manifest-sha256','bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007','--conditioning-manifest',str(T/pf['conditioning']['file']),'--conditioning-manifest-sha256',pf['conditioning']['sha256'],'--asset-manifest',str(D/'ASSETS-v1.json'),'--asset-manifest-sha256','fcaa021e7aa7ecc50ecd14f51e81448a758dc6c8352b65d5ea7aa657d1fe31a8','--port','8195']
rev=subprocess.check_output(['git','-C',str(IR),'rev-parse','HEAD'],text=True,timeout=5).strip()
snapshot={'schema':'q33-e2e-runspec/v1','argv':argv,'run':str(run),'trial':str(trial),'infra':{'repository':'InferRSI','revision':rev},'runtime':{'backend':'ComfyUI native v0.43.0 dynamic VRAM','device':'CUDA0 RTX5070','source_manifest_sha256':'bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007'},'model':{'TE':'qwen3vl32b H3 NVFP4 AWQ','DiT':'FastH3 Trim8 NVFP4','assets_sha256':sha(D/'ASSETS-v1.json')},'scenario':{'prompts':['frozen bounce-audio','frozen hands-face-mug'],'width':384,'height':256,'frames':73,'fps':24,'steps':8,'seed':424242,'precomputed_TE_reused':False,'between_stage_commit_policy':'remaining_owned_private_growth_plus_original_running_floor_min_pre_post','maximum_prompt_posts':4,'maximum_free_posts':3,'native_default_async_streams':2,'history_wait_policy':'one_observer_done_wake_then_original_timeout'},'source_freeze_sha256':FREEZE_SHA,'launcher_sha256':sha(Path(__file__)),'controller_sha256':CONTROLLER_SHA,'preflight':pre,'claim':'INCONCLUSIVE','timestamp':datetime.now(timezone.utc).isoformat()}
save(Q/'results'/('E2E-RUNSPEC-'+nonce+'.json'),snapshot)
print(json.dumps({'event':'dispatch','trial_relative_to_iteration':trial.relative_to(Q).as_posix(),'run_basename':run.name}),flush=True)
sys.path.insert(0,str(IR/'src'));from inferrsi.infra.executors import local
assert Path(local.__file__).resolve()==IR/'src/inferrsi/infra/executors/local.py'
record=local.execute_command(tuple(argv),IR,trial,time.monotonic()+2730);save(trial/'execution.json',record)
print(json.dumps({'event':'completed','status':record['status'],'returncode':record['returncode'],'seconds':record['elapsed_seconds'],'trial_relative_to_iteration':trial.relative_to(Q).as_posix(),'run_basename':run.name}),flush=True)
