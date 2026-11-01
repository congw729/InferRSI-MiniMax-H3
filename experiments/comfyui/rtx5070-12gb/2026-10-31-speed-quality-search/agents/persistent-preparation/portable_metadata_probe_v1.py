from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,hashlib,sys,argparse,subprocess
p=argparse.ArgumentParser();p.add_argument('--notes-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.notes_root;Q=R/'iterations/2026-10-31-speed-quality-search';C=Q/'agents/timing-controller';D=Q/'agents/persistent-preparation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=mod(D/'persistent_stage_b_v1.py','persistent_metadata');h=mod(R/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py','q31_metadata_helper')
b=SimpleNamespace(iteration_root=Q,notes_root=R,profile=C/'profiles/trim-8.json',profile_sha256=sha(C/'profiles/trim-8.json'),asset_manifest=C/'assets/trim.json',asset_manifest_sha256=sha(C/'assets/trim.json'),conditioning_manifest=C/'conditioning/trim-8.json',conditioning_manifest_sha256=sha(C/'conditioning/trim-8.json'))
b.loaded_profile=m.load_profile(b,h);m.ASSET_EXPECTED=m.EXPECTED_BY_FAMILY['trim'];m.ASSET_IDS=set(m.ASSET_EXPECTED);g=json.loads((C/'RESOURCE-GATES-RESERVE3-v1.json').read_text());assets=m.load_asset_manifest(b,h,g);manifest,graph=m.load_stage_b_manifest(b,h,g)
assert len(assets)==3 and b.loaded_profile['id']=='trim-8' and graph['prompt']['9']['inputs']['steps']==8 and 'torch' not in sys.modules
r=subprocess.run([sys.executable,'-s','-B',str(D/'persistent_stage_b_v1.py'),'--help'],capture_output=True,timeout=10)
assert r.returncode==0 and not r.stderr and b'--memory-policy {baseline,smart}' in r.stdout
result={'schema':'q33-persistent-portable-metadata/v1','profile':'trim-8','receipt_count':len(assets),'help_returncode':r.returncode,'help_stderr_bytes':len(r.stderr),'help_stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'timing_actual_source_relative_to_iteration':str(Path(m._timing_module.__file__).relative_to(Q)),'timing_sha256':sha(Path(m._timing_module.__file__)),'Torch_imported':False,'GPU':0,'model_files_read':0,'source':'existing portable interpreter with stdlib metadata only','quality_performance':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'metadata_valid':True,'receipts':3,'help_returncode':0,'GPU':0}))
