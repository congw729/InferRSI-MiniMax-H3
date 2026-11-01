"""One-shot real receipt metadata preflight under portable Python; no model imports."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types
R=Path(sys.argv[1]);Q=R/'iterations/2026-10-31-speed-quality-search';C=Q/'agents/turbo-controller'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path,want):
 assert sha(path)==want
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert Path(m.__file__).resolve()==path.resolve();return m
c=load('q33_real_turbo_metadata',C/'turbo_stage_b_v5.py','604ae8578f132e66654924dd08c9515a0be6b25273faa5e962efe93c9f91102c')
v1=load('q33_original_frozen_helpers',R/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py',c.V1_SHA)
gates=json.loads((C/'TURBO-RESOURCE-GATES-v1.json').read_bytes());assert sha(C/'TURBO-RESOURCE-GATES-v1.json')==c.STAGE_GATES_SHA
results=[]
for steps in (6,8):
 pp=C/f'profiles/turbo-native-{steps}.ready.json';p=json.loads(pp.read_bytes())
 a=types.SimpleNamespace(profile=pp,profile_sha256=sha(pp),asset_manifest=(C/p['assets']['file']).resolve(),asset_manifest_sha256=p['assets']['sha256'],conditioning_manifest=(C/p['conditioning']['file']).resolve(),conditioning_manifest_sha256=p['conditioning']['sha256'],iteration_root=Q,notes_root=R)
 a.loaded_profile=c.load_profile(a,v1);c.ASSET_EXPECTED=c.EXPECTED_BY_FAMILY[a.loaded_profile['family']];c.ASSET_IDS=set(c.ASSET_EXPECTED)
 manifest,graph=c.load_stage_b_manifest(a,v1,gates);assets=c.load_asset_manifest(a,v1,gates)
 assert len(assets)==4 and {x['id']for x in assets}==c.ASSET_IDS
 assert {n['class_type']for n in graph['prompt'].values()}==set(c.NODE_FAMILIES['turbo-native'])
 results.append({'profile':p['id'],'profile_sha256':sha(pp),'assets':4,'actual_receipts_complete_and_plan_bound':True,'graph_sha256':manifest['graph']['sha256'],'reserve_vram_gb':gates['reserve_vram_gb']})
timing_path=Q/'agents/timing-controller/timing_events_v1.py';observer=c.load_timing_module(timing_path,Q)
assert observer.__module__=='q33_frozen_native_timing_v1' and 'torch'not in sys.modules
print(json.dumps({'schema':'q33-real-turbo-metadata-preflight/v1','profiles':results,'timing_source_relative_to_iteration':timing_path.relative_to(Q).as_posix(),'timing_sha256':sha(timing_path),'timing_class_module':observer.__module__,'portable_executable_verified_by_parent_argv':True,'no_torch_import':True,'model_dispatches':0,'network_requests':0,'scope':'actual small receipt/profile/conditioning metadata and timing import only; no payload SHA loop, service or GPU'},indent=2))