"""Frozen small hybrid wiring/metadata checks; stdlib only, no model import."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,sys
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--notes-root',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
Q=a.notes_root/'iterations/2026-10-31-speed-quality-search';H=Q/'agents/turbo-hybrid-controller';B=Q/'agents/turbo-controller'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=module(H/'turbo_hybrid_stage_b_v1.py','hybrid_metadata_only');checks=[]
def check(n,ok):
 checks.append({'name':n,'ok':bool(ok)});assert ok,n
def rejects(name,fn):
 try:fn()
 except RuntimeError:check(name,True)
 else:check(name,False)
profile=json.loads((H/'profiles/turbo-hybrid-6.ready.json').read_text());graph=json.loads((H/profile['graph']['file']).read_text());canonical=json.loads((H/profile['canonical_graph']['file']).read_text())
c.validate_profile_graph(profile,graph,canonical);check('fixed_hybrid_graph',True)
for name,node,key,val in [('merge_forbidden','1100','low_vram',True),('strength_fixed','1100','strength',0.9),('sigma_fixed','148','shift_video',8),('sampler_fixed','17','sampler_name','res_multistep'),('steps_fixed','9','steps',8)]:
 changed=copy.deepcopy(graph);changed['prompt'][node]['inputs'][key]=val;rejects(name,lambda:c.validate_profile_graph(profile,changed,canonical))
source=c.verify_hybrid_sources(a.model_root/c.HYBRID_RELATIVE,a.model_root);check('three_copied_sources_exact',len(source['files'])==3)
rejects('outside_deployment_rejected',lambda:c.verify_hybrid_sources(H,a.model_root))
valid=(f'[MiniMaxH3TurboLoRA] full base [bypass]: lora={c.TURBO_FILENAME} strength=1.0 | 259 backbone modules, 209 bypass adapters, 1 injections, 50 int8 fc2 via merge | model=MiniMaxH3 weight_dtype=torch.int8 weight_dev=cpu\n').encode().replace(b'\\n',b'\n')
canary=c.parse_hybrid_application_log(valid);check('synthetic_author_log_full_coverage',canary['total_targets']==259 and canary['bypass_targets']==209 and canary['merge_targets']==50)
for name,raw in [('missing_author_log',b''),('pruned_rejected',valid.replace(b'full base',b'pruned base')),('partial_mapping_rejected',valid.replace(b'209 bypass',b'208 bypass')),('duplicate_log_rejected',valid+valid)]:rejects(name,lambda:c.parse_hybrid_application_log(raw))
old=ast.parse((B/'turbo_stage_b_v6.py').read_text());new=ast.parse((H/'turbo_hybrid_stage_b_v1.py').read_text())
def named(tree,n):return next(x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==n)
for n in ['cleanup_owned_v2','sample','http','guarded_hash','verify_conditioning','conditioning_header','load_timing_module']:
 check('unchanged_'+n,ast.dump(named(old,n),include_attributes=False)==ast.dump(named(new,n),include_attributes=False))
source_text=(H/'turbo_hybrid_stage_b_v1.py').read_text();check('CLI_single_allowlist_and_original_protections','"--whitelist-custom-nodes", HYBRID_PACKAGE' in source_text and '"--disable-all-custom-nodes"' in source_text and '"--extra-model-paths-config", str(a.hybrid_root / "extra-paths.yaml")' in source_text and '"--vram-headroom", "1.0"' in source_text)
map_path=B/'REAL-LORA-HEADER-NATIVE-MAP-v1.json';check('recorded_header_evidence_identity',sha(map_path)=='cadd5ae14a7cf84572f156e083d39f4c1cd9e92757482aeb69c6411caf7672f7');rows=json.loads(map_path.read_text())['rows'];check('author_native_keys_all259',len(rows)==259 and all('diffusion_model.'+r['lora_module']+'.weight'==r['native_state_dict_key'] and r['header_shape_match'] for r in rows));check('source_derived_fc2_split',sum(r['base_dtype']=='I8' and r['lora_module'].endswith('.mlp.fc2') for r in rows)==50)
hp=a.notes_root/'iterations/2026-10-23-original-quant-audit/metadata/comfy-headers-v2-retry/minimax_h3_fl2va_int8_convrot.safetensors.header.raw';check('original_header_identity',sha(hp)=='7513055dc7e4b8f67e1c8afa2126fb48e899f9c7b70d25655dad599d8c2e6783');base=json.loads(hp.read_bytes()[8:]);check('full_header_no_curve_branch','adaln_t_table' not in base and 'time_embedder.proj_in.weight' in base)
v1=module(a.notes_root/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py','frozen_helper_metadata');args=SimpleNamespace(profile=H/'profiles/turbo-hybrid-6.ready.json',profile_sha256=sha(H/'profiles/turbo-hybrid-6.ready.json'),asset_manifest=(H/profile['assets']['file']).resolve(),asset_manifest_sha256=profile['assets']['sha256'],conditioning_manifest=(H/profile['conditioning']['file']).resolve(),conditioning_manifest_sha256=profile['conditioning']['sha256'],iteration_root=Q,notes_root=a.notes_root)
args.loaded_profile=c.load_profile(args,v1);c.ASSET_EXPECTED=c.EXPECTED_BY_FAMILY['turbo-hybrid'];c.ASSET_IDS=set(c.ASSET_EXPECTED);gates=json.loads((H/'TURBO-RESOURCE-GATES-v1.json').read_text());assets=c.load_asset_manifest(args,v1,gates);c.load_stage_b_manifest(args,v1,gates);check('four_real_receipts_profile_graph_bindings',len(assets)==4)
check('no_torch_import','torch' not in sys.modules)
result={'schema':'q33-hybrid-offline-result/v1','checks':checks,'completed':len(checks),'dispatches':0,'models':0,'GPU':0,'author_node_imported':False,'real_mapping_scope':'Reuse real pinned header A/B shapes plus source-derived key mapping; actual hook application only future runtime log; no numerical quality proof','quality_performance':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'completed':len(checks),'dispatches':0,'all_ok':all(x['ok'] for x in checks)}))
