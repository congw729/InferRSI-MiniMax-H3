"""Frozen narrow Turbo6 corpus metadata/wiring gates; no model imports or GPU."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,hashlib,importlib.util,json,sys
p=argparse.ArgumentParser();p.add_argument('--notes-root',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.notes_root;Q=R/'iterations/2026-10-31-speed-quality-search';D=Q/'agents/turbo-corpus';B=Q/'agents/turbo-controller';C=Q/'agents/corpus-video'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=mod(D/'turbo_corpus_stage_b_v1.py','turbo_corpus_metadata');h=mod(R/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py','q31_metadata_helper');checks=[]
def ck(n,ok):checks.append({'name':n,'ok':bool(ok)});assert ok,n
def reject(fn):
 try:fn()
 except RuntimeError:return True
 return False
canonical=json.loads((D/'canonical/original.json').read_text());native=json.loads((B/'graphs/turbo-native-6.json').read_text());gates=json.loads((B/'TURBO-RESOURCE-GATES-v1.json').read_text())
for case in ('bounce-audio','hands-face-mug'):
 pp=D/'profiles'/(case+'.json');pf=json.loads(pp.read_text());gp=D/pf['graph']['file'];graph=json.loads(gp.read_text());m.validate_profile_graph(pf,graph,canonical)
 normalized=copy.deepcopy(graph)
 for node,key in [('1001','conditioning_name'),('92','filename_prefix')]:normalized['prompt'][node]['inputs'][key]=native['prompt'][node]['inputs'][key]
 ck(case+'_only_condition_and_prefix',normalized==native)
 b=SimpleNamespace(iteration_root=Q,notes_root=R,profile=pp,profile_sha256=sha(pp),asset_manifest=D/pf['assets']['file'],asset_manifest_sha256=pf['assets']['sha256'],conditioning_manifest=D/pf['conditioning']['file'],conditioning_manifest_sha256=pf['conditioning']['sha256'])
 b.loaded_profile=m.load_profile(b,h);m.ASSET_EXPECTED=m.EXPECTED_BY_FAMILY['turbo-native'];m.ASSET_IDS=set(m.ASSET_EXPECTED);assets=m.load_asset_manifest(b,h,gates);manifest,_=m.load_stage_b_manifest(b,h,gates)
 ck(case+'_real_four_assets_and_two_provenance_receipts',len(assets)==4 and manifest['conditioning']['tensors']['conditioning']['shape']==[1,41 if case=='bounce-audio' else 49,5120])
 alias=a.model_root/'comfy-original/models/embeddings'/manifest['conditioning']['basename'];receipt=json.loads((R/manifest['provenance']['stage_a_receipt']['relative_to_notes']).read_text());source=Path(receipt['artifact']['absolute_path']) if 'absolute_path' in receipt['artifact'] else None
 ck(case+'_existing_alias_stat',alias.is_file() and alias.stat().st_size==manifest['conditioning']['bytes'])
 for label,node,key,value in [('strength','1100','strength_model',0.9),('shift','148','shift_video',10),('steps','9','steps',8)]:
  changed=copy.deepcopy(graph);changed['prompt'][node]['inputs'][key]=value;ck(case+'_reject_'+label,reject(lambda:m.validate_profile_graph(pf,changed,canonical)))
 changed=copy.deepcopy(graph);changed['prompt']['1100']['class_type']='MiniMaxH3TurboLoRA';ck(case+'_reject_hybrid',reject(lambda:m.validate_profile_graph(pf,changed,canonical)))
old=ast.parse((B/'turbo_stage_b_v6.py').read_text());new=ast.parse((D/'turbo_corpus_stage_b_v1.py').read_text());corpus=ast.parse((C/'corpus_stage_b_v1.py').read_text())
def fun(t,n):return next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==n)
ck('reused_corpus_provenance_validator',ast.dump(fun(corpus,'load_stage_b_manifest'),include_attributes=False)==ast.dump(fun(new,'load_stage_b_manifest'),include_attributes=False))
for name in ('cleanup_owned_v2','sample','http','guarded_hash','verify_conditioning','load_asset_manifest','load_timing_module'):
 ck('native_unchanged_'+name,ast.dump(fun(old,name),include_attributes=False)==ast.dump(fun(new,name),include_attributes=False))
ck('correct_controller_own_root','a.iteration_root / "agents/turbo-corpus"' in (D/'turbo_corpus_stage_b_v1.py').read_text())
ck('torch_not_imported','torch' not in sys.modules)
r={'schema':'q33-turbo-corpus-offline/v1','checks':checks,'completed':len(checks),'GPU':0,'model_loads':0,'controller_sha256':sha(D/'turbo_corpus_stage_b_v1.py'),'scope':'Metadata/real receipts/alias stat and unchanged protected-function AST; no model/header body read or service; no quality proof','quality_performance':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
print(json.dumps({'completed':len(checks),'all_ok':all(x['ok'] for x in checks),'GPU':0}))
