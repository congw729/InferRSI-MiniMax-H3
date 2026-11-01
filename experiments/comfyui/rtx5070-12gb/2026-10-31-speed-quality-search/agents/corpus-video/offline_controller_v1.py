import argparse,ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=Path(__file__).resolve().parent;t=v.parent/'timing-controller';new=v/'corpus_stage_b_v1.py'
spec=importlib.util.spec_from_file_location('corpus_controller_unit',new);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def ck(name,ok):checks.append({'name':name,'ok':bool(ok)});assert ok,name
oldtree=ast.parse((t/'speed_stage_b_v6.py').read_text());newtree=ast.parse(new.read_text())
def funcs(tree):return {x.name:ast.dump(x,include_attributes=False) for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
old=funcs(oldtree);cur=funcs(newtree)
for name in ['sample','event','guarded_hash','http','verify_conditioning','cleanup_owned_v2','load_asset_manifest','conditioning_header','verify_dynamic_vram_startup']:
 ck('unchanged_'+name,old[name]==cur[name])
for corpus,case in m.CORPUS_CASES.items():
 for family,steps in [('original',50),('original',20),('trim',8),('trim',6)]:
  canonical=json.loads((v/'canonical'/(family+'.json')).read_text());p={'family':family,'steps':steps,'id':family+'-'+str(steps),'role':'reference' if family=='original' and steps==50 else 'candidate','corpus_id':corpus,'prompt_sha256':case['prompt_sha256'],'conditioning_sha256':'1'*64,'canonical_graph':{'sha256':m.CANONICAL_SHA[family]}}
  graph=copy.deepcopy(canonical);graph['prompt']['9']['inputs']['steps']=steps;graph['prompt']['92']['inputs']['filename_prefix']='q33-corpus/'+corpus+'/'+p['id'];graph['prompt']['1001']['inputs']['conditioning_name']='h3-conditioning-'+p['conditioning_sha256']+'.safetensors'
  m.validate_profile_graph(p,graph,canonical);ck(corpus+'_'+p['id']+'_allowed_graph',True)
  bad=copy.deepcopy(graph);bad['prompt']['9']['inputs']['steps']=steps+1
  try:m.validate_profile_graph(p,bad,canonical);rejected=False
  except RuntimeError:rejected=True
  ck(corpus+'_'+p['id']+'_reject_step_change',rejected)
  bad=copy.deepcopy(p);bad['conditioning_sha256']=None
  try:m.validate_profile_graph(bad,graph,canonical);rejected=False
  except RuntimeError:rejected=True
  ck(corpus+'_'+p['id']+'_reject_missing_condition',rejected)
ck('reserve3_identity',m.STAGE_GATES_SHA=='13672a9bbe8946c28c98b3bab0f906f6ddfe67735ff7f77929e58539a4a580dd')
ck('headroom_request_preserved','"--vram-headroom", "1.0"' in new.read_text())
ck('no_torch_import','torch' not in sys.modules)
out={'schema':'q33-corpus-controller-offline/v1','unit_graph_only_not_real_ready_profile':True,'checks':checks,'check_count':len(checks),'model_dispatches':0,'claim':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'all_ok':all(x['ok'] for x in checks)}))
