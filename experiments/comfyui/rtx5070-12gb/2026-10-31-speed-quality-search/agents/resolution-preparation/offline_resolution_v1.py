import argparse,ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser(allow_abbrev=False)
for k in ['notes-root','portable-root','result']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();v=Path(__file__).resolve().parent;t=v.parent/'timing-controller';q=v.parents[1];q31=a.notes_root/'iterations/2026-10-28-comfy-trim-first-video';checks=[]
def ck(n,ok):checks.append({'name':n,'ok':bool(ok)});assert ok,n
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((v/'PLAN-v2.json').read_text())
for row in plan['source_evidence']:ck('native_identity_'+row['relative_to_portable'],sha(a.portable_root/row['relative_to_portable'])==row['sha256'])
tree=ast.parse((a.portable_root/'ComfyUI/comfy_extras/nodes_minimax_h3.py').read_text(encoding='utf-8'));functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['align_frame_count','video_latent_t','temporal_shape']]
ns={'FPS':24,'AUDIO_LATENT_FPS':40};exec(compile(ast.Module(body=functions,type_ignores=[]),'<extracted_native_temporal>','exec'),ns)
ck('native_temporal73',ns['temporal_shape'](73)==(73,22,122));ck('native_spatial_patch_divisible',576%32==0 and 384%32==0 and (576//16)%2==0 and (384//16)%2==0)
video=22*(384//16//2)*(576//16//2);total=video+2*122+29
ck('native_geometry_tokens',video==4752 and total==5025 and total<12288);ck('same_aspect_ratio',576*256==384*384)
base=json.loads((t/'graphs/trim-8.json').read_text());graph=json.loads((v/'graphs/trim-8-576x384.json').read_text())
def diff(x,y,path=()):
 if type(x)!=type(y):return[path]
 if isinstance(x,dict):return sum([diff(x.get(k),y.get(k),path+(k,)) for k in set(x)|set(y)],[])
 if isinstance(x,list):return[path] if x!=y else[]
 return[path] if x!=y else[]
ck('exact_graph_delta',set(diff(base,graph))=={('prompt','1002','inputs','width'),('prompt','1002','inputs','height'),('prompt','92','inputs','filename_prefix')})
ck('all_graph_refs',all(x[0] in graph['prompt'] for n in graph['prompt'].values() for x in n['inputs'].values() if isinstance(x,list) and len(x)==2 and isinstance(x[0],str)))
new=v/'resolution_stage_b_v1.py';spec=importlib.util.spec_from_file_location('resolution_unit',new);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
profile=json.loads((v/'profiles/trim-8-576x384.json').read_text());canonical=json.loads((v/'canonical/trim.json').read_text());m.validate_profile_graph(profile,graph,canonical);ck('candidate_graph_accepted',True)
for node,key,value in [('15','noise_seed',424243),('9','steps',6),('1002','width',640)]:
 bad=copy.deepcopy(graph);bad['prompt'][node]['inputs'][key]=value
 try:m.validate_profile_graph(profile,bad,canonical);rejected=False
 except RuntimeError:rejected=True
 ck('reject_unfrozen_'+key,rejected)
def defs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.walk(ast.parse(p.read_text())) if isinstance(x,ast.FunctionDef)}
o=defs(t/'speed_stage_b_v6.py');n=defs(new)
for name in ['sample','http','guarded_hash','verify_conditioning','cleanup_owned_v2','load_asset_manifest','load_stage_b_manifest','conditioning_header']:ck('unchanged_'+name,o[name]==n[name])
old=(q31/'tools/validate_first_video_media_v1.py').read_text();newmedia=(v/'validate_resolution_media_v1.py').read_text();reversed_text=newmedia.replace('"width": 576, "height": 384','"width": 384, "height": 256').replace('(576, 384)','(384, 256)').replace('(384, 576, 3)','(256, 384, 3)')
ck('media_exact_five_geometry_replacements',reversed_text==old and newmedia.count('"width": 576, "height": 384')==2 and newmedia.count('(576, 384)')==2 and newmedia.count('(384, 576, 3)')==1)
ck('media_resource_gates_identical',(v/'MEDIA-RESOURCE-GATES-v1.json').read_bytes()==(q31/'plan/MEDIA-VALIDATION-GATES-v1.json').read_bytes())
ck('media_expected_only_width_height',json.loads((v/'MEDIA-EXPECTED-v1.json').read_text())=={'width':576,'height':384,'frames':73,'fps':24})
ck('no_torch_import','torch' not in sys.modules)
out={'schema':'q33-resolution-targeted-checks/v1','checks':checks,'check_count':len(checks),'source_derived_geometry':{'video_tokens':video,'total_tokens':total,'latent':[1,24,22,24,36]},'model_dispatches':0,'media_dispatches':0,'claim':'INCONCLUSIVE'}
with a.result.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'all_ok':True,'checks':len(checks),'tokens':total}))
