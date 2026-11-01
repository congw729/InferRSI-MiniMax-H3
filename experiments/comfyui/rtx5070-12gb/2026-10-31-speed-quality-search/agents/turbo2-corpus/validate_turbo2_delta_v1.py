"""Minimal model-plane Turbo2 metadata checks; no Torch, model loads or GPU."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,hashlib,importlib.util,json,sys
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--notes-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.notes_root;Q=R/'iterations/2026-10-31-speed-quality-search';D=Q/'agents/turbo2-corpus';B=Q/'agents/turbo4-corpus'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):return json.loads(p.read_text(encoding='utf8'))
checks=[]
def ck(n,v):checks.append({'name':n,'ok':bool(v)});assert v,n
def rejects(fn):
 try:fn()
 except RuntimeError:return True
 return False
s=(D/'turbo2_corpus_stage_b_v1.py').read_text(encoding='utf8')
for old,new in [('steps == 4, "native Turbo4 corpus family/steps"','steps == 2, "native Turbo2 corpus family/steps"'),('"agents/turbo4-corpus"','"agents/turbo2-corpus"'),('"turbo4-corpus-video-stage-b"','"turbo2-corpus-video-stage-b"'),('corpus in CORPUS_CASES and profile.get','corpus == "bounce-audio" and corpus in CORPUS_CASES and profile.get')]:s=s.replace(new,old)
ck('controller reverse literal changes exact baseline',s==(B/'turbo4_corpus_stage_b_v1.py').read_text(encoding='utf8') and sha(B/'turbo4_corpus_stage_b_v1.py')=='a94784c03a22de99eef9ebb379c0d7f741d0cb5923fffba3fae1cf5c0ad79aaf')
m=load(D/'turbo2_corpus_stage_b_v1.py','metadata_turbo4');h=load(R/'iterations/2026-10-28-comfy-trim-first-video/tools/comfy_first_video_v1.py','metadata_q31');canonical=read(D/'canonical/original.json');gates=read(Q/'agents/timing-controller/RESOURCE-GATES-RESERVE3-v1.json')
for case in ('bounce-audio',):
 pp=D/'profiles'/(case+'.json');pf=read(pp);graph=read(D/pf['graph']['file']);base=read(B/'graphs'/(case+'.json'));m.validate_profile_graph(pf,graph,canonical)
 normalized=copy.deepcopy(graph);normalized['prompt']['9']['inputs']['steps']=4;normalized['prompt']['92']['inputs']['filename_prefix']=base['prompt']['92']['inputs']['filename_prefix'];ck(case+' only steps and prefix',normalized==base)
 b=SimpleNamespace(iteration_root=Q,notes_root=R,profile=pp,profile_sha256=sha(pp),asset_manifest=D/pf['assets']['file'],asset_manifest_sha256=pf['assets']['sha256'],conditioning_manifest=D/pf['conditioning']['file'],conditioning_manifest_sha256=pf['conditioning']['sha256']);b.loaded_profile=m.load_profile(b,h);m.ASSET_EXPECTED=m.EXPECTED_BY_FAMILY['turbo-native'];m.ASSET_IDS=set(m.ASSET_EXPECTED)
 assets=m.load_asset_manifest(b,h,gates);manifest,_=m.load_stage_b_manifest(b,h,gates);ck(case+' actual4assets and real provenance metadata',len(assets)==4 and manifest['conditioning']['tensors']['conditioning']['shape']==[1,41 if case=='bounce-audio' else 49,5120])
 old=read(B/'conditioning'/(case+'.json'));ck(case+' exact conditioning and provenance',old['conditioning']==manifest['conditioning'] and old['provenance']==manifest['provenance'])
 wrong=copy.deepcopy(graph);wrong['prompt']['9']['inputs']['steps']=4;ck(case+' reject4steps',rejects(lambda:m.validate_profile_graph(pf,wrong,canonical)))
bad=copy.deepcopy(pf);bad['corpus_id']='hands-face-mug';ck('reject hands profile',rejects(lambda:m.validate_profile_graph(bad,graph,canonical)))
ck('asset canonical byte identities',all((D/x).read_bytes()==(B/x).read_bytes() for x in ['assets/turbo-native.ready.json','canonical/original.json']))
ck('no Torch imported','torch' not in sys.modules)
with a.output.open('x',encoding='utf8') as f:json.dump({'schema':'q33-turbo2-metadata-result/v1','checks':checks,'count':len(checks),'complete':True,'model_gpu_runs':0,'quality_performance':'INCONCLUSIVE'},f,indent=2)
print(json.dumps({'complete':True,'checks':len(checks),'GPU':0}))
