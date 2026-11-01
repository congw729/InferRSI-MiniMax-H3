"""Bind fixed corpus profiles to actual completed Stage-A and value-check receipts; no model access."""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
CONTROLLER_SHA="74de2ccded6ec7415e183c13d162caba653b54b5bcc99270a924c9391d58c7d1"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,why):
 if not ok:raise RuntimeError(why)
def unique(pairs):
 out={}
 for k,v in pairs:require(k not in out,'duplicate JSON key');out[k]=v
 return out
def parse(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(RuntimeError('nonfinite JSON')))
def encode(obj):return (json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for key in ['notes-root','stage-a-receipt','value-report']:p.add_argument('--'+key,type=Path,required=True)
 for key in ['stage-a-receipt-sha256','value-report-sha256']:p.add_argument('--'+key,required=True)
 p.add_argument('--corpus',choices=['bounce-audio','hands-face-mug'],required=True)
 a=p.parse_args();r=a.notes_root.resolve();q=r/'iterations/2026-10-31-speed-quality-search';v=Path(__file__).resolve().parent
 require(v==q/'agents/corpus-video','fixed corpus-video root')
 controller=v/'corpus_stage_b_v1.py';require(sha(controller.read_bytes())==CONTROLLER_SHA,'controller drift')
 spec=importlib.util.spec_from_file_location('q33_corpus_bound_controller',controller);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 objects={}
 def frozen(path,expected,limit):
  path=Path(path).resolve()
  raw=objects[path] if path in objects else path.read_bytes()
  require(len(raw)<=limit and sha(raw)==expected,'metadata identity');return parse(raw)
 receipt=a.stage_a_receipt.resolve();values=a.value_report.resolve()
 require(receipt.is_relative_to(q/'results') and values.is_relative_to(q/'results'),'receipts under Q33 results')
 rr=frozen(receipt,a.stage_a_receipt_sha256,65536);vr=frozen(values,a.value_report_sha256,65536)
 out=v/'ready'/a.corpus;require(not out.exists(),'never replace prior bound metadata')
 ca=vr.get('artifact',{});require(set(['bytes','sha256','tensors','metadata_keys','option_keys','options_sha256'])<=set(ca),'complete value artifact required')
 c={k:ca[k] for k in ['bytes','sha256','tensors','metadata_keys','option_keys','options_sha256']};c['basename']='h3-conditioning-'+c['sha256']+'.safetensors'
 provenance={'stage_a_receipt':{'relative_to_notes':receipt.relative_to(r).as_posix(),'sha256':a.stage_a_receipt_sha256},'value_report':{'relative_to_notes':values.relative_to(r).as_posix(),'sha256':a.value_report_sha256}}
 profiles=[]
 for family,steps in [('original',50),('original',20),('trim',8),('trim',6)]:
  name=family+'-'+str(steps);canonical=v/'canonical'/(family+'.json');assets=v/'assets'/(family+'.json')
  graph=copy.deepcopy(parse(canonical.read_bytes()));graph['prompt']['9']['inputs']['steps']=steps
  graph['prompt']['1001']['inputs']['conditioning_name']=c['basename'];graph['prompt']['92']['inputs']['filename_prefix']='q33-corpus/'+a.corpus+'/'+name
  gp=out/(name+'.graph.json');objects[gp.resolve()]=encode(graph)
  cond={'schema':'q33-corpus-video-conditioning/v1','conditioning':c,'graph':{'relative_path':gp.relative_to(q).as_posix(),'sha256':sha(objects[gp.resolve()])},'provenance':provenance}
  cp=out/(name+'.conditioning.json');objects[cp.resolve()]=encode(cond)
  descriptor=lambda path:{'file':path.relative_to(v).as_posix(),'sha256':sha(objects[path.resolve()] if path.resolve() in objects else path.read_bytes())}
  profile={'schema':'q33-corpus-video-profile/v1','id':name,'family':family,'steps':steps,'role':'reference' if name=='original-50' else 'candidate','dispatch_ready':True,'dispatch_note':'Requires parent review, source freeze and per-run resource preflight; quality remains INCONCLUSIVE','model_subdir':'comfy-'+family+'/models','graph':descriptor(gp),'canonical_graph':descriptor(canonical),'assets':descriptor(assets),'conditioning':descriptor(cp),'corpus_id':a.corpus,'prompt_sha256':m.CORPUS_CASES[a.corpus]['prompt_sha256'],'conditioning_sha256':c['sha256']}
  pp=out/(name+'.profile.json');objects[pp.resolve()]=encode(profile)
  args=SimpleNamespace(profile=pp,profile_sha256=sha(objects[pp.resolve()]),asset_manifest=assets,asset_manifest_sha256=profile['assets']['sha256'],conditioning_manifest=cp,conditioning_manifest_sha256=profile['conditioning']['sha256'],iteration_root=q,notes_root=r)
  helper=SimpleNamespace(frozen_json=frozen);args.loaded_profile=m.load_profile(args,helper);m.load_stage_b_manifest(args,helper,{'max_schema_bytes':1048576});profiles.append(pp)
 require('torch' not in sys.modules,'no torch import')
 out.mkdir(parents=True,exist_ok=False)
 for path,raw in objects.items():
  with path.open('xb') as f:f.write(raw)
 report={'schema':'q33-bound-corpus-metadata/v1','corpus_id':a.corpus,'conditioning_sha256':c['sha256'],'conditioning_bytes':c['bytes'],'profiles':[{'path':x.relative_to(v).as_posix(),'sha256':sha(x.read_bytes())} for x in profiles],'files':[{'path':x.relative_to(v).as_posix(),'bytes':len(raw),'sha256':sha(raw)} for x,raw in objects.items()],'model_dispatches':0,'payload_access':False,'claim':'INCONCLUSIVE'}
 with (out/'BINDING-v1.json').open('xb') as f:f.write(encode(report))
 print(json.dumps({'corpus_id':a.corpus,'profiles':report['profiles'],'conditioning_sha256':c['sha256']}))
if __name__=='__main__':main()
