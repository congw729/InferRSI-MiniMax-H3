import argparse,ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--notes-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
v=Path(__file__).resolve().parent;q=v.parents[1];old=a.notes_root/'iterations/2026-10-28-comfy-trim-first-video/validator/real_conditioning_values_v3.py';new=v/'corpus_conditioning_values_v1.py'
checks=[]
def ck(name,ok):
 checks.append({'name':name,'ok':bool(ok)});assert ok,name
s=old.read_text();n=new.read_text()
ck('value_scanner_exact_unchanged',s[s.index('        before=target.stat()'):s.index("        out.update(complete=True")]==n[n.index('        before=target.stat()'):n.index("        out.update(complete=True")])
so=ast.parse(s);sn=ast.parse(n)
for name in ['MAX_FILE','MAX_HEADER','CHUNK','DEADLINE_SECONDS']:
 def get(t):return next(ast.dump(x,include_attributes=False) for x in t.body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id==name for y in x.targets))
 ck('resource_'+name,get(so)==get(sn))
ck('cleanup_single_post_guard_preserved',"check('receipt_cleanup_and_single_prompt',receipt.get('owned_root_exited') is True and receipt.get('job_close_requested') is True and receipt.get('prompt_posts')==1)" in n)
ck('no_hardcoded_old_token_count','594360' not in n and 'ad5c2652' not in n and '==29' not in n)
spec=importlib.util.spec_from_file_location('values_unit',new);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f=json.loads((v/'VALUES-FREEZE-v1.json').read_text())
def req(name,ok):
 if not ok:raise m.Rejected(name)
for case,c in f['cases'].items():
 unit={'mode':'quality-corpus-stage-a','status':'conditioning_artifact_received_unvalidated','corpus_id':case,'prompt_sha256':c['prompt_sha256'],'driver_sha256':f['stage_a_driver_sha256'],'graph_sha256':c['graph_sha256']}
 m.validate_corpus_receipt(unit,f,case,req);ck(case+'_unit_identity_accept',True)
 for key in ['prompt_sha256','graph_sha256','driver_sha256','corpus_id','mode']:
  bad=copy.deepcopy(unit);bad[key]='wrong';rejected=False
  try:m.validate_corpus_receipt(bad,f,case,req)
  except m.Rejected:rejected=True
  ck(case+'_unit_reject_'+key,rejected)
ck('no_torch_import','torch' not in sys.modules)
out={'schema':'q33-corpus-values-offline/v1','unit_metadata_only_not_real_receipts':True,'model_dispatches':0,'network_requests':0,'checks':checks,'check_count':len(checks),'claim':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf-8') as h:json.dump(out,h,indent=2);h.write('\n')
print(json.dumps({'checks':len(checks),'all_ok':all(x['ok'] for x in checks)}))
