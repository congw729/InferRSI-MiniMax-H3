"""Small offline headroom delta gates; stdlib only, no model/GPU."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,copy,hashlib,importlib.util,json,sys
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--portable-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=Path(__file__).resolve().parent;B=D.parent/'offload-preparation';Q=D.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
checks=[]
def ck(n,v):checks.append({'name':n,'ok':bool(v)});assert v,n
base=B/'async_persistent_stage_b_v1.py';candidate=D/'headroom_persistent_stage_b_v1.py';s=candidate.read_text(encoding='utf8')
for old,new in [('"agents/offload-preparation"','"agents/headroom-preparation"'),('"async-persistent-video-stage-b"','"headroom-persistent-video-stage-b"'),('"requested_aimdo_vram_headroom_gib": 1.0','"requested_aimdo_vram_headroom_gib": 0.75'),('"--vram-headroom", "1.0"','"--vram-headroom", "0.75"')]:s=s.replace(new,old)
ck('controller reverse exact baseline',s==base.read_text(encoding='utf8') and sha(base)=='be5ed4634a212e0d68d9b014f04ac2ca8e8547ade62c93967e73ccdecb9a8bfe')
b=load(base,'metadata_async_baseline');c=load(candidate,'metadata_headroom_candidate');m=load(D/'validate_headroom_persistent_media_v1.py','metadata_headroom_media')
def command(path):
 ns=[n for n in ast.walk(ast.parse(path.read_text(encoding='utf8'))) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets) and isinstance(n.value,ast.List)];assert len(ns)==1
 return eval(compile(ast.Expression(ns[0].value),'<actual-command>','eval'),{'a':SimpleNamespace(portable_python=Path('E:/owned/python.exe'),comfy_root=Path('E:/owned/ComfyUI'),port=8193),'gates':{'reserve_vram_gb':3},'models_root':Path('E:/owned/models'),'str':str})
bc=b.cache_command(command(base),'classic');cc=c.cache_command(command(candidate),'classic');norm=list(cc);norm[norm.index('--vram-headroom')+1]='1.0';ck('only CLI numeric headroom changes',norm==bc and cc[cc.index('--vram-headroom')+1]=='0.75')
ck('async no-pin classic reserve unchanged','--disable-async-offload' not in cc and '--disable-pinned-memory' in cc and '--cache-classic' in cc and cc[cc.index('--reserve-vram')+1]=='3')
g=json.loads((Q/'agents/timing-controller/graphs/trim-8.json').read_text());ck('two graph bytes semantic equal',all(c.request_graph(g,i)==b.request_graph(g,i) for i in (1,2)))
cli=a.portable_root/'ComfyUI/comfy/cli_args.py';manifest=json.loads((D/'SOURCE-SEMANTICS-v1.json').read_text());row=next(x for x in manifest['sources'] if x['relative_to_portable']=='ComfyUI/comfy/cli_args.py');ck('actual native CLI source hash',sha(cli)==row['sha256'])
t=ast.parse(cli.read_text(encoding='utf8'));calls=[x for x in ast.walk(t) if isinstance(x,ast.Call) and x.args and isinstance(x.args[0],ast.Constant) and x.args[0].value=='--vram-headroom'];assert len(calls)==1
parser=argparse.ArgumentParser(allow_abbrev=False);eval(compile(ast.Expression(calls[0]),'<native-cli-declaration>','eval'),{'parser':parser,'float':float});n=parser.parse_args(['--vram-headroom','0.75']);ck('native declaration accepts exact0.75GiB',n.vram_headroom==0.75 and int(n.vram_headroom*1024**3)==805306368)
ck('media exact sourcefilename and SHA',m.CONTROLLER_SHA==sha(candidate) and "sha(D/'headroom_persistent_stage_b_v1.py')==CONTROLLER_SHA" in (D/'validate_headroom_persistent_media_v1.py').read_text())
ck('media requires0.75 requested field',"r.get('requested_aimdo_vram_headroom_gib')==0.75" in (D/'validate_headroom_persistent_media_v1.py').read_text())
ck('no Torch imported','torch' not in sys.modules)
with a.output.open('x',encoding='utf8') as f:json.dump({'schema':'q33-headroom075-offline/v1','checks':checks,'count':len(checks),'complete':True,'GPU':0,'native_declaration_only_not_Comfy_launch':True,'quality_performance':'INCONCLUSIVE'},f,indent=2)
print(json.dumps({'checks':len(checks),'complete':True,'GPU':0}))
