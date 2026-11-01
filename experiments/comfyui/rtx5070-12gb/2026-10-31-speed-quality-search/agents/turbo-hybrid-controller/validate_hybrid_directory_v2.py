"""One targeted directory regression; stdlib AST slice only, no runtime import."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,hashlib,json,sys
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
old=a.baseline.read_text(encoding='utf8');new=a.candidate.read_text(encoding='utf8')
anchor='        for path in dirs.values():\n            path.mkdir()\n'
added='        (dirs["base"] / "custom_nodes").mkdir()\n'
assert old.count(anchor)==1 and new==old.replace(anchor,anchor+added)
assert a.run_dir.drive.upper()=='E:' and not a.run_dir.exists() and a.run_dir.parent.is_dir()
a.run_dir.mkdir()
t=ast.parse(new);main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
block=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any(isinstance(s,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='dirs' for k in s.targets) for s in n.body))
start=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='dirs' for k in n.targets))
slice_=block.body[start:start+3]
assert [type(n).__name__ for n in slice_]==['Assign','For','Expr']
ns={'a':SimpleNamespace(run_dir=a.run_dir)}
exec(compile(ast.Module(body=slice_,type_ignores=[]),str(a.candidate),'exec'),ns)
dirs=ns['dirs'];target=dirs['base']/'custom_nodes'
assert target.is_dir() and not list(target.iterdir()) and len(list(a.run_dir.iterdir()))==7
assert all(p.is_dir() for p in dirs.values()) and 'torch' not in sys.modules
result={'schema':'q33-hybrid-directory-regression/v2','completed':1,'checks':[{'name':'actual_candidate_directory_slice_creates_empty_custom_nodes_only','ok':True}],'candidate_sha256':sha(a.candidate),'baseline_sha256':sha(a.baseline),'scope':'exact one-line diff; execute only directory AST slice in fresh E task output, no service/Torch/author node/model/GPU','run_id':a.run_dir.name,'directories_created':8,'new_files_in_run':0,'dispatches':0,'quality_performance':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'completed':1,'dispatches':0,'candidate_sha256':result['candidate_sha256']}))
