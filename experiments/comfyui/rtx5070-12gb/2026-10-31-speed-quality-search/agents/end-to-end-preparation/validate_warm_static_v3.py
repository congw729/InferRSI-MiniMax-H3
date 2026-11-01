"""Repair only static scope verification; retain earlier 11 arithmetic outcomes."""
import ast,copy,hashlib,json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--freeze-sha256',required=True);a=p.parse_args();d=Path(__file__).resolve().parent;fp=d/'WARM-STATIC-FREEZE-v3.json'
assert hashlib.sha256(fp.read_bytes()).hexdigest()==a.freeze_sha256
for x in json.loads(fp.read_text())['files']:assert (d/x['path']).stat().st_size==x['bytes'] and hashlib.sha256((d/x['path']).read_bytes()).hexdigest()==x['sha256']
def funcs(p):return {n.name:n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
def dump(x):return ast.dump(x,include_attributes=False)
old=funcs(d/'end_to_end_v1.py');new=funcs(d/'end_to_end_v2.py');checks=[]
def check(n,v):checks.append({'name':n,'ok':bool(v)});assert v,n
check('other_inherited_functions_exact',all(dump(old[k])==dump(new[k]) for k in old if k not in ['main','run_one']))
class Reverse(ast.NodeTransformer):
 def __init__(self):self.count=0
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='warm_admit':self.count+=1;return ast.parse('e2e.fresh_ready(state["last_sample"])',mode='eval').body
  return node
reverse=Reverse();reversed_node=reverse.visit(copy.deepcopy(new['run_one']));check('run_one_exact_two_call_reverse',reverse.count==2 and dump(reversed_node)==dump(old['run_one']))
h1=funcs(d/'e2e_helpers_v1.py');h2=funcs(d/'e2e_helpers_v2.py');check('all_original_helpers_exact',all(dump(h1[k])==dump(h2[k]) for k in h1))
s=(d/'end_to_end_v2.py').read_text();check('initial_cold_gate_retained',s.count('require(e2e.fresh_ready(state["last_sample"]), "initial external admission 12/16.5/6")')==1);check('warm_three_call_sites',s.count('warm_admit()')==4)
out={'schema':'q33-warm-static-scope/v3','complete':True,'checks':checks,'check_count':len(checks),'prior_arithmetic_successes_preserved':11,'candidate_changed':False,'model_dispatches':0,'claim':'INCONCLUSIVE'}
with (d/'WARM-STATIC-RESULT-v3.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'static_checks':len(checks),'arithmetic_not_repeated':11}))
