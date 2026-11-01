import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent
v1=(D/'attention_stage_b_v1.py').read_text()
v2=(D/'attention_stage_b_v2.py').read_text()
old='Path(__file__).resolve().parent == a.iteration_root / "agents/timing-controller", "Q33/Q31 roots")'
new=old.replace('agents/timing-controller','agents/attention-preparation')
assert v1.count(old)==1 and v2.count(new)==1 and v2.replace(new,old)==v1
root=D.parents[3]
a=SimpleNamespace(iteration_root=D.parents[1],notes_root=root,q31_root=root/'iterations/2026-10-28-comfy-trim-first-video')
main=next(x for x in ast.parse(v2).body if isinstance(x,ast.FunctionDef) and x.name=='main')
call=next(x for x in ast.walk(main) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='require' and len(x.args)==2 and isinstance(x.args[1],ast.Constant) and x.args[1].value=='Q33/Q31 roots')
code=compile(ast.Expression(call.args[0]),'<actual-root-check>','eval')
assert eval(code,{'Path':Path,'a':a,'__file__':str(D/'attention_stage_b_v2.py')})
assert not eval(code,{'Path':Path,'a':a,'__file__':str(D.parent/'other/attention_stage_b_v2.py')})
obj={'schema':'q33-attention-root-fix-result/v2','exact_reverse_patch':True,'actual_main_root_expression_accepts_dedicated_directory':True,'rejects_other_directory':True,'no_main_service_gpu_executed':True,'candidate_sha256':hashlib.sha256((D/'attention_stage_b_v2.py').read_bytes()).hexdigest()}
with (D/'ROOT-FIX-RESULT-v2.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
print(json.dumps(obj))