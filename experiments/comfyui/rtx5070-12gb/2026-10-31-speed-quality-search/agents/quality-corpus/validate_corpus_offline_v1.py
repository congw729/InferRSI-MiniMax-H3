import argparse,ast,hashlib,importlib.util,json,pathlib,sys
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument("--candidate-dir",type=pathlib.Path,required=True);p.add_argument("--notes-root",type=pathlib.Path,required=True);a=p.parse_args();C=a.candidate_dir.resolve();R=a.notes_root.resolve()
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location("corpus_candidate",C/"conditioning_corpus_v1.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
mf=json.loads((C/"CORPUS-MANIFEST-v1.json").read_text());base=json.loads((C/"CANONICAL-STAGE-A-v1.json").read_text())
protocol=json.loads((R/"iterations/2026-10-31-speed-quality-search/agents/quality-review/QUALITY-SCREEN-PROTOCOL-v1.json").read_text())
texts={x["id"]:x["prompt"] for x in protocol["suggested_next_prompts_not_executed"]}
checks=[]
for cid,case in mf["cases"].items():
 g=json.loads((C/case["graph"]["file"]).read_text())
 m.validate_corpus_graph(cid,case,g,base)
 assert case["prompt"]==texts[cid] and case["expected_conditioning_sha256"] is None and not case["stageB_ready"]
 checks.append(cid+" exact prompt, graph and no manufactured conditioning identity")
before=ast.parse((R/"iterations/2026-10-28-comfy-trim-first-video/tools/conditioning_stage_v1.py").read_text());after=ast.parse((C/"conditioning_corpus_v1.py").read_text())
def fun(t,n):return next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==n)
for name in ("sample","http","conditioning_header","find_conditioning_artifact","verify_dynamic_vram_startup"):
 assert ast.dump(fun(before,name),include_attributes=False)==ast.dump(fun(after,name),include_attributes=False),name
 checks.append("unchanged "+name)
assert "torch" not in sys.modules
print(json.dumps({"schema":"q33-corpus-offline/v1","checks":checks,"all_observed":True,"model_posts":0,"network":False},indent=2))
