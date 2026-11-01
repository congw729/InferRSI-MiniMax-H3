"""Q33 frozen offline validator: no network, no GPU, no model imports."""
import argparse, ast, copy, importlib.util, json, pathlib, sys, tempfile
sys.dont_write_bytecode = True
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--candidate-dir",type=pathlib.Path,required=True)
    p.add_argument("--baseline-controller",type=pathlib.Path,required=True)
    a=p.parse_args(); root=a.candidate_dir.resolve();sys.path.insert(0,str(root))
    m=load("q33_timing_events_v1",root/"timing_events_v1.py")
    c=load("q33_speed_controller_check",root/"speed_stage_b_v1.py")
    checks=[]
    def check(name,ok):
        checks.append({"name":name,"ok":bool(ok)})
        if not ok: raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError):check(name,True)
        else:check(name,False)
    nodes={"14":"SamplerCustomAdvanced","10":"VAEDecode"}
    def new():return m.NodeTimingCollector(nodes,"prompt-a","client-a")
    def msg(state,node="14",prompt="prompt-a"):
        return {"type":"progress_state","data":{"prompt_id":prompt,"nodes":{node:{"state":state,"value":0,"max":50,"node_id":node,"prompt_id":prompt,"display_node_id":node,"parent_node_id":None,"real_node_id":node}}}}
    def ready(x):x.ingest({"type":"status","data":{"sid":"client-a","status":{}}},1)
    x=new();ready(x);x.ingest(msg("running"),1000000000);x.ingest(msg("running"),2000000000);x.ingest(msg("finished"),4000000000);x.ingest(msg("finished"),5000000000)
    check("first state transitions only",x.summary()["nodes"]["14"]["observed_wall_seconds"]==3)
    check("missing node null",x.summary()["nodes"]["10"]["observed_wall_seconds"] is None)
    x=new();ready(x);x.ingest(msg("running",prompt="other"),2)
    check("foreign prompt ignored",x.summary()["nodes"]["14"]["start_received_ns"] is None)
    x=new();ready(x);x.ingest(msg("finished"),2)
    check("finish without start null",x.summary()["nodes"]["14"]["observed_wall_seconds"] is None)
    x=new();ready(x);x.ingest(msg("running"),2);x.ingest({"type":"execution_cached","data":{"prompt_id":"prompt-a","nodes":["14"],"timestamp":123}},3);x.ingest(msg("finished"),4)
    check("cached null",x.summary()["nodes"]["14"]["cached"] and x.summary()["nodes"]["14"]["observed_wall_seconds"] is None)
    x=new();ready(x);x.ingest(msg("running"),2);x.ingest({"type":"progress","data":{"prompt_id":"prompt-a","node":"14","value":50,"max":50}},3)
    check("max progress not finish",x.summary()["nodes"]["14"]["finish_received_ns"] is None)
    x=new();ready(x);x.ingest(msg("running"),2);x.ingest(msg("finished"),4);x.invalidate("disconnect")
    check("disconnect invalidates intervals",x.summary()["nodes"]["14"]["observed_wall_seconds"] is None)
    reject("wrong sid rejected",lambda:new().ingest({"type":"status","data":{"sid":"other"}},1))
    x=new();ready(x);reject("unknown node rejected",lambda:x.ingest(msg("running",node="999"),2))
    x=new();ready(x);x.ingest(msg("running"),4);reject("clock regression rejected",lambda:x.ingest(msg("finished"),3))
    x=new();ready(x);bad=msg("running");bad["data"]["nodes"]["14"]["max"]=float("nan")
    reject("nonfinite progress rejected",lambda:x.ingest(bad,2))
    with tempfile.TemporaryDirectory() as temp:
        path=pathlib.Path(temp)/"events.jsonl";w=m.BoundedJsonl(path,max_bytes=80,max_messages=2)
        w.write({"a":1});w.write({"a":2})
        reject("message count bounded",lambda:w.write({"a":3}));w.close()
        path=pathlib.Path(temp)/"bytes.jsonl";w=m.BoundedJsonl(path,max_bytes=8,max_messages=20)
        reject("encoded bytes bounded",lambda:w.write({"big":"x"*20}));w.close()
    for profile_path in sorted((root/"profiles").glob("*.json")):
        profile=json.loads(profile_path.read_text(encoding="utf-8"))
        graph=json.loads((root/profile["graph"]["file"]).read_text(encoding="utf-8"))
        canonical=json.loads((root/profile["canonical_graph"]["file"]).read_text(encoding="utf-8"))
        c.validate_profile_graph(profile,graph,canonical)
        check("profile "+profile["id"],True)
        bad=copy.deepcopy(graph);bad["prompt"]["15"]["inputs"]["noise_seed"]=0
        reject("seed change rejected "+profile["id"],lambda:c.validate_profile_graph(profile,bad,canonical))
    p50=json.loads((root/"profiles/original-50.json").read_text());g50=json.loads((root/p50["graph"]["file"]).read_text());base=json.loads((root/p50["canonical_graph"]["file"]).read_text())
    bad=copy.deepcopy(p50);bad["family"]="trim"
    reject("Trim50 cannot be original baseline",lambda:c.validate_profile_graph(bad,g50,base))
    bad=copy.deepcopy(g50);bad["prompt"]["999"]={"class_type":"LoraLoader","inputs":{}}
    reject("extra model node rejected",lambda:c.validate_profile_graph(p50,bad,base))
    before=ast.parse(a.baseline_controller.read_text(encoding="utf-8"));after=ast.parse((root/"speed_stage_b_v1.py").read_text(encoding="utf-8"))
    def fun(tree,name):return next(x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    for name in ("sample","http","conditioning_header","bounded_notes_json","verify_dynamic_vram_startup"):
        check("unchanged guard "+name,ast.dump(fun(before,name),include_attributes=False)==ast.dump(fun(after,name),include_attributes=False))
    for path in sorted(root.glob("*_v1.py")):
        compile(path.read_text(encoding="utf-8"),path.name,"exec");check("compile "+path.name,True)
    check("no torch imported","torch" not in sys.modules)
    print(json.dumps({"schema":"q33-timing-offline/v1","checks":checks,"completed":len(checks),"all_observed":all(x["ok"] for x in checks),"network":False,"gpu":False,"quality_performance":"INCONCLUSIVE"},indent=2))
if __name__=="__main__":main()
