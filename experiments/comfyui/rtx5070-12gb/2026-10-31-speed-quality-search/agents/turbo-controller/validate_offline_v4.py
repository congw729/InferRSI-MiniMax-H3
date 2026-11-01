"""Frozen Q33 native Turbo metadata-only checks; no service/network/GPU."""
import argparse,ast,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--baseline",type=Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location("turbo_controller_check",a.root/"turbo_stage_b_v4.py");c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 checks=[]
 def check(name,ok):
  checks.append({"id":name,"ok":bool(ok)})
  if not ok:raise AssertionError(name)
 def reject(name,fn):
  try:fn()
  except (RuntimeError,ValueError,KeyError):check(name,True)
  else:check(name,False)
 def load(p):return json.loads(Path(p).read_text(encoding="utf8"))
 profiles=[]
 for steps in (6,8):
  profile=load(a.root/f"profiles/turbo-native-{steps}.pending.json");graph=load(a.root/profile["graph"]["file"]);canonical=load(a.root/profile["canonical_graph"]["file"])
  c.validate_profile_graph(profile,graph,canonical);check(f"fixed_turbo_{steps}",True)
  check(f"pending_{steps}",profile["dispatch_ready"]is False)
  for field,val in [("strength_model",0.9),("lora_name","other.safetensors"),("model",["148",0])]:
   bad=copy.deepcopy(graph);bad["prompt"]["1100"]["inputs"][field]=val
   reject(f"{steps}_reject_lora_{field}",lambda:c.validate_profile_graph(profile,bad,canonical))
  for node,field,val in [("15","noise_seed",0),("17","sampler_name","dpmpp_2m"),("9","scheduler","beta"),("1002","length",124),("6","unet_name","trim.safetensors")]:
   bad=copy.deepcopy(graph);bad["prompt"][node]["inputs"][field]=val
   reject(f"{steps}_reject_{node}_{field}",lambda:c.validate_profile_graph(profile,bad,canonical))
  badp=copy.deepcopy(profile);badp["steps"]=4
  reject(f"{steps}_reject_4step",lambda:c.validate_profile_graph(badp,graph,canonical))
  badp=copy.deepcopy(profile);badp["role"]="reference"
  reject(f"{steps}_reject_reference",lambda:c.validate_profile_graph(badp,graph,canonical))
  profiles.append((profile,graph))
 profile,graph=profiles[0]
 assets=load(a.root/"assets/turbo-native.pending.json")
 c.ASSET_EXPECTED=c.EXPECTED_BY_FAMILY["turbo-native"];c.ASSET_IDS=set(c.ASSET_EXPECTED)
 args=types.SimpleNamespace(asset_manifest=a.root/"assets/turbo-native.pending.json",asset_manifest_sha256="unused",loaded_profile=profile,notes_root=a.root)
 class V:
  def __init__(self,doc):self.doc=doc;self.calls=0
  def frozen_json(self,*args):self.calls+=1;return copy.deepcopy(self.doc)
 v=V(assets);reject("pending_receipt_before_payload",lambda:c.load_asset_manifest(args,v,{"max_schema_bytes":1048576}));check("pending_only_manifest_read",v.calls==1)
 for name,mut in [("missing_lora",lambda x:x["assets"].pop()),("changed_lora_sha",lambda x:x["assets"][-1].update(sha256="0"*64)),("extra_asset",lambda x:x["assets"].append(copy.deepcopy(x["assets"][0])))]:
  x=copy.deepcopy(assets);mut(x);v=V(x);reject(name,lambda:c.load_asset_manifest(args,v,{"max_schema_bytes":1048576}))
 good={"input":{"required":{"model":["MODEL"],"lora_name":[[c.TURBO_FILENAME]],"strength_model":["FLOAT"]},"optional":{}},"output":["MODEL"]}
 c.validate_lora_schema(good,graph);check("native_lora_schema",True)
 for name,mut in [("missing_file",lambda x:x["input"]["required"].update(lora_name=[["other"]])),("wrong_output",lambda x:x.update(output=["CLIP"])),("extra_required",lambda x:x["input"]["required"].update(extra=["FLOAT"]))]:
  x=copy.deepcopy(good);mut(x);reject(name,lambda:c.validate_lora_schema(x,graph))
 before=ast.parse(a.baseline.read_text(encoding="utf8"));after=ast.parse((a.root/"turbo_stage_b_v4.py").read_text(encoding="utf8"))
 def fun(tree,name):return next(n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
 for name in ("sample","http","guarded_hash","conditioning_header","bounded_notes_json","verify_dynamic_vram_startup","cleanup_owned_v2"):
  check("unchanged_"+name,ast.dump(fun(before,name),include_attributes=False)==ast.dump(fun(after,name),include_attributes=False))
 check("gates_same",c.STAGE_GATES_SHA=="dd9658a253e1e0c73bd1d1fe9034055622289a9f6748486fb55761bfa195eecf")
 check("core_manifest_same",c.SOURCE_MANIFEST_SHA=="bf0d3e3b09f8130053735eaa6645bcc4ff753833a97ef4ae306d7d223283f007")
 check("no_timing_import_at_module_entry","q33_frozen_native_timing_v1"not in sys.modules)
 reject("timing_wrong_path",lambda:c.load_timing_module(a.root/"wrong.py",a.root))
 check("no_torch_import","torch"not in sys.modules)
 result={"schema":"q33-turbo-controller-offline/v4","checks":checks,"completed":len(checks),"all_observed":all(x["ok"]for x in checks),"candidate_sha256":hashlib.sha256((a.root/"turbo_stage_b_v4.py").read_bytes()).hexdigest(),"scope":"offline graph/metadata/pending receipt/schema/AST identity only; no real LoRA header/model/body or dispatch, no network/GPU"}
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
