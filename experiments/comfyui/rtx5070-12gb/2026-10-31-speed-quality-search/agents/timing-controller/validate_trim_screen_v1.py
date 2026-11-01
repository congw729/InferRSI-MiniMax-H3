import argparse,copy,hashlib,importlib.util,json,pathlib,sys
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument("--candidate-dir",type=pathlib.Path,required=True);p.add_argument("--notes-root",type=pathlib.Path,required=True);a=p.parse_args();C=a.candidate_dir.resolve();R=a.notes_root.resolve()
def load(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v3=load("q33_screen_v3",C/"speed_stage_b_v3.py");v4=load("q33_screen_v4",C/"speed_stage_b_v4.py")
old=(C/"speed_stage_b_v3.py").read_bytes();new=(C/"speed_stage_b_v4.py").read_bytes()
assert old.count(b'"trim":{8,6}')==1 and new.replace(b'"trim":{8,6,4}',b'"trim":{8,6}',1)==old
checks=["one literal source change reversible"]
base=json.loads((C/"canonical/trim.json").read_text())
old6=json.loads((C/"profiles/trim-6.json").read_text());assert old6["dispatch_ready"] is False
for filename,mod,steps in [("trim-6-screen-v1",v3,6),("trim-4-screen-v1",v4,4)]:
 pf=json.loads((C/f"profiles/{filename}.json").read_text());g=json.loads((C/pf["graph"]["file"]).read_text())
 mod.validate_profile_graph(pf,g,base)
 assert pf["dispatch_ready"] is True and pf["family"]=="trim" and pf["steps"]==steps and pf["role"]=="candidate"
 assert g["prompt"]["127"]["inputs"]==base["prompt"]["127"]["inputs"]
 if steps==6:assert pf["graph"]==old6["graph"] and pf["conditioning"]==old6["conditioning"]
 checks.append(filename+" exact graph/config")
pf4=json.loads((C/"profiles/trim-4-screen-v1.json").read_text());g4=json.loads((C/pf4["graph"]["file"]).read_text())
try:v3.validate_profile_graph(pf4,g4,base)
except RuntimeError:checks.append("old v3 continues rejecting4")
else:raise AssertionError("old admission changed")
assert "torch" not in sys.modules
print(json.dumps({"schema":"q33-trim-screen-offline/v1","checks":checks,"all_observed":True,"models":0,"network":False},indent=2))
