"""One targeted path/selection regression; never creates run or model directories."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('q33_pair_v2',a.candidate);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
model=Path('E:/synthetic-models/MiniMax-H3').resolve();parent=Path('E:/synthetic-experiments/runs').resolve();run=parent/'owned-one'
assert not run.is_relative_to(model)
assert m.run_relative(run,parent)=='owned-one'
for invalid in (parent.parent/'unowned',parent):
 try:m.run_relative(invalid,parent)
 except ValueError:pass
 else:raise AssertionError('out-of-root run accepted')
b={'baseline_profile':'trim-8','candidate_profile':'trim-6-screen-v1'}
m.require_selected_pair(b,'trim-8','trim-6-screen-v1')
try:m.require_selected_pair(b,'trim-8','trim-4-screen-v1')
except ValueError:pass
else:raise AssertionError('unselected pair accepted')
r={'schema':'q33-rsi-path-regression/v2','case':'run parent independent of model root','checks':{'record_relative_to_explicit_parent':True,'outside_parent_rejected':True,'parent_itself_rejected':True,'selected_8_vs_6_only':True},'all_observed':True,'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_dispatches':0,'network_requests':0,'directories_created':0}
with a.out.open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))