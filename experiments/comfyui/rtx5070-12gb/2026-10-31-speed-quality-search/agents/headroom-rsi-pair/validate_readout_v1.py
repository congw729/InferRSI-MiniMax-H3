"""Frozen narrow readout checks; existing small metadata only, no model/media execution."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--candidate-sha256',required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args()
assert hashlib.sha256(a.candidate.read_bytes()).hexdigest()==a.candidate_sha256
sp=importlib.util.spec_from_file_location('warm_pair_candidate',a.candidate);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
Q=a.candidate.resolve().parents[2];rows=[];fixtures={}
def read(p):return json.loads(p.read_bytes())
for side,tname,mname in [('A','inferrsi-async-classic-98274aade90b40659e74f9a86b4b2563','async-persistent-media-classic-f895e5c6377d43608ffd1bed59790508'),('B','inferrsi-headroom075-classic-6a072b6e64c64711a405e57e111591ed','headroom075-media-classic-6ada4efd150b483fafea25173bfe0551')]:
 t=Q/'results'/tname;r=read(t/'result.json');argv=read(t/'execution.json')['argv'];run=Path(argv[argv.index('--run-dir')+1]);hist=[read(run/x['history_relative_to_run']) for x in r['requests']];media=read(Q/'results'/mname/'result.json');fixtures[side]=(r,hist,media)
 values=m.extract_warm_metrics(r,hist,media,side);assert values['warm_second_native_history_seconds']==r['requests'][1]['metrics']['native_prompt_wall_seconds'];rows.append({'gate':'actual_metadata_'+side,'completed':True})
def reject(name,mut):
 r,h,media=copy.deepcopy(fixtures['A']);mut(r,h,media)
 try:m.extract_warm_metrics(r,h,media,'A')
 except (ValueError,KeyError,TypeError,IndexError):rows.append({'gate':name,'completed':True});return
 raise AssertionError(name+' accepted')
reject('missing_second_request',lambda r,h,m:r['requests'].pop())
reject('first_media_failed',lambda r,h,m:m['requests'][0].update(status='failed'))
reject('changed_second_seed',lambda r,h,m:r['requests'][1].update(seed=424242))
reject('changed_second_mp4',lambda r,h,m:r['requests'][1]['artifact'].update(sha256='0'*64))
reject('cached_second_sampler',lambda r,h,m:r['requests'][1]['node_timing']['nodes']['14'].update(cached=True))
reject('seven_steps',lambda r,h,m:r['requests'][1]['node_timing']['nodes']['14'].update(progress_value=7))
reject('owned_not_reclaimed',lambda r,h,m:r.update(owned_root_exited=False))
reject('wrong_controller',lambda r,h,m:r.update(driver_sha256='0'*64))
reject('wrong_headroom',lambda r,h,m:r.update(requested_aimdo_vram_headroom_gib=0.75))
def missing_endpoint(r,h,m):
 pid=r['requests'][1]['prompt_id'];h[1][pid]['status']['messages']=[x for x in h[1][pid]['status']['messages'] if x[0]!='execution_success']
reject('missing_history_endpoint',missing_endpoint)
# Pure filename-content snapshot helper: no dispatch and no runtime imports.
fixture=a.result.parent/'snapshot-fixture-v1';fixture.mkdir();source=fixture/'tiny.txt';source.write_text('frozen\n',encoding='utf8');manifest=fixture/'bundle.json';manifest.write_text(json.dumps({'schema':'q33-rsi-source-bundle/v1','files':[{'root':'unit','path':'tiny.txt','bytes':source.stat().st_size,'sha256':m.sha(source)}]}),encoding='utf8');hs=m.sha(manifest)
assert m.base.source_snapshot(manifest,hs,{'unit':fixture})['available'];source.write_text('changed\n',encoding='utf8');assert not m.base.source_snapshot(manifest,hs,{'unit':fixture})['available'];rows.append({'gate':'source_drift_rejected','completed':True})
result={'schema':'q33-headroom-rsi-readout-checks/v1','candidate_sha256':a.candidate_sha256,'checks':rows,'completed':len(rows),'gpu_dispatches':0,'model_or_media_execution':False,'scope':'two actual small metadata readouts plus pure in-memory negative mutations and tiny unit source file; not actual lifecycle execution','claim':'INCONCLUSIVE'}
with a.result.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'completed':len(rows),'model_dispatches':0}))
