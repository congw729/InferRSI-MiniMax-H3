"""Read existing Q33 evidence only. Never dispatch or import Torch. New JSON output only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, re
MAX_JSON=8*1024**2

def number(x):return x if type(x) in (int,float) and math.isfinite(x) else None

def safe_text(x):
 if x is None:return None
 x=str(x)[:512]
 x=re.sub(r'https?://\S+','<url>',x)
 x=re.sub(r'[A-Za-z]:[\\/][^\s\"\']*','<path>',x)
 return x

def extrema(rows, result=None):
 rows=list(rows);result=result or {}
 if isinstance(result.get('last_sample'),dict):rows.append(result['last_sample'])
 def nums(key):return [number(x.get(key)) for x in rows if isinstance(x,dict) and number(x.get(key)) is not None]
 def maximum(key,reported):
  values=nums(key);v=number(result.get(reported))
  if v is not None:values.append(v)
  return max(values) if values else None
 def minimum(key):
  v=nums(key);return min(v) if v else None
 gpu=[number(x.get('gpu_board',{}).get('free_bytes')) for x in rows if isinstance(x,dict) and isinstance(x.get('gpu_board'),dict)]
 gpu=[x for x in gpu if x is not None]
 return {'sample_rows_including_last_sample':len(rows),'owned_rss_peak_bytes':maximum('owned_rss_sum_bytes','peak_owned_rss_sum_bytes'),'owned_private_peak_bytes':maximum('owned_private_sum_bytes','peak_owned_private_sum_bytes'),'gpu_board_free_min_bytes':min(gpu) if gpu else None,'host_available_physical_min_bytes':minimum('available_physical_bytes'),'host_available_commit_min_bytes':minimum('available_commit_bytes'),'scope':'sampled owned process sums / whole-board GPU, not continuous CUDA allocator maxima'}

def native_seconds(record):
 if not isinstance(record,dict):return None
 endpoints={}
 for row in record.get('status',{}).get('messages',[]):
  if isinstance(row,list) and len(row)==2 and row[0] in ('execution_start','execution_success') and isinstance(row[1],dict):
   if row[0] in endpoints:return None
   endpoints[row[0]]=number(row[1].get('timestamp'))
 a,b=endpoints.get('execution_start'),endpoints.get('execution_success')
 return (b-a)/1000 if a is not None and b is not None and b>a else None

class Reader:
 def __init__(self,notes,runs):self.notes=notes;self.runs=runs;self.inputs={};self.issues=[]
 def label(self,p):
  p=p.resolve()
  if p.is_relative_to(self.notes):return 'notes:'+p.relative_to(self.notes).as_posix()
  if p.is_relative_to(self.runs):return 'runs:'+p.relative_to(self.runs).as_posix()
  raise ValueError('read path outside explicit evidence roots')
 def raw(self,p,limit=MAX_JSON):
  label=self.label(p)
  if not p.is_file():return None
  before=p.stat()
  if before.st_size>limit:self.issues.append({'input':label,'issue':'size_bound'});return None
  b=p.read_bytes();after=p.stat();stable=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  self.inputs[label]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'stable_during_read':stable}
  if not stable:self.issues.append({'input':label,'issue':'changed_during_read'})
  return b
 def js(self,p):
  b=self.raw(p)
  if b is None:return None
  try:return json.loads(b)
  except (ValueError,UnicodeError):self.issues.append({'input':self.label(p),'issue':'invalid_or_incomplete_json'});return None
 def lines(self,p,limit=64*1024**2):
  b=self.raw(p,limit)
  if b is None:return []
  rows=[]
  for i,line in enumerate(b.splitlines()):
   if not line.strip():continue
   try:rows.append(json.loads(line))
   except (ValueError,UnicodeError):self.issues.append({'input':self.label(p),'issue':'invalid_or_incomplete_jsonl_line','line':i+1})
  return rows

def child(root,rel):
 if not isinstance(rel,str):return None
 p=Path(rel)
 if p.is_absolute() or p.drive or '..' in p.parts:return None
 p=(root/p).resolve()
 return p if p.is_relative_to(root) else None

def media(reader,run,artifact):
 found=[];expected=artifact.get('sha256') if isinstance(artifact,dict) else None
 if not run or not run.is_dir():return {'status':'not_available','matches':[]}
 for p in sorted(run.glob('media-inspection*/validation-summary.json'))[:32]:
  s=reader.js(p)
  if not isinstance(s,dict) or not expected or s.get('expected_input_sha256')!=expected:continue
  dp=p.parent/'decode-result.json';d=reader.js(dp) or {};binding=reader.inputs.get(reader.label(dp),{}).get('sha256')==s.get('decoder_result_sha256')
  complete=(s.get('status')=='inspection_ready_human_review_pending' and s.get('owned_root_exited') is True and s.get('worker_returncode')==0 and binding and d.get('input_sha256')==expected and d.get('full_demux_eof') is True and d.get('container_closed') is True and d.get('input_unchanged') is True)
  v=d.get('video',{});a=d.get('audio',{})
  found.append({'directory_relative_to_run':p.parent.relative_to(run).as_posix(),'status':safe_text(s.get('status')),'structural_complete':complete,'decoder_result_hash_binding':binding,'worker_returncode':s.get('worker_returncode'),'owned_root_exited':s.get('owned_root_exited'),'video':{k:v.get(k) for k in ('decoded_frames','width','height','pts_rate','decoded_duration_seconds','not_all_black','not_temporally_constant')},'audio':{k:a.get(k) for k in ('sample_rate','channels','samples_per_channel','sample_count_duration_seconds','all_values_finite','not_all_zero_native_samples')},'pngs':[{k:x.get(k) for k in ('name','frame','bytes','sha256')} for x in d.get('pngs',[])],'quality':'not_assessed_by_summary'})
 return {'status':'structurally_complete_human_review_pending' if any(x['structural_complete'] for x in found) else ('incomplete' if found else 'not_available'),'matches':found}

def request_row(reader,run,result,req,rows):
 persistent=req is not None;req=req or result;idx=req.get('request_index') if persistent else None
 hp=child(run,req.get('history_relative_to_run','history.raw.json')) if run else None
 h=reader.js(hp) if hp else None;record=None;prompt=req.get('prompt_id')
 if isinstance(h,dict):
  if prompt in h:record=h[prompt]
  elif not prompt and len(h)==1:record=next(iter(h.values()))
 timing=req.get('node_timing')
 tp=child(run,req.get('timing_relative_to_run',result.get('node_timing_relative_to_run','timing/summary.json'))) if run else None
 on_disk=reader.js(tp) if tp else None
 if isinstance(on_disk,dict):timing=on_disk
 timing=timing if isinstance(timing,dict) else {};nodes=timing.get('nodes',{});graph=record.get('prompt',[None,None,{}])[2] if isinstance(record,dict) and len(record.get('prompt',[]))>2 else {}
 graph=graph if isinstance(graph,dict) else {};latent=next((x.get('inputs',{}) for x in graph.values() if isinstance(x,dict) and x.get('class_type')=='EmptyMiniMaxH3LatentAV'),{})
 artifact=req.get('artifact',{});artifact=artifact if isinstance(artifact,dict) else {}
 selected=[x for x in rows if not persistent or x.get('request_index')==idx]
 rr=dict(req)
 if persistent:
  snapshots=[]
  if isinstance(req.get('after_request_sample'),dict):snapshots.append(req['after_request_sample'])
  ls=result.get('last_sample')
  if isinstance(ls,dict) and ls.get('request_index')==idx:snapshots.append(ls)
  selected+=snapshots;rr.pop('last_sample',None)
 else:rr=result
 actual_seed=graph.get('15',{}).get('inputs',{}).get('noise_seed');steps=graph.get('9',{}).get('inputs',{}).get('steps')
 return {'request_index':idx,'status':safe_text(req.get('status',result.get('status'))),'seed':actual_seed if actual_seed is not None else req.get('seed'),'steps':steps,'native_prompt_wall_seconds':native_seconds(record),'request_elapsed_seconds':number(req.get('request_elapsed_seconds')),'history_success':record.get('status',{}).get('completed') if isinstance(record,dict) else None,'history_relative_to_run':hp.relative_to(run).as_posix() if hp and hp.is_file() else None,'actual_geometry':{'width':latent.get('width'),'height':latent.get('height'),'frames':latent.get('length')},'dense_attention':graph.get('128',{}).get('inputs',{}).get('attention'),'nodes':{str(k):{f:v.get(f) for f in ('class_type','observed_wall_seconds','cached','progress_value','progress_max','last_state')} for k,v in nodes.items() if isinstance(v,dict)},'timing':{k:timing.get(k) for k in ('ready','terminal','invalid_reasons','observer_thread_exited','error_type')},'resources':extrema(selected,rr),'artifact':{k:artifact.get(k) for k in ('bytes','sha256')},'media':media(reader,run,artifact)}

def group_model(profile):
 if profile.startswith('original-'):return 'MiniMax-H3-original-INT8'
 if profile.startswith('trim-'):return 'FastH3-Trim-NVFP4'
 if profile.startswith('turbo-hybrid-'):return 'MiniMax-H3-INT8-Turbo-author-hybrid'
 if profile.startswith('turbo-'):return 'MiniMax-H3-INT8-Turbo-native-LoRA'
 return 'unknown'

def summarize(notes,runs):
 q=notes/'iterations/2026-10-31-speed-quality-search';reader=Reader(notes,runs);entries=[]
 for sp in sorted((q/'results').glob('RUNSPEC-*.json')):
  s=reader.js(sp)
  if not isinstance(s,dict):continue
  trial=Path(s.get('trial','')).resolve();run=Path(s.get('run','')).resolve()
  if not trial.is_relative_to(q/'results') or run.parent!=runs:
   reader.issues.append({'input':reader.label(sp),'issue':'runspec trial/run boundary'});continue
  r=reader.js(trial/'result.json') or {};ex=reader.js(trial/'execution.json') or {};resources=reader.lines(trial/'resources.jsonl')
  argv=s.get('argv',[]);controller=next((Path(x).name for x in argv if isinstance(x,str) and x.endswith('.py')),None)
  def arg(name):return argv[argv.index(name)+1] if name in argv and argv.index(name)+1<len(argv) else None
  mode=r.get('mode') or arg('--mode');profile=r.get('profile_id') or s.get('profile') or 'unknown';corpus=r.get('corpus_id') or s.get('corpus_id') or s.get('corpus') or 'toycar'
  kind='persistent' if 'requests' in r or 'persistent' in str(controller) else 'resolution' if 'resolution' in str(controller) else 'attention' if 'attention' in str(controller) else 'turbo-corpus' if 'turbo_corpus' in str(controller) else 'corpus' if 'corpus' in str(controller) else 'single'
  reqs=[request_row(reader,run,r,x,resources) for x in r['requests']] if isinstance(r.get('requests'),list) else [request_row(reader,run,r,None,resources)]
  sv0=number(r.get('source_verification_start_ns'));sv1=number(r.get('source_verification_end_ns'))
  entries.append({'runspec_relative_to_notes':sp.relative_to(notes).as_posix(),'run_name':run.name,'trial_relative_to_notes':trial.relative_to(notes).as_posix(),'timestamp':s.get('timestamp'),'model':group_model(profile),'profile':profile,'corpus':corpus,'mode':mode,'kind':kind,'controller':controller,'controller_sha256':r.get('driver_sha256') or s.get('controller_sha256'),'scenario_declared':{k:v for k,v in s.get('scenario',{}).items() if k in ('width','height','frames','fps','seed','seeds')},'memory_policy':r.get('memory_policy',s.get('memory_policy')),'cache_policy':r.get('cache_policy',s.get('cache_policy')),'requested_aimdo_vram_headroom_gib':r.get('requested_aimdo_vram_headroom_gib'),'status':r.get('status','result_not_available'),'execution_status':ex.get('status'),'execution_returncode':ex.get('returncode'),'prompt_posts':r.get('prompt_posts'),'completed_requests':r.get('completed_requests'),'controller_wall_seconds':number(r.get('elapsed_seconds')),'executor_wall_seconds':number(ex.get('elapsed_seconds')),'source_verification_wall_seconds':(sv1-sv0)/1e9 if sv0 is not None and sv1 is not None else None,'error_type':safe_text(r.get('error_type')),'error':safe_text(r.get('error')),'cleanup':{k:r.get(k) for k in ('job_close_requested','owned_root_exited','owned_returncode','cleanup_diagnostics')},'resources':extrema(resources,r),'asset_identities':[{k:x.get(k) for k in ('id','bytes','sha256')} for x in r.get('verified_assets',[]) if isinstance(x,dict)],'requests':reqs})
 pairs=[]
 for p in sorted((q/'results/rsi-pairs').glob('*')):
  if not p.is_dir():continue
  st=reader.js(p/'status.json') or {};e=reader.js(p/'evaluation.json') or {};env=reader.js(p/'environment.json') or {};events=reader.lines(p/'events.jsonl');order=[]
  for event in events:
   if event.get('phase')!='measurement':continue
   side=event.get('side');i=event.get('trial')
   if side not in ('baseline','candidate') or type(i) is not int:continue
   t=reader.js(p/side/f'trial-{i:03}'/'result.json') or {}
   order.append({'side':side,'trial':i,'profile':t.get('profile'),'run_name':Path(t['run_relative_to_runs_parent']).name if isinstance(t.get('run_relative_to_runs_parent'),str) else None,'metrics':t.get('metrics'),'media_status':t.get('media_structural_validation')})
  pair={'run_relative_to_notes':p.relative_to(notes).as_posix(),'status':st.get('state'),'verdict':e.get('verdict'),'reasons':e.get('reasons'),'metric_key':e.get('metric_key'),'comparison_scope':e.get('comparison_scope'),'baseline':e.get('baseline'),'candidate':e.get('candidate'),'descriptive_improvement_pct':e.get('improvement_pct'),'statistical_significance':e.get('statistical_significance'),'source_identity_kind':env.get('source_identity_kind'),'source_verified':e.get('source_integrity',{}).get('verified'),'actual_order':order}
  pairs.append(pair)
  for x in entries:
   tags=[{'pair_relative_to_notes':pair['run_relative_to_notes'],'side':t['side'],'trial':t['trial']} for t in order if t['run_name']==x['run_name']]
   if tags:x.setdefault('rsi_pair_membership',[]).extend(tags)
 groups={}
 for i,e in enumerate(entries):
  geometry=e['requests'][0]['actual_geometry'] if e['requests'] else {}
  key=' | '.join(str(e[k]) for k in ('model','profile','corpus','mode','kind','memory_policy','cache_policy'))+' | '+json.dumps(geometry,sort_keys=True)
  groups.setdefault(key,[]).append(i)
 return {'schema':'q33-readonly-results-summary/v1','snapshot_utc':datetime.now(timezone.utc).isoformat(),'claim':'INCONCLUSIVE','coverage':'Only discovered Q33 RUNSPEC files and actual linked evidence; missing/active files preserved as unavailable, no history reconstructed from logs','timing_scope':'native history graph wall and receiver-observed node wall; controller separately includes startup/checks; no GPU-compute or service-performance claim','entries':entries,'groups':[{'group':k,'entry_indices':v} for k,v in sorted(groups.items())],'rsi_pairs_separate':pairs,'issues':reader.issues,'input_identities':reader.inputs,'limitations':['No model or media re-execution and no weight reads','Human perceptual verdict is not inferred from structural media checks','Fresh services do not imply cold OS filesystem cache','Resources include last_sample and use gpu_board; sampled extrema can miss unsampled peaks','Prior frozen evidence and scripts are unchanged']}

def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for n in ('notes-root','runs-parent','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();notes=a.notes_root.resolve();runs=a.runs_parent.resolve();out=a.output.resolve();q=notes/'iterations/2026-10-31-speed-quality-search'
 if not out.is_relative_to(q) or out.exists() or not out.parent.is_dir():raise RuntimeError('new output must be under Q33')
 result=summarize(notes,runs)
 encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
 if re.search(r'[A-Za-z]:[\\/]',encoded):raise RuntimeError('absolute Windows path in summary')
 with out.open('x',encoding='utf-8') as f:f.write(encoded+'\n')
 print(json.dumps({'entries':len(result['entries']),'pairs':len(result['rsi_pairs_separate']),'issues':len(result['issues']),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
if __name__=='__main__':main()