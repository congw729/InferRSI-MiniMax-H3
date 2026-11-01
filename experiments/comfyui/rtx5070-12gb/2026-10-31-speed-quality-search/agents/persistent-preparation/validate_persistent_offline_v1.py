"""Frozen bounded persistent-loop offline tests; stdlib only, no service/model/GPU."""
import argparse,ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('q33_persistent_offline_candidate',a.candidate);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
checks=[]
def ck(n,v):checks.append({'name':n,'ok':bool(v)});assert v,n
def rejects(fn):
 try:fn()
 except RuntimeError:return True
 return False
base=['ppy','--disable-smart-memory','--cache-none','--vram-headroom','1.0','--reserve-vram','3.0']
ck('two_memory_policies_single_flag_delta',m.memory_command(base,'baseline')==base and m.memory_command(base,'smart')==[x for x in base if x!='--disable-smart-memory'] and rejects(lambda:m.memory_command(base,'other')))
g=json.loads(a.graph.read_text());r1=m.request_graph(g,1);r2=m.request_graph(g,2)
for r in (r1,r2):r['prompt']['92']['inputs']['filename_prefix']=g['prompt']['92']['inputs']['filename_prefix']
ck('two_graphs_only_prefix_change',r1==r2==g and rejects(lambda:m.request_graph(g,3)))
seen=[];m.two_requests(lambda i:seen.append(i));ck('exact_two_serial_calls',seen==[1,2])
for fail in (1,2):
 seen=[]
 def one(i):
  seen.append(i)
  if i==fail:raise RuntimeError('synthetic stop')
 ck('fail_fast_request_'+str(fail),rejects(lambda:m.two_requests(one)) and seen==list(range(1,fail+1)))
st={'prompt_posts':0,'requests':[]};m.begin_request(st,1,'p1');m.begin_post(st);bad=rejects(lambda:m.begin_post(st));ck('no_duplicate_or_early_second_post',bad and rejects(lambda:m.begin_request(st,2,'p2')))
st['requests'][0]['status']='complete';m.begin_request(st,2,'p2');m.begin_post(st);ck('second_post_total_bound',st['prompt_posts']==2 and rejects(lambda:m.begin_post(st)) and rejects(lambda:m.begin_request(st,3,'p3')))
nt={'prompt_id':'p1','terminal':'execution_success','invalid_reasons':[],'ready':True,'error_type':None,'observer_thread_exited':True,'raw_received_bytes':100,'nodes':{'14':{'cached':False,'observed_wall_seconds':1.0,'progress_value':8,'progress_max':8}}}
h={'status':{'messages':[['execution_start',{'timestamp':1000,'prompt_id':'p1'}],['execution_success',{'timestamp':2000,'prompt_id':'p1'}]]}}
ck('uncached_eight_step_timing',m.request_metrics(nt,h,'p1')['native_prompt_wall_seconds']==1.0)
for key,value in [('cached',True),('progress_value',7),('progress_max',6)]:
 bad=copy.deepcopy(nt);bad['nodes']['14'][key]=value;ck('reject_'+key,rejects(lambda:m.request_metrics(bad,h,'p1')))
bad=copy.deepcopy(nt);bad['terminal']=None;ck('reject_missing_terminal',rejects(lambda:m.request_metrics(bad,h,'p1')))
old=ast.parse(a.baseline.read_text());new=ast.parse(a.candidate.read_text())
def function(t,n):return next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==n)
ck('original_cleanup_ast',ast.dump(function(old,'cleanup_owned_v2'),include_attributes=False)==ast.dump(function(new,'cleanup_owned_v2'),include_attributes=False))
main=function(new,'main');calls=[x for x in ast.walk(main) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='cleanup_owned_v2'];finals=[x for x in ast.walk(main) if isinstance(x,ast.Try) and any(isinstance(z,ast.Expr) and isinstance(z.value,ast.Call) and isinstance(z.value.func,ast.Name) and z.value.func.id=='cleanup_owned_v2' for z in x.finalbody)]
ck('one_finally_cleanup_call',len(calls)==len(finals)==1)
ck('no_torch_import','torch' not in sys.modules)
result={'schema':'q33-persistent-offline-result/v1','checks':checks,'completed':len(checks),'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'GPU':0,'service':0,'model':0,'scope':'Synthetic loop/CLI/graph/timing gates plus unchanged cleanup AST; not real Windows Job/network cancellation or model residency proof','quality_performance':'INCONCLUSIVE'}
with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
print(json.dumps({'completed':len(checks),'all_ok':all(x['ok'] for x in checks),'GPU':0}))
