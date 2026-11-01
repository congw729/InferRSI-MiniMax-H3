from pathlib import Path
import ast,asyncio,hashlib,importlib.util,json
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('history_marker_observer',D/'timing_events_history_v7.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def c():
 x=m.NodeTimingCollector({},'prompt','client');x.ingest({'type':'status','data':{'sid':'client'}},1);return x
def msg(kind,**fields):return {'type':kind,'data':{'prompt_id':'prompt',**fields}}
x=c();x.ingest(msg('executing',node=None),2);assert not x.should_finish() and x.history_published_marker_received_ns is None
x.ingest(msg('execution_success'),3);assert not x.should_finish() and x.success_received_ns==3
x.ingest({'type':'executing','data':{'prompt_id':'wrong','node':None}},4);assert not x.should_finish()
x.ingest(msg('executing'),5);assert not x.should_finish()
x.ingest(msg('executing',node='node'),6);assert not x.should_finish()
rows=x.ingest(msg('executing',node=None),7);assert x.should_finish() and x.success_received_ns==3 and x.history_published_marker_received_ns==7 and len(rows)==1
assert not x.ingest(msg('executing',node=None),8) and x.history_published_marker_received_ns==7
for kind in ('execution_error','execution_interrupted'):
 y=c();y.ingest(msg(kind),2);assert y.should_finish() and y.invalid_reasons and y.success_received_ns is None
observer=m.NativeWSObserver(1,'prompt','client',{},D,D,0)
async def cancelled():raise asyncio.CancelledError()
observer._run=cancelled;observer._thread_main();assert observer.done.is_set() and observer.error=='CancelledError' and observer.collector.invalid_reasons
old=ast.parse((D/'end_to_end_v6.py').read_text());new=ast.parse((D/'end_to_end_v7.py').read_text());by=lambda tree:{n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)};a,b=by(old),by(new)
for name in ('sample','cleanup_owned_v2','warm_admit','http','wait_history_once','begin_post','request_metrics'):assert ast.dump(a[name])==ast.dump(b[name]),name
obsold=by(ast.parse((D.parent/'timing-controller/timing_events_v1.py').read_text()));obsnew=by(ast.parse((D/'timing_events_history_v7.py').read_text()))
for name in ('_thread_main','finish','write','close'):assert ast.dump(obsold[name])==ast.dump(obsnew[name]),name
oldrun=ast.unparse(obsold['_run']).replace('if self.collector.terminal is not None:', 'if self.collector.should_finish():');assert oldrun==ast.unparse(obsnew['_run'])
assert 'history_authoritative_observer_stop_without_marker' in (D/'end_to_end_v7.py').read_text()
print(json.dumps({'case_order_wrong_prompt_missing_node_null_before_success_positive_once':True,'error_interrupted_cancel_retained':True,'protected_controller_AST_functions':7,'protected_observer_AST_functions':4,'run_loop_only_break_predicate_changed':True,'model_dispatches':0}))
