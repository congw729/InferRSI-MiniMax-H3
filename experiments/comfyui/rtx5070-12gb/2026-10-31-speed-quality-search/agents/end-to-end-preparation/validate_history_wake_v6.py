from pathlib import Path
import ast,json,threading,time
D=Path(__file__).resolve().parent
old=ast.parse((D/'end_to_end_v5.py').read_text());new=ast.parse((D/'end_to_end_v6.py').read_text());by=lambda t:{n.name:n for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
a,b=by(old),by(new)
for name in ('sample','cleanup_owned_v2','warm_admit','begin_post','begin_request','http','guarded_hash','request_metrics'):
 assert ast.dump(a[name])==ast.dump(b[name]),name
ns={'time':time};exec(compile(ast.Module(body=[b['wait_history_once']],type_ignores=[]),'wait','exec'),ns);wait=ns['wait_history_once']
done=threading.Event();t=time.perf_counter();used,woke=wait(done,1.0,False);no_signal=time.perf_counter()-t;assert not used and not woke and no_signal>=0.95
# Signal is consumed once; the next permanently-set state still uses timed sleep.
done.set();t=time.perf_counter();used,woke=wait(done,1.0,False);early=time.perf_counter()-t;assert used and woke
before=time.perf_counter();used,again=wait(done,1.0,used);fallback=time.perf_counter()-before;assert used and not again and fallback>=0.95
class Cancelled:
 def wait(self,timeout):raise KeyboardInterrupt()
try:wait(Cancelled(),1.0,False)
except KeyboardInterrupt:pass
else:raise AssertionError('cancel swallowed')
s=(D/'end_to_end_v6.py').read_text();assert s.count('wait_history_once(observer.done,')==1 and 'current["history_early_wake_count"]=1' in s
assert 'state["history_polls"] < gates["max_history_polls"]' in s and 'sample()\n                require(proc.poll() is None' in s
compile(s,'end_to_end_v6.py','exec')
print(json.dumps({'protected_AST_functions':8,'no_signal_wait_seconds':no_signal,'first_set_event_wait_seconds':early,'consumed_event_fallback_seconds':fallback,'cancel_propagated':True,'single_wait_site':True,'model_dispatches':0}))
