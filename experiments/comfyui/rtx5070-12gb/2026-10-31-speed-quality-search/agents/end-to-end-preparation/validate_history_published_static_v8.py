from pathlib import Path
import ast,json
D=Path(__file__).resolve().parent
by=lambda p:{n.name:n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a=by(D.parent/'timing-controller/timing_events_v1.py');b=by(D/'timing_events_history_v7.py')
old=ast.unparse(a['_run']).replace('if self.collector.terminal is not None:', 'if self.collector.should_finish():')
assert old==ast.unparse(b['_run'])
assert 'history_authoritative_observer_stop_without_marker' in (D/'end_to_end_v7.py').read_text()
compile((D/'end_to_end_v7.py').read_text(),'controller','exec');compile((D/'timing_events_history_v7.py').read_text(),'observer','exec')
print(json.dumps({'async_run_only_break_predicate_changed':True,'history_authoritative_missing_marker_fallback_present':True,'AST_compile_both':True,'candidate_changed':False,'model_dispatches':0}))
