from pathlib import Path
import ast,hashlib,json,tempfile
D=Path(__file__).resolve().parent
old=(D/'end_to_end_v4.py').read_text();new=(D/'end_to_end_v5.py').read_text()
a='"--disable-pinned-memory", "--disable-async-offload", "--disable-smart-memory"'
b='"--disable-pinned-memory", "--disable-smart-memory"'
c='    return {"startup_log_bytes": len(raw),'
d='    require(b"Using async weight offloading with 2 streams" in raw, "native two-stream async marker missing")\n'+c
assert old.count(a)==1 and old.count(c)==1
assert new==old.replace(a,b).replace(c,d)
assert new.replace(b,a).replace(d,c)==old
compile(new,'end_to_end_v5.py','exec')
t=ast.parse(new);ns={'hashlib':hashlib};exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('require','verify_dynamic_vram_startup')],type_ignores=[]),'startup','exec'),ns)
with tempfile.TemporaryDirectory() as p:
 path=Path(p)/'log';path.write_bytes(b'DynamicVRAM support detected and enabled\nUsing async weight offloading with 2 streams\n');ns['verify_dynamic_vram_startup'](path)
 path.write_bytes(b'DynamicVRAM support detected and enabled\n')
 try:ns['verify_dynamic_vram_startup'](path)
 except RuntimeError:pass
 else:raise AssertionError('missing async marker accepted')
print(json.dumps({'exact_two_edit_reverse':True,'only_one_runtime_flag_removed':True,'AST_compile':True,'startup_positive_and_missing_marker_rejection':True,'model_dispatches':0}))
