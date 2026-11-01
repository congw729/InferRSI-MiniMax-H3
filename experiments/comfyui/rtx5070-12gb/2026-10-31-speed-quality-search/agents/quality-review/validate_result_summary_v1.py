import importlib.util,json,hashlib,re
from pathlib import Path
D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('q33_summary',D/'summarize_q33_results_v1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=m.extrema([{'owned_private_sum_bytes':100,'gpu_board':{'free_bytes':1000},'gpu':{'free_bytes':1}}],{'last_sample':{'owned_private_sum_bytes':200,'owned_rss_sum_bytes':150,'gpu_board':{'free_bytes':900}}})
assert x['owned_private_peak_bytes']==200 and x['owned_rss_peak_bytes']==150 and x['gpu_board_free_min_bytes']==900
assert m.native_seconds(None) is None and m.native_seconds({'status':{'messages':[['execution_start',{'timestamp':1000}],['execution_success',{'timestamp':2250}]]}})==1.25
notes=D.parents[3];runs=Path('E:/summary-synthetic-uncreated');reader=m.Reader(notes,runs)
req={'request_index':1,'seed':424242,'node_timing':{},'after_request_sample':{'request_index':1,'owned_private_sum_bytes':160,'gpu_board':{'free_bytes':950}}}
r=m.request_row(reader,runs/'fake-run',{'last_sample':{'request_index':2,'owned_private_sum_bytes':999,'gpu_board':{'free_bytes':1}}},req,[{'request_index':1,'owned_private_sum_bytes':100,'gpu_board':{'free_bytes':1000}},{'request_index':2,'owned_private_sum_bytes':999,'gpu_board':{'free_bytes':1}}])
assert r['resources']['owned_private_peak_bytes']==160 and r['resources']['gpu_board_free_min_bytes']==950 and r['native_prompt_wall_seconds'] is None and r['media']['status']=='not_available'
assert not re.search(r'[A-Za-z]:[\\/]',m.safe_text('C:\\private\\file.log'))
o={'schema':'q33-summary-targeted-check/v1','checks_completed':4,'scope':'resource last_sample/gpu_board; null native timing; persistent request filtering/missing media; Windows path redaction','script_sha256':hashlib.sha256((D/'summarize_q33_results_v1.py').read_bytes()).hexdigest(),'gpu_decoder_network':False}
with (D/'RESULT-SUMMARY-CHECK-v1.json').open('x',encoding='utf-8') as f:json.dump(o,f,indent=2)
print(json.dumps(o))
