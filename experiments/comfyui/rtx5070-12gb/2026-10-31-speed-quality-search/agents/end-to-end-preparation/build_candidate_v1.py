from pathlib import Path
import hashlib,json,ast,difflib,sys
Q=Path(sys.argv[1]);D=Q/'agents/end-to-end-preparation';old=(Q/'agents/persistent-preparation/persistent_stage_b_v1.py').read_text();s=old
AS='fcaa021e7aa7ecc50ecd14f51e81448a758dc6c8352b65d5ea7aa657d1fe31a8';GA='5f7d0cff0c911151fd2318e2be5e8c953503034d66259673edf851f24d2274f8';HS='8d78df645d789a448116542f818b764f28d93f3e6333378da9ae75863c3846d6'
def rep(a,b):
 global s
 assert s.count(a)==1,(a[:90],s.count(a));s=s.replace(a,b)
def block(start,end,value):
 global s
 x=s.index(start);y=s.index(end,x);s=s[:x]+value+s[y:]
rep('Q33 bounded two-request persistent Trim8 infra probe with native node timing.','Q33 one-service real bounce TE -> explicit release -> Trim8 with native timing.')
rep('EXTRA_GATES = {',f'''E2E_ASSETS_SHA = "{AS}"
STAGE_A_GRAPH_SHA = "{GA}"
E2E_HELPER_SHA = "{HS}"
_e2e_path = Path(__file__).resolve().parent / "e2e_helpers_v1.py"
if hashlib.sha256(_e2e_path.read_bytes()).hexdigest() != E2E_HELPER_SHA: raise RuntimeError("frozen lifecycle helper drift")
_e2e_spec = importlib.util.spec_from_file_location("q33_e2e_helpers_v1", _e2e_path)
e2e = importlib.util.module_from_spec(_e2e_spec); _e2e_spec.loader.exec_module(e2e)
if Path(e2e.__file__).resolve() != _e2e_path.resolve(): raise RuntimeError("actual lifecycle helper source")
STAGE_A_NODES = {{'CLIPLoader':'nodes','VAELoader':'nodes','MiniMaxH3ImageToVideo':'comfy_extras.nodes_minimax_h3','SaveConditioning':'comfy_extras.nodes_cond'}}
EXTRA_GATES = {{''')
rep('require(a.asset_manifest==ap.resolve() and a.asset_manifest_sha256==profile["assets"]["sha256"]','require(a.asset_manifest==Path(__file__).resolve().parent / "ASSETS-v1.json" and a.asset_manifest_sha256==E2E_ASSETS_SHA')
rep('len(selected) == 3','len(selected) == 4');rep('"three fixed original assets required"','"four fixed new-prompt assets required"')
s=s.replace('persistent-video-stage-b','new-prompt-trim8');rep('"agents/persistent-preparation", "Q33/Q31 roots"','"agents/end-to-end-preparation", "Q33/Q31 roots"');rep('choices=["baseline", "smart"], default="baseline"','choices=["baseline"], default="baseline"')
rep('    STAGE_NODES=NODE_FAMILIES[a.loaded_profile["family"]]\n    ASSET_EXPECTED=EXPECTED_BY_FAMILY[a.loaded_profile["family"]];ASSET_IDS=set(ASSET_EXPECTED)', '''    STAGE_NODES={**NODE_FAMILIES["trim"],**STAGE_A_NODES}
    ASSET_EXPECTED=copy.deepcopy(EXPECTED_BY_FAMILY["trim"])
    ASSET_EXPECTED["text-encoder"]={'id':'text-encoder','bytes':15687142551,'sha256':'35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6','relative_path':'comfy-trim/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'}
    ASSET_IDS=set(ASSET_EXPECTED)''')
rep('    conditioning = manifest["conditioning"]','''    conditioning = None
    stage_a_path=Path(__file__).resolve().parent/"STAGE-A-GRAPH-v1.json"
    stage_a_graph=v1.frozen_json(stage_a_path,STAGE_A_GRAPH_SHA,gates["max_schema_bytes"])
    require(set(stage_a_graph)=={"prompt"} and {x["class_type"] for x in stage_a_graph["prompt"].values()}==set(STAGE_A_NODES),"StageA fixed nodes")
    stage_a_prompt_sha=hashlib.sha256(stage_a_graph["prompt"]["104"]["inputs"]["prompt"].encode()).hexdigest()''')
rep('== set(STAGE_NODES),\n            "fixed graph node set mismatch"','== set(NODE_FAMILIES["trim"]),\n            "fixed graph node set mismatch"')
rep('"prompt_posts": 0, "http_bytes": 0, "history_polls": 0,','''"prompt_posts": 0, "http_bytes": 0, "history_polls": 0, "free_posts":0,
             "stage_a_graph_sha256":STAGE_A_GRAPH_SHA,"prompt_sha256":stage_a_prompt_sha,
             "fresh_between_stages":{"physical_bytes":12*2**30,"commit_bytes":int(16.5*2**30),"gpu_free_bytes":6*2**30},''')
rep('"finite_tensor_values_checked": False','"finite_tensor_values_checked_in_stage_a": True')
block('        require((method == "GET" and route in allowed_get)', '        connection = http_client.HTTPConnection', '''        require((method == "GET" and route in allowed_get) or
                (method == "POST" and route == "/prompt" and 1 <= state["prompt_posts"] == len(state["requests"]) <= 2 and state["requests"][-1].get("post_attempted") is True) or
                (method=="POST" and route=="/free" and state["free_posts"]==1 and state["completed_requests"]==1 and state["prompt_posts"]==1 and body==b'{"unload_models":true,"free_memory":true}'), "HTTP operation outside fixed task")
''')
rep('            require(response.getheader("Content-Type", "").split(";")[0] == "application/json",\n                    "non-JSON local response")','''            if route!="/free":
                require(response.getheader("Content-Type", "").split(";")[0] == "application/json", "non-JSON local response")''')
rep('            return response.status, json.loads(b"".join(parts), object_pairs_hook=v1.unique_json)','''            if route=="/free":
                require(response.status==200 and b"".join(parts)==b"", "native free empty-200 response")
                return response.status, {}
            return response.status, json.loads(b"".join(parts), object_pairs_hook=v1.unique_json)''')
rep('        sample(force=True, startup=True)\n        for relative, expected in STAGE_SOURCE_SHA256.items():','''        sample(force=True, startup=True)
        require(e2e.fresh_ready(state["last_sample"]), "initial external admission 12/16.5/6")
        for relative, expected in STAGE_SOURCE_SHA256.items():''')
block('        embeddings = models_root / "embeddings"','        with socket.socket', '''        embeddings = models_root / "embeddings"
        e2e.no_links(embeddings)
        require(embeddings.is_dir() and embeddings.resolve()==embeddings,"native embeddings root")
        conditioning_path=None
        sample(force=True,startup=True)

''')
rep('if name in ("UNETLoader", "VAELoader", "ConditioningLoader"):','if name in ("UNETLoader", "VAELoader", "CLIPLoader"):')
block('                field, ids = {"UNETLoader"','        event("native_schema_verification_finished")', '''                field, ids = {"UNETLoader":("unet_name",["6"]),"VAELoader":("vae_name",["11","24"]),"CLIPLoader":("clip_name",["13"])}[name]
                choices=info[name]["input"]["required"][field][0]
                selected_graph=stage_a_graph if name=="CLIPLoader" else graph
                require(type(choices) is list and all(selected_graph["prompt"][i]["inputs"][field] in choices for i in ids),"model file not enumerated")
''')
rep('            nonlocal observer','            nonlocal observer, conditioning, conditioning_path')
rep('            current_graph = request_graph(graph, request_index)','''            current_graph = copy.deepcopy(stage_a_graph) if request_index==1 else e2e.bind_stage_b(graph,conditioning_path.name)
            current["stage"]="text-encoder" if request_index==1 else "trim-video"''')
block('            state["conditioning_pre_post"] = verify_conditioning(', '            render_deadline = ', '''            require(digest(stage_a_path)==STAGE_A_GRAPH_SHA,"StageA graph drift before POST")
            if request_index==2:
                require(state.get("release",{}).get("fresh_admitted") is True,"release/admission required before DiT")
                state["conditioning_pre_post"] = verify_conditioning(conditioning_path,min(deadline,time.monotonic()+60))
                status,info=http("GET","/object_info/ConditioningLoader")
                v1.save_new(request_dir/"ConditioningLoader.schema.raw.json",info)
                require(status==200 and set(info)=={"ConditioningLoader"} and info["ConditioningLoader"]["python_module"]=="comfy_extras.nodes_cond","refreshed conditioning schema source")
                combo=info["ConditioningLoader"]["input"]["required"]["conditioning_name"]
                require(type(combo) is list and len(combo)==2 and combo[0]=="COMBO" and combo[1].get("multiselect") is False and conditioning_path.name in combo[1].get("options",[]),"new condition must be enumerated")
            require(digest(a.conditioning_manifest)==a.conditioning_manifest_sha256,"baseline graph manifest drift")
            require(digest(a.iteration_root/manifest["graph"]["relative_path"])==graph_sha,"baseline graph drift")
            sample(force=True,startup=request_index==1)
            if request_index==2:require(e2e.fresh_ready(state["last_sample"]),"fresh external admission before DiT POST")
''')
x=s.index('                    videos = record["outputs"]');y=s.index('                    current["status"] = "artifact_received_unvalidated"',x);video='\n'.join('    '+line if line else line for line in s[x:y].split('\n'))
s=s[:x]+'''                    if request_index==1:
                        event("conditioning_verification_started")
                        output=e2e.find_conditioning_artifact(dirs["output"],32*2**20,sample)
                        structure=conditioning_header(output,v1.unique_json,32*2**20)
                        value_scan=e2e.scan_conditioning(output,structure,sample,min(deadline,time.monotonic()+60))
                        output_hash,signature=guarded_hash(output,render_deadline)
                        require(signature==structure.pop("file_signature"),"conditioning changed during finite/hash scan")
                        require(output_hash==e2e.REFERENCE_SHA and output.stat().st_size==e2e.REFERENCE_BYTES,"bounce conditioning differs from frozen split-process reference")
                        conditioning={"bytes":output.stat().st_size,"sha256":output_hash,**{k:structure[k] for k in ("tensors","metadata_keys","option_keys","options_sha256")}}
                        current["artifact"]={"relative_to_run":str(output.relative_to(a.run_dir)),"bytes":output.stat().st_size,"sha256":output_hash,"structure":structure}
                        current["conditioning_value_validation"]=value_scan
                        basename="h3-e2e-bounce-"+a.run_dir.name.rsplit("-",1)[-1]+"-"+output_hash+".safetensors"
                        conditioning_path=embeddings/basename
                        require(not conditioning_path.exists() and not conditioning_path.is_symlink(),"new condition alias already exists")
                        os.link(output,conditioning_path)
                        e2e.no_links(conditioning_path)
                        require(conditioning_path.samefile(output),"new conditioning alias identity")
                        current["conditioning_alias"]={"relative_to_models":str(conditioning_path.relative_to(models_root)),"sha256":output_hash,"samefile":True,"created":True}
                        state["conditioning_input"]=verify_conditioning(conditioning_path,min(deadline,time.monotonic()+60))
                        state["conditioning_input"]["relative_to_models"]=str(conditioning_path.relative_to(models_root))
                        event("conditioning_verified_and_registered",artifact=current["artifact"],alias=current["conditioning_alias"],values=value_scan)
                    else:
'''+video+s[y:]
rep('            current["metrics"] = request_metrics(timing, record, state["prompt_id"])','''            current["metrics"] = request_metrics(timing, record, state["prompt_id"])
            if request_index==2:
                state["artifact"]=current["artifact"]
                state["node_timing"]=timing
                state["end_to_end_artifact_wall_seconds"]=time.monotonic()-started''')
rep('        two_requests(run_one)','''        run_one(1)
        event("release_started")
        release_started=time.perf_counter_ns()
        log_path=a.run_dir/"comfy.raw.log";log_offset=log_path.stat().st_size
        require(state["free_posts"]==0,"free must be attempted once")
        state["free_posts"]=1
        state["release"]={"log_offset":log_offset,"post_attempted":True,"reset_path_entered":False,"fresh_admitted":False,"te_object_absence_proven":False,"start_perf_counter_ns":release_started}
        status,_=http("POST","/free",b'{"unload_models":true,"free_memory":true}')
        state["release"]["http_status"]=status
        release_deadline=min(deadline,time.monotonic()+30)
        while True:
            sample(force=True)
            require(proc.poll() is None and time.monotonic()<release_deadline,"release/admission deadline or service exit")
            entered=e2e.reset_path_entered(log_path,log_offset)
            if entered and not state["release"]["reset_path_entered"]:
                state["release"]["reset_path_entered"]=True
                state["release"]["marker_perf_counter_ns"]=time.perf_counter_ns()
                event("reset_path_entered_not_completion_ack")
            if entered and e2e.fresh_ready(state["last_sample"]):
                state["release"]["fresh_admitted"]=True
                state["release"]["fresh_sample"]=copy.deepcopy(state["last_sample"])
                state["release"]["admitted_perf_counter_ns"]=time.perf_counter_ns()
                break
            time.sleep(gates["poll_seconds"])
        state["release"]["elapsed_seconds"]=(time.perf_counter_ns()-release_started)/1e9
        event("release_fresh_admission_finished",release=state["release"])
        run_one(2)''')
s=s.replace('two_artifacts_received_unvalidated','new_prompt_video_received_unvalidated')
rep('        if state["requests"] and "node_timing" in state:\n            state["requests"][-1]["node_timing"] = state.pop("node_timing")','''        if state["requests"] and "node_timing" in state:
            state["requests"][-1]["node_timing"] = state["node_timing"]''')
rep('    sampler = timing.get("nodes", {}).get("14", {})\n    require(sampler.get("cached") is False and sampler.get("progress_value") == 8\n            and sampler.get("progress_max") == 8, "request must execute eight uncached steps")', '''    is_text="104" in record["prompt"][2]
    sampler = timing.get("nodes", {}).get("104" if is_text else "14", {})
    require(sampler.get("cached") is False,"request node must be uncached")
    if not is_text:require(sampler.get("progress_value")==8 and sampler.get("progress_max")==8,"request must execute eight uncached steps")''')
rep('"sampler_observed_wall_seconds": wall, "gpu_compute_only_seconds": None','("text_encoder_observed_wall_seconds" if is_text else "sampler_observed_wall_seconds"): wall, "gpu_compute_only_seconds": None')
rep('        state["stage_source_sha256"] = STAGE_SOURCE_SHA256','''        state["stage_source_sha256"] = STAGE_SOURCE_SHA256
        release_sources={'server.py':'52618f91c663dc6524a028162212e0434fc9239618ddc0efbf7c605027081c8c','main.py':'baaa16d69ee78e08c490eeec20bf27c56b746c070f76cc05b4aa13cbc616bcef','execution.py':'5312ba107f69f2bc70f4009c5d52524b3865dd54fceafdc8649b22c98d6c612f','comfy/model_management.py':'63610e41cf06fccb5eeaa153c5fd3799e01a0c645facba844763781d2036989a','comfy/model_patcher.py':'4f8ccf957a7227942c0e167384ed4e128f038245ad09b9bb547102281be13612','comfy_execution/caching.py':'f71d43f6196330e73674e13795a9d392a95750ad55be2c9d96cf54355794b862'}
        for name,expected in release_sources.items():require(digest(a.comfy_root/name)==expected,"native release source drift")
        state["release_source_sha256"]=release_sources''')
ast.parse(s);compile(s,'end_to_end_v1.py','exec')
for name,value in [('end_to_end_v1.py',s),('persistent-v1-to-e2e-v1.patch',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='persistent_stage_b_v1.py',tofile='end_to_end_v1.py')))]:
 with (D/name).open('x',encoding='utf-8',newline='\n') as f:f.write(value)
print(json.dumps({'lines':len(s.splitlines()),'controller_sha256':hashlib.sha256((D/'end_to_end_v1.py').read_bytes()).hexdigest()}))
