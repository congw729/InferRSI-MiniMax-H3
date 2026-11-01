from pathlib import Path
import argparse,copy,hashlib,importlib.util,json
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('persistent_media_offline',a.candidate);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r={'status':'two_artifacts_received_unvalidated','mode':'persistent-video-stage-b','profile_id':'trim-8','driver_sha256':m.CONTROLLER_SHA,'gates_sha256':m.CONTROLLER_GATES_SHA,'source_manifest_sha256':m.SOURCES_SHA,'memory_policy':'baseline','cache_policy':'none','disable_smart_memory_requested':True,'owned_root_exited':True,'job_close_requested':True,'cleanup_diagnostics':[],'completed_requests':2,'prompt_posts':2,'requests':[{'request_index':i,'prompt_id':'p'+str(i),'status':'complete','submission_outcome':'accepted','post_attempted':True,'artifact':{'relative_to_run':'output/r'+str(i)+'.mp4','sha256':'a'*64,'bytes':100}} for i in (1,2)]}
checks=[]
def ck(n,v):checks.append({'name':n,'ok':bool(v)});assert v,n
def rejects(fn):
 try:fn()
 except RuntimeError:return True
 return False
m.validate_session(r);ck('complete_two_requests_accepted',True)
for name,key,value in [('cleanup','owned_root_exited',False),('job','job_close_requested',False),('partial','completed_requests',1),('wrong_driver','driver_sha256','0'*64),('policy','disable_smart_memory_requested',False)]:
 x=copy.deepcopy(r);x[key]=value;ck('reject_'+name,rejects(lambda:m.validate_session(x)))
x=copy.deepcopy(r);x['memory_policy']='smart';x['disable_smart_memory_requested']=False;m.validate_session(x);ck('smart_source_record_accepted',True)
x=copy.deepcopy(r);x['requests'][1]['prompt_id']='p1';ck('reject_duplicate_prompt',rejects(lambda:m.validate_session(x)))
x=copy.deepcopy(r);x['requests'][1]['artifact']['relative_to_run']='output/r1.mp4';ck('reject_same_output',rejects(lambda:m.validate_session(x)))
ck('relative_artifact_only',m.safe_relative('output/request-1.mp4')==Path('output/request-1.mp4') and rejects(lambda:m.safe_relative('../outside.mp4')) and rejects(lambda:m.safe_relative('C:/outside.mp4')))
with a.output.open('x',encoding='utf8') as f:json.dump({'schema':'q33-persistent-media-offline/v1','checks':checks,'completed':len(checks),'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'model_GPU_media_dispatches':0,'scope':'Synthetic aggregate identity/cleanup and path gate only; no forged real result, no decode run'},f,indent=2)
print(json.dumps({'completed':len(checks),'dispatches':0}))
