"""CPU-only bounded whole-clip decoded-byte identity; no media writes or model imports."""
import argparse,ctypes,hashlib,itertools,json,stat,sys,time
from pathlib import Path,PureWindowsPath
from ctypes import wintypes
class Rejected(Exception): pass
def need(ok,msg):
 if not ok: raise Rejected(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(512*1024),b''): h.update(b)
 return h.hexdigest()
def read_json(path,expected):
 need(path.stat().st_size<=2097152 and sha(path)==expected,'metadata identity');return json.loads(path.read_text(encoding='utf-8'))
def sig(st): return(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
def no_links(path):
 for p in [*reversed(path.parents),path]:
  st=p.lstat();need(not stat.S_ISLNK(st.st_mode) and not(getattr(st,'st_file_attributes',0)&0x400),'link/reparse boundary')
 return path.resolve(strict=True)
def child(root,relative):
 rp=PureWindowsPath(relative);need(not rp.is_absolute() and not rp.drive and '..' not in rp.parts,'relative boundary')
 p=no_links(root.joinpath(*rp.parts));need(p.is_relative_to(root),'resolved boundary');return p
class PM(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
def rss():
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
 api=ctypes.WinDLL('psapi',use_last_error=True);api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PM),wintypes.DWORD];api.GetProcessMemoryInfo.restype=wintypes.BOOL
 v=PM();v.cb=ctypes.sizeof(v);need(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(v),v.cb),'RSS query');return int(v.WorkingSetSize)
def jhash(obj): return hashlib.sha256(json.dumps(obj,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for name in ['notes-root','lab-root','infra-root','freeze','result']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--freeze-sha256',required=True);a=p.parse_args();start=time.perf_counter();root=Path(__file__).resolve().parent
 r=no_links(a.notes_root);lab=no_links(a.lab_root);infra=no_links(a.infra_root);q=r/'iterations/2026-10-31-speed-quality-search'
 need(root==q/'agents/decoded-content-identity' and lab.drive.upper()=='E:','fixed evidence/root boundary')
 need(a.result.parent.resolve().is_relative_to(q/'results') and a.result.parent.is_dir() and not a.result.exists(),'new result boundary')
 f=read_json(a.freeze,a.freeze_sha256)
 for x in f['files']:
  v=child(root,x['path']);need(v.stat().st_size==x['bytes'] and sha(v)==x['sha256'],'frozen source drift')
 for x in f['infra_files']:
  v=child(infra,x['path']);need(v.stat().st_size==x['bytes'] and sha(v)==x['sha256'],'InferRSI source drift')
 plan=json.loads((root/'PLAN-TWO-PROMPT-PUBLISHED-v1.json').read_text());manifest=json.loads((root/'INPUTS-TWO-PROMPT-PUBLISHED-v1.json').read_text());need(1<=len(manifest['inputs'])<=10,'input count bound');expected=plan['expected']
 out={'schema':'q33-decoded-content-identity/v1','complete':False,'claim':'INCONCLUSIVE','quality_verdict':'INCONCLUSIVE','model_dispatches':0,'network_requests':0,'media_writes':0,'worker_sha256':sha(Path(__file__)),'freeze_sha256':a.freeze_sha256,'inputs_sha256':sha(root/'INPUTS-TWO-PROMPT-PUBLISHED-v1.json'),'peak_own_rss_bytes':0,'items':[]}
 def guard():
  need(time.perf_counter()-start<plan['whole_seconds'],'CPU whole deadline');v=rss();out['peak_own_rss_bytes']=max(out['peak_own_rss_bytes'],v);need(v<=plan['max_worker_rss_bytes'],'CPU RSS cap')
 try:
  import av,numpy as np
  out['versions']={'pyav':av.__version__,'numpy':np.__version__,'ffmpeg_libraries':av.library_versions};guard()
  for item in manifest['inputs']:
   guard();t=time.perf_counter();receipt=read_json(child(q,item['trial_relative_to_iteration'])/'result.json',item['result_sha256']);need(receipt['owned_root_exited'] and receipt['job_close_requested'],'generation cleanup binding')
   need(all(receipt[k]==v for k,v in item['generation_identity'].items()),'generation source binding');idx=item['request_index']
   if idx and receipt['status']=='two_new_prompt_videos_received_unvalidated':
    need(receipt['mode']=='two-new-prompts-trim8' and idx in (2,4) and receipt['prompt_posts']==receipt['completed_requests']==4 and receipt['free_posts']==3 and len(receipt['requests'])==4 and not receipt['cleanup_diagnostics'],'actual four-POST receipt')
    need(len(receipt['releases'])==3 and all(x['fresh_admitted'] and x['reset_path_entered'] and x['warm_admission']['admitted'] for x in receipt['releases']),'actual four-POST free admission')
    case,reference={2:('bounce-audio','70a19cfeb46fe9411ac6f3fbfea63c253056d28cffb9bb28f269da4c2134d170'),4:('hands-face-mug','ea6cb031df1988576ee7eb4cc9d8d4c38a393e025f92de57d4aae8227858e2dc')}[idx]
    req=receipt['requests'][idx-1];text=receipt['requests'][idx-2]
    need(req['request_index']==idx and req['case']==case and req['artifact']==item['artifact'] and req['status']=='complete','four-POST request artifact')
    need(text['case']==case and text['artifact']['sha256']==reference and text['conditioning_value_validation']['finite_values_checked'],'four-POST actual conditioning')
   elif idx:
    need(receipt['status']=='two_artifacts_received_unvalidated' and len(receipt['requests'])==2,'persistent receipt')
    req=receipt['requests'][idx-1];need(req['request_index']==idx and req['artifact']==item['artifact'],'request artifact binding')
   else:
    need(receipt['status'] in ['artifact_received_unvalidated','new_prompt_video_received_unvalidated'] and receipt['artifact']==item['artifact'],'single receipt binding')
    if receipt['status']=='new_prompt_video_received_unvalidated':
     need(receipt['mode']=='new-prompt-trim8' and receipt['prompt_posts']==receipt['completed_requests']==2 and receipt['free_posts']==1 and receipt['release']['fresh_admitted'] and not receipt['cleanup_diagnostics'] and receipt['requests'][1]['artifact']==item['artifact'],'actual E2E receipt binding')
     need(receipt['requests'][0]['artifact']['sha256']=='70a19cfeb46fe9411ac6f3fbfea63c253056d28cffb9bb28f269da4c2134d170','actual E2E bounce conditioning')
   run=child(lab,item['run_relative_to_lab']);need(run.parent.name=='runs','owned run boundary');media=child(run,item['media_relative_to_run'])
   prior=read_json(media/'validation-summary.json',item['media_summary_sha256']);decoded=read_json(media/'decode-result.json',item['decode_report_sha256']);art=item['artifact']
   need(prior['worker_returncode']==0 and prior['owned_root_exited'] and prior['expected_input_sha256']==art['sha256'] and prior['decoder_result_sha256']==item['decode_report_sha256'],'prior media summary binding')
   need(decoded['full_demux_eof'] and decoded['input_unchanged'] and decoded['input_sha256']==art['sha256'],'prior decoder binding')
   source=child(run,art['relative_to_run']);need(source.is_relative_to(run/'output'),'output boundary');before=source.stat();need(before.st_size==art['bytes'] and before.st_size<=plan['max_video_bytes'],'input size bound');need(sha(source)==art['sha256'],'full SHA before');guard()
   vh=hashlib.sha256();ah=hashlib.sha256();video_meta=[];audio_meta=[];vbytes=abytes=samples=packets=0;audio_format=None;vlast=alast=None
   with source.open('rb') as handle:
    need(sig(handle.fileno() and __import__('os').fstat(handle.fileno()))==sig(before),'open identity')
    with av.open(handle,mode='r',format='mov',options={'enable_drefs':'0','use_absolute_path':'0'}) as container:
     need(len(container.streams)==2 and len(container.streams.video)==len(container.streams.audio)==1,'one video and audio only');vs=container.streams.video[0];aus=container.streams.audio[0]
     for stream in (vs,aus):stream.codec_context.thread_count=1
     need(vs.average_rate.numerator==int(expected['fps'])*vs.average_rate.denominator,'video average rate')
     for packet in container.demux(vs,aus):
      guard();packets+=1;need(packets<=4096,'packet cap')
      for frame in packet.decode():
       guard();need(frame.pts is not None and frame.time_base is not None,'exact frame PTS');ts=frame.pts*frame.time_base
       if isinstance(frame,av.VideoFrame):
        need(len(video_meta)<expected['video_frames'] and frame.width==expected['width'] and frame.height==expected['height'],'video geometry/count');need(vlast is None or ts>vlast,'monotonic video PTS');vlast=ts
        arr=frame.to_ndarray(format='rgb24');need(arr.dtype==np.uint8 and arr.shape==(expected['height'],expected['width'],3),'RGB24 shape/dtype')
        payload=arr.tobytes(order='C');vh.update(payload);vbytes+=len(payload);video_meta.append({'pts':frame.pts,'time_base':str(frame.time_base),'seconds_exact':str(ts),'source_pixel_format':frame.format.name});del payload,arr
       elif isinstance(frame,av.AudioFrame):
        need(len(audio_meta)<2048 and 0<frame.samples<=65536,'audio frame bounds');need(alast is None or ts>alast,'monotonic audio PTS');alast=ts
        channels=[c.name for c in frame.layout.channels];need(frame.sample_rate==expected['audio_rate'] and len(channels)==expected['audio_channels'],'audio rate/channels')
        arr=frame.to_ndarray();fmt={'native_format':frame.format.name,'native_dtype':arr.dtype.str,'native_planar':frame.format.is_planar,'layout':frame.layout.name,'channel_order':channels,'rate':frame.sample_rate,'canonical_layout':'sample-major interleaved','canonical_dtype':arr.dtype.newbyteorder('<').str}
        need(audio_format is None or audio_format==fmt,'stable audio format');audio_format=fmt;need(np.isfinite(arr).all(),'finite native decoded audio')
        if frame.format.is_planar:
         need(arr.shape==(len(channels),frame.samples),'planar shape');canonical=arr.T
        else:
         need(arr.size==frame.samples*len(channels),'packed size');canonical=arr.reshape(frame.samples,len(channels))
        canonical=canonical.astype(canonical.dtype.newbyteorder('<'),copy=False);payload=canonical.tobytes(order='C');ah.update(payload);abytes+=len(payload);samples+=frame.samples;need(samples<=expected['audio_samples_per_channel'],'total audio sample cap')
        audio_meta.append({'pts':frame.pts,'time_base':str(frame.time_base),'seconds_exact':str(ts),'samples_per_channel':frame.samples});del payload,canonical,arr
       else:raise Rejected('unexpected decoded frame type')
    need(sig(__import__('os').fstat(handle.fileno()))==sig(before),'open identity after EOF')
   need(len(video_meta)==expected['video_frames'] and samples==expected['audio_samples_per_channel'],'complete video/audio extent');guard();need(sha(source)==art['sha256'] and sig(source.stat())==sig(before),'full SHA/stat unchanged after')
   result={'id':item['id'],'generation_result_sha256':item['result_sha256'],'generation_identity':item['generation_identity'],'media_summary_sha256':item['media_summary_sha256'],'prior_decode_report_sha256':item['decode_report_sha256'],'source_sha256':art['sha256'],'source_bytes':art['bytes'],'source_relative_to_run':art['relative_to_run'],'run_relative_to_lab':item['run_relative_to_lab'],'video':{'decoded_frames':len(video_meta),'shape':[expected['height'],expected['width'],3],'dtype':'uint8','layout':'RGB24 HWC C-order','bytes_hashed':vbytes,'all_frames_rgb_sha256':vh.hexdigest(),'timeline_sha256':jhash(video_meta),'frames':video_meta},'audio':{'format':audio_format,'decoded_frames':len(audio_meta),'samples_per_channel':samples,'bytes_hashed':abytes,'all_samples_pcm_sha256':ah.hexdigest(),'timeline_sha256':jhash(audio_meta),'frames':audio_meta},'full_demux_eof':True,'input_unchanged':True,'elapsed_seconds':time.perf_counter()-t}
   out['items'].append(result);print(json.dumps({'id':item['id'],'rgb_sha256':vh.hexdigest(),'pcm_sha256':ah.hexdigest()}),flush=True)
  pairs=[]
  for x,y in itertools.combinations(out['items'],2):
   fields={'container_bytes_equal':x['source_sha256']==y['source_sha256'],'all_rgb_bytes_equal':x['video']['all_frames_rgb_sha256']==y['video']['all_frames_rgb_sha256'],'video_timeline_equal':x['video']['timeline_sha256']==y['video']['timeline_sha256'],'all_pcm_bytes_equal':x['audio']['all_samples_pcm_sha256']==y['audio']['all_samples_pcm_sha256'],'audio_format_equal':x['audio']['format']==y['audio']['format'],'audio_timeline_equal':x['audio']['timeline_sha256']==y['audio']['timeline_sha256']};pairs.append({'a':x['id'],'b':y['id'],**fields})
  out['pairwise']=pairs;out['all_inputs_decoded_rgb_and_pcm_equal']=all(x['all_rgb_bytes_equal'] and x['all_pcm_bytes_equal'] and x['audio_format_equal'] for x in pairs);out['all_inputs_timelines_equal']=all(x['video_timeline_equal'] and x['audio_timeline_equal'] for x in pairs);out['complete']=True
 except BaseException as exc:out.update(error_type=type(exc).__name__,reason=str(exc) if isinstance(exc,Rejected) else 'external operation failed; details omitted')
 out['elapsed_seconds']=time.perf_counter()-start;out['torch_imported']='torch' in sys.modules
 data=json.dumps(out,indent=2)+'\n';need(len(data.encode())<=plan['max_result_bytes'],'result cap')
 with a.result.open('x',encoding='utf-8') as f:f.write(data)
 return 0 if out['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
