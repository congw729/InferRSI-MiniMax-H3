"""Frozen CPU-only temporal contact sheet extraction; no model, audio review or networking."""
import argparse,ctypes,hashlib,json,os,stat,sys,time
from pathlib import Path,PureWindowsPath
from ctypes import wintypes
class Rejected(Exception):pass
def need(ok,msg):
 if not ok:raise Rejected(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(512*1024),b''):h.update(b)
 return h.hexdigest()
def read_json(path,expected):
 need(path.stat().st_size<=1024**2 and sha(path)==expected,'metadata identity');return json.loads(path.read_text(encoding='utf-8'))
def sig(st):return(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
def no_links(path):
 for p in [*reversed(path.parents),path]:
  st=p.lstat();need(not stat.S_ISLNK(st.st_mode) and not(getattr(st,'st_file_attributes',0)&0x400),'link/reparse boundary')
 return path.resolve(strict=True)
class PM(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
def rss():
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
 api=ctypes.WinDLL('psapi',use_last_error=True);api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PM),wintypes.DWORD];api.GetProcessMemoryInfo.restype=wintypes.BOOL
 value=PM();value.cb=ctypes.sizeof(value);need(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb),'own RSS query');return int(value.WorkingSetSize)
def main():
 p=argparse.ArgumentParser(allow_abbrev=False)
 for n in ['notes-root','lab-root','freeze','manifest','result']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--freeze-sha256',required=True);p.add_argument('--manifest-sha256',required=True);a=p.parse_args();start=time.perf_counter();root=Path(__file__).resolve().parent
 r=no_links(a.notes_root);lab=no_links(a.lab_root);q=r/'iterations/2026-10-31-speed-quality-search';need(root==q/'agents/video-contact-sheets' and lab.drive.upper()=='E:','fixed root boundary')
 need(a.result.parent.resolve().is_relative_to(q/'results') and a.result.parent.is_dir() and not a.result.exists(),'new result boundary')
 frozen=read_json(a.freeze,a.freeze_sha256)
 for x in frozen['files']:
  file=root/x['path'];need(file.resolve().is_relative_to(root) and file.stat().st_size==x['bytes'] and sha(file)==x['sha256'],'frozen source drift')
 plan=json.loads((root/'PLAN-v1.json').read_text());need(a.manifest.resolve().parent==root,'frozen local manifest boundary');need(any((root/x['path']).resolve()==a.manifest.resolve() and x['sha256']==a.manifest_sha256 for x in frozen['files']),'manifest missing from freeze');manifest=read_json(a.manifest,a.manifest_sha256)
 out={'schema':'q33-temporal-contact-sheets/v1','complete':False,'claim':'INCONCLUSIVE','quality_verdict':'INCONCLUSIVE','full_motion_playback_reviewed':False,'audio_listened':False,'model_dispatches':0,'network_requests':0,'worker_sha256':sha(Path(__file__)),'freeze_sha256':a.freeze_sha256,'items':[],'peak_own_rss_bytes':0}
 try:
  import av
  from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
  out['versions']={'pyav':av.__version__,'pillow':pillow_version};font=ImageFont.load_default(size=18)
  for item in manifest['inputs']:
   t=time.perf_counter();deadline=min(start+plan['whole_seconds'],t+plan['per_video_seconds'])
   def guard():
    need(time.perf_counter()<deadline,'CPU deadline');value=rss();out['peak_own_rss_bytes']=max(value,out['peak_own_rss_bytes']);need(value<=plan['max_worker_rss_bytes'],'CPU RSS cap')
   receipt_path=q/item['trial_relative_to_iteration']/'result.json';receipt=read_json(receipt_path,item['result_sha256']);need(receipt['status']=='artifact_received_unvalidated' and receipt['owned_root_exited'] and receipt['artifact']==item['artifact'],'actual source receipt')
   rel=Path(item['run_relative_to_lab']);need(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in ['comfy-original','comfy-trim'] and rel.parts[1]=='runs','owned run boundary');run=no_links(lab/rel)
   media=run/'media-inspection-v1';prior=read_json(media/'validation-summary.json',item['prior_media_summary_sha256']);need(prior['worker_returncode']==0 and prior['owned_root_exited'],'prior media inspection')
   art=item['artifact'];rp=PureWindowsPath(art['relative_to_run']);need(not rp.is_absolute() and rp.parts[0]=='output' and '..' not in rp.parts,'artifact boundary');source=no_links(run.joinpath(*rp.parts));before=source.stat()
   need(source.is_relative_to(run/'output') and before.st_size==art['bytes'] and before.st_size<=plan['max_video_bytes'],'source size/boundary');guard();need(sha(source)==art['sha256'],'full SHA before')
   dest=media/'contact-sheet-12-v1';need(not dest.exists(),'new contact-sheet directory required');dest.mkdir();indices=plan['selected_indices'];picked={};count=0;last=-1.0
   with av.open(str(source),mode='r') as container:
    need(len(container.streams.video)==1,'one video stream');stream=container.streams.video[0];stream.thread_count=1;need(float(stream.average_rate)==24,'native24fps')
    for frame in container.decode(stream):
     guard();need(count<73 and frame.width==384 and frame.height==256 and frame.pts is not None,'frame bound/geometry/PTS');seconds=float(frame.pts*frame.time_base);need(seconds>last,'monotonic PTS');last=seconds
     if count in indices:picked[count]=(frame.to_image().convert('RGB'),seconds)
     count+=1
   need(count==73 and list(picked)==indices,'complete73frames/selection');guard()
   grid=Image.new('RGB',(4*384,3*(256+plan['label_height'])),(18,18,18));draw=ImageDraw.Draw(grid)
   for n,index in enumerate(indices):
    image,seconds=picked[index];x=(n%4)*384;y=(n//4)*(256+plan['label_height']);grid.paste(image,(x,y));draw.text((x+8,y+260),f'frame {index:02d} | {seconds:.3f} s',font=font,fill=(245,245,245))
   png=dest/'contact-sheet-12.png'
   with png.open('xb') as f:grid.save(f,format='PNG',compress_level=6)
   need(png.stat().st_size<=plan['max_png_bytes'],'PNG size cap');guard();need(sha(source)==art['sha256'] and sig(source.stat())==sig(before),'source unchanged after full SHA')
   result={'profile':item['profile'],'run_relative_to_lab':item['run_relative_to_lab'],'source_relative_to_run':art['relative_to_run'],'source_bytes':art['bytes'],'source_sha256':art['sha256'],'full_decoded_video_frames':count,'video_decode_reached_eof':True,'source_unchanged':True,'selected_frames':[{'frame':i,'pts_seconds':picked[i][1]} for i in indices],'grid_relative_to_run':png.relative_to(run).as_posix(),'grid_dimensions':list(grid.size),'grid_bytes':png.stat().st_size,'grid_sha256':sha(png),'elapsed_seconds':time.perf_counter()-t,'claim':'INCONCLUSIVE','audio_listened':False}
   with (dest/'contact-sheet-result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
   out['items'].append(result)
   for im,_ in picked.values():im.close()
   grid.close();print(json.dumps({'profile':item['profile'],'grid_sha256':result['grid_sha256'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
  out['complete']=True
 except BaseException as exc:out.update(error_type=type(exc).__name__,reason=str(exc) if isinstance(exc,Rejected) else 'external operation failed; details omitted')
 out['elapsed_seconds']=time.perf_counter()-start;out['torch_imported']='torch' in sys.modules
 with a.result.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2);f.write('\n')
 return 0 if out['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
