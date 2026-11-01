"""Frozen, narrow new-prompt lifecycle helpers; stdlib only, no model imports."""
import copy,hashlib,os,stat,struct,time
from pathlib import Path
FRESH={'available_physical_bytes':12*2**30,'available_commit_bytes':int(16.5*2**30)}
REFERENCE_SHA='70a19cfeb46fe9411ac6f3fbfea63c253056d28cffb9bb28f269da4c2134d170'
REFERENCE_BYTES=840216
RESET_MARKER=b'Disabling intermediate node cache.'
def require(ok,reason):
 if not ok:raise RuntimeError(reason)
def identity(st):return st.st_size,st.st_mtime_ns,st.st_dev,st.st_ino
def no_links(path):
 for part in [*reversed(path.parents),path]:
  s=part.lstat();require(not stat.S_ISLNK(s.st_mode) and not (getattr(s,'st_file_attributes',0)&0x400),'conditioning link/reparse boundary')
 return path.resolve(strict=True)
def fresh_ready(row):
 return all(row[k]>=v for k,v in FRESH.items()) and row['gpu_board']['free_bytes']>=6*2**30
def reset_path_entered(path,offset):
 require(type(offset) is int and offset>=0,'reset offset')
 size=path.stat().st_size;require(offset<=size<=4*2**20 and size-offset<=2**20,'release log truncation/size bound')
 with path.open('rb') as f:f.seek(offset);raw=f.read(2**20+1)
 require(len(raw)<=2**20,'release log read bound')
 return RESET_MARKER in raw

def scan_conditioning(path,structure,sample,deadline):
 no_links(path);before=path.stat();require(8<before.st_size<=32*2**20,'conditioning file bound')
 require(set(structure['tensors'])=={'conditioning','minimax_token_tags'} and structure['option_keys']==[] and structure['options_sha256']==hashlib.sha256(b'{}').hexdigest(),'pure text exact tensors/options')
 cond=structure['tensors']['conditioning'];tags=structure['tensors']['minimax_token_tags']
 require(cond['dtype']=='F32' and cond['shape']==[1,41,5120] and tags['dtype']=='I64' and tags['shape']==[41],'fixed bounce tensor dtype/shape')
 total=nonzero=0
 with path.open('rb') as f:
  opened=os.fstat(f.fileno());require(identity(opened)==identity(before),'conditioning open identity')
  for name,t in structure['tensors'].items():
   begin,end=t['data_offsets'];f.seek(8+structure['header_bytes']+begin);remaining=end-begin;fmt='<q' if name=='minimax_token_tags' else '<I';width=8 if name=='minimax_token_tags' else 4;count=0
   while remaining:
    sample();require(time.monotonic()<deadline,'conditioning finite deadline');n=min(512*1024,remaining);raw=f.read(n);require(len(raw)==n and len(raw)%width==0,'conditioning short chunk')
    for (bits,) in struct.iter_unpack(fmt,raw):
     if name=='minimax_token_tags':require(bits==1,'token tags must be all one')
     else:require(bits&0x7f800000!=0x7f800000,'nonfinite conditioning');nonzero+=int(bool(bits&0x7fffffff))
     count+=1
    remaining-=n
   require(count==(41 if name=='minimax_token_tags' else 41*5120),'element count');total+=count
  require(identity(os.fstat(f.fileno()))==identity(opened),'conditioning handle changed')
 require(nonzero>0 and identity(path.stat())==identity(before),'conditioning allzero/path changed')
 return {'finite_values_checked':True,'all_token_tags_one':True,'elements_checked':total,'nonzero_conditioning_values':nonzero,'roundtrip_equivalence_checked':False}

def bind_stage_b(graph,basename):
 require(type(basename) is str and basename.startswith('h3-e2e-bounce-') and basename.endswith('.safetensors') and Path(basename).name==basename,'content-address alias basename')
 require(graph['prompt']['9']['inputs']['steps']==8 and graph['prompt']['15']['inputs']['noise_seed']==424242,'fixed Trim8 step/seed')
 result=copy.deepcopy(graph);result['prompt']['1001']['inputs']['conditioning_name']=basename;result['prompt']['92']['inputs']['filename_prefix']='q33-e2e/bounce-video'
 return result

def find_conditioning_artifact(root, max_bytes, sample):
    """Only this new output tree; reject links/reparse points and excessive entries."""
    root = root.resolve()
    files = []
    count = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        sample()
        parent = Path(directory)
        require(parent.resolve().is_relative_to(root) and len(parent.relative_to(root).parts) <= 8,
                "conditioning output traversal boundary")
        for name in dirs + names:
            count += 1
            require(count <= 128, "conditioning output entry bound")
            path = parent / name
            st = path.lstat()
            require(not path.is_symlink() and not (getattr(st, "st_file_attributes", 0) & 0x400)
                    and path.resolve().is_relative_to(root), "conditioning output link/reparse boundary")
            if name in names:
                require(path.is_file() and path.suffix.lower() == ".safetensors",
                        "unexpected conditioning output file")
                require(8 < st.st_size <= max_bytes, "conditioning output size guard")
                files.append(path)
                require(len(files) <= 1, "multiple conditioning artifacts")
    require(len(files) == 1, "conditioning artifact missing")
    return files[0]



def warm_admission(row,after_memory,gates):
    """Budget only remaining private growth; use conservative pre/post available memory."""
    owned=row['owned_private_sum_bytes'];cap=gates['max_owned_private_bytes'];floor=gates['running_commit_bytes']
    require(type(owned) is int and type(cap) is int and type(floor) is int and 0<=owned<=cap and floor>0,'warm owned/cap domain')
    names=('available_physical_bytes','available_commit_bytes')
    for values in (row,after_memory):
        require(all(type(values[k]) is int and values[k]>=0 for k in names),'warm available memory domain')
    require(after_memory['available_physical_bytes']>=gates['running_physical_bytes'] and after_memory['available_commit_bytes']>=floor,'post-read running system floors')
    effective={k:min(row[k],after_memory[k]) for k in names}
    gpu=row['gpu_board']['free_bytes'];require(type(gpu) is int and gpu>=0,'warm GPU value')
    remaining=cap-owned;required=remaining+floor
    return {'admitted':effective['available_physical_bytes']>=12*2**30 and effective['available_commit_bytes']>=required and gpu>=6*2**30,
            'current_owned_private_bytes':owned,'same_owned_private_cap_bytes':cap,'remaining_private_growth_bytes':remaining,'same_system_commit_floor_bytes':floor,'required_available_commit_bytes':required,
            'effective_available_physical_bytes':effective['available_physical_bytes'],'effective_available_commit_bytes':effective['available_commit_bytes'],
            'post_read_available_physical_bytes':after_memory['available_physical_bytes'],'post_read_available_commit_bytes':after_memory['available_commit_bytes'],
            'gpu_board_free_bytes':gpu,'TE_object_absence_proven':False}
