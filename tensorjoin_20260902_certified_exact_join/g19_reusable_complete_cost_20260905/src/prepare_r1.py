"""Copy OWL and fix only the observed pinned-host deallocation pair."""
import difflib
import shutil
from common import *
old=H/'adapter_a0';new=H/'adapter_r1';assert not new.exists()
shutil.copytree(old,new,symlinks=True)
(new/'OWL').unlink() # Only the newly copied symlink, never shared upstream data.
shutil.copytree(P/'adapters/rthiss_g7_a0/OWL',new/'OWL',symlinks=True,
                ignore=shutil.ignore_patterns('.git','__pycache__'))
p=new/'OWL/owl/DeviceMemory.h';text=p.read_text()
prefix,tail=text.split('  struct PinnedHostMem {',1)
assert tail.count('cudaFree(ptr)')==2
fixed=prefix+'  struct PinnedHostMem {'+tail.replace('cudaFree(ptr)','cudaFreeHost(ptr)')
p.write_text(fixed)
(H/'artifacts/pinned_host_r1.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),fixed.splitlines(True),fromfile='pinned OWL DeviceMemory.h',tofile='G19 private OWL DeviceMemory.h')))
changes=[]
for f in (new/'OWL').rglob('*'):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(new/'OWL');base=P/'adapters/rthiss_g7_a0/OWL'/rel
    a,b=sha(base),sha(f)
    if a!=b:changes.append(dict(path=str(rel),before=a,after=b))
assert len(changes)==1 and changes[0]['path']=='owl/DeviceMemory.h'
write(H/'artifacts/owl_r1_changes.json',dict(changes=changes,scope='Only two cudaFree -> cudaFreeHost substitutions in PinnedHostMem'))
for original,destination,replacements in [
 ('operators.py','operators_r1.py',[('build_on_a0','build_on_r1'),('build_off_a0','build_off_r1')]),
 ('run_slot.py','run_slot_r1.py',[('from operators import make','from operators_r1 import make'),('check_frozen();root=','check_frozen();check_r1();root='),
  ('from common import *','from common import *\nfrom common_r1 import check_r1'),('results/admission.json','results/admission_r1.json')]),
 ('supervise.py','supervise_r1.py',[('src/run_slot.py','src/run_slot_r1.py'),('check_frozen();command=','check_frozen();check_r1();command='),
  ('from common import *','from common import *\nfrom common_r1 import check_r1')])]:
    s=(H/'src'/original).read_text()
    for a,b in replacements:assert a in s;s=s.replace(a,b)
    p=H/'src'/destination;assert not p.exists();p.write_text(s)
print('R1 PREPARED; shared upstream unchanged')
