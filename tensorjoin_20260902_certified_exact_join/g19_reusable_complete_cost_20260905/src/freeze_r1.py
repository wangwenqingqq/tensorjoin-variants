"""Freeze the private OWL release fix and R1 execution layer additively."""
import subprocess
from common import *
check_frozen()
files=[H/'ADDENDUM_PINNED_LIFETIME_R1.md',H/'artifacts/allocator_probe',
       H/'artifacts/owl_r1_changes.json',H/'artifacts/pinned_host_r1.diff']
files += [H/'src'/name for name in ['allocator_probe.cu','probe_allocator.py','prepare_r1.py',
          'operators_r1.py','run_slot_r1.py','supervise_r1.py','common_r1.py','freeze_r1.py','admit_r1.py']]
files += [p for p in (H/'adapter_r1').rglob('*') if p.is_file() and not p.is_symlink()]
changes=[]
for p in (H/'adapter_r1/OWL').rglob('*'):
    if not p.is_file() or p.is_symlink():continue
    rel=p.relative_to(H/'adapter_r1/OWL');old=P/'adapters/rthiss_g7_a0/OWL'/rel
    if sha(old)!=sha(p):changes.append(str(rel))
assert changes==['owl/DeviceMemory.h'],changes
for build,diag in [('build_off_r1',0),('build_on_r1',1)]:
    b=H/build;f=b/'CMakeFiles/RT-HiSS.dir/flags.make'
    text=f.read_text();assert '-DDATASET_DIM=512' in text and f'-DG19_DIAGNOSTICS={diag}' in text and '--ftz=false' in text
    files += [b/n for n in ['libRT-HiSS.so','OWL/owl/libowl.so','CMakeCache.txt','CMakeFiles/RT-HiSS.dir/flags.make']]
    for kind in ['sass','ptx','resource-usage']:
        p=H/'raw'/f'{build}.{kind}';assert not p.exists()
        p.write_text(subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-'+kind,str(b/'libRT-HiSS.so')],text=True))
write(H/'artifacts/frozen_r1.json',{str(p.relative_to(P)):sha(p) for p in sorted(set(files))})
print('FROZEN_R1',len(set(files)))
