"""Freeze G19 execution sources, built variants and referenced evidence."""
import re
import subprocess
from common import *

files=[]
for folder in [H/'src',H/'adapter_a0',P/'src',P/'g16_gpu_preparation_20260905/src',P/'g15_strong_control_20260905/src']:
    files += [f for f in folder.iterdir() if f.is_file() and not f.is_symlink()]
files += [H/'PROTOCOL.md',H/'DESIGN.md',H/'ATTEMPTS.md',H/'artifacts/inputs.json',
          P/'g18_rthiss_conservative_repair_20260905/src/bounds.py',
          P/'g17_rthiss_pair_contract_20260905/src/validate_export.py']
for c in cases():files += [P/c['path'],P/c['g19_oracle_path']]
resource=[]
for build,diag in [('build_off_a0',0),('build_on_a0',1)]:
    b=H/build;flags=b/'CMakeFiles/RT-HiSS.dir/flags.make'
    text=flags.read_text();assert '-DDATASET_DIM=512' in text and f'-DG19_DIAGNOSTICS={diag}' in text
    assert '--ftz=false' in text and 'compute_120' in text
    for f in ['libRT-HiSS.so','OWL/owl/libowl.so','CMakeCache.txt','CMakeFiles/RT-HiSS.dir/flags.make']:
        files.append(b/f)
    binary=b/'libRT-HiSS.so'
    for kind in ['sass','ptx','resource-usage']:
        path=H/'raw'/f'{build}.{kind}';assert not path.exists()
        path.write_text(subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-'+kind,str(binary)],text=True))
    resource.append(dict(build=build,binary_sha256=sha(binary),libowl_sha256=sha(b/'OWL/owl/libowl.so')))
write(H/'artifacts/frozen_execution.json',{str(f.relative_to(P)):sha(f) for f in sorted(set(files))})
write(H/'artifacts/builds.json',resource)
print('FROZEN',len(set(files)))

