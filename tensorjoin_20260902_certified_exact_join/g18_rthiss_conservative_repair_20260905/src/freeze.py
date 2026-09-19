"""Freeze builds, source, prerequisites, input and independent validators."""
import hashlib,json,subprocess,re
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent;G=P/'g17_rthiss_pair_contract_20260905'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
flags=(H/'build_r1/CMakeFiles/RT-HiSS.dir/flags.make').read_text();assert '-DDATASET_DIM=512 ' in flags and '--ftz=false' in flags
res=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(H/'build_r1/RT-HiSS')],text=True)
(H/'raw/resources.txt').write_text(res)
block=next(b for b in res.split('Function ') if 'identifyNeighborsGridPrimitiveSharedQueryShared' in b)
reg=int(re.search(r'REG:(\d+)',block)[1]);stack=int(re.search(r'STACK:(\d+)',block)[1]);local=int(re.search(r'LOCAL:(\d+)',block)[1]);assert reg<=64 and stack==local==0,block
(H/'results/static_admission.json').write_text(json.dumps(dict(registers=reg,stack=stack,local=local,launch_1024_admitted=True,not_performance=True),indent=2)+'\n')
files=[H/'PROTOCOL.md',H/'NUMERICAL_DESIGN.md',P/'src/run_g5_guarded_process.py',G/'src/validate_export.py',G/'data/manifest_r1.json']
files+=list((H/'src').glob('*.py'))+list((H/'src').glob('*.cu'))+list((H/'src').glob('*.cuh'))
files += [p for p in (H/'adapter_a0').iterdir() if p.is_file()]+list((H/'data').iterdir())
files += [G/x for x in ['data/'+p.name for p in (G/'data').glob('*.f32')]]
files += [P/'data/g2a_cifar4096'/n for n in ['vectors_f32.npy','oracle_pairs_u64.npy','metadata.json']]
files += [H/n for n in ['artifacts/predicate_probe','build_r1/RT-HiSS','build_r1/OWL/owl/libowl.so','build_r1/CMakeCache.txt','build_r1/CMakeFiles/RT-HiSS.dir/flags.make']]
# The exact native maps/results used for comparison are also immutable inputs.
for p in (G/'results').glob('case_native_*.json'):files.append(p)
files.append(G/'results/case_none_boundary_zero32_a0.json')
for p in (G/'raw').glob('*/*_map.u32'):files.append(p)
out=H/'artifacts/frozen_execution.json';assert not out.exists();out.write_text(json.dumps({str(p.relative_to(P)):sha(p) for p in files},indent=2)+'\n');print('FROZEN',len(files),reg,stack,local)
