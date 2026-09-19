"""Hash intended source, input, build configuration and selected executable."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[HERE/'PROTOCOL.md',HERE/'data/manifest.json',PROJECT/'src/run_g5_guarded_process.py']
paths+=list((HERE/'src').glob('*.py'))
paths += [PROJECT/'data/g2a_cifar4096'/name for name in ['vectors_f32.npy','oracle_pairs_u64.npy','metadata.json']]
paths+=[p for p in (HERE/'adapter_a0').iterdir() if p.is_file()]
paths+=list((HERE/'data').glob('*.f32'))
paths += [HERE/'build_a0/RT-HiSS',HERE/'build_a0/OWL/owl/libowl.so',HERE/'build_a0/CMakeCache.txt',HERE/'build_a0/CMakeFiles/RT-HiSS.dir/flags.make']
for name in ['CMakeCache.txt','CMakeFiles/RT-HiSS.dir/flags.make']:
    p=HERE/'build_a0'/name;target=HERE/'artifacts'/('build_'+Path(name).name)
    assert not target.exists();shutil.copyfile(p,target)
path=HERE/'artifacts/frozen_execution.json';assert not path.exists()
path.write_text(json.dumps({str(p.relative_to(PROJECT)):sha(p) for p in paths},indent=2)+'\n')
print('FROZEN_FILES',len(paths))
