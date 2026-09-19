"""Freeze G19 admitted/rejected evidence; preserve selected binaries remotely."""
import datetime
from common import *
assert str(P).startswith('/home/')
out=H/'artifacts/raw_evidence_manifest.json';assert not out.exists()
selected={H/b/n for b in ['build_off_a0','build_on_a0','build_off_r1','build_on_r1']
          for n in ['libRT-HiSS.so','OWL/owl/libowl.so','CMakeCache.txt','CMakeFiles/RT-HiSS.dir/flags.make']}
assert all(p.is_file() for p in selected)
rows=[]
for p in sorted(H.rglob('*')):
 if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p==out:continue
 if any(x.startswith('build_') for x in p.relative_to(H).parts[:-1]) and p not in selected:continue
 assert 'root_routes_before_g19_local' not in p.parts and p.name!='root_route_update_local.json'
 rows.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p),
   remote_only=p in selected or p.suffix in ['.nsys-rep','.ncu-rep','.sqlite','.so'] or p.name=='allocator_probe'))
write(out,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host='gpu-host-8',
 file_count=len(rows),files=rows,scope='All G19 sources, private OWL, exact IDs, raw/results, guards, decisions, host-local route backups, selected builds, actual compiled caches and profiler reports. Nonselected build intermediates excluded.',
 external_prerequisites='Frozen original G17/G18/core inputs and read-only installed CUDA/OptiX/torch/cuBLAS; see frozen manifests.'))
print('ARCHIVED',len(rows),'FILES',sum(r['remote_only'] for r in rows),'REMOTE_ONLY',sum(r['bytes'] for r in rows),'BYTES')
