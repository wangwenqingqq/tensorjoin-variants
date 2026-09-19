"""Inventory immutable G17 evidence; selected builds/reports may stay remote."""
import datetime,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parents[1];P=H.parent
out=H/'artifacts/raw_evidence_manifest.json';assert not out.exists()
selected=[]
for folder in ['build_a0','build_upstream_d512_a0']:
 selected += [H/folder/n for n in ['RT-HiSS','OWL/owl/libowl.so','CMakeCache.txt','CMakeFiles/RT-HiSS.dir/flags.make']]
files=[]
for p in H.rglob('*'):
 if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
 if any(x in p.parts for x in ['build_a0','build_upstream_d512_a0']) and p not in selected:continue
 if p==out:continue
 files.append(p)
rows=[]
for p in sorted(files):
 rows.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
 remote_only=any(x in p.parts for x in ['build_a0','build_upstream_d512_a0']) or p.suffix in ['.nsys-rep','.sqlite'] or p.name=='pair_replay'))
out.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host='gpu-host-8',file_count=len(rows),files=rows,
 scope='All G17 frozen sources/inputs/raw results, guard copies, decisions, selected builds and NSYS artifacts; full build intermediates remain remote but are not asserted as evidence',
 external_prerequisite='Pinned upstream and OWL plus read-only OptiX9.1; see PROTOCOL and upstream_sources.json'),indent=2)+'\n')
print('ARCHIVED',len(rows),'files;',sum(r['remote_only'] for r in rows),'remote-only')
