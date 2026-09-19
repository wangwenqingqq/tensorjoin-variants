"""Freeze G18 evidence inventory; large selected artifacts remain remote."""
import datetime
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parents[1]
P = H.parent
assert str(P).startswith('/home/'), 'Create the authoritative inventory on the execution host.'
out = H / 'artifacts/raw_evidence_manifest.json'
assert not out.exists(), 'Never overwrite a frozen inventory.'
selected = {H / folder / name
            for folder in ['build_a0', 'build_r1']
            for name in ['RT-HiSS', 'OWL/owl/libowl.so', 'CMakeCache.txt',
                         'CMakeFiles/RT-HiSS.dir/flags.make']}
assert all(p.is_file() for p in selected)
rows = []
for p in sorted(H.rglob('*')):
    if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p == out:
        continue
    if any(x in p.parts for x in ['build_a0', 'build_r1']) and p not in selected:
        continue
    # Host-local route backups are intentionally never copied to this host.
    assert 'root_routes_before_g18_local' not in p.parts
    assert p.name != 'root_route_update_local.json'
    rows.append(dict(path=str(p.relative_to(P)), bytes=p.stat().st_size,
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                     remote_only=p in selected or p.suffix in ['.nsys-rep', '.sqlite']
                     or p.name == 'predicate_probe'))
out.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    host='gpu-host-8', file_count=len(rows), files=rows,
    scope='All G18 sources, inputs, raw/results, copied guards, decisions, host-local route backups, selected admitted/rejected builds and NSYS reports. Full build intermediates remain remote, not asserted as evidence.',
    external_prerequisites='Pinned G17/upstream/OWL, read-only OptiX 9.1; see PROTOCOL and frozen execution records.'), indent=2) + '\n')
print('ARCHIVED', len(rows), 'files;', sum(r['remote_only'] for r in rows), 'remote-only')
