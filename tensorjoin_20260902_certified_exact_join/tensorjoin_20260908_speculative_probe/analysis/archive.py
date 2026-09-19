"""Archive actual imported dependencies; large immutable arrays stay referenced."""
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

freeze=json.loads((HERE/'artifacts/timing_freeze.json').read_text())
records=[]
for path,want in freeze.items():
    p=Path(path);assert sha(p)==want,path
    dest=None
    if p.is_relative_to(HERE):
        dest=str(p.relative_to(HERE))
    elif p.suffix!='.npy':
        rel=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('external')/p.relative_to('@TENSORJOIN_ROOT@')
        q=HERE/'inherited'/rel;q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():assert sha(q)==want,q
        else:shutil.copy2(p,q)
        dest=str(q.relative_to(HERE))
    records.append(dict(source=path,sha256=want,archived=dest,
        note='immutable array reference' if dest is None else ''))
(HERE/'artifacts/dependency_archive.json').write_text(json.dumps(records,indent=2))
print(json.dumps(dict(verified_dependencies=len(records),archived=sum(r['archived'] is not None for r in records))))
