"""Archive G15/G16 receipts, frozen code and failures; hash large profiler reports."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parent
OLD=PROJECT/'g15_strong_control_20260905'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    for directory in (OLD,HERE):
        assert json.loads((directory/'results/public_screen.json').read_text())['complete']
    frozen=json.loads((HERE/'artifacts/frozen_screen_sources.json').read_text())
    paths=set();large={};guards=[]
    for rel,h in frozen.items():
        source=PROJECT/rel;assert sha(source)==h,rel
        target=HERE/'artifacts/source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source,target);assert sha(target)==h
        paths.add(source)
    for g in sorted((PROJECT/'results').glob('g5_guard_g1[56]*.json')):
        r=json.loads(g.read_text());paths.add(g);guards.append(dict(path=str(g.relative_to(PROJECT)),admitted=r['admitted'],sha256=sha(g)))
        assert r['gpu_uuid']=='GPU-16f27f5a-dfcd-48e0-bb39-bebbe4009245'
        for key in ('raw_log','preflight_log','occupancy_log'):
            if r.get(key):
                p=PROJECT/r[key];assert sha(p)==r[key+'_sha256'],p;paths.add(p)
        if r.get('triton_cache'):
            paths.update(x for x in (PROJECT/r['triton_cache']).rglob('*') if x.is_file())
    for directory in (OLD,HERE):
        paths.update(x for x in directory.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    # Historical draft only: preserve identity, do not pretend it was updated.
    paths.add(PROJECT/'overleaf_bundle/main.tex')
    for p in sorted(paths.copy()):
        if p.suffix in ('.nsys-rep','.ncu-rep','.sqlite') or p.stat().st_size>32<<20:
            paths.remove(p);large[str(p.relative_to(PROJECT))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    manifest=dict(guards=guards,remote_only_large_artifacts=large,
       files={str(p.relative_to(PROJECT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)},
       scope='private source/evidence archive; includes vendor-selected SASS for audit, not public redistribution')
    target=HERE/'artifacts/raw_evidence_manifest.json'
    with target.open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
    paths.add(target)
    with (HERE/'artifacts/transfer_files.txt').open('x') as f:
        for p in sorted(paths):f.write(str(p.relative_to(PROJECT))+'\n')
    print(json.dumps(dict(files_to_sync=len(paths),guards=len(guards),failed_guards=sum(not x['admitted'] for x in guards),
        transfer_bytes=sum(p.stat().st_size for p in paths),remote_only=len(large))))
if __name__=='__main__':main()
