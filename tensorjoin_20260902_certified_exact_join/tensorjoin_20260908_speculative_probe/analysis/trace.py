"""Use actual CUDA kernel intervals, not event envelopes, to measure overlap."""
import hashlib
import json
import sqlite3
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
NT=440391;CHUNK=4096;NG=(NT+CHUNK-1)//CHUNK

def merged(items):
    out=[]
    for a,b in sorted(items):
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def overlap(a,b):
    a=merged(a);b=merged(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total

records={}
for method in ['pipeline','draft_serial','speculative']:
    path=HERE/'raw'/f'profile_{method}.sqlite'
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    fields={v['name'] for v in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    namefield='shortName' if 'shortName' in fields else 'demangledName'
    rows=[dict(v) for v in db.execute(f'''select k.start,k.end,k.streamId,k.gridX,s.value as name
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.{namefield}=s.id order by k.start''')]
    db.close()
    plans=[v for v in rows if 'plan_tiles' in v['name']]
    scans=[v for v in rows if 'scan_queue' in v['name']]
    if method=='pipeline':
        ver=plans;draft=[];firstscans=scans
        assert len(ver)==NG and len(scans)==NG
        current=ver
    else:
        draft=plans[:1];assert draft[0]['gridX']==NT
        ver=plans[1:];assert len(scans)==2*NG
        firstscans=scans[::2]
        if method=='draft_serial':
            assert len(ver)==1 and ver[0]['gridX']==NT
            current=ver*NG
        else:
            assert len(ver)==NG
            current=ver
    vi=[(v['start'],v['end']) for v in ver]
    si=[(v['start'],v['end']) for v in scans]
    first=[(v['start'],v['end']) for v in firstscans]
    allover=overlap(vi,si)
    sameover=sum(max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
        for a,b in zip(current,firstscans))
    done=sum(a['end']<=b['start'] for a,b in zip(current,firstscans))
    record=dict(groups=NG,kernels=len(rows),verify_kernels=len(ver),scan_kernels=len(scans),
        draft_kernels=len(draft),verify_streams=sorted({v['streamId'] for v in ver}),
        scan_streams=sorted({v['streamId'] for v in scans}),
        verify_kernel_ms=sum(b-a for a,b in vi)/1e6,
        scan_kernel_ms=sum(b-a for a,b in si)/1e6,
        draft_kernel_ms=sum(v['end']-v['start'] for v in draft)/1e6,
        verify_scan_overlap_ms=allover/1e6,
        verify_firstscan_overlap_ms=overlap(vi,first)/1e6,
        current_verification_done_before_scan=done,
        current_verification_draft_overlap_ms=sameover/1e6,
        trace_sqlite_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    records[method]=record
summary=dict(methods=records,scope='one separately profiled original-threshold query per method; profiler times excluded from retained timing',
    measurement='intersection of unions of actual kernel start/end intervals from Nsight Systems')
(HERE/'analysis/trace_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
