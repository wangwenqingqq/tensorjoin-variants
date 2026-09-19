"""Validate all saved outcomes and trace arithmetic without re-running the GPU."""
import hashlib
import json
import sqlite3
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
refs=json.loads((HERE/'inherited/precision_routing_20260908_threshold_sweep/artifacts/reference_manifest.json').read_text())['cells']
calls=0;source_records=0
for p in sorted((HERE/'results').glob('*.json')):
    r=json.loads(p.read_text())
    if 'phase' not in r:continue
    assert r['pass_'],p
    for v in sum((r.get(k,[]) for k in ['warmups','samples','diagnostics']),[]):
        ref=refs[v['cell']]
        assert v['output_count']==ref['count'] and v['output_sha256']==ref['sha256'],(p,v['cell'],v['method'])
        if v.get('canaries_pass') is not None:assert v['canaries_pass']
        calls+=1
    source_records+=1
summary=json.loads((HERE/'analysis/summary.json').read_text())
assert calls==summary['full_calls']
trace=json.loads((HERE/'analysis/trace_summary.json').read_text())
checks=[]
for method,r in trace['methods'].items():
    db=sqlite3.connect(HERE/'raw'/f'profile_{method}.sqlite')
    rows=db.execute('select k.start,k.end,k.gridX,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start').fetchall()
    db.close();plans=[v for v in rows if 'plan_tiles' in v[3]]
    ver=plans if method=='pipeline' else plans[1:]
    scans=[v for v in rows if 'scan_queue' in v[3]]
    events=[]
    for typ,vals in enumerate([ver,scans]):
        for a,b,_,_ in vals:events.extend([(a,typ,1),(b,typ,-1)])
    events.sort();active=[0,0];total=0;previous=events[0][0]
    for stamp,typ,delta in events:
        if active[0]>0 and active[1]>0:total+=stamp-previous
        active[typ]+=delta;previous=stamp
    assert active==[0,0]
    assert abs(total/1e6-r['verify_scan_overlap_ms'])<1e-12
    checks.append(dict(method=method,overlap_ns=total,pass_=True))
record=dict(pass_=True,full_calls=calls,result_records=source_records,
    trace_checks=checks,summary_sha256=hashlib.sha256((HERE/'analysis/summary.json').read_bytes()).hexdigest())
(HERE/'analysis/final_checks.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
