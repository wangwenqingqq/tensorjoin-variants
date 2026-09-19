"""Independent scalar statistics, without the NumPy aggregation implementation."""
import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
s=json.loads((HERE/'analysis/summary.json').read_text())
grid={};checks=0
for process in range(3):
    rec=json.loads((HERE/'results'/f'confirm_{process}.json').read_text())
    for ci,cell in enumerate(s['cells']):
        for mi,method in enumerate(s['methods']):
            a=sorted(v['query_seconds']*1000 for v in rec['samples'] if v['cell']==cell and v['method']==method)
            assert len(a)==4
            median=(a[1]+a[2])/2
            assert abs(median-s['process_medians_ms'][process][ci][mi])<1e-9
            grid[process,cell,method]=median;checks+=1
for method in s['methods']:
    mixture=[statistics.mean(grid[p,c,method] for c in s['cells']) for p in range(3)]
    assert all(abs(a-b)<1e-9 for a,b in zip(mixture,s['mixtures'][method]['process_ms']))
    assert abs(statistics.mean(mixture)-s['mixtures'][method]['mean_ms'])<1e-9
    checks+=4
winner=min(['original_full','project64','packed16'],key=lambda m:s['mixtures'][m]['mean_ms'])
assert winner==s['strongest_fixed_control']
gate=all(s['mixtures']['speculative']['process_ms'][p]<=.95*s['mixtures'][m]['process_ms'][p]
    for p in range(3) for m in ['pipeline',winner])
assert gate==s['pass_gate']
(HERE/'analysis/statistical_audit.json').write_text(json.dumps(dict(pass_=True,
    scalar_checks=checks,gate=gate,strongest_fixed_control=winner,
    summary_sha256=hashlib.sha256((HERE/'analysis/summary.json').read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(pass_=True,scalar_checks=checks,gate=gate)))
