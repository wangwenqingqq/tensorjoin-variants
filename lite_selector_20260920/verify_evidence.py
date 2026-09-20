"""Verify the stopped G0 artifact on CPU without implying campaign completion."""
import hashlib
import json
from pathlib import Path

from replay_failure import main as replay

ROOT=Path(__file__).resolve().parent


def main():
    r=ROOT/'results'; manifest=json.loads((r/'EVIDENCE_MANIFEST.json').read_text())
    for name,h in manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((ROOT/'workloads.jsonl').read_bytes()).hexdigest()==manifest['workloads_sha256']
    for name,h in manifest['diagnostic_helper_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    data=[json.loads((r/(label+'.json')).read_text()) for label in ('g0_probe_v1','g0_regression_v1','g0_validate_v1')]
    assert [d['pass'] for d in data]==[True,True,False]
    assert len(data[1]['records'])==68 and len(data[2]['records'])==49
    for d in data:
        assert d['source_sha256']==data[0]['source_sha256']
        assert d['contract_sha256']==hashlib.sha256((ROOT/'CONTRACT.yaml').read_bytes()).hexdigest()
        raw=[json.loads(s) for s in (r/(d['label']+'.jsonl')).read_text().splitlines()]
        assert raw==d['records']
        for row in raw:
            assert row['correct'] and row['split']!='test' and row['role']!='sample'
        compiled=json.loads((r/(d['label']+'.compiled.json')).read_text())
        known={v['cubin_sha256'] for v in compiled.values()}
        assert all(set(row['kernels'].values())<=known for row in raw)
    for name,h in data[0]['source_sha256'].items():
        assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==h,name
    configs=[json.loads(s) for s in (ROOT/'workloads.jsonl').read_text().splitlines()]
    assert len(configs)==96 and len({c['parent_id'] for c in configs})==24
    assert {split:sum(c['split']==split for c in configs) for split in ('train','validation','test')}=={'train':48,'validation':16,'test':32}
    assert all(float(c['threshold_d2']).hex()==c['threshold_hex'] for c in configs)
    assert not list(r.glob('g1_bench*'))
    assert json.loads((ROOT/'model.json').read_text())['status']=='not_trained'
    replay()
    print(json.dumps({'artifact_consistent':True,'G0_pass':False,'G1_G2_G3':'NOT_RUN',
                      'source_files':len(data[0]['source_sha256']),'scope':'offline evidence replay, not a GPU gate'}))


if __name__=='__main__':main()
