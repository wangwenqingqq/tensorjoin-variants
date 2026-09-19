#!/usr/bin/env python3
"""Audit the exact owned JIT cubins produced by the G9 correctness process."""

import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'artifacts/g9_check_a3'
    a=json.loads((ROOT/'results/g9_check_a3.json').read_text())
    assert a['correctness_pass']
    records={}
    for h,k in a['compiled'].items():
        binary=p/(h+'.cubin')
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==h
        res=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(binary)],text=True)
        sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/nvdisasm',str(binary)],text=True)
        (p/(h+'.resources.txt')).write_text(res)
        (p/(h+'.sass')).write_text(sass)
        ptx=(p/(h+'.ptx')).read_text()
        assert k['n_spills']==0 and 'STACK:0' in res and 'LOCAL:0' in res
        name=k['metadata']['name']
        if name=='limb_dot':
            assert 'mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32' in ptx and 'IMMA' in sass
        if name in ('fp64_subset','tc_certificate'):
            tag=k['tags'][0].split('/')[0]
            t=next(c['threshold'] for c in a['cases'] if c['name']==tag)
            assert '0d'+struct.pack('>d',t).hex().upper() in ptx
        if name=='fp64_subset':
            for value in (1.0-2.0**-40,1.0+2.0**-40):
                assert '0d'+struct.pack('>d',value).hex().upper() in ptx
        lines=[]
        ops=[]
        for line in sass.splitlines():
            m=re.match(r'\s*/\*[0-9a-fA-F]+\*/\s+(.*)',line)
            if m:
                instruction=m.group(1).strip()
                lines.append(instruction)
                op=re.match(r'(?:@!?P\d+\s+)?([A-Z][A-Z0-9]+)',instruction)
                if op:
                    ops.append(op.group(1))
            elif re.match(r'\s*\.L_[^:]+:',line):
                lines.append(line.strip())
        normalized='\n'.join(lines)+'\n'
        (p/(h+'.normalized.sass')).write_text(normalized)
        records[h]=dict(name=name,tags=k['tags'],n_regs=k['n_regs'],n_spills=k['n_spills'],
            triton_shared_metadata=k['metadata']['shared'],resource_text=res,
            normalized_sass_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
            static_opcodes={op:ops.count(op) for op in sorted(set(ops))})
    report=dict(pass_static=True,normalization='g9-owned-nvdisasm-lines-v1: instruction text plus local labels; addresses removed; constants/operands/predication preserved',
                caveat='Static instruction sites are not dynamically executed instruction or traffic counts.',kernels=records)
    target=ROOT/'results/g9_static_audit_a0.json'
    assert not target.exists()
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('PASS',len(records),'runtime JIT kernels; native INT8 MMA; no spills/stack/local; exact FP64 constants')
    for h,k in records.items():
        if k['name']=='limb_dot':
            print('limb_dot',h,'registers',k['n_regs'],'IMMA sites',k['static_opcodes'].get('IMMA',0))


if __name__=='__main__':
    main()
