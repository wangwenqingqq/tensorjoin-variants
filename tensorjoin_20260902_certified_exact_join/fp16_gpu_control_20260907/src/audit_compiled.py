"""Static directed-rounding and resource audit of the actual captured kernels."""
from pathlib import Path
import collections,hashlib,json,re
HERE=Path(__file__).resolve().parents[1]
records=[]
for f in sorted((HERE/'artifacts/compiled').glob('*.ptx')):
    ptx=f.read_text();sass=f.with_suffix('.sass.r1.txt').read_text();md=json.loads(f.with_suffix('.json').read_text())
    ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([\w.]+)',sass);counts=collections.Counter(ops)
    assert '.ftz' not in ptx and '.FTZ' not in sass
    assert not any(k.split('.')[0] in ['LDL','STL'] for k in counts)
    assert md['n_spills']==0 and md['global_scratch_size']==md['profile_scratch_size']==0
    resource=f.with_suffix('.resource.r1.txt').read_text();assert 'STACK:0' in resource and 'LOCAL:0' in resource
    if f.stem.startswith('class_'):
        assert '.f64' not in ptx and not any(k.startswith(('DADD','DMUL','DFMA')) for k in counts)
        assert counts['FADD.RP']==36 and counts['FADD.RM']==12 and counts['FMUL.RP']==28
        plain=[x.strip() for x in sass.splitlines() if re.search(r'\b(?:FADD|FMUL) ',x)]
        assert len(plain)==8
        for line in plain:
            if 'FMUL ' in line:assert ', -2 ;' in line
            else:assert re.search(r'FADD R\d+, RZ, -R\d+ ;',line)
    else:
        assert counts['DADD']>0 and counts['DMUL']>0
        assert 'cvt.rp.f32.f64' in ptx and 'cvt.rm.f32.f64' in ptx
    norm=[]
    for line in sass.splitlines():
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(.+?)\s*;',line)
        if m:norm.append(m[1]+' '+' '.join(m[2].split()))
    data=('\n'.join(norm)+'\n').encode();dest=f.with_suffix('.selected.normalized.sass')
    if dest.exists():assert dest.read_bytes()==data
    else:dest.write_bytes(data)
    records.append({'kernel':f.stem,'pass':True,'cubin_sha256':hashlib.sha256(f.with_suffix('.cubin').read_bytes()).hexdigest(),
                    'normalized_sass_sha256':hashlib.sha256(data).hexdigest(),'opcodes':dict(counts),'metadata':md,'resource':resource})
assert len(records)==9
with (HERE/'results/compiled_static_audit.json').open('x') as h:json.dump({'pass':True,'scope':'static instruction/resource gates, not universal numerical proof or throughput','records':records},h,indent=2)
print('PASS',len(records),'captured programs; no FTZ/local/spills; directed FP32 interval paths')
