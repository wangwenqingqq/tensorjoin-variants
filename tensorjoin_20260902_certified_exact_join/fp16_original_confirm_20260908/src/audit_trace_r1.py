"""Normalize runtime-selected NCU SASS without treating a mnemonic as a proof."""
import collections,csv,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
records=[]
for f in sorted((HERE/'raw').glob('ncu_half_*_source.csv')):
    label=f.name.removesuffix('_source.csv')
    with f.open() as h: rows=list(csv.reader(h))
    name=rows[0][1]; header=rows[1]; source=[x for x in rows[2:] if x and x[0].startswith('0x')]
    base=int(source[0][0],16);static=collections.Counter();dynamic=collections.Counter();norm=[]
    for x in source:
        ins=' '.join(x[1].split()); op=re.sub(r'^@!?P\d+\s+','',ins).split()[0]
        static[op]+=1
        value=x[header.index('Instructions Executed')].replace(',','')
        dynamic[op]+=int(value or 0)
        if op.split('.')[0] in ['BRA','CALL','JMP','JMX','BSSY']:
            ins=re.sub(r'0x[0-9a-f]+',lambda m:hex(int(m[0],16)-base),ins)
        norm.append(f'{int(x[0],16)-base:06x} {ins}')
    data=('\n'.join(norm)+'\n').encode();out=HERE/'artifacts'/f'{label}_selected_r1.normalized.sass'
    if out.exists():assert out.read_bytes()==data
    else:out.write_bytes(data)
    log=(HERE/'raw'/f'{label}.log').read_text().splitlines()
    raw=list(csv.DictReader([x for x in log if x.startswith('"')]))
    launch=[x for x in raw if x.get('ID')=='0']; assert len(launch)==1
    keys=['Kernel Name','Block Size','Grid Size','launch__registers_per_thread','launch__shared_mem_per_block_dynamic','launch__stack_size','sass__inst_executed_register_spilling']
    result=json.loads((HERE/'results'/f'{label}.json').read_text());guard=json.loads((HERE/'results'/f'{label}_guard.json').read_text())
    mma={k:v for k,v in static.items() if 'MMA' in k};assert set(mma)=={'HMMA.16816.F32'}
    records.append({'label':label,'name':name,'shape':result['shape'],'numeric_pass':result['pass'],'guard_pass':guard['pass'],
        'exact_dot_checks':len(result['dot_checks']),'selected_normalized_sha256':hashlib.sha256(data).hexdigest(),
        'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'launch':{k:launch[0].get(k) for k in keys},
        'static_opcodes':dict(static),'dynamic_warp_instructions':dict(dynamic),'libraries':result['libraries']})
assert len(records)==5
assert records[1]['selected_normalized_sha256']==records[2]['selected_normalized_sha256']
assert records[3]['selected_normalized_sha256']==records[4]['selected_normalized_sha256']
r={'pass':True,'scope':'runtime-selected NCU SASS and finite dot checks; conditional library error model, no speed or universal numeric proof',
   'normalization':'relative instruction PCs and relative branch/call/reconvergence targets; no register renaming or operand deletion',
   'records':records}
with (HERE/'results/selected_trace_audit_r1.json').open('x') as f:json.dump(r,f,indent=2)
for x in records:
 print(x['label'],x['guard_pass'],x['name'],x['selected_normalized_sha256'],x['launch'],{k:v for k,v in x['static_opcodes'].items() if any(w in k for w in ['HMMA','F2F','LDL','STL'])})
