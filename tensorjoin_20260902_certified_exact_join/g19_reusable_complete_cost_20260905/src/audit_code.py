"""Close actual-launch/code identity gates, never use profiler durations as speed."""
import collections
import csv
import re
import sqlite3
import struct
import subprocess
from common import *
from common_r1 import check_r1
from common_r2 import check_r2
from common_r3 import check_r3

def functions(text):
    result={}
    for block in re.split(r'\s*Function : ',text)[1:]:
        name=block.splitlines()[0].strip();inst=[]
        for line in block.splitlines()[1:]:
            m=re.search(r'/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;',line)
            if m:inst.append(' '.join(m.group(1).split())+';')
        if inst:result[name]=inst
    return result
def op(s):return re.sub(r'^@!?\w+\s+','',s).split()[0].rstrip(';')
def stats(inst):
    return dict(instructions=len(inst),normalized_sass_sha256=hashlib.sha256(('\n'.join(inst)+'\n').encode()).hexdigest(),opcodes=dict(collections.Counter(map(op,inst))))

check_frozen();check_r1();check_r2();check_r3()
assert json.loads((H/'results/admission_r3.json').read_text())['passed']
assert json.loads((H/'results/runtime_collection.json').read_text())['passed']
all_launches={};selected_launches={};ranges={}
for method in ['rt','tc','fp32']:
    rid=f'{method}_profile_nsys_r3';dbpath=H/'artifacts'/f'{rid}.sqlite'
    assert not dbpath.exists()
    subprocess.run(['nsys','export','--type','sqlite','--output',str(dbpath),str(H/'artifacts'/f'{rid}.nsys-rep')],check=True)
    db=sqlite3.connect(dbpath);db.row_factory=sqlite3.Row
    nv=list(db.execute('select * from NVTX_EVENTS'))
    strings=dict(db.execute('select id,value from StringIds'))
    def label(r):
        return (r['text'] if 'text' in r.keys() else None) or strings.get(r['textId'] if 'textId' in r.keys() else None,'')
    nv=[dict(start=r['start'],end=r['end'],text=label(r)) for r in nv if label(r).startswith(f'G19_{method}_cifar4096_')]
    assert len(nv)==2 and all(r['end']>r['start'] for r in nv),nv
    launches=[dict(r) for r in db.execute('''select s.value as name,k.start,k.end,k.registersPerThread,
       k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.staticSharedMemory,k.dynamicSharedMemory,
       k.localMemoryPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName''')]
    selected=[r for r in launches if any(v['start']<=r['start'] and r['end']<=v['end'] for v in nv)]
    assert selected
    all_launches[method]=launches;selected_launches[method]=selected;ranges[method]=nv

rt=[]
old=json.loads((P/'g18_rthiss_conservative_repair_20260905/results/code_audit.json').read_text())
old_compression=next(v for v in old['selected'] if v['target']=='compressResultMask')['normalized_sass_sha256']
for build in ['build_off_a0','build_on_a0','build_off_r1','build_on_r1']:
    funcs=functions((H/'raw'/f'{build}.sass').read_text())
    for target in ['identifyNeighborsGridPrimitiveSharedQueryShared','compressResultMask']:
        matched=[(name,code) for name,code in funcs.items() if target in name];assert len(matched)==1
        name,code=matched[0];row=dict(build=build,target=target,symbol=name,**stats(code))
        ops=row['opcodes'];assert not any(re.match(r'^(?:LDL|STL|[A-Z]*MMA)',o) for o in ops)
        if target=='compressResultMask':assert row['normalized_sass_sha256']==old_compression
        else:
            assert ops.get('FFMA',0)>0 and ops.get('DADD',0)>0 and ops.get('DMUL',0)>0
            assert not any('.FTZ' in o and o.startswith(('FADD','FFMA','DADD','DMUL')) for o in ops)
        if build=='build_off_r1':
            row['runtime']=[v for v in selected_launches['rt'] if target in v['name']]
            assert len(row['runtime'])==2
            if target.startswith('identify'):
                assert all(r['registersPerThread']<=64 and r['blockX']==1024 and r['dynamicSharedMemory']==47104
                           and r['localMemoryPerThread']==0 for r in row['runtime'])
        rt.append(row)
for mode in ['off','on']:
    for target in ['identifyNeighborsGridPrimitiveSharedQueryShared','compressResultMask']:
        a=next(r for r in rt if r['build']==f'build_{mode}_a0' and r['target']==target)
        b=next(r for r in rt if r['build']==f'build_{mode}_r1' and r['target']==target)
        assert a['normalized_sass_sha256']==b['normalized_sass_sha256'],'Host-only release fix changed CUDA code'

owned=[]
real=next(c for c in cases() if c['name']=='cifar4096')
threshold_hex='0d'+struct.pack('>d',real['reference_threshold']).hex().upper()
for method in ['tc','fp32']:
    capture=json.loads((H/'results'/f'inner_{method}_runtime_bind_r3.json').read_text())
    assert capture['correctness']['exact_contract_pass']
    for obj in capture['objects'].values():
        cubin=next(r for name,r in obj['files'].items() if name.endswith('.cubin'))
        ptx=next(r for name,r in obj['files'].items() if name.endswith('.ptx'))
        assert sha(P/cubin['path'])==cubin['sha256'] and sha(P/ptx['path'])==ptx['sha256']
        text=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(P/cubin['path'])],text=True)
        destination=H/'raw'/f"{method}_{obj['hash']}.sass";assert not destination.exists();destination.write_text(text)
        matches=[(name,inst) for name,inst in functions(text).items() if obj['kernel'] in name];assert len(matches)==1
        symbol,inst=matches[0];row=dict(method=method,kernel=obj['kernel'],object_hash=obj['hash'],cubin=cubin,ptx=ptx,
            constants=obj['constants'],resources=obj['metadata'],**stats(inst))
        ops=row['opcodes'];assert not any(o.startswith(('LDL','STL')) for o in ops)
        if obj['kernel']=='analytic_certificate_ragged_safe_i64':assert any('IMMA' in o for o in ops)
        else:assert not any('MMA' in o for o in ops)
        if obj['kernel'] in ['classify_pedantic_panel','refine_ambiguous_fp64_i64']:
            assert threshold_hex.lower() in (P/ptx['path']).read_text().lower(),'Stored FP64 threshold not visible in PTX'
        assert any(obj['kernel'] in l['name'] for l in selected_launches[method])
        owned.append(row)
    expected={'gpu_quantize_metadata','analytic_certificate_ragged_safe_i64','certified_fp32_filter_i64','refine_ambiguous_fp64_i64'} if method=='tc' else {'gpu_norm_metadata','classify_pedantic_panel','refine_ambiguous_fp64_i64'}
    assert expected.issubset({r['kernel'] for r in owned if r['method']==method})

# Exact selected library function is freshly exported by NCU, not inferred from
# container/name equality with an earlier campaign.
source=H/'raw/fp32_runtime_bind_r3_source.csv';csvrows=list(csv.reader(source.open()))
assert csvrows[0][0]=='Kernel Name' and 'magma_sgemmEx_kernel' in csvrows[0][1]
addresses={int(r[0],16) for r in csvrows[2:]};base=min(addresses);inst=[];rebased=[0]
for r in csvrows[2:]:
    s=' '.join(r[1].split())
    if op(s).split('.')[0] in ['BRA','BSSY']:
        def fix(m):
            value=int(m.group(0),16)
            if value in addresses:rebased[0]+=1;return f'rel+0x{value-base:x}'
            return m.group(0)
        s=re.sub(r'0x[0-9a-fA-F]+',fix,s)
    inst.append(s)
lib=dict(function=csvrows[0][1],**stats(inst),rebased_control_targets=rebased[0],source_sha256=sha(source))
assert lib['opcodes'].get('FFMA',0)>0 and not any('MMA' in o or 'TF32' in o for o in lib['opcodes'])
lib['runtime']=[r for r in selected_launches['fp32'] if 'magma_sgemmEx_kernel' in r['name']]
assert len(lib['runtime'])==2
assert all((r['gridX'],r['gridY'],r['blockX'],r['blockY'])==(64,64,8,16) for r in lib['runtime'])
legacy=json.loads((P/'g16_gpu_preparation_20260905/results/precision_audit.json').read_text())
lib['same_g16_selected_function']=lib['normalized_sass_sha256']==legacy['selected_gemm_cases'][0]['normalized_sass_sha256']
lib['containers']={name:sha(name) for name in legacy['library_containers_sha256']}
lib['same_g16_containers']=lib['containers']==legacy['library_containers_sha256']
lib['parent_code_object']='Unresolved; selected runtime function freshly exported by NCU, not guessed from static symbols.'

# Equality of diagnostic work is checked per actual input, never by requiring
# a byte-identical cross-run point permutation.
g18=json.loads((P/'g18_rthiss_conservative_repair_20260905/results/closure_checks.json').read_text())
matrix=json.loads((H/'results/inner_rt_on_matrix_none_r1.json').read_text())
work=[]
for r in matrix['rows']:
    previous=next(v for v in g18['full_operator_observations'] if v['case']==r['case'])['work']
    expected=[previous[k] for k in ['safe_reject','safe_accept','terminal_accept','terminal_reject','fp32_dimensions','fp64_dimensions']]
    assert r['predicate_counters']==expected
    work.append(dict(case=r['case'],iteration=r['iteration'],counters=expected))

result=dict(passed=True,rt_selected=rt,owned_actual_compiled_objects=owned,library_selected=lib,
    nsys_operation_launches=selected_launches,nvtx_ranges=ranges,rt_work_equals_g18=work,
    threshold_constant_hex=threshold_hex,performance_from_profiler=False,optix_driver_jit_identity=False,
    scope='Runtime-observed CUDA code and actual owned compiled-object binding; no OptiX code identity or profiler-speed claim.')
write(H/'results/code_audit.json',result)
print(json.dumps(dict(passed=True,rt_functions=len(rt),owned_functions=len(owned),
    library_instructions=lib['instructions'],library_same_g16=lib['same_g16_selected_function'])),flush=True)
