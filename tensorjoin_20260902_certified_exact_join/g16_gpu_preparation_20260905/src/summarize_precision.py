"""Bind actual full-operator launches to exported selected-function SASS."""

import collections
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path

from g2b_public_common import atomic_json, sha256_file

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent


def opcode(source):
    source = re.sub(r'^@!?[A-Z0-9.]+\s+', '', source.strip())
    return source.split()[0].rstrip(';')


def extra_audit(result):
    old = PROJECT/'g15_strong_control_20260905'
    old_precision = json.loads((old/'results/precision_audit.json').read_text())
    assert result['library_containers_sha256'] == old_precision['library_containers_sha256']
    assert {x['normalized_sass_sha256'] for x in result['selected_gemm_cases']} == {x['normalized_sass_sha256'] for x in old_precision['selected_gemm_cases']}
    assert {x['cubin_sha256'] for x in result['classifier_variants']} == {x['cubin_sha256'] for x in old_precision['classifier_variants']}
    tc = json.loads((HERE/'results/gpu_prepare_compatibility_full_a0.json').read_text())
    oldtc = json.loads((old/'results/chunked_compatibility_full_a0.json').read_text())
    work_keys = ['scheduled_tiles','batch_count','g3b_direct_accept_upper_pairs',
       'g3b_ambiguous_upper_pairs','fp32_direct_accept_upper_pairs','fp32_direct_reject_upper_pairs',
       'fp32_bitwise_equal_upper_pairs','fp64_refined_upper_pairs','accepted_upper_pairs','overflow_events']
    for k in work_keys: assert tc[k] == oldtc[k], k
    key='selected_cache_artifacts_after_timer'
    for name, entry in tc[key].items():
        for suffix in ('cubin','ptx'): assert entry[suffix+'_sha256']==oldtc[key][name][suffix+'_sha256'],name
    new_metadata=[]
    for kernel,label in [('gpu_quantize_metadata','tc_full_a0'),('gpu_norm_metadata','fp32_full_a0')]:
        cache=PROJECT/f'artifacts/g5_g16_{label}_triton_cache'
        files=list(cache.rglob(kernel+'.cubin')); assert len(files)==1
        cubin=files[0]; text=subprocess.check_output(['nvdisasm',str(cubin)],text=True)
        output=HERE/f'raw/{kernel}.sass'
        with output.open('x') as f: f.write(text)
        instructions=[]
        for line in text.splitlines():
            m=re.search(r'/\*[0-9a-f]+\*/\s*(.*?);',line)
            if m: instructions.append(' '.join(m.group(1).split()))
        counts=collections.Counter(opcode(x) for x in instructions)
        assert instructions and any(x.startswith('DADD') for x in counts)
        spills={k:v for k,v in counts.items() if k.split('.')[0] in ('LDL','STL')}
        assert not spills and not any('MMA' in k for k in counts)
        meta=json.loads(cubin.with_suffix('.json').read_text())
        new_metadata.append(dict(kernel=kernel,cubin=str(cubin.relative_to(PROJECT)),cubin_sha256=sha256_file(cubin),
             ptx_sha256=sha256_file(cubin.with_suffix('.ptx')),resources=meta,instruction_count=len(instructions),
             static_opcodes=dict(counts),local_traffic=spills,sass_sha256=sha256_file(output)))
    db=sqlite3.connect(HERE/'artifacts/nsys_tc_a0.sqlite')
    tc_launches=db.execute("""select s.value,k.registersPerThread,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
       k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k
       join StringIds s on s.id=k.demangledName group by s.value,k.registersPerThread,k.gridX,k.gridY,k.gridZ,
       k.blockX,k.blockY,k.blockZ,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread""").fetchall()
    tc_names=collections.Counter()
    for row in tc_launches: tc_names[row[0]]+=row[-1]
    assert tc_names['gpu_quantize_metadata']==2
    assert result['nsys_name_counts']['gpu_norm_metadata']==2
    for label in ('tc_memcheck_a0','metadata_synccheck_a0'):
        assert 'ERROR SUMMARY: 0 errors' in (PROJECT/f'raw/g5_g16_{label}.log').read_text()
    assert json.loads((HERE/'results/safety_gates_r1.json').read_text())['complete']
    return dict(same_g15_library_and_selected_functions=True,same_g5_stage_code_and_work=True,
       tc_work={k:tc[k] for k in work_keys},metadata=new_metadata,tc_nsys_launches=tc_launches,
       tc_nsys_name_counts=dict(tc_names),tc_nsys_sqlite_sha256=sha256_file(HERE/'artifacts/nsys_tc_a0.sqlite'),
       lifecycle_limit='early screen only; no formal or sustained60K/Graph/arbitrary stream admission')


def main():
    assert json.loads((HERE/'results/precision_collection.json').read_text())['complete']
    db = sqlite3.connect(HERE/'artifacts/nsys_public_a0.sqlite')
    launches = db.execute('''select s.value, k.registersPerThread, k.gridX,k.gridY,k.gridZ,
        k.blockX,k.blockY,k.blockZ,k.staticSharedMemory,k.dynamicSharedMemory,
        k.localMemoryPerThread,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id=k.demangledName group by s.value,k.registersPerThread,
        k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.staticSharedMemory,
        k.dynamicSharedMemory,k.localMemoryPerThread''').fetchall()
    names = collections.Counter()
    for row in launches:
        names[row[0]] += row[-1]
    dot_names = {name: count for name, count in names.items() if 'sgemm' in name.lower()}
    assert len(dot_names) == 1 and sum(dot_names.values()) == 123
    assert names['classify_pedantic_panel'] == 123
    assert names['refine_ambiguous_fp64_i64'] == 121
    cases = []
    for i in range(3):
        path = HERE/f'raw/ncu_shape{i}_a0_source.csv'
        rows = list(csv.reader(path.open()))
        assert rows[0][0] == 'Kernel Name' and rows[0][1] in dot_names
        header = rows[1]
        # NCU source operands contain process-relocated branch destinations.
        # Rebase only observed BRA/BSSY destinations that are real PCs in this function.
        addresses = {int(row[0], 16) for row in rows[2:]}
        base_address = min(addresses)
        relocation_count = 0
        instructions = []
        for row in rows[2:]:
            source = ' '.join(row[1].split())
            if opcode(source).split('.')[0] in ('BRA', 'BSSY'):
                def rebase(match):
                    nonlocal relocation_count
                    value = int(match.group(0), 16)
                    if value in addresses:
                        relocation_count += 1
                        return f'rel+0x{value-base_address:x}'
                    return match.group(0)
                source = re.sub(r'0x[0-9a-fA-F]+', rebase, source)
            instructions.append(source)
        assert relocation_count == 72, relocation_count
        histogram = collections.Counter(opcode(s) for s in instructions)
        dynamic = collections.Counter()
        index = header.index('Instructions Executed')
        for row in rows[2:]:
            dynamic[opcode(row[1])] += int(row[index].replace(',', '') or '0')
        forbidden = {op: count for op, count in histogram.items() if 'MMA' in op or 'TF32' in op}
        assert not forbidden and histogram['FFMA'] > 0
        normalized = '\n'.join(instructions) + '\n'
        raw = list(csv.reader((HERE/f'raw/ncu_shape{i}_a0_raw.csv').open()))
        metrics = dict(zip(raw[0], raw[2]))
        cases.append({'shape_index': i, 'function': rows[0][1],
                      'source_csv_sha256': sha256_file(path),
                      'normalized_sass_sha256': hashlib.sha256(normalized.encode()).hexdigest(),
                      'instruction_count': len(instructions), 'rebased_control_targets': relocation_count, 'static_opcodes': dict(histogram),
                      'dynamic_warp_pc_instructions': dict(dynamic), 'forbidden_mma_tf32': forbidden,
                      'ncu_launch_metrics': metrics})
    # The selected function must be identical across all three full/tail shapes.
    assert len({c['normalized_sass_sha256'] for c in cases}) == 1
    own = []
    cache = PROJECT/'artifacts/g5_g16_fp32_full_a0_triton_cache'
    for i, cubin in enumerate(sorted(cache.rglob('classify_pedantic_panel.cubin'))):
        text = subprocess.check_output(['nvdisasm', str(cubin)], text=True)
        output = HERE/f'raw/classifier_{i}.sass'
        with output.open('x') as handle:
            handle.write(text)
        instructions = []
        for line in text.splitlines():
            match = re.search(r'/\*[0-9a-f]+\*/\s*(.*?);', line)
            if match:
                instructions.append(' '.join(match.group(1).split()))
        histogram = collections.Counter(opcode(s) for s in instructions)
        assert instructions and not any('MMA' in op for op in histogram)
        spills = {op: n for op, n in histogram.items() if op.split('.')[0] in ('LDL', 'STL')}
        assert not spills
        own.append({'cubin': str(cubin.relative_to(PROJECT)), 'cubin_sha256': sha256_file(cubin),
                    'normalized_sass_sha256': hashlib.sha256(('\n'.join(instructions)+'\n').encode()).hexdigest(),
                    'instruction_count': len(instructions), 'static_opcodes': dict(histogram), 'local_traffic': spills})
    assert len(own) == 3
    terminal = list(cache.rglob('refine_ambiguous_fp64_i64.cubin'))
    assert len(terminal) == 1
    assert sha256_file(terminal[0]) == 'c2b3362119abfbeb3df85e96788705f6ff4c79414e3a74e2d5d5f3355eff00f8'
    compatibility = json.loads((HERE/'results/fp32_gpu_compatibility_full_a0.json').read_text())
    library = Path(compatibility['library']['library'])
    containers = {str(library): sha256_file(library)}
    other = library.with_name('libcublasLt.so.13')
    if other.is_file():
        containers[str(other.resolve())] = sha256_file(other)
    assert containers[str(library)] == compatibility['library']['library_sha256']
    for label in ('fp32_memcheck_a0', 'fp32_synccheck_a0'):
        raw = (PROJECT/f'raw/g5_g16_{label}.log').read_text()
        assert 'ERROR SUMMARY: 0 errors' in raw, label
    result = {'precision_gate_pass': True, 'scope': 'frozen FP32 full/tail dispatch and new classifier only',
              'nsys_launches': launches, 'nsys_name_counts': dict(names),
              'nsys_sqlite_sha256': sha256_file(HERE/'artifacts/nsys_public_a0.sqlite'),
              'library_containers_sha256': containers, 'selected_gemm_cases': cases,
              'classifier_variants': own, 'terminal_matches_g5_p4': True,
              'normalization': 'v2: remove address column; rebase 72 BRA/BSSY targets only when target is an observed function PC; preserve all other operands/predicates; collapse whitespace',
              'parent_code_object': 'unresolved; actual selected function exported by NCU, not guessed from static symbols',
              'sanitizer_scope': 'tested device access and synchronization; leak-free shutdown not claimed',
              'performance_from_profiler': False}
    result['g16_extra'] = extra_audit(result)
    atomic_json(HERE/'results/precision_audit.json', result)
    print(json.dumps({'precision_gate_pass': True, 'gemm_instructions': cases[0]['instruction_count'],
                      'gemm_ffma': cases[0]['static_opcodes']['FFMA'], 'gemm_hash': cases[0]['normalized_sass_sha256'],
                      'classifier_variants': len(own), 'all_full_dot_launches': sum(dot_names.values())}))


if __name__ == '__main__':
    main()
