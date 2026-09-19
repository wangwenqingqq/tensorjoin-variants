"""One isolated method process; full-operator tests and strictly gated timing."""
import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from common import *
from common_r1 import check_r1
from operators_r1 import make

def cache_hashes():
    root=Path(os.environ['TRITON_CACHE_DIR'])
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.suffix in ['.cubin','.ptx']}

def main():
    a=argparse.ArgumentParser();a.add_argument('--method',choices=['rt_on','rt','tc','fp32'],required=True)
    a.add_argument('--kind',choices=['matrix','safety','stress','profile','screen'],required=True)
    a.add_argument('--record-id',required=True);args=a.parse_args()
    assert args.record_id.replace('_','').isalnum()
    assert platform.node()=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES']=='3'
    assert os.environ['NVIDIA_TF32_OVERRIDE']=='0' and os.environ['OMP_NUM_THREADS']=='8'
    check_frozen();check_r1();root=H/'raw'/args.record_id;root.mkdir(exist_ok=False)
    cache=H/'artifacts'/'cache'/args.record_id;assert not cache.exists()
    os.environ['TRITON_CACHE_DIR']=str(cache);cache.mkdir(parents=True)
    torch.set_num_threads(1);torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32=False
    inputs=cases();loaded={c['name']:load(c) for c in inputs}
    byname={c['name']:c for c in inputs}
    rows=[];engine_records=[];passed=False
    r=dict(method=args.method,kind=args.kind,record_id=args.record_id,pid=os.getpid(),
           torch_version=torch.__version__,gpu=str(torch.cuda.get_device_properties(0)),
           source_manifest_sha256=sha(H/'artifacts/frozen_execution.json'),public_timing=args.kind=='screen')
    try:
        if args.kind=='screen':
            gate=json.loads((H/'results/admission_r1.json').read_text());assert gate['passed']
            assert json.loads((H/'results/code_audit.json').read_text())['passed']
        cycles=2 if args.kind=='stress' and args.method=='rt' else 1
        for cycle in range(cycles):
            start=time.perf_counter();engine=make(args.method)
            engine_record=dict(cycle=cycle,initialization_seconds_diagnostic=time.perf_counter()-start)
            if args.method=='fp32':engine_record['cublas']=engine.blas.record()
            if args.method in ['rt','rt_on']:engine_record['library_sha256']=sha(engine.path)
            if args.kind=='matrix':names=[c['name'] for c in inputs];names+=list(reversed(names)) if args.method.startswith('rt') else []
            else:names=['cifar4096','boundary_zero32']
            # Exact shapes/threshold constants compile outside any public timer.
            for name in dict.fromkeys(names):engine.prepare(byname[name]['n'],byname[name])
            if args.kind=='stress':
                names=['cifar4096','boundary_zero32']*(25 if args.method=='rt' else 16)
            elif args.kind=='safety':names=['cifar4096','boundary_zero32','boundary_zero32','cifar4096']
            elif args.kind=='profile':names=['cifar4096','cifar4096']
            elif args.kind=='screen':names=['cifar4096']*7
            # Preparation did not retain any method-specific input or output.
            engine_record['after_prepare_memory']=memory()
            memory_rows=[]
            for i,name in enumerate(names):
                case=byname[name];source,oracle=loaded[name]
                # Independent owned source allocations alternate addresses; the
                # previous source stays alive until this clone has been allocated.
                x=source.copy();x.flags.writeable=False
                before_source=hashlib.sha256(x.tobytes()).hexdigest()
                target=root/f'c{cycle}_i{i}_{name}'
                if args.method=='rt_on':target.mkdir()
                if args.kind=='screen' and i==2:engine_record['cache_before_timing']=cache_hashes()
                torch.cuda.synchronize()
                if args.kind=='profile':torch.cuda.nvtx.range_push(f'G19_{args.method}_{name}_call{i}')
                t0=time.perf_counter_ns()
                output,work=engine.run(x,case,target if args.method=='rt_on' else None)
                torch.cuda.synchronize()
                # A materialized owned host output is returned before stopping.
                elapsed=(time.perf_counter_ns()-t0)*1e-9
                if args.kind=='profile':torch.cuda.nvtx.range_pop()
                row=dict(cycle=cycle,iteration=i,case=name,host_pointer=x.ctypes.data,
                         output_pointer=output.ctypes.data,work=work,audit=validate(output,oracle,len(x)))
                if args.kind=='screen':row['warmup']=i<2;row['public_seconds']=elapsed if i>=2 else None
                assert row['audit']['exact'],row
                assert hashlib.sha256(x.tobytes()).hexdigest()==before_source==case['source_sha256']
                if args.method=='rt_on':
                    sys.path.insert(0,str(P/'g17_rthiss_pair_contract_20260905/src'))
                    from validate_export import validate as validate_export
                    row['independent_decoder']=validate_export(target,case,P)
                    assert row['independent_decoder']['adapter_structure_pass'] and row['independent_decoder']['frozen_reference_pass']
                    assert row['independent_decoder']['unique_rt_candidate_pairs']==len(x)**2
                    counters=np.fromfile(target/'g18_stats.u64',dtype='<u8')
                    row['predicate_counters']=counters.tolist()
                    assert int(counters[:4].sum())==len(x)**2
                    assert int(counters[1]+counters[2])==len(output)
                elif args.kind in ['matrix','safety']:
                    output.astype('<u8',copy=False).tofile(root/f'c{cycle}_i{i}_{name}.u64')
                del output,work
                gc.collect()
                mem=memory();row['after_call_memory']=mem;memory_rows.append(mem)
                rows.append(row)
                with (root/'calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if args.kind=='screen':
                engine_record['cache_after_timing']=cache_hashes()
                assert engine_record['cache_before_timing']==engine_record['cache_after_timing']
            if args.kind=='stress':
                stable=memory_rows[2:]
                engine_record['device_used_range']=max(m['cuda_used'] for m in stable)-min(m['cuda_used'] for m in stable)
                engine_record['rss_range']=max(m['rss_bytes'] for m in stable)-min(m['rss_bytes'] for m in stable)
                assert engine_record['device_used_range']<=16*1024**2
                assert engine_record['rss_range']<=64*1024**2
                if args.method in ['tc','fp32']:assert max(m['torch_allocated'] for m in stable)==min(m['torch_allocated'] for m in stable)==0
            start=time.perf_counter();engine.close();del engine;gc.collect();torch.cuda.synchronize()
            engine_record['destruction_seconds_diagnostic']=time.perf_counter()-start
            engine_record['after_destroy_memory']=memory();engine_records.append(engine_record)
        passed=True
    except Exception as e:r['error']=repr(e)
    r.update(rows=rows,engine_records=engine_records,cache=cache_hashes(),
             correctness=dict(exact_contract_pass=passed),performance_admitted=False,novelty_pass=False)
    write(H/'results'/f'inner_{args.record_id}.json',r)
    print(json.dumps(dict(record_id=args.record_id,passed=passed,calls=len(rows),error=r.get('error'))),flush=True)
    return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())

