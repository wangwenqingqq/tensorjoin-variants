import gc
import os
import traceback
from executor import *

def remap_tests():
    rng = np.random.default_rng(2026090810)
    perm = rng.permutation(N).astype(np.int64)
    pg = torch.from_numpy(perm).to('cuda')
    records = []
    for count in [0,1,31,256,10007]:
        r, c = rng.integers(0,N,(2,count),dtype=np.int64)
        ids = np.minimum(r,c)*N+np.maximum(r,c)
        if count:
            ids[0] = N*N-1
        device_ids = torch.from_numpy(ids).to('cuda')
        buffer = torch.full((count+64,),-117,device='cuda',dtype=torch.int64)
        part = buffer[32:32+count]
        if count:
            k = remap_ids[(triton.cdiv(count,256),)](device_ids,pg,part,count,NN=N,B=256,num_warps=4)
            capture_remap(k)
        rr, cc = perm[ids//N],perm[ids%N]
        want = np.minimum(rr,cc)*N+np.maximum(rr,cc)
        assert np.array_equal(part.cpu().numpy(),want)
        assert torch.all(buffer[:32]==-117) and torch.all(buffer[32+count:]==-117)
        records.append(dict(count=count,pass_=True))
    return records

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    result = dict(started=time.time(),pass_=False,samples=[],diagnostics=[])
    try:
        selection = json.loads((SHARED/'artifacts/selection.json').read_text())
        assert selection['chosen'] == 'gpu_radix'
        assert json.loads((SHARED/'artifacts/admission.json').read_text())['pass']
        result['remap_tests'] = remap_tests()
        x = output.full_source()
        p = plan(x)
        op, radix = output.Matrix(),output.Radix()
        for cell in CELLS:
            for method in METHODS:
                for mode in MODES:
                    rec, kernel = call(op,radix,x,cell,method,mode,p,exact=True)
                    capture_remap(kernel)
                    result['samples'].append(rec)
                    print(json.dumps(rec),flush=True)
                rec, kernel = call(op,radix,x,cell,method,'geometric',p,exact=True,diagnostic=True)
                capture_remap(kernel)
                result['diagnostics'].append(rec)
                print(json.dumps(rec),flush=True)
            op.capture()
            gc.collect()
            torch.cuda.empty_cache()
        rec, kernel = call(op,radix,x,CELLS[0],'F8','geometric',p,exact=True,cold=True)
        result['cold_admission'] = rec
        capture_remap(kernel)
        result.update(pass_=True,compiled=op.capture(),ended=time.time(),cub_version=radix.version)
        result['pass'] = True
        save_json(HERE/'artifacts/admission.json',result)
        paths = [*sorted((HERE/'src').glob('*.py')), HERE/'PROTOCOL.md',HERE/'NUMERICAL_SCOPE.md',
            HERE/'targets_r2.json',HERE/'artifacts/selection.json',HERE/'artifacts/admission.json',
            *sorted((HERE/'artifacts').glob('cifar60000_*')), *sorted((HERE/'artifacts').glob('remap*')),
            *sorted((SHARED/'src').glob('*.py')),SHARED/'artifacts/output.so',
            SHARED/'artifacts/build.json',SHARED/'artifacts/selection.json',
            *sorted((output.OLD/'src').glob('*.py')),*sorted(output.OLD.glob('*manifest.json')),
            output.SWEEP/'artifacts/thresholds.json',output.SWEEP/'artifacts/reference_manifest.json']
        paths += list((output.SWEEP/'artifacts/compiled').glob('*'))
        freeze = {str(f):output.sha(f) for f in paths if f.is_file()}
        freeze.update({str(ROOT/f):h for f,h in op.identities.items()})
        save_json(HERE/'artifacts/timing_freeze.json',freeze)
        print(json.dumps(dict(admission_pass=True,configurations=len(result['samples']),
                              frozen_files=len(freeze))),flush=True)
    except Exception:
        result['pass'] = False
        result['exception'] = traceback.format_exc()
        result['ended'] = time.time()
        save_json(HERE/'results/admission_failed_a0.json',result)
        raise

if __name__ == '__main__':
    main()
