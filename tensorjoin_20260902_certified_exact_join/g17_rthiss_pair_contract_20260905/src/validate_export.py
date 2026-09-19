"""Independent bitmap decoding, RT coverage, and native/reference predicate audit."""
import ctypes,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist,squareform

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def validate(directory,case,project):
    p=Path(directory);receipt=json.loads((p/'export.json').read_text());n=case['n']
    read=lambda name:np.fromfile(p/name,dtype='<u4')
    point=read('point_map.u32');dims=read('dimension_map.u32')
    assert np.array_equal(np.sort(point),np.arange(n)) and np.array_equal(np.sort(dims),np.arange(512))
    x=np.fromfile(project/case['path'],dtype='<f4').reshape(n,512)
    actual=np.fromfile(p/'pairs.u64',dtype='<u8');decoded=[];coverage=np.zeros((n,n),bool);batches=[]
    for b in range(receipt['batches']):
        get=lambda name:read(f'b{b}_{name}.u32')
        mask,bits,groups,prefix,counts,starts,candidates,grouped=map(get,('mask','compressed','groups','prefix','counts','starts','candidates','grouped'))
        setbits=np.flatnonzero(np.unpackbits(mask.view(np.uint8),bitorder='little')).astype(np.uint64)
        assert np.array_equal(np.sort(bits.astype(np.uint64)),setbits), 'compressed/raw bitmap mismatch'
        assert np.array_equal(np.sort(grouped),np.arange(n))
        batch=[]
        for i,c in enumerate(counts):
            ps=grouped[groups[i]:groups[i+1]];qs=candidates[starts[i]:starts[i]+c]
            assert int(prefix[i+1]-prefix[i])==(len(ps)*int(c)+31)//32
            coverage[np.ix_(point[qs],point[ps])]=True
            # Primitive-local uncompressed matrix is independent of compressed decoder.
            local=np.unpackbits(mask[prefix[i]:prefix[i+1]].view(np.uint8),bitorder='little')
            assert not local[len(ps)*int(c):].any(),'padding bit set'
            if c:
                pi,qi=np.nonzero(local[:len(ps)*int(c)].reshape(len(ps),int(c)))
                ids=point[qs[qi]].astype(np.uint64)*n+point[ps[pi]].astype(np.uint64)
                batch.extend(ids.tolist())
        decoded.extend(batch);batches.append(dict(batch=b,accepted_bits=len(setbits),candidate_entries=len(candidates),mask_words=len(mask)))
    reconstructed=np.sort(np.asarray(decoded,dtype=np.uint64))
    assert np.array_equal(actual,reconstructed),'original-ID decoder mismatch'
    assert len(actual)==receipt['pair_count'] and (len(actual)<2 or np.all(actual[1:]>actual[:-1]))
    assert not len(actual) or actual[-1]<n*n
    assert coverage.reshape(-1)[actual.astype(np.int64)].all()
    distances=squareform(pdist(x.astype(np.float64),metric='sqeuclidean'))
    frozen=np.flatnonzero(distances.reshape(-1)<=case['reference_threshold']).astype(np.uint64)
    native=np.flatnonzero(distances.reshape(-1)<=receipt['native_threshold_d2']).astype(np.uint64)
    if case['oracle_path']:
        original=np.load(project/case['oracle_path']);assert np.array_equal(frozen,original),'Frozen G2A oracle drift'
    missing=np.setdiff1d(frozen,actual,assume_unique=True);extra=np.setdiff1d(actual,frozen,assume_unique=True)
    native_missing=np.setdiff1d(native,actual,assume_unique=True);native_extra=np.setdiff1d(actual,native,assume_unique=True)
    candidate_misses=missing[~coverage.reshape(-1)[missing.astype(np.int64)]]
    lib=ctypes.CDLL('libm.so.6');fma=lib.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
    replay=[]
    for kind,ids in [('missing',missing[:32]),('extra',extra[:32])]:
        for pair in ids:
            a,b=divmod(int(pair),n);total=0.;step=0
            for d in dims:
                delta=ctypes.c_float(float(x[a,d])-float(x[b,d])).value
                total=fma(delta,delta,total);step+=1
                if (step%4==0 or step==512) and total>receipt['native_threshold_d2']:break
            decision=total<=receipt['native_threshold_d2']
            replay.append(dict(kind=kind,pair=int(pair),row=a,column=b,rt_covered=bool(coverage[a,b]),
                fp64_distance=float(distances[a,b]),native_fp32_sum=float(total),replayed_dims=step,
                native_cpu_decision=decision,observed_gpu_decision=kind=='extra',
                native_cpu_matches_observed=decision==(kind=='extra')))
    return dict(adapter_structure_pass=True,all_original_ids_decoded=True,output_pairs=len(actual),output_sha256=digest(actual),
        frozen_oracle_pairs=len(frozen),frozen_oracle_sha256=digest(frozen),frozen_reference_pass=not(len(missing) or len(extra)),
        missing_pairs=len(missing),extra_pairs=len(extra),first_missing=missing[:32].tolist(),first_extra=extra[:32].tolist(),
        native_threshold_reference_pass=not(len(native_missing) or len(native_extra)),native_threshold_oracle_pairs=len(native),
        native_threshold_missing=len(native_missing),native_threshold_extra=len(native_extra),
        unique_rt_candidate_pairs=int(coverage.sum()),all_pair_count=n*n,rt_candidate_missing_frozen_pairs=len(candidate_misses),
        first_rt_candidate_missing=candidate_misses[:32].tolist(),self_pairs=int(np.count_nonzero(actual//n==actual%n)),
        reference_threshold=case['reference_threshold'],native_threshold=receipt['native_threshold_d2'],
        scalar_fmaf_replays=replay,replay_complete=len(missing)<=32 and len(extra)<=32,
        batches=batches,precision_scope='scipy FP64 direct-distance reference; native scalar C fmaf diagnostic; not exact-real proof')
