"""Full ID coverage and ground-truth empty-tile ceiling; CPU, not latency."""
import hashlib,json,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from screen import HERE,ROOT,N,D,B,THRESHOLD,hadamard,sha

def main():
    started=time.time();p=HERE/'artifacts/reference_ids_u64.bin'
    assert sha(p)=='13cae87e6a0f75a13f3e9a7881b33c42ab77680f6d13e99dcc00fa2de5963495'
    ids=np.fromfile(p,dtype='<u8');assert len(ids)==3926078 and np.all(ids[1:]>ids[:-1])
    ref=json.loads((HERE/'results/reference_a0.json').read_text())
    guard=json.loads((HERE/'results/reference_a0_guard.json').read_text())
    assert ref['pass'] and guard['pass'] and guard['exit_code']==0
    structural=json.loads((HERE/'results/structural_a0.json').read_text());assert structural['complete']
    masks=np.load(HERE/'artifacts/masks.npz');orders=np.load(HERE/'artifacts/orders.npz');tr,tc=masks['tr'],masks['tc'];M=(N+B-1)//B
    assert len(tr)==440391
    result={'started':started,'reference_sha256':sha(p),'reference_count':len(ids),'reference_pid':guard['pid'],'rows':[],'orders':[]}
    occupancy={}
    for key in orders.files:
        order=orders[key];inv=np.empty(N,dtype=np.int64);inv[order]=np.arange(N)
        rr=inv[(ids//N).astype(np.int64)]//B;cc=inv[(ids%N).astype(np.int64)]//B
        r,c=np.minimum(rr,cc),np.maximum(rr,cc)
        occupied=np.zeros(M*M,dtype=bool);occupied[r*M+c]=True
        occ=occupied.reshape(M,M)[tr,tc]
        occupancy[key]=occ
        result['orders'].append({'order':key,'answer_occupied_tiles':int(occ.sum()),'ideal_empty_tiles':int((~occ).sum()),'ideal_empty_tile_fraction':float((~occ).mean()),'ceiling_scope':'ground truth; zero-cost perfect tile-rejection oracle, not achievable runtime'})
        for name in masks.files:
            if not name.startswith(key+'__'):continue
            mask=masks[name];assert mask.dtype==bool and mask.shape==occ.shape
            bad=np.count_nonzero(mask&occ)
            dense=np.zeros((M,M),dtype=bool);dense[tr,tc]=mask
            false_neg=int(np.count_nonzero(dense[r,c]))
            assert bad==false_neg==0,(name,bad,false_neg)
            result['rows'].append({'name':name,'pruned_tiles':int(mask.sum()),'false_negative_directed_ids':false_neg,'false_pruned_answer_tiles':int(bad),'pass':True})
    np.savez_compressed(HERE/'artifacts/oracle_occupancy.npz',**occupancy)
    # Independent exact-coordinate transform replay on real data.
    x=np.load(ROOT/'data/g2b_cifar60000/vectors_f32.npy',mmap_mode='r')
    rng=np.random.default_rng(20260908);sel=rng.choice(N,8,replace=False);y,e=hadamard(x[sel])
    errors=[]
    for i,src in enumerate(sel):
        for k in [0,1,17,511]:
            actual=sum(((-1 if (k&j).bit_count()%2 else 1)*Fraction(float(x[src,j])) for j in range(D)),Fraction(0))
            diff=abs(Fraction(float(y[i,k]))-actual)
            assert diff<=Fraction(float(e[i]))
            errors.append(float(diff))
    result['fraction_transform_checks']={'count':len(errors),'maximum_error':max(errors),'pass':True}
    result['pass']=True;result['ended']=time.time();result['structural_gate_pass']=bool(structural['any_count_gate_pass'])
    result['gpu_candidate_admitted']=result['structural_gate_pass']
    result['decision']='ADMIT_BOUNDED_GPU_PROTOTYPE' if result['structural_gate_pass'] else 'STOP_FIXED_DIRECTIONAL_ENVELOPE_FRONTEND'
    (HERE/'results/audit_a0.json').open('x').write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
