"""Independent direct-reference spot checks of histogram aggregation/bounds."""
import hashlib, json, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
SWEEP=ROOT/'precision_routing_20260908_threshold_sweep'

def main():
    started=time.time()
    result=json.loads((HERE/'results/probe.json').read_text())
    assert result['pass_'] and result['configuration_count']==288
    x=np.load(ROOT/'data/g2b_cifar60000/vectors_f32.npy',mmap_mode='r')
    refs={c['name']:np.load(SWEEP/'artifacts'/f'reference_{c["name"]}.npy',mmap_mode='r') for c in result['cells']}
    rng=np.random.Generator(np.random.PCG64(2026090817))
    checked=[]
    for layout in result['layouts']:
        order=np.load(HERE/'artifacts'/f'order_{layout}.npy')
        for size in [32,64,128]:
            groups=(len(order)+size-1)//size
            rr,cc=np.triu_indices(groups)
            bounds=np.load(HERE/'artifacts'/f'bounds_{layout}_{size}.npz')
            allcounts={c:np.load(HERE/'artifacts'/f'counts_{layout}_{size}_{c}.npz')['accepted'] for c in refs}
            # Random tiles plus diagonals, ragged edges, and tiles with the
            # greatest number of true positives. This selection is QA only.
            select=set(map(int,rng.choice(len(rr),8,replace=False)))
            select.update([0,len(rr)-1,groups-1])
            for c in ['k4','original','k1024']:
                select.update(map(int,np.argsort(allcounts[c])[-3:]))
            for idx in sorted(select):
                a=order[rr[idx]*size:min((rr[idx]+1)*size,len(order))]
                b=order[cc[idx]*size:min((cc[idx]+1)*size,len(order))]
                ids=(a.astype(np.uint64)[:,None]*len(order)+b[None,:]).reshape(len(a),len(b))
                if rr[idx]==cc[idx]:ids=ids[np.triu_indices(len(a))]
                else:ids=ids.ravel()
                for c,ref in refs.items():
                    pos=np.searchsorted(ref,ids)
                    valid=pos<len(ref)
                    matches=np.zeros(len(ids),dtype=bool)
                    matches[valid]=ref[pos[valid]]==ids[valid]
                    assert int(matches.sum())==int(allcounts[c][idx]),(layout,size,idx,c)
                # Direct FP64 CPU distances independently validate interval
                # coverage on selected blocks, separate from membership checks.
                diff=x[a].astype(np.float64)[:,None,:]-x[b].astype(np.float64)[None,:,:]
                d2=np.sum(diff*diff,axis=2)
                for family in ['box','sphere','pivot','combined']:
                    assert bounds[family+'_lower'][idx] <= float(d2.min()),(layout,size,idx,family,'lower')
                    assert bounds[family+'_upper'][idx] >= float(d2.max()),(layout,size,idx,family,'upper')
                checked.append(dict(layout=layout,size=size,row=int(rr[idx]),col=int(cc[idx]),pairs=len(ids)))
            print(json.dumps(dict(verified_layout=layout,size=size,blocks=len(select))),flush=True)
    report=dict(pass_=True,seconds=time.time()-started,blocks=len(checked),
        membership_checks=len(checked)*len(refs),bound_checks=len(checked)*4,
        selected_pair_evaluations=sum(z['pairs'] for z in checked),records=checked,
        scope='Independent sampled direct membership and FP64 interval QA; main census validates all block decisions.')
    with (HERE/'results/independent_verification.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
