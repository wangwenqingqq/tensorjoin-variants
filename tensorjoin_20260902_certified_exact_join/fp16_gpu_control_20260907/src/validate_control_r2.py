"""GPU predicate admission with saved metadata, intervals and terminal outputs."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
import platform
import time
import traceback
import numpy as np
import torch
from control_r2 import Control, HERE, ROOT, N, D, CAP, T, GAMMA, full_source, output_check


def hash_ids(a):return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()


def check_metadata(v,half,md,x,tag):
    selected=list(range(32))+[255,256,511,57344,59999]
    z=half.cpu().numpy()
    expected=x.astype(np.float16);expected[np.abs(expected)<2**-14]=0
    assert np.array_equal(z,expected),'FP16 conversion/FTZ mismatch'
    m=[a.cpu().numpy().copy() for a in md];checks=[]
    for i in selected:
        a=[F(float(t)) for t in x[i]];b=[F(float(t)) for t in z[i]]
        nx=sum(t*t for t in a);nz=sum(t*t for t in b);nr=sum((q-r)**2 for q,r in zip(a,b))
        lo,hi,lz,er=[F(float(q[i])) for q in m]
        ok=lo<=nx<=hi and lz*lz>=nz and er*er>=nr
        checks.append({'row':i,'pass':ok})
    np.savez_compressed(HERE/'results'/f'{tag}_metadata.npz',rows=selected,x=x[selected],z=z[selected],
                        lo=m[0][selected],hi=m[1][selected],lz=m[2][selected],er=m[3][selected])
    assert all(t['pass'] for t in checks),'metadata bound failed'
    return m,checks


def interval_audit(dots,md,row,col,lower,upper):
    nxlo,nxhi,lz,er=[x.astype(np.float64) for x in md]
    rows,cols=dots.shape;violations=0
    for off in range(0,rows,64):
        i=np.arange(row+off,row+min(off+64,rows))[:,None];j=np.arange(col,col+cols)[None,:]
        p=dots[off:off+len(i)].astype(np.float64)
        b=GAMMA*lz[i]*lz[j]+er[i]*lz[j]+lz[i]*er[j]+er[i]*er[j]+2048*2**-126
        R=2*b+(nxhi[i]+nxhi[j])*2**-38+1.0000001044244144e-12
        lo=np.maximum(nxlo[i]+nxlo[j]-2*p-R,0);hi=nxhi[i]+nxhi[j]-2*p+R
        violations+=int(np.count_nonzero(lower[off:off+len(i)]>lo)+np.count_nonzero(upper[off:off+len(i)]<hi))
    rng=np.random.default_rng(202609073)
    cells=[(i,j) for i in range(min(rows,16)) for j in range(min(cols,16))]
    cells += [(int(i),int(j)) for i,j in zip(rng.integers(rows,size=128),rng.integers(cols,size=128))]
    exact=[]
    for i,j in cells:
        a,b=row+i,col+j;p=F(float(dots[i,j]))
        l1,l2=F(float(md[2][a])),F(float(md[2][b]));e1,e2=F(float(md[3][a])),F(float(md[3][b]))
        rad=F(float(GAMMA))*l1*l2+e1*l2+l1*e2+e1*e2+F(2048,2**126)
        il,jl=F(float(md[0][a])),F(float(md[0][b]));ih,jh=F(float(md[1][a])),F(float(md[1][b]))
        R=2*rad+(ih+jh)*F(1,2**38)+F(1.0000001044244144e-12)
        lo=max(F(0),il+jl-2*p-R);hi=ih+jh-2*p+R
        ok=F(float(lower[i,j]))<=lo and F(float(upper[i,j]))>=hi
        exact.append({'i':i,'j':j,'pass':ok})
    return {'all_cell_fp64_expression_violations':violations,'exact_rational_checks':exact,
            'pass':violations==0 and all(x['pass'] for x in exact)},cells


def one_dataset(o,x,panels,tag,record,hold):
    v=torch.from_numpy(x).to('cuda');assert v.data_ptr() not in [q.data_ptr() for q in hold];hold.append(v)
    half,md=o.prepare(v);hostmd,checks=check_metadata(v,half,md,x,tag)
    dr={'tag':tag,'input_pointer':v.data_ptr(),'metadata_checks':checks,'panels':[]};record['datasets'].append(dr)
    out=torch.empty(CAP,device='cuda',dtype=torch.int64);amb=torch.empty_like(out);fp64=torch.empty_like(out)
    for index,(row,col,rows,cols) in enumerate(panels):
        dots=torch.empty((rows,cols),device='cuda');o.blas.gemm(half[row:row+rows],half[col:col+cols],dots)
        c=torch.zeros(4,device='cuda',dtype=torch.int32)
        lo,hi=o.classify(dots,md,out,amb,c,row,col,dump=True)
        first=c.cpu().numpy().copy();direct,n1,over,reject=map(int,first)
        direct_ids=np.sort(out[:direct].cpu().numpy().copy());amb_ids=np.sort(amb[:n1].cpu().numpy().copy())
        # The non-dump production program must give exactly the same classes.
        cprod=torch.zeros_like(c);o.classify(dots,md,out,amb,cprod,row,col,dump=False)
        assert np.array_equal(cprod.cpu().numpy(),first)
        assert np.array_equal(np.sort(out[:direct].cpu().numpy()),direct_ids)
        assert np.array_equal(np.sort(amb[:n1].cpu().numpy()),amb_ids)
        interval,cells=interval_audit(dots.cpu().numpy(),hostmd,row,col,lo.cpu().numpy(),hi.cpu().numpy())
        sample_i=np.array([i for i,j in cells]);sample_j=np.array([j for i,j in cells])
        np.savez_compressed(HERE/'results'/f'{tag}_panel{index}_interval.npz',i=sample_i+row,j=sample_j+col,
                            p=dots.cpu().numpy()[sample_i,sample_j],lo=lo.cpu().numpy()[sample_i,sample_j],
                            hi=hi.cpu().numpy()[sample_i,sample_j],
                            metadata=np.array([[m[sample_i+row],m[sample_j+col]] for m in hostmd]))
        offsets=np.arange(rows*cols,dtype=np.int64);ii=row+offsets//cols;jj=col+offsets%cols
        ids=ii[ii<=jj]*N+jj[ii<=jj];assert direct+n1+reject==len(ids) and over==0
        rp=torch.from_numpy(ids).to('cuda');rc=torch.zeros(6,device='cuda',dtype=torch.int32)
        reference=torch.empty_like(rp);o.terminal(v,rp,reference,rc,len(ids))
        truth=np.sort(reference[:int(rc[0].item())].cpu().numpy().copy());assert rc[2].item()==0
        c2=torch.zeros(6,device='cuda',dtype=torch.int32);c2[0]=direct
        o.stage2(v,amb,out,fp64,c2,n1);mid=c2.cpu().numpy().copy()
        fp64_ids=np.sort(fp64[:int(mid[3])].cpu().numpy().copy())
        o.terminal(v,fp64,out,c2,int(mid[3]));last=c2.cpu().numpy().copy();assert last[2]==0
        got=np.sort(out[:int(last[0])].cpu().numpy().copy())
        good=np.array_equal(truth,got) and not np.setdiff1d(direct_ids,truth).size and not np.setdiff1d(truth,np.r_[direct_ids,amb_ids]).size
        pr={'offset':[row,col],'shape':[rows,cols],'pairs':len(ids),'count':len(got),'hash':hash_ids(got),
            'direct':direct,'ambiguous':n1,'reject':reject,'fp64_inputs':len(fp64_ids),'interval':interval,
            'exact_output':bool(good),'pass':bool(good and interval['pass'])}
        dr['panels'].append(pr)
        np.savez_compressed(HERE/'results'/f'{tag}_panel{index}_ids.npz',reference=truth,actual=got,direct=direct_ids,ambiguous=amb_ids,fp64=fp64_ids)
        print(json.dumps({k:v for k,v in pr.items() if k!='interval'}),flush=True)
        assert pr['pass'],('panel failed',tag,index)
    return o.capture()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixture','public','full'],required=True);p.add_argument('--label',required=True);a=p.parse_args()
    dest=HERE/'results'/f'{a.label}.json';assert not dest.exists()
    r={'label':a.label,'mode':a.mode,'started':time.time(),'pid':os.getpid(),'host':platform.node(),'pass':False,'datasets':[],'speed_claim':False,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')};o=None;hold=[]
    try:
        assert r['host']=='gpu-host-8' and os.environ['CUDA_VISIBLE_DEVICES'] in ['0','2']
        o=Control();r.update(original_identities=o.identities,binding=o.blas.record(),libraries=o.library_hashes)
        hold=[]
        if a.mode=='fixture':
            x=np.zeros((N,D),np.float32);x[:512]=np.load(ROOT/'fp16_dense_gate0_20260907/results/fixture_a0_fixture512.npy')
            for i in range(2):r['compiled']=one_dataset(o,x,[(0,0,512,512)],a.label+f'_churn{i}',r,hold)
        elif a.mode=='public':
            assert json.loads((HERE/'results/gpu0_fixture_a2.json').read_text())['pass']
            r['compiled']=one_dataset(o,full_source(),[(0,0,4096,4096),(0,57344,4096,2656),(57344,57344,2656,2656)],a.label,r,hold)
        else:
            assert json.loads((HERE/'results/gpu0_public_a2.json').read_text())['pass']
            out,rec=o.run('C',full_source());rec['hash']=output_check(out);rec['count']=len(out)
            r['full']=rec;r['compiled']=o.capture()
        r['pass']=True
    except Exception:r['exception']=traceback.format_exc();print(r['exception'],flush=True)
    finally:
        hold.clear()
        if o is not None:o.close()
        o=None
        import gc
        gc.collect();torch.cuda.empty_cache();r['ended']=time.time()
        with dest.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({'pass':r['pass'],'path':str(dest)}),flush=True)
    return 0 if r['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
