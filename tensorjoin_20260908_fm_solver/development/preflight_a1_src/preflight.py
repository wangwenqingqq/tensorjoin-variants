import os
import traceback
import torch
from problem import *
from certificate import minima,certified_reject,lower_bounds
from models import Solver,configure,initialize_matched,parameter_count,heun,predict


def capture(kernel):
    if kernel is None:return None
    assert kernel.n_spills==0
    result={}
    import hashlib
    stem='certificate_'+hashlib.sha256(kernel.asm['cubin']).hexdigest()[:16]
    for ext in ['cubin','ptx','llir','ttgir','ttir']:
        data=kernel.asm[ext];data=data if isinstance(data,bytes) else data.encode()
        p=HERE/'artifacts'/(stem+'.'+ext)
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
        result[ext]=sha(p)
    return result


def cpu_min(q,tr,tc):
    return np.asarray([np.min(np.sum((q[r*64:min(N,(r+1)*64),None,:]-q[None,c*64:min(N,(c+1)*64),:])**2,axis=-1)) for r,c in zip(tr,tc)])


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4';configure()
    result=dict(started=time.time(),pass_=False,certificates=[])
    # Analytic ODE checks use FP64 and a nontrivial initial vector.
    zz=torch.tensor([-.7,.2,1.3],device='cuda',dtype=torch.float64)
    constant=heun(lambda z,t:torch.ones_like(z)*.3,zz,8)
    assert torch.max(abs(constant-(zz+.3))).item()<1e-14
    errors=[]
    for steps in [2,4,8,16]:
        z=heun(lambda z,t:.5*z,zz,steps)
        errors.append(float(torch.max(abs(z-zz*np.exp(.5))).item()))
    assert all(errors[i+1]<errors[i]/3.4 for i in range(3))
    back=heun(lambda z,t:.5*z,z,16,start=1.,end=0.)
    assert torch.max(abs(back-zz)).item()<1e-5
    result['analytic_heun_errors']=errors
    prep=json.loads((HERE/'artifacts/preparation.json').read_text())
    q=np.load(HERE/'artifacts/features_f64.npy')[np.load(HERE/'artifacts/permutation.npy')]
    margin=prep['numerical_margin']['error'];scale2=prep['scale2']
    qg=torch.from_numpy(np.ascontiguousarray(q.T,dtype=np.float32)).to('cuda')
    rng=np.random.default_rng(2026090902)
    tr=rng.integers(0,NB,128,dtype=np.int32);tc=rng.integers(0,NB,128,dtype=np.int32)
    tr=np.r_[tr,[0,NB-1,0,NB-2]].astype(np.int32)
    tc=np.r_[tc,[0,NB-1,NB-1,NB-1]].astype(np.int32)
    gpu,dims,kernel=minima(qg,tr,tc);assert np.all(dims==32)
    truth=cpu_min(q,tr,tc)
    lower=lower_bounds(gpu,margin,scale2)
    assert np.all(lower<=truth/scale2+1e-13)
    assert np.all(abs(gpu.astype(np.float64)-truth)<=margin)
    result['actual_projection_checks']=dict(tiles=len(tr),max_abs_error=float(abs(gpu-truth).max()),margin=margin,kernel=capture(kernel))
    x=source()[np.load(HERE/'artifacts/permutation.npy')].astype(np.float64)
    raw_truth=cpu_min(x,tr[:24],tc[:24])
    assert np.all(lower[:24]<=raw_truth+1e-13)
    for c in CELLS:
        early,used,kernel=minima(qg,tr,tc,c['T'],margin,scale2)
        assert np.array_equal(certified_reject(early,c['T'],margin,scale2),certified_reject(gpu,c['T'],margin,scale2))
        assert set(used.tolist())<={8,16,32}
        result['certificates'].append(dict(cell=c['name'],dimensions=used.tolist(),kernel=capture(kernel)))
    del x,qg
    # Synthetic identities, tiny differences, large dynamic range, ragged tails.
    synthetic=np.zeros((N,32),np.float64)
    synthetic[:64]=rng.uniform(-1,1,(64,32)).astype(np.float32)
    synthetic[64:128]=synthetic[:64]+np.float32(2**-20)
    synthetic[128:192]=rng.normal(0,1e-20,(64,32)).astype(np.float32)
    synthetic[192:256]=np.float32(.75)
    synthetic[-32:]=rng.uniform(-1,1,(32,32)).astype(np.float32)
    # Treat the represented FP32 rows as the original 32-coordinate input.
    synthetic=synthetic.astype(np.float32).astype(np.float64)
    sm=numerical_margin(synthetic)['error']
    tr=np.asarray([0,0,1,2,2,3,0,1,2,3,NB-1],np.int32)
    tc=np.asarray([0,1,2,2,3,3,NB-1,NB-1,NB-1,NB-1,NB-1],np.int32)
    qg=torch.from_numpy(np.ascontiguousarray(synthetic.T,dtype=np.float32)).to('cuda')
    gpu,_,kernel=minima(qg,tr,tc);truth=cpu_min(synthetic,tr,tc)
    assert np.all(lower_bounds(gpu,sm,1.)<=truth+1e-13)
    boundary_checks=0
    for threshold in np.unique(np.r_[0.,truth,np.nextafter(truth,-np.inf),np.nextafter(truth,np.inf)]):
        early,_,kernel=minima(qg,tr,tc,float(threshold),sm,1.)
        rejected=certified_reject(early,threshold,sm,1.)
        assert not np.any(rejected & (truth<=threshold))
        assert np.array_equal(rejected,certified_reject(gpu,threshold,sm,1.))
        boundary_checks+=len(tr)
    result['synthetic_certificate_checks']=dict(tiles=len(tr),boundary_checks=boundary_checks,max_abs_error=float(abs(gpu-truth).max()),margin=sm)
    norm=json.loads((HERE/'artifacts/normalizer.json').read_text())
    torch.manual_seed(2026090903)
    direct=Solver('direct',norm).to('cuda');fm=Solver('fm',norm).to('cuda');initialize_matched(direct,fm)
    assert parameter_count(direct)==5697 and parameter_count(fm)==5761
    features=np.load(HERE/'artifacts/pair_features.npy')[:129]
    coarse=np.load(HERE/'artifacts/coarse_state.npy')[:129]
    z0=initial_state(coarse,CELLS[0]['T'])
    assert np.array_equal(predict(direct,features,z0,CELLS[0]['T']),z0)
    assert np.array_equal(predict(fm,features,z0,CELLS[0]['T'],steps=4),z0)
    # Check condition caching on a nonzero field, including a time column.
    with torch.no_grad():
        fm.head.weight.normal_(0,.03);fm.time_weight.normal_(0,.1)
        f=torch.from_numpy(features).to('cuda');z=torch.from_numpy(z0).to('cuda')
        c=fm.conditions(f,CELLS[0]['T']);embedded=fm.embedding(c)
        a=heun(lambda zz,t:fm(c,zz,t),z,4)
        b=heun(lambda zz,t:fm.field(embedded,zz,t),z,4)
        assert torch.equal(a,b)
    result.update(pass_=True,ended=time.time(),parameters=dict(direct=5697,fm=5761),cached_condition_affine_bitwise_equal=True,
        zero_models_recover_data_initial_state=True)
    save_json(HERE/'artifacts/preflight.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save_json(HERE/'results'/f'preflight_failed_{int(time.time())}.json',dict(exception=traceback.format_exc(),time=time.time()))
        raise
