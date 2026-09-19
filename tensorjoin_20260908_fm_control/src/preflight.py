"""Analytic ODE, assignment and ragged-grouping checks; no model selection."""
import os
import math
import torch
from scipy.optimize import linear_sum_assignment
from models import heun,Transport,initialize_matched,configure,parameter_count
from grouping import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    configure()
    z=torch.linspace(-2,2,128,dtype=torch.float64).reshape(4,32)
    shift=torch.linspace(-.2,.3,32,dtype=torch.float64)
    constant=lambda x,t:shift.expand_as(x)
    const=heun(constant,z,8)
    constant_error=float(torch.max(torch.abs(const-z-shift)))
    assert constant_error<1e-14
    linear=lambda x,t:.5*x
    errors={n:float(torch.max(torch.abs(heun(linear,z,n)-z*math.exp(.5)))) for n in [8,16,32]}
    assert errors[8]>3.7*errors[16] and errors[16]>3.7*errors[32]
    reverse=heun(linear,heun(linear,z,32),32,start=1.,end=0.)
    reverse_error=float(torch.max(torch.abs(reverse-z)))
    assert reverse_error<2e-6
    direct=Transport('direct',np.zeros(32),1.,1.)
    fm=Transport('fm',np.zeros(32),1.,1.)
    initialize_matched(direct,fm)
    assert parameter_count(direct)==24864 and parameter_count(fm)==24992
    assert torch.equal(direct(z.float()),z.float())
    assert torch.equal(heun(fm,z.float(),8),z.float())
    cost=np.array([[9.,1.,9.],[9.,9.,1.],[1.,9.,9.]])
    rows,cols=linear_sum_assignment(cost)
    assert np.array_equal(rows,np.arange(3)) and np.array_equal(cols,[1,2,0])
    sizes=[1,63,64,65,129,1000]
    rng=np.random.default_rng(2026090830)
    for n in sizes:
        coords=rng.normal(size=(n,32))
        for rule in RULES:
            perm=permutation(coords,rule)
            assert np.array_equal(np.sort(perm),np.arange(n))
            leaves=[perm[i:i+64] for i in range(0,n,64)]
            assert len(leaves)==(n+63)//64 and all(len(x)==64 for x in leaves[:-1])
    assert not torch.cuda.is_initialized()
    save_json(HERE/'artifacts/preflight.json',dict(pass_=True,
        constant_field_max_error=constant_error,linear_field_errors=errors,
        reverse_linear_max_error=reverse_error,identity_initializations=True,
        exact_assignment_known_case=True,ragged_sizes=sizes,
        method='Heun really evaluates a time-dependent field twice per step.'))
    print(json.dumps(dict(preflight_pass=True,linear_errors=errors)),flush=True)


if __name__=='__main__':main()
