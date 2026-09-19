import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CONDITION_DIM=21
HIDDEN=64


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


class Solver(nn.Module):
    def __init__(self,family,normalizer):
        super().__init__();assert family in ['direct','fm'];self.family=family
        for name in ['mean','scale','threshold_mean','threshold_scale']:
            self.register_buffer(name,torch.as_tensor(normalizer[name],dtype=torch.float32))
        self.condition_layer=nn.Linear(CONDITION_DIM,HIDDEN)
        self.state_weight=nn.Parameter(torch.empty(HIDDEN))
        nn.init.uniform_(self.state_weight,-1/math.sqrt(CONDITION_DIM+1),1/math.sqrt(CONDITION_DIM+1))
        if family=='fm':self.time_weight=nn.Parameter(torch.zeros(HIDDEN))
        self.hidden=nn.Linear(HIDDEN,HIDDEN)
        self.head=nn.Linear(HIDDEN,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def conditions(self,features,threshold):
        base=(features-self.mean)/self.scale
        tt=torch.full((len(base),1),float(threshold),device=base.device,dtype=base.dtype)
        return torch.cat([base,(tt-self.threshold_mean)/self.threshold_scale],dim=1)

    def embedding(self,conditions):return self.condition_layer(conditions)

    def field(self,embedding,state,t):
        h=embedding+state.reshape(-1,1)*self.state_weight
        if self.family=='fm':
            if isinstance(t,torch.Tensor):t=t.reshape(-1,1)
            h=h+t*self.time_weight
        return self.head(F.silu(self.hidden(F.silu(h)))).reshape(-1)

    def forward(self,conditions,state,t=0.):
        value=self.field(self.embedding(conditions),state,t)
        return value if self.family=='fm' else state+value


def parameter_count(model):return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def initialize_matched(direct,fm):
    for name,p in direct.named_parameters():dict(fm.named_parameters())[name].copy_(p)
    fm.time_weight.zero_()


@torch.no_grad()
def heun(field,z,steps,start=0.,end=1.,trajectory=False):
    assert steps>0
    dt=(end-start)/steps;states=[z.clone()] if trajectory else None
    for k in range(steps):
        t=start+k*dt;a=field(z,t);b=field(z+dt*a,t+dt)
        z=z+(dt*.5)*(a+b)
        if trajectory:states.append(z.clone())
    return (z,torch.stack(states)) if trajectory else z


def load_model(path,device='cuda'):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    state=obj['state_dict']
    normalizer={k:state[k] for k in ['mean','scale','threshold_mean','threshold_scale']}
    model=Solver(obj['family'],normalizer);model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model,features,initial,threshold,steps=4,chunk=65536):
    values=[]
    for off in range(0,len(features),chunk):
        f=torch.from_numpy(np.ascontiguousarray(features[off:off+chunk],dtype=np.float32)).to('cuda')
        z=torch.from_numpy(np.ascontiguousarray(initial[off:off+chunk],dtype=np.float32)).to('cuda')
        c=model.conditions(f,threshold)
        # This affine term is independent of ODE state/time and is computed once.
        embedded=model.embedding(c)
        if model.family=='fm':z=heun(lambda zz,t:model.field(embedded,zz,t),z,steps)
        else:z=z+model.field(embedded,z,0.)
        values.append(z.cpu().numpy().copy())
    return np.concatenate(values) if values else np.empty(0,np.float32)
