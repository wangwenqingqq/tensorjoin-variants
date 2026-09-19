import numpy as np
import torch
from torch import nn


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


class Transport(nn.Module):
    def __init__(self,family,mean,scale,target_std):
        super().__init__()
        assert family in ['direct','fm']
        self.family=family
        self.solver_steps=16 if family=='fm' else 0
        self.rule='first'
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(33 if family=='fm' else 32,128),nn.SiLU(),
            nn.Linear(128,128),nn.SiLU(),nn.Linear(128,32))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def normalize(self,q):
        return (q-self.mean)/self.scale

    def forward(self,z,t=None):
        if self.family=='fm':
            assert t is not None
            if isinstance(t,(float,int)):
                tt=torch.full((len(z),1),float(t),device=z.device,dtype=z.dtype)
            else:
                tt=t.reshape(-1,1)
            return self.net(torch.cat([z,tt],dim=1))
        return z+self.net(z)


@torch.no_grad()
def initialize_matched(direct,fm):
    fm.net[0].weight[:,:32].copy_(direct.net[0].weight)
    fm.net[0].weight[:,32].zero_()
    fm.net[0].bias.copy_(direct.net[0].bias)
    for layer in [2,4]:
        fm.net[layer].weight.copy_(direct.net[layer].weight)
        fm.net[layer].bias.copy_(direct.net[layer].bias)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def heun(field,z,steps,start=0.,end=1.,trajectory=False):
    assert steps>0
    dt=(end-start)/steps
    states=[z.clone()] if trajectory else None
    for k in range(steps):
        t=start+k*dt
        first=field(z,t)
        second=field(z+dt*first,t+dt)
        z=z+(0.5*dt)*(first+second)
        if trajectory:states.append(z.clone())
    return (z,torch.stack(states)) if trajectory else z


def load_model(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['state_dict']
    model=Transport(saved['family'],state['mean'],state['scale'],state['target_std'])
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def mapped_coordinates(model,q,device='cuda',steps=None):
    x=torch.from_numpy(np.ascontiguousarray(q,dtype=np.float32)).to(device)
    z=model.normalize(x)
    if model.family=='fm':
        z=heun(model,z,model.solver_steps if steps is None else steps)
    else:
        z=model(z)
    return z.cpu().numpy().copy()
