import numpy as np
import torch
from torch import nn

class ScoreMap(nn.Module):
    def __init__(self,family,mean,std):
        super().__init__()
        self.family=family
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        if family=='linear':
            self.residual=nn.Linear(32,1)
            last=self.residual
        elif family=='mlp':
            self.residual=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
            last=self.residual[-1]
        else:
            raise ValueError(family)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    def forward(self,q):
        norm=(q-self.mean)/self.std
        return norm[:,0]+self.residual(norm).squeeze(-1)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def load_model(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['state_dict']
    model=ScoreMap(saved['family'],state['mean'],state['std'])
    model.load_state_dict(state)
    return model.to(device).eval()

def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

@torch.no_grad()
def mapped_scores(model,q,device='cuda'):
    # This function defines host-to-host mapping, with weights already loaded.
    return model(torch.from_numpy(np.ascontiguousarray(q,dtype=np.float32)).to(device)).cpu().numpy().copy()
