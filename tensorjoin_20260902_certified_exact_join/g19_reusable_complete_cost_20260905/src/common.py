"""Immutable inputs, validation and per-call evidence for G19."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parents[1];P=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.asarray(x,dtype='<u8').tobytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def check_frozen():
    for path,value in json.loads((H/'artifacts/frozen_execution.json').read_text()).items():
        assert sha(P/path)==value,path
def cases():
    return json.loads((H/'artifacts/inputs.json').read_text())['inputs']
def load(case):
    assert sha(P/case['path'])==case['source_sha256']
    assert sha(P/case['g19_oracle_path'])==case['g19_oracle_sha256']
    x=np.fromfile(P/case['path'],dtype='<f4').reshape(case['n'],512).copy()
    reference=np.fromfile(P/case['g19_oracle_path'],dtype='<u8')
    assert np.isfinite(x).all() and np.max(np.abs(x))<=1 and x.flags.c_contiguous
    assert reference.size==case['g19_oracle_count']
    x.flags.writeable=False
    return x,reference
def validate(actual,reference,n):
    assert actual.dtype==np.uint64 and actual.flags.c_contiguous and actual.flags.owndata
    missing=np.setdiff1d(reference,actual);extra=np.setdiff1d(actual,reference)
    ordered=bool(np.all(actual[1:]>actual[:-1]))
    valid=bool(np.all(actual<np.uint64(n)*np.uint64(n)))
    return dict(pairs=len(actual),sha256=digest(actual),missing=len(missing),extra=len(extra),
                ordered_unique=ordered,in_range=valid,exact=bool(np.array_equal(actual,reference) and ordered and valid))
def memory():
    import torch
    torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
    status=Path('/proc/self/status').read_text()
    rss=int(next(l.split()[1] for l in status.splitlines() if l.startswith('VmRSS:')))*1024
    return dict(cuda_used=total-free,torch_allocated=torch.cuda.memory_allocated(),
                torch_reserved=torch.cuda.memory_reserved(),rss_bytes=rss)

