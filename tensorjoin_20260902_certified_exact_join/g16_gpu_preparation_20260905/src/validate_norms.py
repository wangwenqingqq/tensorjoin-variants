"""Independent norm enclosure tests, including the whole public input."""
import json
from pathlib import Path
import numpy as np
import torch
from g2b_public_common import load_pageable_source, atomic_json, sha256_file
from validate_metadata import data_cases
from gpu_norms import prepare_norm_device

HERE = Path(__file__).resolve().parents[1]
source = load_pageable_source()
records = []
for name, vectors in data_cases(source):
    gpu = tuple(x.cpu().numpy() for x in prepare_norm_device(torch.from_numpy(vectors).to('cuda')))
    # CPU extended precision is an independent positive-sum reference.
    reference = np.sum(vectors.astype(np.longdouble)**2, axis=1, dtype=np.longdouble)
    n,r,u = [x.astype(np.longdouble) for x in gpu]
    norm_bad = int(np.count_nonzero((n-r > reference) | (n+r < reference)))
    upper_bad = int(np.count_nonzero(u*u < reference))
    record = dict(name=name,n=len(vectors),norm_enclosure_violations=norm_bad,l2_upper_violations=upper_bad,
                  all_pass=norm_bad==upper_bad==0,
                  max_center_relative_error=float(np.max(np.abs(n-reference)/np.maximum(reference,np.longdouble(2)**-1022))))
    records.append(record)
    print('NORM_CASE '+json.dumps(record),flush=True)
    if not record['all_pass']:
        break
atomic_json(HERE/'results/norm_metadata_a0.json', dict(records=records,
            longdouble_precision=np.finfo(np.longdouble).nmant,
            kernel_sha256=sha256_file(HERE/'src/gpu_norms.py'),
            scope='metadata enclosure, not complete join',
            correctness=dict(exact_contract_pass=len(records)==7 and all(x['all_pass'] for x in records))))
