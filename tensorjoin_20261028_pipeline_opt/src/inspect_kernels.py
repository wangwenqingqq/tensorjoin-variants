#!/usr/bin/env python3
"""Retain generated-code evidence without changing the timed benchmark source."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
import bench
import run_g2b_tensorjoin_public as baseline
from g2b_public_common import PROJECT, atomic_json, load_contract

out=PROJECT/'raw/kernel_static'
out.mkdir(parents=True,exist_ok=True)
records=[]
class Capture:
    def __init__(self,name,kernel):self.name,self.kernel=name,kernel
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            k=self.kernel[grid](*args,**kwargs)
            cubin=k.asm['cubin']
            path=out/(self.name+'.cubin')
            path.write_bytes(cubin)
            (out/(self.name+'.ptx')).write_text(k.asm['ptx'])
            sass=subprocess.check_output(['nvdisasm',str(path)],text=True)
            (out/(self.name+'.sass')).write_text(sass)
            normalized='\n'.join(line.strip() for line in sass.splitlines() if line.strip() and not line.lstrip().startswith('//'))
            records.append({'name':self.name,'registers':k.n_regs,'spills':k.n_spills,
                'shared_bytes':k.metadata.shared,'metadata':str(k.metadata),
                'cubin_sha256':hashlib.sha256(cubin).hexdigest(),
                'sass_text_sha256':hashlib.sha256(sass.encode()).hexdigest(),
                'normalized_sass_sha256':hashlib.sha256(normalized.encode()).hexdigest(),
                'ptx_target':[x for x in k.asm['ptx'].splitlines() if '.target' in x or '.version' in x]})
            return k
        return launch

for module,names in [(bench,['scan_device_start','filter_device_count','refine_device_count']),
    (baseline,['triangular_int8_certificate_compact_i64','filter_ambiguous_fp32_i64','refine_ambiguous_fp64_i64'])]:
    for name in names:setattr(module,name,Capture(name,getattr(module,name)))
threshold=float(load_contract()['threshold_d2'])
baseline.compile_exact_specializations(bench.CAPACITY,threshold)
bench.compile_candidates(threshold)
atomic_json(PROJECT/'results/kernel_static.json',records)
print(json.dumps(records,indent=2),flush=True)
