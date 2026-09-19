"""Diagnostic launcher: load retained cubins without Triton compilation."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
MANIFEST = json.loads((HERE/'frozen_manifest.json').read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def verify():
    out = {}
    for key, spec in MANIFEST.items():
        for ext, expected in spec['hashes'].items():
            p = ROOT/(spec['stem']+ext)
            actual = sha(p)
            if actual != expected: raise RuntimeError(f'Frozen identity mismatch: {p}')
            out[str(p.relative_to(ROOT))] = actual
    return out

class Driver:
    def __init__(self):
        self.identities = verify()
        torch.cuda.init()
        assert torch.cuda.device_count() == 1
        torch.empty(1,device='cuda')
        assert torch.cuda.get_device_capability() == (12,0)
        self.lib = C.CDLL('libcuda.so.1')
        self.modules, self.functions, self.abi = [], {}, {}
        self.bind('cuModuleLoad',[C.POINTER(C.c_void_p),C.c_char_p])
        self.bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self.bind('cuFuncGetParamInfo',[C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)])
        self.bind('cuLaunchKernel',[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)])
        for key,spec in MANIFEST.items():
            meta=json.loads((ROOT/(spec['stem']+'.json')).read_text())
            assert meta['global_scratch_size']==meta['profile_scratch_size']==0
            assert meta['num_warps']==4 and meta['num_ctas']==1
            ptx=(ROOT/(spec['stem']+'.ptx')).read_text()
            decl=ptx.split('.visible .entry '+spec['name']+'(')[1].split(')')[0]
            kinds=re.findall(r'\.param \.(u64|f32)\b',decl)
            expect=['u64']*spec['pointer_args']+['f32']*spec['float_args']+['u64']*2
            assert kinds==expect and '.reqntid 128' in ptx
            module,fun=C.c_void_p(),C.c_void_p()
            self.check(self.lib.cuModuleLoad(C.byref(module),str(ROOT/(spec['stem']+'.cubin')).encode()),'module_load')
            self.check(self.lib.cuModuleGetFunction(C.byref(fun),module,spec['name'].encode()),'function')
            self.modules.append(module); self.functions[key]=fun
            info=[]; position=0
            for index,kind in enumerate(kinds):
                size=8 if kind=='u64' else 4
                position=(position+size-1)//size*size
                off,got=C.c_size_t(),C.c_size_t()
                self.check(self.lib.cuFuncGetParamInfo(fun,index,C.byref(off),C.byref(got)),'param_info')
                assert (off.value,got.value)==(position,size), (key,index,off.value,got.value,position,size)
                info.append({'index':index,'offset':off.value,'size':got.value,'type':kind}); position+=size
            self.abi[key]={'parameters':info,'block':[128,1,1],'dynamic_shared':spec['shared']}
    def bind(self,name,args):
        f=getattr(self.lib,name);f.argtypes=args;f.restype=C.c_int
    @staticmethod
    def check(rc,where):
        if rc: raise RuntimeError(f'CUDA Driver {where}: CUresult={rc}')
    def launch(self,key,grid,pointers,floats=()):
        spec=MANIFEST[key]
        assert len(pointers)==spec['pointer_args'] and len(floats)==spec['float_args']
        if grid==0:return
        assert 0<grid<2**31
        vals=[C.c_uint64(t.data_ptr()) for t in pointers]+[C.c_float(x) for x in floats]+[C.c_uint64(0),C.c_uint64(0)]
        params=(C.c_void_p*len(vals))(*(C.cast(C.byref(x),C.c_void_p) for x in vals))
        stream=torch.cuda.current_stream().cuda_stream
        self.check(self.lib.cuLaunchKernel(self.functions[key],grid,1,1,128,1,1,spec['shared'],stream,params,None),key)
