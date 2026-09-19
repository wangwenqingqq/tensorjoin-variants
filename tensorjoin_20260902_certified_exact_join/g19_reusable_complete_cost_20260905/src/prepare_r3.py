"""Generate additive owned-cuBLAS R3 sources; never alter a frozen predecessor."""
from common import *
import difflib
s=(P/'src/run_h2b_p1_pedantic_correctness.py').read_text()
original=s[s.index('class CuBLAS:'):s.index('\n\n@dataclass\nclass PedanticState:')]
s=original.replace('class CuBLAS:','class OwnedCuBLAS:')
old='''        self.handle_value = int(torch.cuda.current_blas_handle())
        self.handle = ctypes.c_void_p(self.handle_value)'''
new='''        signatures = {
            'cublasCreate_v2': [ctypes.POINTER(handle_type)],
            'cublasDestroy_v2': [handle_type],
            'cublasSetStream_v2': [handle_type, ctypes.c_void_p],
            'cublasGetStream_v2': [handle_type, ctypes.POINTER(ctypes.c_void_p)],
            'cublasSetWorkspace_v2': [handle_type, ctypes.c_void_p, ctypes.c_size_t],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments; function.restype = ctypes.c_int
        torch_library = ctypes.CDLL(str(Path(torch.__file__).resolve().parent / 'lib/libtorch_cuda.so'))
        query = getattr(torch_library, '_ZN2at4cuda22getChosenWorkspaceSizeEv')
        query.argtypes = []; query.restype = ctypes.c_size_t
        self.workspace_bytes = int(query())
        assert self.workspace_bytes == 8519680, self.workspace_bytes
        self.workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device='cuda')
        self.stream = int(torch.cuda.current_stream().cuda_stream)
        self.handle = handle_type()
        self._check(self.library.cublasCreate_v2(ctypes.byref(self.handle)), 'cublasCreate_v2')
        self.handle_value = int(self.handle.value)
        self.closed = False
        self._check(self.library.cublasSetStream_v2(self.handle, ctypes.c_void_p(self.stream)), 'cublasSetStream_v2')
        self._check(self.library.cublasSetWorkspace_v2(self.handle, ctypes.c_void_p(self.workspace.data_ptr()), self.workspace_bytes), 'cublasSetWorkspace_v2')
        selected_stream = ctypes.c_void_p()
        self._check(self.library.cublasGetStream_v2(self.handle, ctypes.byref(selected_stream)), 'cublasGetStream_v2')
        assert int(selected_stream.value or 0) == self.stream'''
assert old in s;s=s.replace(old,new)
s=s.replace('        if query.dtype != torch.float32',"        assert not self.closed and int(torch.cuda.current_stream().cuda_stream) == self.stream\n        if query.dtype != torch.float32")
s=s.replace('            "handle": self.handle_value,','            "handle": self.handle_value,\n            "ownership": "engine-owned",\n            "workspace_bytes": self.workspace_bytes,\n            "stream": self.stream,')
s+='''\n    def close(self):
        assert not self.closed
        torch.cuda.synchronize()
        self._check(self.library.cublasDestroy_v2(self.handle), 'cublasDestroy_v2')
        self.closed = True
        self.workspace = None
'''
imports='''"""Typed owned-handle counterpart of the immutable H2B/G16 GEMM binding."""
import ctypes
import ctypes.util
import os
from pathlib import Path
import torch
from run_h2b_p1_pedantic_correctness import (sha256_file, CUBLAS_STATUS_SUCCESS,
    CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_POINTER_MODE_HOST, CUBLAS_PEDANTIC_MATH,
    CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT)

'''
(H/'src/owned_cublas_r3.py').write_text(imports+s)
(H/'artifacts/owned_cublas_r3.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='immutable_CuBLAS',tofile='OwnedCuBLAS_R3')))
s=(H/'src/operators_r2.py').read_text().replace('class FP32:', 'from owned_cublas_r3 import OwnedCuBLAS\n\nclass FP32:')
s=s.replace('self.blas=fp.BoundCuBLAS()','self.blas=OwnedCuBLAS()').replace('        # The handle is borrowed from PyTorch, not owned by this wrapper.','        self.blas.close()')
(H/'src/operators_r3.py').write_text(s)
for name in ['run_slot','supervise']:
    s=(H/'src'/f'{name}_r2.py').read_text().replace('operators_r2','operators_r3').replace('run_slot_r2','run_slot_r3').replace('admission_r2.json','admission_r3.json')
    s=s.replace('from common_r2 import check_r2','from common_r2 import check_r2\nfrom common_r3 import check_r3').replace('check_r2();','check_r2();check_r3();')
    (H/'src'/f'{name}_r3.py').write_text(s)
s=(H/'src/admit_r2.py').read_text()
s=s.replace('from common_r2 import check_r2','from common_r2 import check_r2\nfrom common_r3 import check_r3').replace('check_r2();','check_r2();check_r3();')
s=s.replace('admission_r2.json','admission_r3.json',1).replace('admission_r1.json','admission_r2.json')
s=s.replace("'tc_safety_memcheck_r1'","'fp32_safety_memcheck_r2'")
s=s.replace("if r['passed']];assert len(records)==6","if r['passed'] and not (r['method']=='fp32' and r['kind']=='matrix')];assert len(records)==6")
s=s.replace("[('tc','safety','memcheck'),('fp32','safety','memcheck')]","[('fp32','matrix','none'),('fp32','safety','memcheck')]")
s=s.replace('admission_r2_slots','admission_r3_slots').replace("{tool}_r2'","{tool}_r3'").replace('supervise_r2.py','supervise_r3.py').replace('ADMISSION_R2_END','ADMISSION_R3_END')
(H/'src/admit_r3.py').write_text(s)
(H/'src/common_r3.py').write_text('''from common import *
def check_r3():
    for path,value in json.loads((H/'artifacts/frozen_r3.json').read_text()).items():assert sha(P/path)==value,path
''')
(H/'src/freeze_r3.py').write_text('''from common import *
from common_r1 import check_r1
from common_r2 import check_r2
check_frozen();check_r1();check_r2()
files=[H/'ADDENDUM_OWNED_CUBLAS_R3.md',H/'artifacts/owned_cublas_r3.diff']
files += [H/'src'/n for n in ['prepare_r3.py','owned_cublas_r3.py','operators_r3.py','run_slot_r3.py','supervise_r3.py','common_r3.py','freeze_r3.py','admit_r3.py']]
write(H/'artifacts/frozen_r3.json',{str(p.relative_to(P)):sha(p) for p in files})
print('FROZEN_R3',len(files))
''')
print('Prepared additive R3 sources')
