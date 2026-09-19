"""Build a typed bridge against the exact installed ATen allocator API."""
import subprocess
import torch
from common import *
package=Path(torch.__file__).resolve().parent
command=['c++','-O2','-std=c++17','-shared','-fPIC',
         '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch.compiled_with_cxx11_abi())),
         '-I'+str(package/'include'),'-I'+str(package/'include/torch/csrc/api/include'),
         str(H/'src/host_allocator.cpp'),'-L'+str(package/'lib'),'-ltorch_cpu','-lc10',
         '-Wl,-rpath,'+str(package/'lib'),'-o',str(H/'artifacts/host_allocator.so')]
assert not (H/'artifacts/host_allocator.so').exists()
write(H/'artifacts/host_allocator_build.json',dict(command=command,torch_version=torch.__version__,
    headers={str(package/'include'/f):sha(package/'include'/f) for f in ['ATen/core/CachingHostAllocator.h','ATen/cuda/CachingHostAllocator.h']}))
subprocess.run(command,check=True)
